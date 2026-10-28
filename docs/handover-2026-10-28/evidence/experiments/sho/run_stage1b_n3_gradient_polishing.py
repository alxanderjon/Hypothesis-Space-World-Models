#!/usr/bin/env python3
"""Ultra-local safeguarded-Newton polishing from persisted 2-D N2 endpoints."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.n3_box_qp import dimension_general_box_qp_direction

from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    LOWER,
    UPPER,
    _posthoc_reference,
    _prefix_losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ORIGINAL_RESULTS = Path(
    "outputs/sho/stage1B/learned_powell_2d/per_trajectory_results.csv"
)
CONTINUATION_RESULTS = Path(
    "outputs/sho/stage1B/learned_powell_2d_continuation/continuation_per_trajectory.csv"
)
OUTPUT_DIR = Path("outputs/sho/stage1B/n3_gradient_polishing")
TRAJECTORY_COUNT = 256
SNAPSHOT_STEPS = (0, 1, 2, 5, 10, 20)
# Compatibility sentinel for older experiment wrappers.  The deterministic
# reference loop has no accepted-step resource stop.
MAX_ACCEPTED_STEPS = math.inf
MAX_BACKTRACKS = 52
CONDITION_LIMIT = 1.0e12
DAMPING_RELATIVE_FLOOR = 1.0e-12
MAX_DAMPING_TRIALS = 14
IMPROVEMENT_THRESHOLDS = (1.0e-20, 1.0e-24, 1.0e-28)
FLOAT64_STAGNATION_LOSS_DECREASE_ULPS_MAX = 512.0
FLOAT64_STAGNATION_FULL_PREDICTED_DECREASE_ULPS_MIN = 512.0
FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE = 3
FLOAT64_STAGNATION_STOP_REASON = "FLOAT64_STAGNATION_TAIL"


def _updated_float64_stagnation_count(
    previous_count: int,
    previous_loss: float,
    accepted_loss: float,
    full_predicted_decrease: float,
) -> int:
    """Update persistence after one accepted step; rejected trials never enter."""
    loss_spacing = abs(float(np.spacing(np.float64(previous_loss))))
    loss_decrease_ulps = (previous_loss - accepted_loss) / loss_spacing
    full_predicted_decrease_ulps = full_predicted_decrease / loss_spacing
    if (
        loss_decrease_ulps <= FLOAT64_STAGNATION_LOSS_DECREASE_ULPS_MAX
        and full_predicted_decrease_ulps
        > FLOAT64_STAGNATION_FULL_PREDICTED_DECREASE_ULPS_MIN
    ):
        return previous_count + 1
    return 0


@dataclass(slots=True)
class N2Endpoint:
    trajectory_id: int
    theta: np.ndarray
    saved_loss: float
    source: str
    category: str
    previous_plateau_case: bool


@dataclass(slots=True)
class PolishResult:
    trajectory_id: int
    endpoint: N2Endpoint
    theta_history: list[np.ndarray]
    loss_history: list[float]
    accepted_steps: int
    rejected_newton_steps: int
    actual_loss_evaluations: int
    backtracking_rejections: int
    accepted_with_backtracking: int
    hessian_evaluations: int
    damping_steps: int
    condition_numbers: list[float]
    damping_values: list[float]
    gradient_norms: list[float]
    stop_reason: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_endpoints() -> list[N2Endpoint]:
    original = _read_csv(ORIGINAL_RESULTS)
    continuation_rows = [
        row
        for row in _read_csv(CONTINUATION_RESULTS)
        if int(row["checkpoint_budget"]) == 1500
    ]
    continuation = {int(row["trajectory_id"]): row for row in continuation_rows}
    plateau_ids = {
        trajectory_id
        for trajectory_id, row in continuation.items()
        if row["final_failure_classification"] == "direction_update_stagnation"
    }
    if len(original) != TRAJECTORY_COUNT or len(continuation) != 70:
        raise RuntimeError("Persisted N2 endpoint counts do not match 256/70")
    if len(plateau_ids) != 6:
        raise RuntimeError(f"Expected six previous plateau cases, found {plateau_ids}")
    endpoints: list[N2Endpoint] = []
    for trajectory_id, original_row in enumerate(original):
        if trajectory_id in continuation:
            row = continuation[trajectory_id]
            theta = np.asarray((float(row["best_a"]), float(row["best_b"])))
            loss = float(row["best_prefix_loss"])
            source = "n2_continuation_endpoint"
        else:
            theta = np.asarray(
                (float(original_row["final_a"]), float(original_row["final_b"]))
            )
            loss = float(original_row["final_loss"])
            source = "n2_original_endpoint"
        if loss <= 1.0e-20:
            category = "n2_at_or_below_1e-20"
        elif loss <= 1.0e-18:
            category = "n2_1e-20_to_1e-18"
        else:
            category = "n2_above_1e-18"
        endpoints.append(
            N2Endpoint(
                trajectory_id=trajectory_id,
                theta=theta,
                saved_loss=loss,
                source=source,
                category=category,
                previous_plateau_case=trajectory_id in plateau_ids,
            )
        )
    return endpoints


def _scalar_loss(theta: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    return _prefix_losses(theta[None, :], position[None, :])[0]


def _gradient_hessian(
    theta: torch.Tensor, position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    differentiable = theta.detach().clone().requires_grad_(True)
    loss = _scalar_loss(differentiable, position)
    gradient = torch.autograd.grad(loss, differentiable, create_graph=True)[0]
    rows = [
        torch.autograd.grad(gradient[index], differentiable, retain_graph=True)[0]
        for index in range(2)
    ]
    hessian = torch.stack(rows)
    hessian = 0.5 * (hessian + hessian.T)
    return loss.detach(), gradient.detach(), hessian.detach()


def _newton_direction(
    gradient: torch.Tensor, hessian: torch.Tensor
) -> tuple[torch.Tensor | None, float, float, str]:
    if not bool(torch.isfinite(gradient).all() and torch.isfinite(hessian).all()):
        return None, math.inf, math.nan, "nonfinite_derivatives"
    singular_values = torch.linalg.svdvals(hessian)
    smallest = float(torch.min(singular_values))
    largest = float(torch.max(singular_values))
    condition = largest / smallest if smallest > 0.0 else math.inf
    eigenvalues = torch.linalg.eigvalsh(hessian)
    minimum_eigenvalue = float(torch.min(eigenvalues))
    scale = max(float(torch.max(torch.abs(eigenvalues))), torch.finfo(DTYPE).tiny)
    direct_allowed = minimum_eigenvalue > 0.0 and condition <= CONDITION_LIMIT
    initial_damping = 0.0
    if not direct_allowed:
        initial_damping = max(
            scale * DAMPING_RELATIVE_FLOOR,
            -minimum_eigenvalue + scale * DAMPING_RELATIVE_FLOOR,
        )
    identity = torch.eye(len(gradient), dtype=DTYPE)
    for attempt in range(MAX_DAMPING_TRIALS):
        if attempt == 0 and direct_allowed:
            damping = 0.0
        else:
            base = initial_damping if initial_damping > 0.0 else scale * DAMPING_RELATIVE_FLOOR
            damping = base * (10.0 ** (attempt if not direct_allowed else attempt - 1))
        try:
            direction = torch.linalg.solve(hessian + damping * identity, -gradient)
        except torch.linalg.LinAlgError:
            continue
        if bool(torch.isfinite(direction).all()) and float(torch.dot(gradient, direction)) < 0.0:
            return direction, condition, damping, "ok"
    return None, condition, math.nan, "no_descent_direction"


def _polish(endpoint: N2Endpoint, position: torch.Tensor) -> PolishResult:
    theta = torch.tensor(endpoint.theta, dtype=DTYPE)
    with torch.no_grad():
        current_loss = float(_scalar_loss(theta, position))
    theta_history = [theta.detach().cpu().numpy().copy()]
    loss_history = [current_loss]
    rejected_newton_steps = 0
    actual_loss_evaluations = 0
    backtracking_rejections = 0
    accepted_with_backtracking = 0
    hessian_evaluations = 0
    damping_steps = 0
    condition_numbers: list[float] = []
    damping_values: list[float] = []
    gradient_norms: list[float] = []
    stop_reason = "not_started"
    float64_stagnation_accepted_steps = 0

    while True:
        _, gradient, hessian = _gradient_hessian(theta, position)
        hessian_evaluations += 1
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        gradient_norms.append(gradient_norm)
        if not math.isfinite(gradient_norm):
            rejected_newton_steps += 1
            stop_reason = "nonfinite_gradient"
            break
        if gradient_norm == 0.0:
            rejected_newton_steps += 1
            stop_reason = "floating_point_zero_gradient"
            break
        direction, box_qp = dimension_general_box_qp_direction(
            theta, gradient, hessian, LOWER, UPPER, _newton_direction)
        condition = box_qp["condition"]
        damping = box_qp["damping"]
        direction_status = box_qp["route_status"]
        condition_numbers.append(condition)
        damping_values.append(damping)
        if damping > 0.0 and math.isfinite(damping):
            damping_steps += 1
        if direction is None:
            rejected_newton_steps += 1
            stop_reason = direction_status
            break

        directional_derivative = float(torch.dot(gradient, direction))
        full_predicted_decrease = -float(box_qp["quadratic_value"])
        accepted = False
        unchanged_candidate = False
        for backtrack in range(MAX_BACKTRACKS + 1):
            alpha = 2.0 ** (-backtrack)
            candidate = theta + alpha * direction
            if bool(torch.equal(candidate, theta)):
                unchanged_candidate = True
                break
            if not bool(torch.all(candidate >= LOWER) and torch.all(candidate <= UPPER)):
                backtracking_rejections += 1
                continue
            with torch.no_grad():
                candidate_loss = float(_scalar_loss(candidate, position))
            actual_loss_evaluations += 1
            if candidate_loss < current_loss:
                previous_loss = current_loss
                float64_stagnation_accepted_steps = (
                    _updated_float64_stagnation_count(
                        float64_stagnation_accepted_steps,
                        previous_loss,
                        candidate_loss,
                        full_predicted_decrease,
                    )
                )
                theta = candidate.detach()
                current_loss = candidate_loss
                theta_history.append(theta.cpu().numpy().copy())
                loss_history.append(current_loss)
                accepted_with_backtracking += int(backtrack > 0)
                accepted = True
                break
            backtracking_rejections += 1

        if (
            accepted
            and float64_stagnation_accepted_steps
            >= FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE
        ):
            stop_reason = FLOAT64_STAGNATION_STOP_REASON
            break

        if not accepted:
            rejected_newton_steps += 1
            predicted_decrease = max(0.0, -directional_derivative)
            loss_spacing = abs(float(np.spacing(np.float64(current_loss))))
            if unchanged_candidate or predicted_decrease <= 32.0 * loss_spacing:
                stop_reason = "floating_point_no_further_improvement"
            else:
                stop_reason = "backtracking_no_improvement"
            break

    return PolishResult(
        trajectory_id=endpoint.trajectory_id,
        endpoint=endpoint,
        theta_history=theta_history,
        loss_history=loss_history,
        accepted_steps=len(loss_history) - 1,
        rejected_newton_steps=rejected_newton_steps,
        actual_loss_evaluations=actual_loss_evaluations,
        backtracking_rejections=backtracking_rejections,
        accepted_with_backtracking=accepted_with_backtracking,
        hessian_evaluations=hessian_evaluations,
        damping_steps=damping_steps,
        condition_numbers=condition_numbers,
        damping_values=damping_values,
        gradient_norms=gradient_norms,
        stop_reason=stop_reason,
    )


def _snapshot(result: PolishResult, requested_step: int) -> tuple[np.ndarray, float, int]:
    actual_step = min(requested_step, result.accepted_steps)
    return (
        result.theta_history[actual_step],
        result.loss_history[actual_step],
        actual_step,
    )


def _per_trajectory_rows(
    results: list[PolishResult], references: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        reference = references[result.trajectory_id]
        initial_loss = result.loss_history[0]
        previous_loss: float | None = None
        saved_loss_relative_error = abs(initial_loss - result.endpoint.saved_loss) / max(
            result.endpoint.saved_loss, torch.finfo(DTYPE).tiny
        )
        for requested_step in SNAPSHOT_STEPS:
            theta, loss, actual_step = _snapshot(result, requested_step)
            rows.append(
                {
                    "trajectory_id": result.trajectory_id,
                    "n2_source": result.endpoint.source,
                    "n2_loss_category": result.endpoint.category,
                    "previous_six_plateau_case": int(result.endpoint.previous_plateau_case),
                    "requested_accepted_n3_steps": requested_step,
                    "actual_accepted_n3_steps": actual_step,
                    "best_prefix_loss": loss,
                    "best_a": theta[0],
                    "best_b": theta[1],
                    "parameter_error_posthoc": float(np.linalg.norm(theta - reference)),
                    "improved_from_n2": int(loss < initial_loss),
                    "worsened_from_n2": int(loss > initial_loss),
                    "improvement_factor": initial_loss / loss if loss > 0.0 else math.inf,
                    "improved_since_previous_checkpoint": (
                        "" if previous_loss is None else int(loss < previous_loss)
                    ),
                    "incremental_improvement_factor": (
                        "" if previous_loss is None else previous_loss / loss
                    ),
                    "reached_1e-20": int(loss < 1.0e-20),
                    "reached_1e-24": int(loss < 1.0e-24),
                    "reached_1e-28": int(loss < 1.0e-28),
                    "total_accepted_steps": result.accepted_steps,
                    "total_rejected_newton_steps": result.rejected_newton_steps,
                    "actual_n3_loss_evaluations": result.actual_loss_evaluations,
                    "backtracking_rejections": result.backtracking_rejections,
                    "accepted_steps_requiring_backtracking": result.accepted_with_backtracking,
                    "hessian_evaluations": result.hessian_evaluations,
                    "damping_steps": result.damping_steps,
                    "median_hessian_condition": (
                        float(np.median(result.condition_numbers))
                        if result.condition_numbers
                        else ""
                    ),
                    "worst_hessian_condition": (
                        max(result.condition_numbers) if result.condition_numbers else ""
                    ),
                    "stop_reason": result.stop_reason,
                    "saved_vs_audited_n2_loss_relative_error": saved_loss_relative_error,
                }
            )
            previous_loss = loss
    return rows


def _finite_factor(values: list[float]) -> np.ndarray:
    maximum = np.finfo(np.float64).max
    return np.asarray([value if math.isfinite(value) else maximum for value in values])


def _summary_rows(per_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = (
        "all",
        "n2_at_or_below_1e-20",
        "n2_1e-20_to_1e-18",
        "n2_above_1e-18",
        "previous_six_plateau_cases",
        "trajectory_230",
    )
    output: list[dict[str, object]] = []
    for group in groups:
        if group == "all":
            group_rows = per_rows
        elif group == "previous_six_plateau_cases":
            group_rows = [row for row in per_rows if int(row["previous_six_plateau_case"]) == 1]
        elif group == "trajectory_230":
            group_rows = [row for row in per_rows if int(row["trajectory_id"]) == 230]
        else:
            group_rows = [row for row in per_rows if row["n2_loss_category"] == group]
        for requested_step in SNAPSHOT_STEPS:
            selected = [
                row
                for row in group_rows
                if int(row["requested_accepted_n3_steps"]) == requested_step
            ]
            losses = [float(row["best_prefix_loss"]) for row in selected]
            errors = [float(row["parameter_error_posthoc"]) for row in selected]
            factors = _finite_factor([float(row["improvement_factor"]) for row in selected])
            incremental = _finite_factor(
                [
                    float(row["incremental_improvement_factor"])
                    for row in selected
                    if row["incremental_improvement_factor"] != ""
                ]
            )
            stops = Counter(row["stop_reason"] for row in selected)
            output.append(
                {
                    "group": group,
                    "requested_accepted_n3_steps": requested_step,
                    "trajectories": len(selected),
                    "median_prefix_loss": float(np.median(losses)) if losses else "",
                    "worst_prefix_loss": max(losses, default=""),
                    "median_parameter_error_posthoc": float(np.median(errors)) if errors else "",
                    "worst_parameter_error_posthoc": max(errors, default=""),
                    "improved_from_n2_fraction": (
                        sum(int(row["improved_from_n2"]) for row in selected) / len(selected)
                        if selected
                        else ""
                    ),
                    "worsened_from_n2_fraction": (
                        sum(int(row["worsened_from_n2"]) for row in selected) / len(selected)
                        if selected
                        else ""
                    ),
                    "improvement_factor_q25": float(np.quantile(factors, 0.25)) if len(factors) else "",
                    "improvement_factor_median": float(np.median(factors)) if len(factors) else "",
                    "improvement_factor_q75": float(np.quantile(factors, 0.75)) if len(factors) else "",
                    "improvement_factor_worst": float(np.max(factors)) if len(factors) else "",
                    "improved_since_previous_checkpoint_fraction": (
                        sum(
                            int(row["improved_since_previous_checkpoint"])
                            for row in selected
                            if row["improved_since_previous_checkpoint"] != ""
                        )
                        / len(selected)
                        if selected and requested_step > 0
                        else ""
                    ),
                    "median_incremental_improvement_factor": (
                        float(np.median(incremental)) if len(incremental) else ""
                    ),
                    "reached_1e-20_count": sum(int(row["reached_1e-20"]) for row in selected),
                    "reached_1e-24_count": sum(int(row["reached_1e-24"]) for row in selected),
                    "reached_1e-28_count": sum(int(row["reached_1e-28"]) for row in selected),
                    "median_total_accepted_steps": (
                        float(np.median([int(row["total_accepted_steps"]) for row in selected]))
                        if selected
                        else ""
                    ),
                    "total_rejected_newton_steps": sum(
                        int(row["total_rejected_newton_steps"]) for row in selected
                    ),
                    "total_backtracking_rejections": sum(
                        int(row["backtracking_rejections"]) for row in selected
                    ),
                    "backtracking_acceptance_fraction": (
                        sum(int(row["accepted_steps_requiring_backtracking"]) for row in selected)
                        / max(1, sum(int(row["total_accepted_steps"]) for row in selected))
                        if selected
                        else ""
                    ),
                    "damping_usage_fraction": (
                        sum(int(row["damping_steps"]) for row in selected)
                        / max(1, sum(int(row["hessian_evaluations"]) for row in selected))
                        if selected
                        else ""
                    ),
                    "median_hessian_condition": (
                        float(
                            np.median(
                                [
                                    float(row["median_hessian_condition"])
                                    for row in selected
                                    if row["median_hessian_condition"] != ""
                                ]
                            )
                        )
                        if selected
                        else ""
                    ),
                    "worst_hessian_condition": (
                        max(
                            [
                                float(row["worst_hessian_condition"])
                                for row in selected
                                if row["worst_hessian_condition"] != ""
                            ],
                            default="",
                        )
                    ),
                    "stop_reason_counts": ";".join(
                        f"{key}:{value}" for key, value in sorted(stops.items())
                    ),
                }
            )
    return output


def _plot(path: Path, per_rows: list[dict[str, object]]) -> None:
    steps = np.asarray(SNAPSHOT_STEPS)

    def series(rows: list[dict[str, object]], reducer) -> np.ndarray:
        return np.asarray(
            [
                reducer(
                    [
                        float(row["best_prefix_loss"])
                        for row in rows
                        if int(row["requested_accepted_n3_steps"]) == step
                    ]
                )
                for step in SNAPSHOT_STEPS
            ]
        )

    all_rows = per_rows
    median = series(all_rows, np.median)
    q25 = series(all_rows, lambda values: np.quantile(values, 0.25))
    q75 = series(all_rows, lambda values: np.quantile(values, 0.75))
    worst = series(all_rows, np.max)
    plateau_rows = [row for row in per_rows if int(row["previous_six_plateau_case"]) == 1]
    id230_rows = [row for row in per_rows if int(row["trajectory_id"]) == 230]
    plateau_median = series(plateau_rows, np.median)
    id230 = series(id230_rows, np.median)

    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    axis.semilogy(steps, median, marker="o", linewidth=2.2, label="all median")
    axis.fill_between(steps, q25, q75, alpha=0.18, label="all IQR")
    axis.semilogy(steps, worst, marker="o", color="0.3", linewidth=1.5, label="all worst")
    axis.semilogy(steps, plateau_median, marker="s", color="tab:orange", linewidth=1.7, label="six plateaus median")
    axis.semilogy(steps, id230, marker="^", color="tab:red", linewidth=1.7, label="ID 230")
    for threshold in IMPROVEMENT_THRESHOLDS:
        axis.axhline(threshold, color="0.78", linestyle="--", linewidth=0.8)
    axis.set_xticks(steps)
    axis.set_xlabel("accepted N3 steps")
    axis.set_ylabel("best prefix loss")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=TRAJECTORY_COUNT)
    args = parser.parse_args()
    for required in (ORIGINAL_RESULTS, CONTINUATION_RESULTS):
        if not required.exists():
            raise FileNotFoundError(required)
    endpoints = _load_endpoints()[: args.limit]
    positions, _, hidden_c = _make_exact_labeled_split(TRAJECTORY_COUNT, FRESH_TEST_SEED)
    references = _posthoc_reference(hidden_c).cpu().numpy()
    results: list[PolishResult] = []
    for endpoint in endpoints:
        result = _polish(endpoint, positions[endpoint.trajectory_id])
        results.append(result)
        if len(results) % 16 == 0 or len(results) == len(endpoints):
            print(
                f"N3 {len(results)}/{len(endpoints)} median accepted="
                f"{np.median([item.accepted_steps for item in results]):.1f}",
                flush=True,
            )
    if args.limit != TRAJECTORY_COUNT:
        print("smoke run complete; outputs not written", flush=True)
        return
    per_rows = _per_trajectory_rows(results, references)
    summary_rows = _summary_rows(per_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "n3_gradient_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "n3_per_trajectory.csv", per_rows)
    _plot(OUTPUT_DIR / "prefix_loss_vs_n3_steps.png", per_rows)
    print(f"wrote N3 diagnostic to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
