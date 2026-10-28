"""Trace the unchanged 5-D E-DIRECT search on the failed Kepler case."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.audit_known_structure import GT, HIGH, LOW, step
from experiments.kepler import run_four_parameter_peelback as old
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/kepler/five_parameter_n1_diagnosis")
WIDTH = HIGH - LOW
CASE_ID = 91_000
CHECKPOINTS = (100, 200, 300, 400, 500, 600, 700, 800, 900,
               1_000, 1_500, 2_000, 3_000, 5_000)
CORRECTABILITY_CHECKPOINTS = (1_000, 1_500, 2_000, 3_000, 5_000)


def objective_factory():
    trajectory = old.trajectories()[0]
    inputs = trajectory[: old.kepler.PREFIX_STATES - 1]
    targets = trajectory[1: old.kepler.PREFIX_STATES]
    torch_inputs = torch.from_numpy(inputs.copy())
    torch_targets = torch.from_numpy(targets.copy())

    def objective(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            residual = step(inputs, np.asarray(theta, dtype=np.float64), np) - targets
            value = float(np.mean(residual * residual))
        return value if math.isfinite(value) else 1.0e100

    def torch_objective(theta: torch.Tensor) -> torch.Tensor:
        return (step(torch_inputs, theta, torch) - torch_targets).square().mean()

    return trajectory, objective, torch_objective


def geometry_snapshot(points: np.ndarray, losses: np.ndarray, data,
                      iteration: int) -> dict:
    physical = LOW + WIDTH * points
    distances = np.linalg.norm(physical - GT[None, :], axis=1)
    nearest = int(np.argmin(distances))
    teacher_unit = (GT - LOW) / WIDTH
    pool_distance = np.linalg.norm(data.points - teacher_unit[None, :], axis=1)
    nearest_pool = int(np.argmin(pool_distance))
    pool_owner = int(data.owners[nearest_pool])
    selected = set(int(value) for value in data.selected)
    selected_sizes = data.sizes[data.selected]
    largest = int(data.selected[int(np.argmax(selected_sizes))])
    best = int(np.argmin(losses))
    return dict(
        iteration=iteration, evaluations=len(losses),
        best_index=best, best_theta=physical[best].tolist(), best_loss=float(losses[best]),
        best_distance_to_teacher=float(distances[best]),
        nearest_teacher_site=nearest,
        nearest_teacher_theta=physical[nearest].tolist(),
        nearest_teacher_distance=float(distances[nearest]),
        nearest_teacher_loss=float(losses[nearest]),
        nearest_teacher_cell_selected=nearest in selected,
        nearest_teacher_cell_size=float(data.sizes[nearest]),
        nearest_pool_to_teacher_distance=float(np.linalg.norm(WIDTH *
                                                    (data.points[nearest_pool] - teacher_unit))),
        nearest_pool_owner=pool_owner,
        nearest_pool_owner_selected=pool_owner in selected,
        selected_sites=[int(value) for value in data.selected],
        potentially_optimal_count=len(data.selected),
        largest_selected_site=largest,
        largest_selected_is_nearest_teacher=largest == nearest,
        pool_size=len(data.points),
        boundary_pool_count=int(np.count_nonzero(np.any((data.points == 0.0) |
                                                        (data.points == 1.0), axis=1))),
    )


def derivative_state(theta: np.ndarray, derivatives) -> dict:
    loss, gradient, hessian = derivatives(torch.from_numpy(theta.copy()), torch.empty(0))
    g = gradient.numpy(); h = hessian.numpy()
    eigenvalues = np.linalg.eigvalsh(h)
    singular = np.linalg.svd(h, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    return dict(loss=float(loss), gradient=g.tolist(), gradient_norm=float(np.linalg.norm(g)),
                hessian_eigenvalues=eigenvalues.tolist(), hessian_condition=condition,
                positive_definite=bool(eigenvalues[0] > 0.0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    trajectory, objective, torch_objective = objective_factory()
    derivatives, _, run_n3 = old._make_n3(torch_objective, LOW, HIGH)
    api = edirect.composition(5)
    continuous_traces: list[dict] = []
    api["_greedy_proposal"] = lambda p, f, c, it: continuous.proposal(
        p, f, c, it, continuous_traces
    )
    points: list[np.ndarray] = []
    losses: list[float] = []
    events: list[dict] = []
    rounds: list[dict] = []
    checkpoint_rows: list[dict] = []
    iteration = 0
    stop = False

    def evaluate(unit: np.ndarray, source: str, owner: int | None = None) -> None:
        theta = LOW + WIDTH * unit
        value = objective(theta)
        points.append(np.asarray(unit).copy()); losses.append(value)
        events.append(dict(evaluation=len(losses), iteration=iteration, source=source,
                           owner=owner, point=np.asarray(unit).tolist(), theta=theta.tolist(),
                           loss=value,
                           distance_to_teacher=float(np.linalg.norm(theta - GT)),
                           boundary=bool(np.any((unit == 0.0) | (unit == 1.0)))))

    for unit in edirect.initial_design(5):
        evaluate(unit, "initial")

    def save() -> None:
        np.savez_compressed(OUT / "trace.npz", points=np.asarray(points),
                            losses=np.asarray(losses))
        (OUT / "events.json").write_text(json.dumps(events))
        (OUT / "rounds.json").write_text(json.dumps(rounds))
        (OUT / "checkpoints.json").write_text(json.dumps(checkpoint_rows, indent=2))
        (OUT / "continuous.json").write_text(json.dumps(continuous_traces))

    while len(losses) < 5_000 and not stop:
        point_array = np.asarray(points); loss_array = np.asarray(losses)
        data = api["_cell_data"](point_array, loss_array, CASE_ID, iteration)
        round_row = geometry_snapshot(point_array, loss_array, data, iteration)
        proposals = api["_cell_proposals"](point_array, loss_array, CASE_ID, iteration)
        proposal_owners = (cKDTree(point_array).query(proposals, k=1)[1]
                           if len(proposals) else np.empty(0, dtype=int))
        round_row["cell_proposals"] = [value.tolist() for value in proposals]
        round_row["cell_proposal_owners"] = [int(value) for value in proposal_owners]
        round_row["nearest_teacher_site_proposed"] = (
            round_row["nearest_teacher_site"] in set(round_row["cell_proposal_owners"])
        )
        round_row["largest_cell_proposed"] = (
            round_row["largest_selected_site"] in set(round_row["cell_proposal_owners"])
        )
        for unit, owner in zip(proposals, proposal_owners):
            if len(losses) >= 5_000:
                break
            evaluate(unit, "cell", int(owner))
        if len(losses) < 5_000:
            before_count = len(continuous_traces)
            unit = api["_greedy_proposal"](
                np.asarray(points), np.asarray(losses), CASE_ID, iteration
            )
            if unit is not None:
                evaluate(unit, "continuous_surrogate")
                trace = continuous_traces[-1]
                round_row["continuous_proposal"] = unit.tolist()
                round_row["continuous_prediction"] = trace["prediction"]
                round_row["continuous_actual_loss"] = float(losses[-1])
                round_row["continuous_distance_to_teacher"] = float(
                    np.linalg.norm((LOW + WIDTH * unit) - GT)
                )
            else:
                round_row["continuous_duplicate_rejected"] = len(continuous_traces) > before_count
        rounds.append(round_row)
        iteration += 1

        for requested in CHECKPOINTS:
            if len(losses) >= requested and not any(row["checkpoint"] == requested
                                                     for row in checkpoint_rows):
                prefix_points = np.asarray(points[:requested])
                prefix_losses = np.asarray(losses[:requested])
                physical = LOW + WIDTH * prefix_points
                best = int(np.argmin(prefix_losses))
                nearest = int(np.argmin(np.linalg.norm(physical-GT[None, :], axis=1)))
                row = dict(
                    checkpoint=requested, best_theta=physical[best].tolist(),
                    best_loss=float(prefix_losses[best]),
                    best_distance_to_teacher=float(np.linalg.norm(physical[best]-GT)),
                    nearest_teacher_theta=physical[nearest].tolist(),
                    nearest_teacher_loss=float(prefix_losses[nearest]),
                    nearest_teacher_distance=float(np.linalg.norm(physical[nearest]-GT)),
                    boundary_evaluations=int(sum(event["boundary"]
                                                 for event in events[:requested])),
                    cell_evaluations=int(sum(event["source"] == "cell"
                                             for event in events[:requested])),
                    continuous_evaluations=int(sum(event["source"] == "continuous_surrogate"
                                                   for event in events[:requested])),
                )
                if requested in CORRECTABILITY_CHECKPOINTS:
                    n3 = old._run_n3(physical[best], float(prefix_losses[best]),
                                     f"n1_{requested}", run_n3)
                    nearest_n3 = old._run_n3(physical[nearest], float(prefix_losses[nearest]),
                                             f"nearest_{requested}", run_n3)
                    row["best_newton"] = n3
                    row["nearest_teacher_site_newton"] = nearest_n3
                    if n3["recovered"]:
                        row["entered_correct_basin"] = True
                        stop = True
                    else:
                        row["entered_correct_basin"] = False
                checkpoint_rows.append(row)
                save()
                if stop:
                    break

    points_array = np.asarray(points); losses_array = np.asarray(losses)
    best = int(np.argmin(losses_array))
    wrong = np.asarray((-0.24999999999988348, -0.9999999999989788, -2.0,
                        0.9911484896709515, 3.0407184018915747))
    line = []
    for fraction in np.linspace(0.0, 1.0, 101):
        theta = wrong + fraction * (GT - wrong)
        line.append(dict(fraction=float(fraction), loss=objective(theta), theta=theta.tolist()))
    local_controls = {}
    for name, theta in (("ground_truth", GT), ("wrong_alias", wrong),
                        ("n1_final", LOW + WIDTH * points_array[best])):
        probes = []
        for scale in (1e-6, 1e-4, 1e-2):
            values = []
            for axis in range(5):
                for sign in (-1.0, 1.0):
                    trial = theta.copy(); trial[axis] += sign * scale * WIDTH[axis]
                    trial = np.clip(trial, LOW, HIGH); values.append(objective(trial))
            probes.append(dict(scale=scale, minimum=min(values), maximum=max(values)))
        local_controls[name] = dict(theta=theta.tolist(), derivative=derivative_state(theta, derivatives),
                                    perturbations=probes)

    selected_rounds = sum(row["nearest_teacher_cell_selected"] for row in rounds)
    proposed_rounds = sum(row["nearest_teacher_site_proposed"] for row in rounds)
    final_report = dict(
        status=("BECAME_NEWTON_CORRECTABLE" if stop else "NOT_CORRECTABLE_AT_5000"),
        case=0, bounds=[LOW.tolist(), HIGH.tolist()], teacher=GT.tolist(),
        evaluations=len(losses), iterations=iteration, checkpoints=checkpoint_rows,
        correct_basin_sampled=any(
            row.get("best_newton", {}).get("recovered", False)
            or row.get("nearest_teacher_site_newton", {}).get("recovered", False)
            for row in checkpoint_rows
        ),
        nearest_teacher_cell_selected_rounds=int(selected_rounds),
        nearest_teacher_site_proposed_rounds=int(proposed_rounds),
        total_rounds=len(rounds),
        minimum_evaluated_teacher_distance=float(np.min(
            np.linalg.norm((LOW + WIDTH * points_array)-GT[None, :], axis=1))),
        ground_truth_loss=objective(GT), local_landscape=local_controls,
        alias_to_teacher_line=line,
        checkpoint_hashes_unchanged=before == hashes(), checkpoint_count=len(before),
    )
    (OUT / "report.json").write_text(json.dumps(final_report, indent=2))
    save()
    print(json.dumps({key: final_report[key] for key in (
        "status", "evaluations", "iterations", "correct_basin_sampled",
        "nearest_teacher_cell_selected_rounds", "nearest_teacher_site_proposed_rounds",
        "minimum_evaluated_teacher_distance", "checkpoint_hashes_unchanged",
    )}, indent=2))


if __name__ == "__main__":
    main()
