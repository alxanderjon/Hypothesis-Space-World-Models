"""Continue the unchanged failed-case 5-D E-DIRECT trajectory to 10,000 sites."""
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

from experiments.audit_known_structure import GT, HIGH, LOW
from experiments.kepler import run_four_parameter_peelback as old
from experiments.kepler.diagnose_five_parameter_n1 import (
    CASE_ID,
    WIDTH,
    geometry_snapshot,
    objective_factory,
)
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SOURCE = Path("outputs/kepler/five_parameter_n1_diagnosis")
OUT = Path("outputs/kepler/five_parameter_n1_continuation")
TARGETS = (2_000, 3_000, 5_000, 7_500, 10_000)
START_ITERATION = 174
WRONG_ALIAS = np.asarray(
    [-0.24999999999988348, -0.9999999999989788, -2.0,
     0.9911484896709515, 3.0407184018915747],
    dtype=np.float64,
)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def load_state() -> tuple[list[np.ndarray], list[float], int, list[dict], list[dict]]:
    resume = OUT / "resume.npz"
    state_path = OUT / "state.json"
    if resume.exists() and state_path.exists():
        saved = np.load(resume)
        state = json.loads(state_path.read_text())
        return (
            [row.copy() for row in saved["points"]],
            saved["losses"].astype(np.float64).tolist(),
            int(state["iteration"]),
            json.loads((OUT / "checkpoints.json").read_text()),
            json.loads((OUT / "round_summaries.json").read_text()),
        )
    saved = np.load(SOURCE / "trace.npz")
    if saved["points"].shape != (1503, 5):
        raise RuntimeError("the exact completed iteration-173 state is unavailable")
    return (
        [row.copy() for row in saved["points"]],
        saved["losses"].astype(np.float64).tolist(),
        START_ITERATION,
        [],
        [],
    )


def save(points, losses, iteration, checkpoints, rounds) -> None:
    atomic_npz(OUT / "resume.npz", points=np.asarray(points), losses=np.asarray(losses))
    atomic_json(OUT / "state.json", dict(iteration=iteration, evaluations=len(losses)))
    atomic_json(OUT / "checkpoints.json", checkpoints)
    atomic_json(OUT / "round_summaries.json", rounds)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    trajectory, objective, torch_objective = objective_factory()
    _, _, run_n3 = old._make_n3(torch_objective, LOW, HIGH)
    api = edirect.composition(5)
    points, losses, iteration, checkpoints, rounds = load_state()
    if len(losses) < 1503 or iteration < START_ITERATION:
        raise RuntimeError("resume state precedes the exact saved round boundary")

    checkpoint_set = {int(row["checkpoint"]) for row in checkpoints}
    stop = False
    while len(losses) < TARGETS[-1] and not stop:
        point_array = np.asarray(points)
        loss_array = np.asarray(losses)
        data = api["_cell_data"](point_array, loss_array, CASE_ID, iteration)
        snapshot = geometry_snapshot(point_array, loss_array, data, iteration)
        proposals = api["_cell_proposals"](
            point_array, loss_array, CASE_ID, iteration
        )
        owners = (
            cKDTree(point_array).query(proposals, k=1)[1]
            if len(proposals) else np.empty(0, dtype=np.int64)
        )
        proposal_distances = []
        for unit, owner in zip(proposals, owners):
            if len(losses) >= TARGETS[-1]:
                break
            theta = LOW + WIDTH * unit
            value = objective(theta)
            points.append(np.asarray(unit).copy())
            losses.append(value)
            proposal_distances.append(float(np.linalg.norm(theta - GT)))

        continuous_trace: list[dict] = []
        continuous_unit = None
        continuous_value = None
        if len(losses) < TARGETS[-1]:
            continuous_unit = continuous.proposal(
                np.asarray(points), np.asarray(losses), CASE_ID, iteration,
                continuous_trace,
            )
            if continuous_unit is not None:
                theta = LOW + WIDTH * continuous_unit
                continuous_value = objective(theta)
                points.append(np.asarray(continuous_unit).copy())
                losses.append(continuous_value)

        rounds.append(dict(
            iteration=iteration,
            evaluations_before=int(snapshot["evaluations"]),
            evaluations_after=len(losses),
            best_loss_before=float(snapshot["best_loss"]),
            best_theta_before=snapshot["best_theta"],
            nearest_teacher_distance=float(snapshot["nearest_teacher_distance"]),
            nearest_teacher_loss=float(snapshot["nearest_teacher_loss"]),
            nearest_teacher_cell_selected=bool(snapshot["nearest_teacher_cell_selected"]),
            nearest_pool_to_teacher_distance=float(snapshot["nearest_pool_to_teacher_distance"]),
            nearest_pool_owner_selected=bool(snapshot["nearest_pool_owner_selected"]),
            potentially_optimal_count=int(snapshot["potentially_optimal_count"]),
            largest_selected_is_nearest_teacher=bool(
                snapshot["largest_selected_is_nearest_teacher"]
            ),
            cell_proposal_count=len(proposal_distances),
            closest_cell_proposal_distance=(min(proposal_distances)
                                            if proposal_distances else None),
            continuous_theta=(
                (LOW + WIDTH * continuous_unit).tolist()
                if continuous_unit is not None else None
            ),
            continuous_loss=(float(continuous_value)
                             if continuous_value is not None else None),
            continuous_distance_to_teacher=(
                float(np.linalg.norm((LOW + WIDTH * continuous_unit) - GT))
                if continuous_unit is not None else None
            ),
            continuous_prediction=(
                float(continuous_trace[-1]["prediction"])
                if continuous_trace else None
            ),
        ))
        iteration += 1

        for target in TARGETS:
            if target in checkpoint_set or len(losses) < target:
                continue
            prefix_points = np.asarray(points[:target])
            prefix_losses = np.asarray(losses[:target])
            physical = LOW + WIDTH * prefix_points
            best_index = int(np.argmin(prefix_losses))
            nearest_index = int(np.argmin(np.linalg.norm(physical - GT, axis=1)))
            best_theta = physical[best_index]
            entry_error = best_theta - GT
            relevant_rounds = [row for row in rounds
                               if row["evaluations_after"] > (checkpoints[-1]["checkpoint"]
                                                              if checkpoints else 1500)
                               and row["evaluations_before"] < target]
            continuous_rows = [row for row in relevant_rounds
                               if row["continuous_loss"] is not None]
            n3 = old._run_n3(best_theta, float(prefix_losses[best_index]),
                             f"n1_{target}", run_n3)
            row = dict(
                checkpoint=target,
                best_index=best_index + 1,
                best_theta=best_theta.tolist(),
                best_loss=float(prefix_losses[best_index]),
                entrywise_error=entry_error.tolist(),
                entrywise_absolute_error=np.abs(entry_error).tolist(),
                parameter_l2_distance=float(np.linalg.norm(entry_error)),
                n1_teacher_equivalent=bool(prefix_losses[best_index] < old.FINAL_GATE),
                newton_correctable=bool(n3["recovered"]),
                offline_n3=n3,
                nearest_evaluated_index=nearest_index + 1,
                nearest_evaluated_theta=physical[nearest_index].tolist(),
                nearest_evaluated_loss=float(prefix_losses[nearest_index]),
                nearest_evaluated_distance=float(np.linalg.norm(
                    physical[nearest_index] - GT
                )),
                wrong_alias_is_best=bool(best_index <= 940),
                wrong_alias_loss=float(objective(WRONG_ALIAS)),
                continuous_proposals=len(continuous_rows),
                continuous_best_loss=(
                    min(row["continuous_loss"] for row in continuous_rows)
                    if continuous_rows else None
                ),
                continuous_closest_teacher_distance=(
                    min(row["continuous_distance_to_teacher"] for row in continuous_rows)
                    if continuous_rows else None
                ),
                nearest_teacher_cell_selected_rounds=sum(
                    row["nearest_teacher_cell_selected"] for row in relevant_rounds
                ),
                nearest_pool_owner_selected_rounds=sum(
                    row["nearest_pool_owner_selected"] for row in relevant_rounds
                ),
                closest_cell_proposal_distance=min(
                    (row["closest_cell_proposal_distance"] for row in relevant_rounds
                     if row["closest_cell_proposal_distance"] is not None),
                    default=None,
                ),
            )
            checkpoints.append(row)
            checkpoint_set.add(target)
            print(json.dumps({key: row[key] for key in (
                "checkpoint", "best_theta", "best_loss", "parameter_l2_distance",
                "newton_correctable", "nearest_evaluated_distance",
                "wrong_alias_is_best",
            )}, indent=2), flush=True)
            if row["n1_teacher_equivalent"]:
                stop = True
                break
        save(points, losses, iteration, checkpoints, rounds)

    result = dict(
        status=("N1_TEACHER_EQUIVALENT" if stop else "REACHED_10000"),
        evaluations=len(losses),
        iterations=iteration,
        checkpoints=checkpoints,
        checkpoint_hashes_unchanged=before == hashes(),
        checkpoint_count=len(before),
    )
    atomic_json(OUT / "report.json", result)
    print(json.dumps({
        "status": result["status"],
        "evaluations": result["evaluations"],
        "iterations": result["iterations"],
        "checkpoint_hashes_unchanged": result["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
