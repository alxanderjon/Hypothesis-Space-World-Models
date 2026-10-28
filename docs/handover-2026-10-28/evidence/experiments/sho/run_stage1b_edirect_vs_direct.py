#!/usr/bin/env python3
"""Standard DIRECT versus eDIRECT for the two-parameter N1 handoff.

The eDIRECT rule implemented here follows Liu et al. (2015), with explicit
numerical choices recorded below so the experiment is reproducible:

* Start from a deterministic 5*n = 10 point Latin-hypercube design.
* Approximate each Voronoi cell using 50*m*n candidate points, capped at
  32,768 for tractability at the 5,000-evaluation emergency limit.  Scrambled
  Sobol points fill the box and an unweighted, uniformly spaced set on all
  four closed-domain edges makes boundary points eligible without an edge
  preference or bonus.
* Define cell size as the largest distance from its sampled site to an
  assigned candidate.  Select potentially-optimal cells with the DIRECT
  lower-right hull and the published epsilon = 1e-4 sufficient-decrease test.
* Use a scalable local Gaussian radial-basis response surface on
  log10(prefix_loss + 1e-300).  Its local leave-one-out error and normalized
  cell size receive equal weight, as in the published eDIRECT rule.
* In the largest selected cell use pure farthest-point exploration.  In each
  other selected cell maximize u*d_bar - (1-u)*fhat_bar over that cell's
  candidate points.  Refit after these evaluations and add one global
  response-surface-minimum candidate, mirroring eDIRECT's greedy step.

The paper used full kriging and a genetic algorithm for the final surrogate
minimization.  Local Gaussian-RBF prediction and deterministic candidate
enumeration are the documented scalable substitutions here; neither sees the
true parameters or gradients of the real prefix loss.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    DRIFT_HEAVY,
    J_ALL,
    P,
)
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    UPPER,
    _posthoc_reference,
    _prefix_losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/edirect_vs_direct")
STANDARD_SOURCE = Path(
    "outputs/sho/stage1B/birect_vs_direct/per_trajectory_comparison.csv"
)
TRAJECTORY_COUNT = 256
HANDOFF_GATE = 1.0e-4
MAX_EVALUATIONS = 5000
EVALUATION_BUDGETS = (160, 320, 640, 1280, 2560, 5000)
INITIAL_COUNT = 10
EDIRECT_EPSILON = 1.0e-4
MC_FACTOR_PER_SITE = 100  # 50 * m * n for n=2.
MC_CANDIDATE_CAP = 32768
RBF_NEIGHBORS = 24
LOSS_FLOOR = 1.0e-300
SEARCH_SEED = SEED + 8_730_000

EDIRECT_RULE = (
    "10-point deterministic Latin hypercube; approximate Voronoi cells with "
    "min(100*m,32768) scrambled-Sobol interior candidates plus uniformly "
    "spaced closed-box boundary candidates (no edge bonus); cell size=max "
    "assigned distance; DIRECT lower-right hull with epsilon=1e-4; local "
    "24-neighbor Gaussian-RBF response on log10(loss+1e-300); exploration "
    "weight=0.5*normalized_cell_size+0.5*normalized_LOO_error; largest "
    "selected cell uses pure farthest-point exploration; other selected cells "
    "maximize u*dbar-(1-u)*fhatbar; refit and enumerate one global surrogate "
    "minimum after every cell-sampling iteration"
)
STANDARD_RULE = (
    "unchanged persisted standard-DIRECT result from the preceding comparison: "
    "center sampling, exact lower-right hull, canonical longest-side trisection, "
    "and its existing evidence-driven edge checks"
)


@dataclass(slots=True)
class SearchOutcome:
    trajectory_id: int
    success: bool
    evaluations: int
    best_loss: float
    best_a: float
    best_b: float
    runtime_seconds: float


@dataclass(slots=True)
class CellProposalData:
    points: np.ndarray
    owners: np.ndarray
    distances: np.ndarray
    sizes: np.ndarray
    selected: np.ndarray


class GaussianLocalResponse:
    """Scalable local Gaussian radial-basis response surface."""

    def __init__(self, points: np.ndarray, losses: np.ndarray) -> None:
        self.points = points
        self.values = np.log10(np.maximum(losses, LOSS_FLOOR))
        self.tree = cKDTree(points)

    def predict(self, query: np.ndarray) -> np.ndarray:
        query = np.atleast_2d(query)
        neighbor_count = min(RBF_NEIGHBORS, len(self.points))
        distances, indices = self.tree.query(query, k=neighbor_count)
        if neighbor_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        bandwidth = np.maximum(distances[:, -1:], 1.0e-12)
        weights = np.exp(-9.0 * np.square(distances / bandwidth)) + 1.0e-15
        return np.sum(weights * self.values[indices], axis=1) / np.sum(
            weights, axis=1
        )

    def loo_errors(self, site_indices: np.ndarray) -> np.ndarray:
        if len(self.points) <= 2:
            return np.ones(len(site_indices), dtype=np.float64)
        neighbor_count = min(RBF_NEIGHBORS + 1, len(self.points))
        distances, indices = self.tree.query(
            self.points[site_indices], k=neighbor_count
        )
        if neighbor_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        result = np.empty(len(site_indices), dtype=np.float64)
        for row, site_index in enumerate(site_indices):
            keep = indices[row] != site_index
            local_indices = indices[row, keep][:RBF_NEIGHBORS]
            local_distances = distances[row, keep][:RBF_NEIGHBORS]
            if not len(local_indices):
                result[row] = 0.0
                continue
            bandwidth = max(float(local_distances[-1]), 1.0e-12)
            weights = np.exp(
                -9.0 * np.square(local_distances / bandwidth)
            ) + 1.0e-15
            prediction = float(
                np.sum(weights * self.values[local_indices]) / np.sum(weights)
            )
            result[row] = abs(self.values[site_index] - prediction)
        return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize(values: np.ndarray, default: float = 0.5) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum <= 1.0e-15 * max(1.0, abs(minimum), abs(maximum)):
        return np.full_like(values, default, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def _initial_design() -> np.ndarray:
    first = (np.arange(INITIAL_COUNT, dtype=np.float64) + 0.5) / INITIAL_COUNT
    # A coprime modular permutation gives deterministic projective stratification.
    second = ((3 * np.arange(INITIAL_COUNT)) % INITIAL_COUNT + 0.5) / INITIAL_COUNT
    return np.column_stack((first, second))


def _physical(normalized: np.ndarray) -> np.ndarray:
    return LOWER + DOMAIN_WIDTH * normalized


def _prefix_losses_numpy(theta: np.ndarray, position: np.ndarray) -> np.ndarray:
    """NumPy-equivalent of the repository's exact float64 prefix loss."""

    theta = np.atleast_2d(theta).astype(np.float64, copy=False)
    prefix = position[:P]
    result = np.zeros(theta.shape[0], dtype=np.float64)
    parameter_a = theta[:, 0]
    parameter_b = theta[:, 1]
    for horizon in J_ALL:
        starts = np.arange(1, P - horizon)
        previous = np.broadcast_to(prefix[starts - 1], (len(theta), len(starts))).copy()
        current = np.broadcast_to(prefix[starts], (len(theta), len(starts))).copy()
        predictions: list[np.ndarray] = []
        for _ in range(horizon):
            following = parameter_a[:, None] * previous + parameter_b[:, None] * current
            predictions.append(following)
            previous, current = current, following
        prediction = np.stack(predictions, axis=-1)
        truth = np.stack(
            [prefix[starts + step] for step in range(1, horizon + 1)], axis=-1
        )[None, :, :]
        result += float(DRIFT_HEAVY[horizon]) * np.mean(
            np.square(prediction - truth), axis=(1, 2)
        )
    return result


def _candidate_pool(
    sample_count: int, trajectory_id: int, iteration: int
) -> np.ndarray:
    requested = min(MC_FACTOR_PER_SITE * sample_count, MC_CANDIDATE_CAP)
    exponent = int(math.ceil(math.log2(max(2, requested))))
    seed = SEARCH_SEED + 104729 * trajectory_id + 1009 * iteration
    interior = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(exponent)[:requested]

    edge_count = min(257, max(9, int(math.ceil(math.sqrt(requested)))))
    edge_coordinate = np.linspace(0.0, 1.0, edge_count, dtype=np.float64)
    zeros = np.zeros_like(edge_coordinate)
    ones = np.ones_like(edge_coordinate)
    boundaries = np.concatenate(
        (
            np.column_stack((edge_coordinate, zeros)),
            np.column_stack((edge_coordinate, ones)),
            np.column_stack((zeros, edge_coordinate)),
            np.column_stack((ones, edge_coordinate)),
        ),
        axis=0,
    )
    return np.unique(np.concatenate((interior, boundaries), axis=0), axis=0)


def _potentially_optimal_cells(
    sizes: np.ndarray, losses: np.ndarray
) -> np.ndarray:
    """DIRECT lower-right hull plus eDIRECT's epsilon condition."""

    order = np.lexsort((np.arange(len(sizes)), losses, sizes))
    groups: list[tuple[float, float, int]] = []
    for index in order:
        size = float(sizes[index])
        loss = float(losses[index])
        if not groups or abs(size - groups[-1][0]) > 1.0e-14:
            groups.append((size, loss, int(index)))
        elif loss < groups[-1][1]:
            groups[-1] = (groups[-1][0], loss, int(index))

    hull: list[int] = []
    for group_index, (size, loss, _) in enumerate(groups):
        while len(hull) >= 2:
            first = groups[hull[-2]]
            second = groups[hull[-1]]
            cross = (second[0] - first[0]) * (loss - first[1]) - (
                second[1] - first[1]
            ) * (size - first[0])
            scale = max(1.0, abs(first[1]), abs(second[1]), abs(loss))
            if cross < -5.0e-14 * scale:
                hull.pop()
            else:
                break
        hull.append(group_index)

    target = float(np.min(losses)) - EDIRECT_EPSILON * abs(float(np.min(losses)))
    selected: list[int] = []
    for position, group_index in enumerate(hull):
        size, loss, sample_index = groups[group_index]
        lower_slope = -math.inf
        upper_slope = math.inf
        if position > 0:
            previous = groups[hull[position - 1]]
            lower_slope = (loss - previous[1]) / (size - previous[0])
        if position < len(hull) - 1:
            following = groups[hull[position + 1]]
            upper_slope = (following[1] - loss) / (following[0] - size)
        required = (loss - target) / size if size > 0.0 else math.inf
        lower = max(lower_slope, required, 0.0)
        if upper_slope > 0.0 and lower <= upper_slope + 5.0e-14:
            selected.append(sample_index)
    if not selected:
        selected.append(int(np.argmax(sizes)))
    return np.asarray(
        sorted(selected, key=lambda index: (sizes[index], losses[index], index)),
        dtype=np.int64,
    )


def _cell_data(
    points: np.ndarray,
    losses: np.ndarray,
    trajectory_id: int,
    iteration: int,
) -> CellProposalData:
    pool = _candidate_pool(len(points), trajectory_id, iteration)
    tree = cKDTree(points)
    distances, owners = tree.query(pool, k=1)
    sizes = np.zeros(len(points), dtype=np.float64)
    np.maximum.at(sizes, owners, distances)
    if len(points) > 1:
        nearest = tree.query(points, k=2)[0][:, 1]
        sizes = np.maximum(sizes, 0.5 * nearest)
    selected = _potentially_optimal_cells(sizes, losses)
    return CellProposalData(pool, owners, distances, sizes, selected)


def _cell_proposals(
    points: np.ndarray,
    losses: np.ndarray,
    trajectory_id: int,
    iteration: int,
) -> np.ndarray:
    data = _cell_data(points, losses, trajectory_id, iteration)
    surface = GaussianLocalResponse(points, losses)
    selected_sizes = data.sizes[data.selected]
    size_weight = _normalize(selected_sizes)
    loo_weight = _normalize(surface.loo_errors(data.selected))
    weights = 0.5 * size_weight + 0.5 * loo_weight
    largest_position = int(np.argmax(selected_sizes))
    weights[largest_position] = 1.0

    proposals: list[np.ndarray] = []
    for selected_position, site_index in enumerate(data.selected):
        mask = data.owners == site_index
        candidates = data.points[mask]
        distances = data.distances[mask]
        if not len(candidates):
            continue
        if selected_position == largest_position:
            proposal_index = int(np.argmax(distances))
        else:
            normalized_distance = _normalize(distances)
            predicted = surface.predict(candidates)
            normalized_prediction = _normalize(predicted)
            weight = float(weights[selected_position])
            score = weight * normalized_distance - (1.0 - weight) * normalized_prediction
            proposal_index = int(np.argmax(score))
        proposal = candidates[proposal_index]
        if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
            proposals.append(proposal)
    if not proposals:
        # The largest cell always has a farthest candidate unless numerical
        # duplication occurs; try the globally farthest candidate in that case.
        nearest = cKDTree(points).query(data.points, k=1)[0]
        for proposal_index in np.argsort(nearest)[::-1]:
            proposal = data.points[proposal_index]
            if nearest[proposal_index] > 1.0e-12:
                proposals.append(proposal)
                break
    return np.asarray(proposals, dtype=np.float64)


def _local_greedy_candidates(points: np.ndarray, losses: np.ndarray) -> np.ndarray:
    tree = cKDTree(points)
    if len(points) > 1:
        nearest = tree.query(points, k=2)[0][:, 1]
    else:
        nearest = np.ones(len(points), dtype=np.float64)
    best_indices = np.argsort(losses)[: min(12, len(points))]
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates: list[np.ndarray] = []
    for index in best_indices:
        base_radius = max(float(nearest[index]), 1.0e-5)
        for fraction in (0.5, 0.25, 0.125, 0.0625):
            trial = points[index][None, :] + fraction * base_radius * directions
            candidates.append(np.clip(trial, 0.0, 1.0))
    return np.concatenate(candidates, axis=0) if candidates else np.empty((0, 2))


def _greedy_proposal(
    points: np.ndarray,
    losses: np.ndarray,
    trajectory_id: int,
    iteration: int,
) -> np.ndarray | None:
    pool = _candidate_pool(len(points), trajectory_id, iteration + 1_000_000)
    local = _local_greedy_candidates(points, losses)
    candidates = np.unique(np.concatenate((pool, local), axis=0), axis=0)
    surface = GaussianLocalResponse(points, losses)
    predictions = surface.predict(candidates)
    for candidate_index in np.argsort(predictions):
        candidate = candidates[candidate_index]
        if np.min(np.linalg.norm(points - candidate[None, :], axis=1)) > 1.0e-12:
            return candidate
    return None


def _search_one_edirect(arguments: tuple[int, np.ndarray, int]) -> SearchOutcome:
    trajectory_id, position, cap = arguments
    started = time.perf_counter()
    points = _initial_design()
    losses = _prefix_losses_numpy(_physical(points), position)
    evaluations = 0
    best_loss = math.inf
    best_point = points[0].copy()
    kept_points: list[np.ndarray] = []
    kept_losses: list[float] = []
    for point, loss in zip(points, losses):
        evaluations += 1
        kept_points.append(point.copy())
        kept_losses.append(float(loss))
        if loss < best_loss:
            best_loss = float(loss)
            best_point = point.copy()
        if best_loss <= HANDOFF_GATE:
            physical_best = _physical(best_point[None, :])[0]
            return SearchOutcome(
                trajectory_id,
                True,
                evaluations,
                best_loss,
                float(physical_best[0]),
                float(physical_best[1]),
                time.perf_counter() - started,
            )
    points = np.asarray(kept_points)
    losses = np.asarray(kept_losses)

    iteration = 0
    while evaluations < cap:
        proposals = _cell_proposals(
            points, losses, trajectory_id, iteration
        )
        for proposal in proposals:
            if evaluations >= cap:
                break
            loss = float(
                _prefix_losses_numpy(_physical(proposal[None, :]), position)[0]
            )
            evaluations += 1
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray([loss])))
            if loss < best_loss:
                best_loss = loss
                best_point = proposal.copy()
            if best_loss <= HANDOFF_GATE:
                physical_best = _physical(best_point[None, :])[0]
                return SearchOutcome(
                    trajectory_id,
                    True,
                    evaluations,
                    best_loss,
                    float(physical_best[0]),
                    float(physical_best[1]),
                    time.perf_counter() - started,
                )

        if evaluations >= cap:
            break
        greedy = _greedy_proposal(
            points, losses, trajectory_id, iteration
        )
        if greedy is not None:
            loss = float(
                _prefix_losses_numpy(_physical(greedy[None, :]), position)[0]
            )
            evaluations += 1
            points = np.concatenate((points, greedy[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray([loss])))
            if loss < best_loss:
                best_loss = loss
                best_point = greedy.copy()
            if best_loss <= HANDOFF_GATE:
                physical_best = _physical(best_point[None, :])[0]
                return SearchOutcome(
                    trajectory_id,
                    True,
                    evaluations,
                    best_loss,
                    float(physical_best[0]),
                    float(physical_best[1]),
                    time.perf_counter() - started,
                )
        iteration += 1

    physical_best = _physical(best_point[None, :])[0]
    return SearchOutcome(
        trajectory_id,
        False,
        evaluations,
        best_loss,
        float(physical_best[0]),
        float(physical_best[1]),
        time.perf_counter() - started,
    )


def _load_standard(references: np.ndarray) -> list[SearchOutcome]:
    rows = [
        row
        for row in _read_csv(STANDARD_SOURCE)
        if row["method"] == "standard_direct"
    ]
    rows.sort(key=lambda row: int(row["trajectory_id"]))
    if len(rows) != TRAJECTORY_COUNT:
        raise RuntimeError("Persisted standard-DIRECT result is incomplete")
    source_references = np.asarray(
        [
            [float(row["reference_a_posthoc"]), float(row["reference_b_posthoc"])]
            for row in rows
        ]
    )
    if not np.array_equal(source_references, references):
        raise RuntimeError("Standard-DIRECT source uses a different trajectory split")
    outcomes = [
        SearchOutcome(
            int(row["trajectory_id"]),
            bool(int(row["success_loss_le_1e-4"])),
            int(row["objective_evaluations"]),
            float(row["best_prefix_loss"]),
            float(row["best_sample_a"]),
            float(row["best_sample_b"]),
            0.0,
        )
        for row in rows
    ]
    if sum(outcome.success for outcome in outcomes) != 201:
        raise RuntimeError("Persisted standard-DIRECT result did not reproduce 201/256")
    return outcomes


def _failure_sets(
    standard: list[SearchOutcome], references: np.ndarray
) -> dict[str, set[int]]:
    failed = {outcome.trajectory_id for outcome in standard if not outcome.success}
    boundary_distance = np.min(
        np.column_stack(
            (
                references[:, 0] - LOWER,
                UPPER - references[:, 0],
                references[:, 1] - LOWER,
                UPPER - references[:, 1],
            )
        ),
        axis=1,
    )
    near = {index for index in failed if boundary_distance[index] <= 0.2}
    very = {index for index in failed if boundary_distance[index] <= 0.05}
    interior = failed - near
    if (len(near), len(very), len(interior)) != (26, 6, 29):
        raise RuntimeError("Previous DIRECT failure cohorts do not match 26/6/29")
    return {"near_edge": near, "very_near_edge": very, "interior": interior}


def _comparison_rows(
    outcomes: dict[str, list[SearchOutcome]], references: np.ndarray
) -> list[dict[str, object]]:
    standard = outcomes["standard_direct"]
    failures = _failure_sets(standard, references)
    rows: list[dict[str, object]] = []
    for method, method_outcomes in outcomes.items():
        for outcome, reference in zip(method_outcomes, references):
            boundary_distance = float(
                min(
                    reference[0] - LOWER,
                    UPPER - reference[0],
                    reference[1] - LOWER,
                    UPPER - reference[1],
                )
            )
            rows.append(
                {
                    "trajectory_id": outcome.trajectory_id,
                    "method": method,
                    "success_loss_le_1e-4": int(outcome.success),
                    "objective_evaluations": outcome.evaluations,
                    "hit_cap": int(not outcome.success),
                    "best_prefix_loss": outcome.best_loss,
                    "best_sample_a": outcome.best_a,
                    "best_sample_b": outcome.best_b,
                    "handoff_parameter_error_posthoc": math.hypot(
                        outcome.best_a - reference[0],
                        outcome.best_b - reference[1],
                    ),
                    "reference_a_posthoc": reference[0],
                    "reference_b_posthoc": reference[1],
                    "distance_to_global_boundary_posthoc": boundary_distance,
                    "is_interior_gt_0p2": int(boundary_distance > 0.2),
                    "is_near_edge_le_0p2": int(boundary_distance <= 0.2),
                    "is_very_near_edge_le_0p05": int(boundary_distance <= 0.05),
                    "was_previous_direct_failed_near_edge": int(
                        outcome.trajectory_id in failures["near_edge"]
                    ),
                    "was_previous_direct_failed_very_near_edge": int(
                        outcome.trajectory_id in failures["very_near_edge"]
                    ),
                    "was_previous_direct_failed_interior": int(
                        outcome.trajectory_id in failures["interior"]
                    ),
                    "edirect_recovered_previous_failure": int(
                        method == "edirect"
                        and outcome.success
                        and outcome.trajectory_id
                        in (failures["near_edge"] | failures["interior"])
                    ),
                    "runtime_seconds": outcome.runtime_seconds,
                    "implemented_search_rule": (
                        EDIRECT_RULE if method == "edirect" else STANDARD_RULE
                    ),
                }
            )
    return rows


def _summary_rows(comparison: list[dict[str, object]]) -> list[dict[str, object]]:
    ordinary_groups = {
        "all": None,
        "interior": "is_interior_gt_0p2",
        "near_edge": "is_near_edge_le_0p2",
        "very_near_edge": "is_very_near_edge_le_0p05",
    }
    previous_groups = {
        "previous_direct_failed_near_edge": "was_previous_direct_failed_near_edge",
        "previous_direct_failed_very_near_edge": "was_previous_direct_failed_very_near_edge",
        "previous_direct_failed_interior": "was_previous_direct_failed_interior",
    }
    result: list[dict[str, object]] = []
    for method in ("standard_direct", "edirect"):
        method_rows = [row for row in comparison if row["method"] == method]
        groups = dict(ordinary_groups)
        if method == "edirect":
            groups.update(previous_groups)
        for subgroup, key in groups.items():
            rows = (
                method_rows
                if key is None
                else [row for row in method_rows if int(row[key]) == 1]
            )
            for budget in EVALUATION_BUDGETS:
                successes = [
                    row
                    for row in rows
                    if int(row["success_loss_le_1e-4"]) == 1
                    and int(row["objective_evaluations"]) <= budget
                ]
                evaluations = [float(row["objective_evaluations"]) for row in successes]
                errors = [
                    float(row["handoff_parameter_error_posthoc"]) for row in successes
                ]
                result.append(
                    {
                        "method": method,
                        "subgroup": subgroup,
                        "evaluation_budget": budget,
                        "trajectory_count": len(rows),
                        "success_count": len(successes),
                        "success_fraction": len(successes) / len(rows),
                        "median_evaluations_among_successes": (
                            float(np.median(evaluations)) if evaluations else ""
                        ),
                        "mean_evaluations_among_successes": (
                            float(np.mean(evaluations)) if evaluations else ""
                        ),
                        "worst_evaluations_among_successes": (
                            int(np.max(evaluations)) if evaluations else ""
                        ),
                        "median_parameter_error_at_handoff_posthoc": (
                            float(np.median(errors)) if errors else ""
                        ),
                        "worst_parameter_error_at_handoff_posthoc": (
                            float(np.max(errors)) if errors else ""
                        ),
                        "failure_count_at_budget": len(rows) - len(successes),
                        "gate": HANDOFF_GATE,
                        "emergency_cap": MAX_EVALUATIONS,
                        "implemented_search_rule": (
                            EDIRECT_RULE if method == "edirect" else STANDARD_RULE
                        ),
                    }
                )
    return result


def _plot(path: Path, comparison: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    groups: tuple[tuple[str, object], ...] = (
        ("all", lambda row: True),
        ("interior", lambda row: int(row["is_interior_gt_0p2"]) == 1),
        ("near edge", lambda row: int(row["is_near_edge_le_0p2"]) == 1),
        (
            "very near edge",
            lambda row: int(row["is_very_near_edge_le_0p05"]) == 1,
        ),
    )
    for axis, (group_name, predicate) in zip(axes.flat, groups):
        for method, color in (("standard_direct", "tab:orange"), ("edirect", "tab:blue")):
            rows = [
                row
                for row in comparison
                if row["method"] == method and predicate(row)
            ]
            success_evaluations = sorted(
                int(row["objective_evaluations"])
                for row in rows
                if int(row["success_loss_le_1e-4"]) == 1
            )
            x_values = [0] + success_evaluations + [MAX_EVALUATIONS]
            y_values = [0.0] + [
                (index + 1) / len(rows) for index in range(len(success_evaluations))
            ] + [len(success_evaluations) / len(rows)]
            axis.step(
                x_values,
                y_values,
                where="post",
                linewidth=2.0,
                color=color,
                label=method.replace("_", " "),
            )
        for budget in EVALUATION_BUDGETS[:-1]:
            axis.axvline(budget, color="0.85", linewidth=0.7, zorder=0)
        axis.set_title(group_name)
        axis.set_xlabel("objective evaluations")
        axis.set_ylabel("fraction reaching loss <= 1e-4")
        axis.set_xlim(0, MAX_EVALUATIONS)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _run_edirect(
    positions: np.ndarray, cap: int, workers: int
) -> list[SearchOutcome]:
    tasks = [(index, positions[index], cap) for index in range(len(positions))]
    if workers <= 1:
        outcomes = []
        for task in tasks:
            outcomes.append(_search_one_edirect(task))
            print(
                f"eDIRECT {len(outcomes)}/{len(tasks)} success="
                f"{sum(outcome.success for outcome in outcomes)}",
                flush=True,
            )
        return outcomes

    outcomes: list[SearchOutcome] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_search_one_edirect, task) for task in tasks]
        for future in as_completed(futures):
            outcomes.append(future.result())
            if len(outcomes) % 8 == 0 or len(outcomes) == len(tasks):
                print(
                    f"eDIRECT {len(outcomes)}/{len(tasks)} success="
                    f"{sum(outcome.success for outcome in outcomes)}",
                    flush=True,
                )
    outcomes.sort(key=lambda outcome: outcome.trajectory_id)
    return outcomes


def _validate_loss_equivalence(positions: torch.Tensor) -> None:
    generator = np.random.default_rng(SEARCH_SEED)
    theta = generator.uniform(LOWER, UPPER, size=(8, 2))
    trajectory_ids = np.arange(8)
    torch_losses = _prefix_losses(
        torch.tensor(theta, dtype=DTYPE), positions[trajectory_ids]
    ).detach().cpu().numpy()
    numpy_losses = np.asarray(
        [
            _prefix_losses_numpy(theta[index : index + 1], positions[index].cpu().numpy())[0]
            for index in trajectory_ids
        ]
    )
    if not np.allclose(torch_losses, numpy_losses, rtol=2.0e-13, atol=1.0e-14):
        raise RuntimeError("NumPy prefix loss does not match repository torch loss")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not STANDARD_SOURCE.exists():
        raise FileNotFoundError(STANDARD_SOURCE)

    positions, _, hidden_c = _make_exact_labeled_split(
        TRAJECTORY_COUNT, FRESH_TEST_SEED
    )
    positions = positions.to(dtype=DTYPE)
    references = _posthoc_reference(hidden_c).cpu().numpy()
    _validate_loss_equivalence(positions)
    standard = _load_standard(references)
    print(
        f"loaded unchanged standard DIRECT: "
        f"{sum(outcome.success for outcome in standard)}/256",
        flush=True,
    )
    print(f"eDIRECT rule: {EDIRECT_RULE}", flush=True)
    edirect = _run_edirect(positions.cpu().numpy(), MAX_EVALUATIONS, args.workers)
    print(
        f"eDIRECT success={sum(outcome.success for outcome in edirect)}/256",
        flush=True,
    )

    comparison = _comparison_rows(
        {"standard_direct": standard, "edirect": edirect}, references
    )
    summary = _summary_rows(comparison)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "edirect_vs_direct_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "per_trajectory_comparison.csv", comparison)
    _plot(OUTPUT_DIR / "handoff_success_vs_evaluations.png", comparison)
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
