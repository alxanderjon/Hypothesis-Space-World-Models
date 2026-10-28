#!/usr/bin/env python3
"""Enhanced eDIRECT with targeted boundary rescue for 2-D N1 localization.

Enhanced eDIRECT keeps the prior Voronoi/DIRECT scaffold but replaces the
local kernel smoother by a proper Gaussian-process (ordinary kriging) mean:

* squared-exponential covariance and a 1e-6 nugget;
* exact Cholesky solve on at most 96 loss-aware and farthest-point sites;
* length scale selected by marginal likelihood from an adaptive nine-value
  grid based on sample spacing;
* analytic GP leave-one-out residuals for selected sites in the fit subset;
* log10(prefix_loss + 1e-300) response to handle the objective's dynamic range.

Voronoi regions use min(256*m, 131072) scrambled-Sobol candidates plus an
unweighted representation of every global face.  The global surrogate step
searches min(512*m, 131072) points across the complete bounded domain, then
performs deterministic multiresolution surrogate-only refinement around its
best predictions and observed low-loss evidence.  No real-loss optimizer or
true parameter is used.

If the 5,000-evaluation phase fails and its best observed point is close to a
global face relative to observed resolution, a single nearest-face strip is
formed.  Its width is 3*max(local 8-neighbor spacing, nominal domain spacing),
clipped to [0.03, domain_width/4].  Only that strip is searched for at most
1,500 new evaluations, using the same GP/eDIRECT machinery and reusing prior
observations inside the strip.  Structured initial strip probes cover the
exact face, progressively inward offsets, and the free coordinate.  No other
face is searched if this one fails.
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
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree, distance
from scipy.stats import qmc

from experiments.sho.run_stage1b_edirect_vs_direct import (
    HANDOFF_GATE,
    LOSS_FLOOR,
    _initial_design,
    _normalize,
    _prefix_losses_numpy,
    _read_csv,
    _write_csv,
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
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/enhanced_edirect_boundary_rescue")
PREVIOUS_RESULTS = Path(
    "outputs/sho/stage1B/edirect_vs_direct/per_trajectory_comparison.csv"
)
TRAJECTORY_COUNT = 256
NORMAL_BUDGET = 5000
RESCUE_BUDGET = 1500
GP_MAX_SITES = 96
GP_BEST_SITES = 64
GP_NUGGET = 1.0e-6
VORONOI_MULTIPLIER = 256
VORONOI_CAP = 131072
GLOBAL_MULTIPLIER = 512
GLOBAL_CAP = 131072
EDIRECT_EPSILON = 1.0e-4
SEARCH_SEED = SEED + 9_410_000

ENHANCED_RULE = (
    "10-point deterministic Latin hypercube; Voronoi regions from "
    "min(256*m,131072) scrambled-Sobol points plus unweighted closed-box "
    "faces; DIRECT lower-right hull with epsilon=1e-4; proper squared-"
    "exponential GP on log10(loss+1e-300), Cholesky solve, nugget=1e-6, "
    "maximum 96 loss-aware/farthest-point sites, marginal-likelihood adaptive "
    "nine-value length-scale grid; GP LOO and cell-size weighting; largest-"
    "cell pure exploration; full-domain min(512*m,131072) surrogate "
    "enumeration plus multiresolution surrogate-only refinement around both "
    "predicted and observed low-loss sites"
)
RESCUE_RULE = (
    "only after enhanced-eDIRECT failure: choose nearest face to best observed "
    "point; local_scale=max(median 8-neighbor distance, domain_width/sqrt(N)); "
    "delta=clip(3*local_scale,0.03,domain_width/4); invoke only when face "
    "distance<=1.25*delta; search that one strip for <=1500 new evaluations "
    "with the same enhanced-eDIRECT machinery, reused in-strip history, exact "
    "face probes, progressive inward probes, and no other boundary search"
)


@dataclass(slots=True)
class PhaseResult:
    success: bool
    new_evaluations: int
    best_loss: float
    best_physical: np.ndarray
    points_normalized: np.ndarray
    physical_points: np.ndarray
    losses: np.ndarray
    runtime_seconds: float


@dataclass(slots=True)
class TrajectoryResult:
    trajectory_id: int
    normal_success: bool
    normal_evaluations: int
    normal_best_loss: float
    normal_best_a: float
    normal_best_b: float
    normal_runtime_seconds: float
    rescue_invoked: bool
    rescue_success: bool
    rescue_evaluations: int
    rescue_face: str
    rescue_delta: float
    rescue_local_scale: float
    rescue_loss_before: float
    rescue_final_loss: float
    rescue_runtime_seconds: float
    final_success: bool
    total_evaluations: int
    final_best_loss: float
    final_best_a: float
    final_best_b: float


class ProperGaussianProcess:
    """Exact GP mean on a deterministic sparse, representative training set."""

    def __init__(
        self,
        points: np.ndarray,
        losses: np.ndarray,
        mandatory_indices: np.ndarray | None = None,
    ) -> None:
        indices = self._training_indices(points, losses, mandatory_indices)
        self.training_indices = indices
        self.points = points[indices]
        response = np.log10(np.maximum(losses[indices], LOSS_FLOOR))
        self.response_mean = float(np.mean(response))
        self.response_scale = max(float(np.std(response)), 1.0e-8)
        normalized_response = (response - self.response_mean) / self.response_scale

        if len(self.points) > 1:
            neighbor_distance = cKDTree(self.points).query(self.points, k=2)[0][:, 1]
            base_length = float(np.median(neighbor_distance))
        else:
            base_length = 0.2
        base_length = float(np.clip(base_length, 0.015, 0.35))
        # A wide log grid is important here: the 128-step objective develops a
        # very narrow well close to the solution, while the early global model
        # must still correlate observations across a sizeable part of the box.
        minimum_length = max(0.0015, 0.03 * base_length)
        maximum_length = min(0.75, max(0.08, 4.0 * base_length))
        length_candidates = np.geomspace(minimum_length, maximum_length, 9)
        squared_distance = distance.cdist(self.points, self.points, "sqeuclidean")
        best_objective = math.inf
        best: tuple[float, tuple[np.ndarray, bool], np.ndarray] | None = None
        for length_scale in length_candidates:
            covariance = np.exp(
                -0.5 * squared_distance / float(length_scale * length_scale)
            )
            covariance.flat[:: len(covariance) + 1] += GP_NUGGET
            try:
                factor = cho_factor(covariance, lower=True, check_finite=False)
                alpha = cho_solve(factor, normalized_response, check_finite=False)
            except np.linalg.LinAlgError:
                continue
            objective = 0.5 * float(normalized_response @ alpha)
            objective += float(np.sum(np.log(np.diag(factor[0]))))
            if objective < best_objective:
                best_objective = objective
                best = (float(length_scale), factor, alpha)
        if best is None:
            raise RuntimeError("Every GP covariance factorization failed")
        self.length_scale, self.factor, self.alpha = best
        self.tree = cKDTree(self.points)
        inverse = cho_solve(
            self.factor, np.eye(len(self.points), dtype=np.float64), check_finite=False
        )
        self.inverse_diagonal = np.diag(inverse)
        self._index_lookup = {
            int(original): local for local, original in enumerate(self.training_indices)
        }

    @staticmethod
    def _training_indices(
        points: np.ndarray,
        losses: np.ndarray,
        mandatory_indices: np.ndarray | None,
    ) -> np.ndarray:
        if len(points) <= GP_MAX_SITES:
            return np.arange(len(points), dtype=np.int64)
        selected: list[int] = []
        if mandatory_indices is not None:
            for index in mandatory_indices:
                if int(index) not in selected and len(selected) < GP_MAX_SITES:
                    selected.append(int(index))
        for index in np.argsort(losses)[:GP_BEST_SITES]:
            if int(index) not in selected and len(selected) < GP_MAX_SITES:
                selected.append(int(index))
        if not selected:
            selected.append(int(np.argmin(losses)))

        minimum_distance = np.min(
            distance.cdist(points, points[np.asarray(selected)]), axis=1
        )
        minimum_distance[np.asarray(selected)] = -math.inf
        while len(selected) < GP_MAX_SITES:
            next_index = int(np.argmax(minimum_distance))
            selected.append(next_index)
            update = np.linalg.norm(points - points[next_index][None, :], axis=1)
            minimum_distance = np.minimum(minimum_distance, update)
            minimum_distance[np.asarray(selected)] = -math.inf
        return np.asarray(selected, dtype=np.int64)

    def predict(self, query: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
        query = np.atleast_2d(query)
        predictions: list[np.ndarray] = []
        denominator = 2.0 * self.length_scale * self.length_scale
        for start in range(0, len(query), chunk_size):
            block = query[start : start + chunk_size]
            squared_distance = distance.cdist(block, self.points, "sqeuclidean")
            covariance = np.exp(-squared_distance / denominator)
            normalized = covariance @ self.alpha
            predictions.append(
                self.response_mean + self.response_scale * normalized
            )
        return np.concatenate(predictions)

    def loo_errors(
        self,
        full_points: np.ndarray,
        full_losses: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        full_response = np.log10(np.maximum(full_losses, LOSS_FLOOR))
        errors = np.empty(len(indices), dtype=np.float64)
        missing_positions: list[int] = []
        missing_queries: list[np.ndarray] = []
        for position, original_index in enumerate(indices):
            local_index = self._index_lookup.get(int(original_index))
            if local_index is None:
                missing_positions.append(position)
                missing_queries.append(full_points[original_index])
            else:
                normalized_error = abs(
                    self.alpha[local_index] / self.inverse_diagonal[local_index]
                )
                errors[position] = self.response_scale * normalized_error
        if missing_queries:
            predicted = self.predict(np.asarray(missing_queries))
            for position, value in zip(missing_positions, predicted):
                errors[position] = abs(full_response[indices[position]] - value)
        return errors


def _map_to_physical(normalized: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])


def _map_to_normalized(physical: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return (physical - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _candidate_pool(
    sample_count: int,
    trajectory_id: int,
    iteration: int,
    multiplier: int,
    cap: int,
) -> np.ndarray:
    requested = min(multiplier * sample_count, cap)
    exponent = int(math.ceil(math.log2(max(2, requested))))
    seed = (
        SEARCH_SEED + 130363 * trajectory_id + 1559 * iteration + 17 * multiplier
    ) % (2**32 - 1)
    interior = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(exponent)[:requested]
    edge_count = min(1025, max(17, int(math.ceil(math.sqrt(requested)))))
    coordinate = np.linspace(0.0, 1.0, edge_count, dtype=np.float64)
    zeros = np.zeros_like(coordinate)
    ones = np.ones_like(coordinate)
    edges = np.concatenate(
        (
            np.column_stack((coordinate, zeros)),
            np.column_stack((coordinate, ones)),
            np.column_stack((zeros, coordinate)),
            np.column_stack((ones, coordinate)),
        ),
        axis=0,
    )
    return np.concatenate((interior, edges), axis=0)


def _potentially_optimal_cells(
    sizes: np.ndarray, losses: np.ndarray
) -> np.ndarray:
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
    minimum = float(np.min(losses))
    target = minimum - EDIRECT_EPSILON * abs(minimum)
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
        if upper_slope > 0.0 and max(lower_slope, required, 0.0) <= upper_slope + 5e-14:
            selected.append(sample_index)
    if not selected:
        selected.append(int(np.argmax(sizes)))
    return np.asarray(
        sorted(selected, key=lambda index: (sizes[index], losses[index], index)),
        dtype=np.int64,
    )


def _cell_proposals(
    points: np.ndarray,
    losses: np.ndarray,
    trajectory_id: int,
    iteration: int,
) -> np.ndarray:
    pool = _candidate_pool(
        len(points), trajectory_id, iteration, VORONOI_MULTIPLIER, VORONOI_CAP
    )
    tree = cKDTree(points)
    candidate_distances, owners = tree.query(pool, k=1)
    sizes = np.zeros(len(points), dtype=np.float64)
    np.maximum.at(sizes, owners, candidate_distances)
    if len(points) > 1:
        nearest = tree.query(points, k=2)[0][:, 1]
        sizes = np.maximum(sizes, 0.5 * nearest)
    selected = _potentially_optimal_cells(sizes, losses)
    process = ProperGaussianProcess(points, losses, selected)
    selected_sizes = sizes[selected]
    weight = 0.5 * _normalize(selected_sizes) + 0.5 * _normalize(
        process.loo_errors(points, losses, selected)
    )
    largest_position = int(np.argmax(selected_sizes))
    weight[largest_position] = 1.0
    proposals: list[np.ndarray] = []
    for selected_position, site_index in enumerate(selected):
        mask = owners == site_index
        candidates = pool[mask]
        distances_in_cell = candidate_distances[mask]
        if not len(candidates):
            continue
        if selected_position == largest_position:
            candidate_index = int(np.argmax(distances_in_cell))
        else:
            normalized_distance = _normalize(distances_in_cell)
            normalized_prediction = _normalize(process.predict(candidates))
            score = weight[selected_position] * normalized_distance - (
                1.0 - weight[selected_position]
            ) * normalized_prediction
            candidate_index = int(np.argmax(score))
        proposal = candidates[candidate_index]
        if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
            proposals.append(proposal)
    if not proposals:
        for candidate_index in np.argsort(candidate_distances)[::-1]:
            if candidate_distances[candidate_index] > 1.0e-12:
                proposals.append(pool[candidate_index])
                break
    return np.asarray(proposals, dtype=np.float64)


def _best_predicted_distinct(
    candidates: np.ndarray,
    predictions: np.ndarray,
    existing: np.ndarray,
) -> np.ndarray | None:
    tree = cKDTree(existing)
    for index in np.argsort(predictions):
        if tree.query(candidates[index], k=1)[0] > 1.0e-12:
            return candidates[index]
    return None


def _global_surrogate_proposal(
    points: np.ndarray,
    losses: np.ndarray,
    trajectory_id: int,
    iteration: int,
) -> np.ndarray | None:
    process = ProperGaussianProcess(points, losses)
    pool = _candidate_pool(
        len(points),
        trajectory_id,
        iteration + 1_000_000,
        GLOBAL_MULTIPLIER,
        GLOBAL_CAP,
    )
    predictions = process.predict(pool)
    top_count = min(24, len(pool))
    top_indices = np.argpartition(predictions, top_count - 1)[:top_count]
    base_resolution = 1.0 / math.sqrt(len(pool))
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    refined: list[np.ndarray] = []
    for index in top_indices:
        for radius in (
            2.0 * base_resolution,
            base_resolution,
            0.5 * base_resolution,
            0.25 * base_resolution,
        ):
            refined.append(np.clip(pool[index][None, :] + radius * directions, 0.0, 1.0))

    # Also resolve the surrogate around actual low-loss evidence.  This is
    # still a surrogate-only candidate search (no objective values are taken
    # here), but avoids depending on a coarse global candidate landing inside
    # an increasingly narrow basin before local kriging can be useful.
    point_tree = cKDTree(points)
    neighbor_count = min(2, len(points))
    best_observed = np.argsort(losses)[: min(24, len(points))]
    nominal_spacing = 1.0 / math.sqrt(len(points))
    for index in best_observed:
        if neighbor_count > 1:
            observed_spacing = float(point_tree.query(points[index], k=2)[0][1])
        else:
            observed_spacing = nominal_spacing
        starting_radius = max(observed_spacing, 0.5 * nominal_spacing)
        for radius_multiplier in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            radius = starting_radius * radius_multiplier
            refined.append(
                np.clip(points[index][None, :] + radius * directions, 0.0, 1.0)
            )
    refined_points = np.concatenate(refined, axis=0)
    refined_predictions = process.predict(refined_points)
    candidates = np.concatenate((pool, refined_points), axis=0)
    all_predictions = np.concatenate((predictions, refined_predictions))
    return _best_predicted_distinct(candidates, all_predictions, points)


def _evaluate_new_points(
    points: np.ndarray,
    losses: np.ndarray,
    proposals: np.ndarray,
    position: np.ndarray,
    bounds: np.ndarray,
    remaining: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    used = 0
    success = False
    for proposal in proposals:
        if used >= remaining:
            break
        if len(points) and np.min(np.linalg.norm(points - proposal[None, :], axis=1)) <= 1e-12:
            continue
        physical = _map_to_physical(proposal[None, :], bounds)
        loss = float(_prefix_losses_numpy(physical, position)[0])
        points = np.concatenate((points, proposal[None, :]), axis=0)
        losses = np.concatenate((losses, np.asarray([loss])))
        used += 1
        if loss <= HANDOFF_GATE:
            success = True
            break
    return points, losses, used, success


def _run_phase(
    trajectory_id: int,
    position: np.ndarray,
    bounds: np.ndarray,
    budget: int,
    seed_points: np.ndarray | None = None,
    seed_losses: np.ndarray | None = None,
    startup_points: np.ndarray | None = None,
    iteration_offset: int = 0,
) -> PhaseResult:
    started = time.perf_counter()
    points = (
        np.empty((0, 2), dtype=np.float64)
        if seed_points is None
        else np.asarray(seed_points, dtype=np.float64).copy()
    )
    losses = (
        np.empty(0, dtype=np.float64)
        if seed_losses is None
        else np.asarray(seed_losses, dtype=np.float64).copy()
    )
    new_evaluations = 0
    initial = _initial_design() if startup_points is None else startup_points
    points, losses, used, success = _evaluate_new_points(
        points, losses, initial, position, bounds, budget
    )
    new_evaluations += used
    iteration = 0
    while not success and new_evaluations < budget:
        proposals = _cell_proposals(
            points,
            losses,
            trajectory_id,
            iteration + iteration_offset,
        )
        points, losses, used, success = _evaluate_new_points(
            points,
            losses,
            proposals,
            position,
            bounds,
            budget - new_evaluations,
        )
        new_evaluations += used
        if success or new_evaluations >= budget:
            break
        proposal = _global_surrogate_proposal(
            points,
            losses,
            trajectory_id,
            iteration + iteration_offset,
        )
        if proposal is not None:
            points, losses, used, success = _evaluate_new_points(
                points,
                losses,
                proposal[None, :],
                position,
                bounds,
                budget - new_evaluations,
            )
            new_evaluations += used
        iteration += 1
    best_index = int(np.argmin(losses))
    physical_points = _map_to_physical(points, bounds)
    return PhaseResult(
        success=bool(losses[best_index] <= HANDOFF_GATE),
        new_evaluations=new_evaluations,
        best_loss=float(losses[best_index]),
        best_physical=physical_points[best_index].copy(),
        points_normalized=points,
        physical_points=physical_points,
        losses=losses,
        runtime_seconds=time.perf_counter() - started,
    )


def _rescue_setup(normal: PhaseResult) -> tuple[str, float, float, np.ndarray] | None:
    best = normal.best_physical
    tree = cKDTree(normal.physical_points)
    neighbor_count = min(9, len(normal.physical_points))
    neighbor_distance = tree.query(best, k=neighbor_count)[0]
    if neighbor_count > 1:
        local_neighbor_spacing = float(np.median(neighbor_distance[1:]))
    else:
        local_neighbor_spacing = DOMAIN_WIDTH
    nominal_spacing = DOMAIN_WIDTH / math.sqrt(len(normal.physical_points))
    local_scale = max(local_neighbor_spacing, nominal_spacing)
    delta = float(np.clip(3.0 * local_scale, 0.03, DOMAIN_WIDTH / 4.0))
    face_distances = np.asarray(
        (best[0] - LOWER, UPPER - best[0], best[1] - LOWER, UPPER - best[1])
    )
    face_index = int(np.argmin(face_distances))
    if face_distances[face_index] > 1.25 * delta:
        return None
    face_names = ("a_lower", "a_upper", "b_lower", "b_upper")
    bounds = np.asarray(((LOWER, UPPER), (LOWER, UPPER)), dtype=np.float64)
    axis = 0 if face_index < 2 else 1
    lower_face = face_index in (0, 2)
    if lower_face:
        bounds[axis] = (LOWER, LOWER + delta)
    else:
        bounds[axis] = (UPPER - delta, UPPER)
    other = 1 - axis
    other_half_width = max(8.0 * local_scale, 0.2 * DOMAIN_WIDTH)
    bounds[other] = (
        max(LOWER, best[other] - other_half_width),
        min(UPPER, best[other] + other_half_width),
    )
    return face_names[face_index], delta, local_scale, bounds


def _strip_startup_points(
    normal: PhaseResult, bounds: np.ndarray, face: str
) -> np.ndarray:
    axis = 0 if face.startswith("a_") else 1
    lower_face = face.endswith("lower")
    other = 1 - axis
    best = normal.best_physical
    inward_fractions = np.asarray((0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0))
    other_values = np.linspace(bounds[other, 0], bounds[other, 1], 9)
    face_value = bounds[axis, 0] if lower_face else bounds[axis, 1]
    inward_sign = 1.0 if lower_face else -1.0
    strip_width = bounds[axis, 1] - bounds[axis, 0]
    physical: list[np.ndarray] = [best.copy()]
    for other_value in other_values:
        for fraction in inward_fractions:
            point = best.copy()
            point[axis] = face_value + inward_sign * fraction * strip_width
            point[other] = other_value
            physical.append(point)
    normalized = _map_to_normalized(np.asarray(physical), bounds)
    return np.unique(np.clip(normalized, 0.0, 1.0), axis=0)


def _search_trajectory(arguments: tuple[int, np.ndarray]) -> TrajectoryResult:
    trajectory_id, position = arguments
    full_bounds = np.asarray(((LOWER, UPPER), (LOWER, UPPER)), dtype=np.float64)
    normal = _run_phase(
        trajectory_id,
        position,
        full_bounds,
        NORMAL_BUDGET,
        iteration_offset=0,
    )
    if normal.success:
        return TrajectoryResult(
            trajectory_id,
            True,
            normal.new_evaluations,
            normal.best_loss,
            float(normal.best_physical[0]),
            float(normal.best_physical[1]),
            normal.runtime_seconds,
            False,
            False,
            0,
            "",
            0.0,
            0.0,
            normal.best_loss,
            normal.best_loss,
            0.0,
            True,
            normal.new_evaluations,
            normal.best_loss,
            float(normal.best_physical[0]),
            float(normal.best_physical[1]),
        )

    setup = _rescue_setup(normal)
    if setup is None:
        return TrajectoryResult(
            trajectory_id,
            False,
            normal.new_evaluations,
            normal.best_loss,
            float(normal.best_physical[0]),
            float(normal.best_physical[1]),
            normal.runtime_seconds,
            False,
            False,
            0,
            "",
            0.0,
            0.0,
            normal.best_loss,
            normal.best_loss,
            0.0,
            False,
            normal.new_evaluations,
            normal.best_loss,
            float(normal.best_physical[0]),
            float(normal.best_physical[1]),
        )

    face, delta, local_scale, bounds = setup
    inside = np.all(
        (normal.physical_points >= bounds[:, 0] - 1.0e-14)
        & (normal.physical_points <= bounds[:, 1] + 1.0e-14),
        axis=1,
    )
    seed_physical = normal.physical_points[inside]
    seed_points = _map_to_normalized(seed_physical, bounds)
    seed_losses = normal.losses[inside]
    startup = _strip_startup_points(normal, bounds, face)
    rescue = _run_phase(
        trajectory_id,
        position,
        bounds,
        RESCUE_BUDGET,
        seed_points=seed_points,
        seed_losses=seed_losses,
        startup_points=startup,
        iteration_offset=2_000_000,
    )
    final_is_rescue = rescue.best_loss < normal.best_loss
    final_loss = rescue.best_loss if final_is_rescue else normal.best_loss
    final_point = rescue.best_physical if final_is_rescue else normal.best_physical
    final_success = final_loss <= HANDOFF_GATE
    return TrajectoryResult(
        trajectory_id,
        False,
        normal.new_evaluations,
        normal.best_loss,
        float(normal.best_physical[0]),
        float(normal.best_physical[1]),
        normal.runtime_seconds,
        True,
        bool(rescue.success),
        rescue.new_evaluations,
        face,
        delta,
        local_scale,
        normal.best_loss,
        rescue.best_loss,
        rescue.runtime_seconds,
        bool(final_success),
        normal.new_evaluations + rescue.new_evaluations,
        final_loss,
        float(final_point[0]),
        float(final_point[1]),
    )


def _load_previous(references: np.ndarray) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(PREVIOUS_RESULTS) if row["method"] == "edirect"]
    rows.sort(key=lambda row: int(row["trajectory_id"]))
    if len(rows) != TRAJECTORY_COUNT:
        raise RuntimeError("Previous eDIRECT result is incomplete")
    source_reference = np.asarray(
        [
            [float(row["reference_a_posthoc"]), float(row["reference_b_posthoc"])]
            for row in rows
        ]
    )
    if not np.array_equal(source_reference, references):
        raise RuntimeError("Previous eDIRECT result used a different test split")
    if sum(int(row["success_loss_le_1e-4"]) for row in rows) != 252:
        raise RuntimeError("Previous eDIRECT result did not reproduce 252/256")
    return rows


def _per_trajectory_rows(
    results: list[TrajectoryResult],
    references: np.ndarray,
    previous: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result, reference, old in zip(results, references, previous):
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
                "trajectory_id": result.trajectory_id,
                "reference_a_posthoc": reference[0],
                "reference_b_posthoc": reference[1],
                "distance_to_global_boundary_posthoc": boundary_distance,
                "is_interior_gt_0p2": int(boundary_distance > 0.2),
                "is_near_edge_le_0p2": int(boundary_distance <= 0.2),
                "is_very_near_edge_le_0p05": int(boundary_distance <= 0.05),
                "previous_edirect_success": int(old["success_loss_le_1e-4"]),
                "previous_edirect_failure": int(old["success_loss_le_1e-4"] == "0"),
                "normal_success": int(result.normal_success),
                "normal_evaluations": result.normal_evaluations,
                "normal_best_loss": result.normal_best_loss,
                "normal_best_a": result.normal_best_a,
                "normal_best_b": result.normal_best_b,
                "normal_parameter_error_posthoc": math.hypot(
                    result.normal_best_a - reference[0],
                    result.normal_best_b - reference[1],
                ),
                "normal_runtime_seconds": result.normal_runtime_seconds,
                "rescue_invoked": int(result.rescue_invoked),
                "rescue_success": int(result.rescue_success),
                "rescue_evaluations": result.rescue_evaluations,
                "rescue_face": result.rescue_face,
                "rescue_delta": result.rescue_delta,
                "rescue_local_scale": result.rescue_local_scale,
                "rescue_loss_before": result.rescue_loss_before,
                "rescue_final_loss": result.rescue_final_loss,
                "rescue_runtime_seconds": result.rescue_runtime_seconds,
                "final_success": int(result.final_success),
                "total_evaluations": result.total_evaluations,
                "final_best_loss": result.final_best_loss,
                "final_best_a": result.final_best_a,
                "final_best_b": result.final_best_b,
                "final_parameter_error_posthoc": math.hypot(
                    result.final_best_a - reference[0],
                    result.final_best_b - reference[1],
                ),
                "was_one_of_previous_four_failures": int(
                    result.trajectory_id in (36, 68, 116, 230)
                ),
                "enhanced_rule": ENHANCED_RULE,
                "rescue_rule": RESCUE_RULE,
            }
        )
    return rows


def _stage_summary(
    rows: list[dict[str, object]], previous: list[dict[str, str]]
) -> list[dict[str, object]]:
    groups = {
        "all": None,
        "interior": "is_interior_gt_0p2",
        "near_edge": "is_near_edge_le_0p2",
        "very_near_edge": "is_very_near_edge_le_0p05",
        "previous_four_failures": "was_one_of_previous_four_failures",
    }
    output: list[dict[str, object]] = []
    stages = (
        (
            "enhanced_edirect_pre_rescue",
            "normal_success",
            "normal_evaluations",
            "normal_parameter_error_posthoc",
        ),
        (
            "enhanced_edirect_plus_boundary_rescue",
            "final_success",
            "total_evaluations",
            "final_parameter_error_posthoc",
        ),
    )
    for stage, success_key, evaluation_key, error_key in stages:
        for subgroup, group_key in groups.items():
            selected = (
                rows
                if group_key is None
                else [row for row in rows if int(row[group_key]) == 1]
            )
            successes = [row for row in selected if int(row[success_key]) == 1]
            evaluations = [float(row[evaluation_key]) for row in successes]
            errors = [float(row[error_key]) for row in successes]
            output.append(
                {
                    "stage": stage,
                    "subgroup": subgroup,
                    "trajectory_count": len(selected),
                    "success_count": len(successes),
                    "success_fraction": len(successes) / len(selected),
                    "median_evaluations_among_successes": float(np.median(evaluations))
                    if evaluations
                    else "",
                    "mean_evaluations_among_successes": float(np.mean(evaluations))
                    if evaluations
                    else "",
                    "worst_evaluations_among_successes": int(np.max(evaluations))
                    if evaluations
                    else "",
                    "median_parameter_error_at_handoff_posthoc": float(
                        np.median(errors)
                    )
                    if errors
                    else "",
                    "worst_parameter_error_at_handoff_posthoc": float(np.max(errors))
                    if errors
                    else "",
                    "failure_count": len(selected) - len(successes),
                    "normal_budget": NORMAL_BUDGET,
                    "rescue_budget": RESCUE_BUDGET,
                    "gate": HANDOFF_GATE,
                    "enhanced_rule": ENHANCED_RULE,
                    "rescue_rule": RESCUE_RULE,
                }
            )

    previous_successes = [row for row in previous if row["success_loss_le_1e-4"] == "1"]
    output.append(
        {
            "stage": "previous_edirect_reference",
            "subgroup": "all",
            "trajectory_count": len(previous),
            "success_count": len(previous_successes),
            "success_fraction": len(previous_successes) / len(previous),
            "median_evaluations_among_successes": float(
                np.median([int(row["objective_evaluations"]) for row in previous_successes])
            ),
            "mean_evaluations_among_successes": float(
                np.mean([int(row["objective_evaluations"]) for row in previous_successes])
            ),
            "worst_evaluations_among_successes": max(
                int(row["objective_evaluations"]) for row in previous_successes
            ),
            "median_parameter_error_at_handoff_posthoc": float(
                np.median(
                    [
                        float(row["handoff_parameter_error_posthoc"])
                        for row in previous_successes
                    ]
                )
            ),
            "worst_parameter_error_at_handoff_posthoc": max(
                float(row["handoff_parameter_error_posthoc"])
                for row in previous_successes
            ),
            "failure_count": len(previous) - len(previous_successes),
            "normal_budget": NORMAL_BUDGET,
            "rescue_budget": 0,
            "gate": HANDOFF_GATE,
            "enhanced_rule": "prior saved eDIRECT implementation",
            "rescue_rule": "",
        }
    )
    return output


def _rescue_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if int(row["rescue_invoked"]) != 1:
            continue
        output.append(
            {
                "trajectory_id": row["trajectory_id"],
                "rescue_success": row["rescue_success"],
                "rescue_face": row["rescue_face"],
                "rescue_delta": row["rescue_delta"],
                "rescue_local_scale": row["rescue_local_scale"],
                "loss_before_rescue": row["rescue_loss_before"],
                "rescue_evaluations": row["rescue_evaluations"],
                "strip_phase_best_loss": row["rescue_final_loss"],
                "final_loss_after_rescue": row["final_best_loss"],
                "final_parameter_error_posthoc": row[
                    "final_parameter_error_posthoc"
                ],
                "distance_to_global_boundary_posthoc": row[
                    "distance_to_global_boundary_posthoc"
                ],
                "reference_a_posthoc": row["reference_a_posthoc"],
                "reference_b_posthoc": row["reference_b_posthoc"],
                "final_best_a": row["final_best_a"],
                "final_best_b": row["final_best_b"],
                "was_one_of_previous_four_failures": row[
                    "was_one_of_previous_four_failures"
                ],
                "rescue_rule": row["rescue_rule"],
            }
        )
    return output


def _plot(
    path: Path, rows: list[dict[str, object]], previous: list[dict[str, str]]
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.7))
    curves: list[tuple[str, list[int], int, str]] = []
    previous_evaluations = sorted(
        int(row["objective_evaluations"])
        for row in previous
        if row["success_loss_le_1e-4"] == "1"
    )
    curves.append(("previous eDIRECT", previous_evaluations, 256, "0.5"))
    normal_evaluations = sorted(
        int(row["normal_evaluations"]) for row in rows if int(row["normal_success"]) == 1
    )
    curves.append(("enhanced eDIRECT", normal_evaluations, 256, "tab:blue"))
    final_evaluations = sorted(
        int(row["total_evaluations"]) for row in rows if int(row["final_success"]) == 1
    )
    curves.append(("enhanced + rescue", final_evaluations, 256, "tab:green"))
    maximum = NORMAL_BUDGET + RESCUE_BUDGET
    for label, evaluations, denominator, color in curves:
        x_values = [0] + evaluations + [maximum]
        y_values = [0.0] + [
            (index + 1) / denominator for index in range(len(evaluations))
        ] + [len(evaluations) / denominator]
        axis.step(x_values, y_values, where="post", linewidth=2.1, label=label, color=color)
    axis.axvline(NORMAL_BUDGET, color="0.25", linestyle="--", linewidth=1.0)
    axis.text(
        NORMAL_BUDGET,
        0.03,
        "rescue begins",
        rotation=90,
        ha="right",
        va="bottom",
        color="0.3",
    )
    axis.set_xlim(0, maximum)
    axis.set_ylim(-0.01, 1.01)
    axis.set_xlabel("total objective evaluations")
    axis.set_ylabel("fraction reaching prefix loss <= 1e-4")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _run_all(positions: np.ndarray, workers: int) -> list[TrajectoryResult]:
    tasks = [(index, positions[index]) for index in range(len(positions))]
    if workers <= 1:
        results = []
        for task in tasks:
            results.append(_search_trajectory(task))
            print(
                f"enhanced {len(results)}/{len(tasks)} normal="
                f"{sum(result.normal_success for result in results)} final="
                f"{sum(result.final_success for result in results)}",
                flush=True,
            )
        return results
    results: list[TrajectoryResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_search_trajectory, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 8 == 0 or len(results) == len(tasks):
                print(
                    f"enhanced {len(results)}/{len(tasks)} normal="
                    f"{sum(result.normal_success for result in results)} final="
                    f"{sum(result.final_success for result in results)}",
                    flush=True,
                )
    results.sort(key=lambda result: result.trajectory_id)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not PREVIOUS_RESULTS.exists():
        raise FileNotFoundError(PREVIOUS_RESULTS)
    positions, _, hidden_c = _make_exact_labeled_split(
        TRAJECTORY_COUNT, FRESH_TEST_SEED
    )
    positions = positions.to(dtype=DTYPE)
    references = _posthoc_reference(hidden_c).cpu().numpy()
    previous = _load_previous(references)
    print("enhanced rule:", ENHANCED_RULE, flush=True)
    print("rescue rule:", RESCUE_RULE, flush=True)
    results = _run_all(positions.cpu().numpy(), args.workers)
    rows = _per_trajectory_rows(results, references, previous)
    summary = _stage_summary(rows, previous)
    rescues = _rescue_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "enhanced_edirect_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "boundary_rescue_results.csv", rescues)
    _write_csv(OUTPUT_DIR / "per_trajectory_results.csv", rows)
    _plot(OUTPUT_DIR / "handoff_success_vs_evaluations.png", rows, previous)
    print(
        f"normal={sum(result.normal_success for result in results)}/256 "
        f"final={sum(result.final_success for result in results)}/256 "
        f"rescues={sum(result.rescue_invoked for result in results)}",
        flush=True,
    )
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
