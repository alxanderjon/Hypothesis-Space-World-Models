#!/usr/bin/env python3
"""Pure-DIRECT scaling on fresh trajectory-unique continuous SHO dynamics."""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic import (
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    P,
    TEST_DATA_SEED,
    UPPER,
    _posthoc_reference,
    _prefix_losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

HALF_WIDTH = DOMAIN_WIDTH / 2.0
DOMAIN_DIAGONAL = math.sqrt(2.0) * DOMAIN_WIDTH
DEFAULT_BUDGETS = (160, 320, 640, 1280, 2560, 5000)
FRESH_TEST_SEED = TEST_DATA_SEED + 730_021
LOSS_EPSILON = 1.0e-300
BOUNDARY_SUBGROUP_WIDTH = 0.1


@dataclass(slots=True)
class DirectRectangle:
    rid: int
    center_a: float
    center_b: float
    exponent_a: int
    exponent_b: int
    loss: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.exponent_a, self.exponent_b)

    @property
    def half_a(self) -> float:
        return HALF_WIDTH / (3.0**self.exponent_a)

    @property
    def half_b(self) -> float:
        return HALF_WIDTH / (3.0**self.exponent_b)

    @property
    def side_a(self) -> float:
        return 2.0 * self.half_a

    @property
    def side_b(self) -> float:
        return 2.0 * self.half_b

    @property
    def half_diagonal(self) -> float:
        return math.hypot(self.half_a, self.half_b)

    @property
    def canonical_cost(self) -> int:
        if self.exponent_a == self.exponent_b:
            return 4
        return 2

    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.center_a - self.half_a,
            self.center_a + self.half_a,
            self.center_b - self.half_b,
            self.center_b + self.half_b,
        )


@dataclass(slots=True)
class SizeGroup:
    heap: list[tuple[float, int]] = field(default_factory=list)


@dataclass(slots=True)
class DirectState:
    trajectory_id: int
    rectangles: dict[int, DirectRectangle]
    groups: dict[tuple[int, int], SizeGroup]
    next_rid: int
    direct_queue: list[int]
    adaptive_evaluations: int
    sample_count: int
    edge_probe_count: int
    sample_keys: set[tuple[float, float]]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _point_key(a_value: float, b_value: float) -> tuple[float, float]:
    return (round(a_value, 14), round(b_value, 14))


def _add_rectangle(state: DirectState, rectangle: DirectRectangle) -> None:
    state.rectangles[rectangle.rid] = rectangle
    group = state.groups.setdefault(rectangle.key, SizeGroup())
    heapq.heappush(group.heap, (rectangle.loss, rectangle.rid))


def _remove_rectangle(state: DirectState, rid: int) -> DirectRectangle:
    return state.rectangles.pop(rid)


def _group_minimum(
    state: DirectState, key: tuple[int, int], group: SizeGroup
) -> DirectRectangle | None:
    while group.heap:
        loss, rid = group.heap[0]
        rectangle = state.rectangles.get(rid)
        if rectangle is None or rectangle.key != key or rectangle.loss != loss:
            heapq.heappop(group.heap)
            continue
        return rectangle
    return None


def _potentially_optimal_ids(state: DirectState) -> list[int]:
    """Exact lower-right hull of current size-class minima for every K>0."""

    minima: list[DirectRectangle] = []
    empty_keys: list[tuple[int, int]] = []
    for key, group in state.groups.items():
        rectangle = _group_minimum(state, key, group)
        if rectangle is None:
            empty_keys.append(key)
        else:
            minima.append(rectangle)
    for key in empty_keys:
        del state.groups[key]

    minima.sort(key=lambda item: (item.half_diagonal, item.loss, item.rid))
    groups: list[tuple[float, float, int]] = []
    for rectangle in minima:
        diagonal = rectangle.half_diagonal
        if not groups or abs(diagonal - groups[-1][0]) > 1.0e-14:
            groups.append((diagonal, rectangle.loss, rectangle.rid))
            continue
        old_diagonal, old_loss, old_rid = groups[-1]
        tolerance = 5.0e-14 * max(1.0, abs(old_loss), abs(rectangle.loss))
        if rectangle.loss < old_loss - tolerance or (
            abs(rectangle.loss - old_loss) <= tolerance and rectangle.rid < old_rid
        ):
            groups[-1] = (old_diagonal, rectangle.loss, rectangle.rid)

    hull: list[int] = []
    for group_index, (diagonal, loss, _) in enumerate(groups):
        while len(hull) >= 2:
            first = groups[hull[-2]]
            second = groups[hull[-1]]
            cross = (second[0] - first[0]) * (loss - first[1]) - (
                second[1] - first[1]
            ) * (diagonal - first[0])
            scale = max(1.0, abs(first[1]), abs(second[1]), abs(loss))
            if cross < -5.0e-14 * scale:
                hull.pop()
            else:
                break
        hull.append(group_index)

    result: list[int] = []
    for position, group_index in enumerate(hull):
        current = groups[group_index]
        if position == 0:
            lower_slope = -math.inf
        else:
            previous = groups[hull[position - 1]]
            lower_slope = (current[1] - previous[1]) / (current[0] - previous[0])
        if position == len(hull) - 1:
            upper_slope = math.inf
        else:
            following = groups[hull[position + 1]]
            upper_slope = (following[1] - current[1]) / (following[0] - current[0])
        if upper_slope > 0.0 and max(lower_slope, 0.0) <= upper_slope + 5.0e-14:
            result.append(current[2])
    return sorted(
        result,
        key=lambda rid: (
            -state.rectangles[rid].half_diagonal,
            state.rectangles[rid].loss,
            rid,
        ),
    )


def _canonical_probes(
    rectangle: DirectRectangle,
) -> tuple[list[tuple[float, float]], list[int]]:
    if rectangle.exponent_a < rectangle.exponent_b:
        dimensions = [0]
    elif rectangle.exponent_b < rectangle.exponent_a:
        dimensions = [1]
    else:
        dimensions = [0, 1]
    points: list[tuple[float, float]] = []
    point_dimensions: list[int] = []
    for dimension in dimensions:
        delta = (rectangle.side_a if dimension == 0 else rectangle.side_b) / 3.0
        for sign in (-1.0, 1.0):
            a_value = rectangle.center_a + (sign * delta if dimension == 0 else 0.0)
            b_value = rectangle.center_b + (sign * delta if dimension == 1 else 0.0)
            points.append((a_value, b_value))
            point_dimensions.append(dimension)
    return points, point_dimensions


def _split_canonical(
    state: DirectState,
    rid: int,
    points: list[tuple[float, float]],
    dimensions: list[int],
    losses: list[float],
) -> DirectRectangle:
    parent = _remove_rectangle(state, rid)
    by_dimension: dict[int, list[tuple[tuple[float, float], float]]] = {}
    for point, dimension, loss in zip(points, dimensions, losses):
        by_dimension.setdefault(dimension, []).append((point, float(loss)))
    split_order = sorted(
        by_dimension,
        key=lambda dimension: min(loss for _, loss in by_dimension[dimension]),
    )
    exponent_a = parent.exponent_a
    exponent_b = parent.exponent_b
    for dimension in split_order:
        child_exponent_a = exponent_a + int(dimension == 0)
        child_exponent_b = exponent_b + int(dimension == 1)
        for point, loss in by_dimension[dimension]:
            child = DirectRectangle(
                state.next_rid,
                point[0],
                point[1],
                child_exponent_a,
                child_exponent_b,
                loss,
            )
            state.next_rid += 1
            _add_rectangle(state, child)
        exponent_a = child_exponent_a
        exponent_b = child_exponent_b
    central = DirectRectangle(
        state.next_rid,
        parent.center_a,
        parent.center_b,
        exponent_a,
        exponent_b,
        parent.loss,
    )
    state.next_rid += 1
    _add_rectangle(state, central)
    return parent


def _unseen_pair(
    state: DirectState,
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return _point_key(*first) not in state.sample_keys and _point_key(*second) not in state.sample_keys


def _boundary_pair_after_split(
    state: DirectState,
    rectangle: DirectRectangle,
    points: list[tuple[float, float]],
    dimensions: list[int],
    losses: list[float],
) -> list[tuple[float, float]] | None:
    by_dimension: dict[int, list[tuple[tuple[float, float], float]]] = {}
    for point, dimension, loss in zip(points, dimensions, losses):
        by_dimension.setdefault(dimension, []).append((point, float(loss)))
    lower_a, upper_a, lower_b, upper_b = rectangle.bounds()
    bounds = ((lower_a, upper_a), (lower_b, upper_b))
    warranted: list[tuple[int, float, tuple[float, float]]] = []
    for dimension in (0, 1):
        entries = sorted(by_dimension.get(dimension, []), key=lambda item: item[0][dimension])
        for value, direction_index in ((LOWER, 0), (UPPER, -1)):
            touches = (
                bounds[dimension][0] <= LOWER + 1.0e-13
                if value == LOWER
                else bounds[dimension][1] >= UPPER - 1.0e-13
            )
            improving = False
            anchor = (rectangle.center_a, rectangle.center_b)
            if len(entries) == 2:
                directed = entries[direction_index]
                opposite = entries[-1 - direction_index]
                improving = directed[1] < min(rectangle.loss, opposite[1])
                anchor = directed[0]
            side = rectangle.side_a if dimension == 0 else rectangle.side_b
            center = rectangle.center_a if dimension == 0 else rectangle.center_b
            near_boundary = abs(center - value) <= max(side, DOMAIN_WIDTH / 9.0)
            if touches or (improving and near_boundary):
                warranted.append((dimension, value, anchor))

    # Check a warranted corner first.
    for first_index, first in enumerate(warranted):
        for second in warranted[first_index + 1 :]:
            if first[0] == second[0]:
                continue
            corner = [rectangle.center_a, rectangle.center_b]
            corner[first[0]] = first[1]
            corner[second[0]] = second[1]
            near = list(corner)
            offset = min(DOMAIN_WIDTH / 54.0, max(rectangle.side_a, rectangle.side_b) / 6.0)
            near[first[0]] += offset if first[1] == LOWER else -offset
            near[second[0]] += offset if second[1] == LOWER else -offset
            pair = [tuple(corner), tuple(near)]
            if _unseen_pair(state, pair[0], pair[1]):
                return pair

    for dimension, value, anchor in warranted:
        edge = [anchor[0], anchor[1]]
        edge[dimension] = value
        near = list(edge)
        side = rectangle.side_a if dimension == 0 else rectangle.side_b
        offset = min(DOMAIN_WIDTH / 54.0, side / 6.0)
        near[dimension] += offset if value == LOWER else -offset
        pair = [tuple(edge), tuple(near)]
        if _unseen_pair(state, pair[0], pair[1]):
            return pair
    return None


def _fallback_boundary_pair(state: DirectState) -> list[tuple[float, float]] | None:
    """Fill a two-evaluation remainder using an unsampled global-edge gap."""

    best_rectangle: DirectRectangle | None = None
    for rectangle in state.rectangles.values():
        lower_a, upper_a, lower_b, upper_b = rectangle.bounds()
        if not (
            lower_a <= LOWER + 1.0e-13
            or upper_a >= UPPER - 1.0e-13
            or lower_b <= LOWER + 1.0e-13
            or upper_b >= UPPER - 1.0e-13
        ):
            continue
        if best_rectangle is None or (
            rectangle.half_diagonal,
            -rectangle.loss,
            -rectangle.rid,
        ) > (
            best_rectangle.half_diagonal,
            -best_rectangle.loss,
            -best_rectangle.rid,
        ):
            best_rectangle = rectangle
    if best_rectangle is None:
        return None
    rectangle = best_rectangle
    lower_a, upper_a, lower_b, upper_b = rectangle.bounds()
    candidates: list[tuple[int, float]] = []
    if lower_a <= LOWER + 1.0e-13:
        candidates.append((0, LOWER))
    if upper_a >= UPPER - 1.0e-13:
        candidates.append((0, UPPER))
    if lower_b <= LOWER + 1.0e-13:
        candidates.append((1, LOWER))
    if upper_b >= UPPER - 1.0e-13:
        candidates.append((1, UPPER))
    for dimension, value in candidates:
        other = 1 - dimension
        center = [rectangle.center_a, rectangle.center_b]
        sides = [rectangle.side_a, rectangle.side_b]
        for along_fraction in (0.0, 1 / 6, -1 / 6, 1 / 3, -1 / 3):
            edge = list(center)
            edge[dimension] = value
            edge[other] = min(UPPER, max(LOWER, edge[other] + along_fraction * sides[other]))
            for divisor in (54.0, 81.0, 162.0, 486.0, 1458.0):
                near = list(edge)
                offset = min(DOMAIN_WIDTH / divisor, max(sides[dimension] / 6.0, DOMAIN_WIDTH / 4374.0))
                near[dimension] += offset if value == LOWER else -offset
                pair = [tuple(edge), tuple(near)]
                if _unseen_pair(state, pair[0], pair[1]):
                    return pair
    return None


def _append_sample(
    state: DirectState,
    point: tuple[float, float],
    loss: float,
    is_edge: bool,
    sample_points: torch.Tensor,
    sample_losses: torch.Tensor,
    sample_is_edge: torch.Tensor,
) -> None:
    index = state.sample_count
    sample_points[state.trajectory_id, index, 0] = point[0]
    sample_points[state.trajectory_id, index, 1] = point[1]
    sample_losses[state.trajectory_id, index] = loss
    sample_is_edge[state.trajectory_id, index] = is_edge
    state.sample_count += 1
    state.adaptive_evaluations += 1
    state.edge_probe_count += int(is_edge)
    state.sample_keys.add(_point_key(*point))


def _true_containing_rectangle(
    state: DirectState, reference: np.ndarray
) -> DirectRectangle:
    containing: list[DirectRectangle] = []
    for rectangle in state.rectangles.values():
        lower_a, upper_a, lower_b, upper_b = rectangle.bounds()
        if (
            lower_a - 1.0e-13 <= reference[0] <= upper_a + 1.0e-13
            and lower_b - 1.0e-13 <= reference[1] <= upper_b + 1.0e-13
        ):
            containing.append(rectangle)
    if not containing:
        raise RuntimeError("DIRECT partition lost the true parameter point")
    return min(
        containing,
        key=lambda rectangle: (
            rectangle.side_a * rectangle.side_b,
            math.hypot(
                rectangle.center_a - reference[0],
                rectangle.center_b - reference[1],
            ),
            rectangle.rid,
        ),
    )


def _snapshot_rows(
    budget: int,
    states: list[DirectState],
    references: np.ndarray,
    sample_points: torch.Tensor,
    sample_losses: torch.Tensor,
    sample_is_edge: torch.Tensor,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state, reference in zip(states, references):
        count = budget + 1
        points = sample_points[state.trajectory_id, :count].cpu().numpy()
        losses = sample_losses[state.trajectory_id, :count].cpu().numpy()
        nearest_error = np.linalg.norm(points - reference[None, :], axis=1)
        best_index = int(np.argmin(losses))
        containing = _true_containing_rectangle(state, reference)
        po_ids = set(_potentially_optimal_ids(state))
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
                "trajectory_id": state.trajectory_id,
                "adaptive_evaluation_budget": budget,
                "total_prefix_loss_evaluations_including_root": count,
                "reference_a_posthoc": float(reference[0]),
                "reference_b_posthoc": float(reference[1]),
                "distance_to_global_boundary_posthoc": boundary_distance,
                "nearest_sampled_parameter_error_posthoc": float(np.min(nearest_error)),
                "best_observed_prefix_loss": float(losses[best_index]),
                "true_rectangle_center_distance_posthoc": math.hypot(
                    containing.center_a - reference[0],
                    containing.center_b - reference[1],
                ),
                "true_rectangle_side_a": containing.side_a,
                "true_rectangle_side_b": containing.side_b,
                "true_rectangle_area": containing.side_a * containing.side_b,
                "true_rectangle_half_diagonal": containing.half_diagonal,
                "true_rectangle_is_potentially_optimal_posthoc": int(
                    containing.rid in po_ids
                ),
                "best_loss_sample_a": float(points[best_index, 0]),
                "best_loss_sample_b": float(points[best_index, 1]),
                "best_loss_sample_parameter_error_posthoc": float(
                    nearest_error[best_index]
                ),
                "global_edge_probes_used": int(
                    bool(sample_is_edge[state.trajectory_id, :count].any())
                ),
                "global_edge_probe_count": int(
                    sample_is_edge[state.trajectory_id, :count].sum().item()
                ),
                "active_rectangle_count": len(state.rectangles),
                "potentially_optimal_rectangle_count": len(po_ids),
            }
        )

    errors = np.asarray([row["nearest_sampled_parameter_error_posthoc"] for row in rows])
    losses = np.asarray([row["best_observed_prefix_loss"] for row in rows])
    center_distances = np.asarray([row["true_rectangle_center_distance_posthoc"] for row in rows])
    areas = np.asarray([row["true_rectangle_area"] for row in rows])
    boundary_distances = np.asarray([row["distance_to_global_boundary_posthoc"] for row in rows])
    near_boundary = boundary_distances <= BOUNDARY_SUBGROUP_WIDTH
    summary: dict[str, object] = {
        "method": "pure_direct_with_evidence_driven_boundary_checks",
        "adaptive_evaluation_budget": budget,
        "total_prefix_loss_evaluations_including_root": budget + 1,
        "trajectories": len(states),
        "median_nearest_sampled_parameter_error": float(np.median(errors)),
        "worst_nearest_sampled_parameter_error": float(np.max(errors)),
        "median_best_observed_prefix_loss": float(np.median(losses)),
        "worst_best_observed_prefix_loss": float(np.max(losses)),
        "fraction_parameter_error_le_1e-1": float(np.mean(errors <= 1.0e-1)),
        "fraction_parameter_error_le_1e-2": float(np.mean(errors <= 1.0e-2)),
        "fraction_parameter_error_le_1e-3": float(np.mean(errors <= 1.0e-3)),
        "fraction_parameter_error_le_1e-4": float(np.mean(errors <= 1.0e-4)),
        "fraction_prefix_loss_le_1e-2": float(np.mean(losses <= 1.0e-2)),
        "fraction_prefix_loss_le_1e-4": float(np.mean(losses <= 1.0e-4)),
        "fraction_prefix_loss_le_1e-8": float(np.mean(losses <= 1.0e-8)),
        "fraction_prefix_loss_le_1e-12": float(np.mean(losses <= 1.0e-12)),
        "fraction_prefix_loss_le_1e-20": float(np.mean(losses <= 1.0e-20)),
        "fraction_true_rectangle_potentially_optimal": float(
            np.mean([row["true_rectangle_is_potentially_optimal_posthoc"] for row in rows])
        ),
        "median_true_rectangle_center_distance": float(np.median(center_distances)),
        "worst_true_rectangle_center_distance": float(np.max(center_distances)),
        "median_true_rectangle_area": float(np.median(areas)),
        "worst_true_rectangle_area": float(np.max(areas)),
        "fraction_trajectories_with_global_edge_probes": float(
            np.mean([row["global_edge_probes_used"] for row in rows])
        ),
        "total_global_edge_probes": int(
            np.sum([row["global_edge_probe_count"] for row in rows])
        ),
        "near_global_boundary_width": BOUNDARY_SUBGROUP_WIDTH,
        "near_global_boundary_trajectory_count": int(np.sum(near_boundary)),
        "near_boundary_median_parameter_error": (
            float(np.median(errors[near_boundary])) if bool(near_boundary.any()) else float("nan")
        ),
        "near_boundary_worst_parameter_error": (
            float(np.max(errors[near_boundary])) if bool(near_boundary.any()) else float("nan")
        ),
        "near_boundary_fraction_parameter_error_le_1e-2": (
            float(np.mean(errors[near_boundary] <= 1.0e-2))
            if bool(near_boundary.any())
            else float("nan")
        ),
        "near_boundary_fraction_true_rectangle_potentially_optimal": (
            float(
                np.mean(
                    np.asarray(
                        [row["true_rectangle_is_potentially_optimal_posthoc"] for row in rows]
                    )[near_boundary]
                )
            )
            if bool(near_boundary.any())
            else float("nan")
        ),
        "mean_active_rectangle_count": float(
            np.mean([row["active_rectangle_count"] for row in rows])
        ),
        "mean_potentially_optimal_rectangle_count": float(
            np.mean([row["potentially_optimal_rectangle_count"] for row in rows])
        ),
    }
    return rows, summary


def _run_continuing_direct(
    positions: torch.Tensor,
    references: np.ndarray,
    budgets: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trajectory_count = positions.shape[0]
    maximum_budget = budgets[-1]
    sample_points = torch.empty(
        (trajectory_count, maximum_budget + 1, 2), dtype=DTYPE
    )
    sample_losses = torch.empty((trajectory_count, maximum_budget + 1), dtype=DTYPE)
    sample_is_edge = torch.zeros(
        (trajectory_count, maximum_budget + 1), dtype=torch.bool
    )
    root_theta = torch.zeros((trajectory_count, 2), dtype=DTYPE)
    root_losses = _prefix_losses(root_theta, positions).detach().cpu()
    sample_points[:, 0] = root_theta
    sample_losses[:, 0] = root_losses

    states: list[DirectState] = []
    for trajectory_id in range(trajectory_count):
        root = DirectRectangle(
            rid=0,
            center_a=0.0,
            center_b=0.0,
            exponent_a=0,
            exponent_b=0,
            loss=float(root_losses[trajectory_id].item()),
        )
        state = DirectState(
            trajectory_id=trajectory_id,
            rectangles={},
            groups={},
            next_rid=1,
            direct_queue=[],
            adaptive_evaluations=0,
            sample_count=1,
            edge_probe_count=0,
            sample_keys={_point_key(0.0, 0.0)},
        )
        _add_rectangle(state, root)
        states.append(state)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    checkpoint_index = 0
    start = time.perf_counter()
    while checkpoint_index < len(budgets):
        checkpoint = budgets[checkpoint_index]
        while any(state.adaptive_evaluations < checkpoint for state in states):
            selections: list[
                tuple[int, int, list[tuple[float, float]], list[int]]
            ] = []
            fallback_pairs: list[tuple[int, tuple[float, float]]] = []
            for state_index, state in enumerate(states):
                remaining = checkpoint - state.adaptive_evaluations
                if remaining <= 0:
                    continue
                if not state.direct_queue:
                    state.direct_queue = _potentially_optimal_ids(state)
                selected_any = False
                while state.direct_queue:
                    affordable_index = next(
                        (
                            index
                            for index, rid in enumerate(state.direct_queue)
                            if rid in state.rectangles
                            and state.rectangles[rid].canonical_cost <= remaining
                        ),
                        None,
                    )
                    if affordable_index is None:
                        break
                    rid = state.direct_queue.pop(affordable_index)
                    rectangle = state.rectangles[rid]
                    points, dimensions = _canonical_probes(rectangle)
                    selections.append((state_index, rid, points, dimensions))
                    remaining -= len(points)
                    selected_any = True
                    if remaining < 2:
                        break
                if not selected_any and remaining >= 2:
                    pair = _fallback_boundary_pair(state)
                    if pair is None:
                        raise RuntimeError("Unable to complete exact DIRECT evaluation budget")
                    fallback_pairs.extend((state_index, point) for point in pair)

            if not selections and not fallback_pairs:
                raise RuntimeError("DIRECT made no progress")

            if selections:
                flat_points: list[tuple[float, float]] = []
                flat_trajectory_ids: list[int] = []
                for state_index, _, points, _ in selections:
                    flat_points.extend(points)
                    flat_trajectory_ids.extend([state_index] * len(points))
                theta = torch.tensor(flat_points, dtype=DTYPE)
                selected_positions = positions[
                    torch.tensor(flat_trajectory_ids, dtype=torch.long)
                ]
                evaluated = _prefix_losses(theta, selected_positions).detach().cpu().tolist()
                cursor = 0
                boundary_requests: list[tuple[int, tuple[float, float]]] = []
                pending_boundary_counts = [0] * len(states)
                planned_direct_counts = [0] * len(states)
                processed_direct_counts = [0] * len(states)
                for state_index, _, points, _ in selections:
                    planned_direct_counts[state_index] += len(points)
                for state_index, rid, points, dimensions in selections:
                    state = states[state_index]
                    local_losses = [float(value) for value in evaluated[cursor : cursor + len(points)]]
                    cursor += len(points)
                    parent = state.rectangles[rid]
                    parent_copy = DirectRectangle(
                        parent.rid,
                        parent.center_a,
                        parent.center_b,
                        parent.exponent_a,
                        parent.exponent_b,
                        parent.loss,
                    )
                    for point, loss in zip(points, local_losses):
                        _append_sample(
                            state,
                            point,
                            loss,
                            False,
                            sample_points,
                            sample_losses,
                            sample_is_edge,
                        )
                    processed_direct_counts[state_index] += len(points)
                    _split_canonical(state, rid, points, dimensions, local_losses)
                    remaining = (
                        checkpoint
                        - state.adaptive_evaluations
                        - (
                            planned_direct_counts[state_index]
                            - processed_direct_counts[state_index]
                        )
                        - pending_boundary_counts[state_index]
                    )
                    if remaining >= 2:
                        pair = _boundary_pair_after_split(
                            state, parent_copy, points, dimensions, local_losses
                        )
                        if pair is not None:
                            boundary_requests.extend((state_index, point) for point in pair)
                            pending_boundary_counts[state_index] += 2

                if boundary_requests:
                    edge_theta = torch.tensor(
                        [point for _, point in boundary_requests], dtype=DTYPE
                    )
                    edge_positions = positions[
                        torch.tensor(
                            [state_index for state_index, _ in boundary_requests],
                            dtype=torch.long,
                        )
                    ]
                    edge_losses = _prefix_losses(edge_theta, edge_positions).detach().cpu().tolist()
                    for (state_index, point), loss in zip(boundary_requests, edge_losses):
                        state = states[state_index]
                        _append_sample(
                            state,
                            point,
                            float(loss),
                            True,
                            sample_points,
                            sample_losses,
                            sample_is_edge,
                        )

            if fallback_pairs:
                theta = torch.tensor([point for _, point in fallback_pairs], dtype=DTYPE)
                fallback_positions = positions[
                    torch.tensor(
                        [state_index for state_index, _ in fallback_pairs], dtype=torch.long
                    )
                ]
                evaluated = _prefix_losses(theta, fallback_positions).detach().cpu().tolist()
                for (state_index, point), loss in zip(fallback_pairs, evaluated):
                    _append_sample(
                        states[state_index],
                        point,
                        float(loss),
                        True,
                        sample_points,
                        sample_losses,
                        sample_is_edge,
                    )

        checkpoint_rows, summary = _snapshot_rows(
            checkpoint,
            states,
            references,
            sample_points,
            sample_losses,
            sample_is_edge,
        )
        summary["cumulative_runtime_seconds"] = time.perf_counter() - start
        all_rows.extend(checkpoint_rows)
        summaries.append(summary)
        print(
            f"budget={checkpoint} median_error={summary['median_nearest_sampled_parameter_error']:.6g} "
            f"worst_error={summary['worst_nearest_sampled_parameter_error']:.6g} "
            f"median_loss={summary['median_best_observed_prefix_loss']:.6g} "
            f"po_fraction={summary['fraction_true_rectangle_potentially_optimal']:.4f} "
            f"elapsed={summary['cumulative_runtime_seconds']:.1f}s",
            flush=True,
        )
        checkpoint_index += 1
    return all_rows, summaries


def _plot(path: Path, summaries: list[dict[str, object]]) -> None:
    budgets = np.asarray([row["adaptive_evaluation_budget"] for row in summaries])
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes[0, 0].plot(
        budgets,
        [row["median_nearest_sampled_parameter_error"] for row in summaries],
        marker="o",
        label="median",
    )
    axes[0, 0].plot(
        budgets,
        [row["worst_nearest_sampled_parameter_error"] for row in summaries],
        marker="o",
        label="worst",
    )
    axes[0, 0].set_ylabel("nearest sampled parameter error")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        budgets,
        [max(float(row["median_best_observed_prefix_loss"]), LOSS_EPSILON) for row in summaries],
        marker="o",
        label="median",
    )
    axes[0, 1].plot(
        budgets,
        [max(float(row["worst_best_observed_prefix_loss"]), LOSS_EPSILON) for row in summaries],
        marker="o",
        label="worst",
    )
    axes[0, 1].set_ylabel("best observed prefix loss")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(frameon=False)

    for threshold, label in (
        ("1e-1", "1e-1"),
        ("1e-2", "1e-2"),
        ("1e-3", "1e-3"),
        ("1e-4", "1e-4"),
    ):
        axes[1, 0].plot(
            budgets,
            [row[f"fraction_parameter_error_le_{threshold}"] for row in summaries],
            marker="o",
            label=label,
        )
    axes[1, 0].set_ylabel("fraction within parameter error")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend(frameon=False, ncol=2)

    axes[1, 1].plot(
        budgets,
        [row["fraction_true_rectangle_potentially_optimal"] for row in summaries],
        marker="o",
        label="true rectangle on PO hull",
    )
    axes[1, 1].plot(
        budgets,
        [row["fraction_prefix_loss_le_1e-2"] for row in summaries],
        marker="o",
        label="loss <= 1e-2",
    )
    axes[1, 1].plot(
        budgets,
        [row["fraction_prefix_loss_le_1e-8"] for row in summaries],
        marker="o",
        label="loss <= 1e-8",
    )
    axes[1, 1].set_ylabel("fraction of trajectories")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("adaptive prefix-loss evaluations")
        axis.grid(alpha=0.25)
    figure.suptitle("Pure DIRECT on trajectory-unique continuous dynamics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sho/stage1B/pure_direct_unique_scaling"),
    )
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_size = 12 if args.smoke else args.test_size
    budgets = (20, 40, 80) if args.smoke else DEFAULT_BUDGETS
    positions, _, hidden_c = _make_exact_labeled_split(test_size, FRESH_TEST_SEED)
    positions = positions.to(dtype=DTYPE)
    references_tensor = _posthoc_reference(hidden_c)
    references = references_tensor.cpu().numpy()
    if len(np.unique(references, axis=0)) != test_size:
        raise RuntimeError("Fresh test dynamics are not unique")
    per_trajectory_rows, summaries = _run_continuing_direct(
        positions, references, budgets
    )
    _write_csv(args.output_dir / "direct_unique_summary.csv", summaries)
    _write_csv(args.output_dir / "per_trajectory_direct.csv", per_trajectory_rows)
    _plot(args.output_dir / "direct_accuracy_vs_evaluations.png", summaries)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
