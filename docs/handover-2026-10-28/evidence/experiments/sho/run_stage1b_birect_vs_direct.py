#!/usr/bin/env python3
"""BIRECT versus center-sampling DIRECT for the 2-D N1 handoff gate.

BIRECT rule used (Paulavicius, Chiter, and Zilinskas, 2018):

* A rectangle owns two samples at one-third and two-thirds along a diagonal.
* Its selection value is the smaller sample loss and its size is two-thirds of
  the Euclidean diagonal length.
* It is bisected along one longest coordinate, with the lowest coordinate
  index breaking ties.
* Each child inherits one parent sample.  Its second sample is obtained by
  shifting the opposite parent sample by half the parent's branching side.

Both methods use the same exact lower-right-hull test (all rectangles optimal
for some positive Lipschitz constant).  The optional sufficient-decrease
epsilon is intentionally omitted for both methods so this experiment isolates
sampling/partition geometry rather than changing the selection heuristic.
"""

from __future__ import annotations

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

from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    DirectRectangle,
    DirectState,
    SizeGroup,
    _add_rectangle,
    _boundary_pair_after_split,
    _canonical_probes,
    _fallback_boundary_pair,
    _make_exact_labeled_split,
    _point_key,
    _potentially_optimal_ids,
    _split_canonical,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    UPPER,
    _posthoc_reference,
    _prefix_losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE


OUTPUT_DIR = Path("outputs/sho/stage1B/birect_vs_direct")
SOURCE_RESULTS = Path(
    "outputs/sho/stage1B/pure_direct_unique_scaling/per_trajectory_direct.csv"
)
TRAJECTORY_COUNT = 256
HANDOFF_GATE = 1.0e-4
MAX_EVALUATIONS = 5000
BIRECT_RULE = (
    "two samples at diagonal 1/3 and 2/3; score=min sample loss; "
    "size=(2/3)*Euclidean diagonal; bisect one longest coordinate with "
    "lowest-index tie break; each child inherits one sample and evaluates "
    "one half-side-shifted sample"
)
SELECTION_RULE = (
    "exact lower-right hull: potentially optimal for some K>0; no optional "
    "sufficient-decrease epsilon, identically for DIRECT and BIRECT"
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
class StandardRuntime:
    state: DirectState
    best_loss: float
    best_a: float
    best_b: float
    evaluations: int
    start_time: float
    finished: bool = False
    success: bool = False


@dataclass(slots=True)
class BirectRectangle:
    rid: int
    lower_a: float
    upper_a: float
    lower_b: float
    upper_b: float
    exponent_a: int
    exponent_b: int
    l_a: float
    l_b: float
    u_a: float
    u_b: float
    l_loss: float
    u_loss: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.exponent_a, self.exponent_b)

    @property
    def side_a(self) -> float:
        return self.upper_a - self.lower_a

    @property
    def side_b(self) -> float:
        return self.upper_b - self.lower_b

    @property
    def size(self) -> float:
        return (2.0 / 3.0) * math.hypot(self.side_a, self.side_b)

    @property
    def score(self) -> float:
        return min(self.l_loss, self.u_loss)


@dataclass(slots=True)
class BirectGroup:
    heap: list[tuple[float, int]] = field(default_factory=list)
    ids: set[int] = field(default_factory=set)


@dataclass(slots=True)
class BirectState:
    trajectory_id: int
    rectangles: dict[int, BirectRectangle]
    groups: dict[tuple[int, int], BirectGroup]
    next_rid: int
    queue: list[int]
    best_loss: float
    best_a: float
    best_b: float
    evaluations: int
    start_time: float
    finished: bool = False
    success: bool = False


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


def _update_best(
    runtime: StandardRuntime | BirectState,
    points: list[tuple[float, float]],
    losses: list[float],
) -> None:
    for point, loss in zip(points, losses):
        if loss < runtime.best_loss:
            runtime.best_loss = float(loss)
            runtime.best_a = float(point[0])
            runtime.best_b = float(point[1])


def _standard_outcomes(positions: torch.Tensor) -> list[SearchOutcome]:
    started = time.perf_counter()
    centers = torch.zeros((TRAJECTORY_COUNT, 2), dtype=DTYPE)
    losses = _prefix_losses(centers, positions).detach().cpu().tolist()
    runtimes: list[StandardRuntime] = []
    for trajectory_id, loss in enumerate(losses):
        root = DirectRectangle(0, 0.0, 0.0, 0, 0, float(loss))
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
        runtime = StandardRuntime(
            state=state,
            best_loss=float(loss),
            best_a=0.0,
            best_b=0.0,
            evaluations=1,
            start_time=started,
        )
        if runtime.best_loss <= HANDOFF_GATE:
            runtime.finished = True
            runtime.success = True
        runtimes.append(runtime)

    while any(not runtime.finished for runtime in runtimes):
        selections: list[
            tuple[int, int, list[tuple[float, float]], list[int]]
        ] = []
        fallback_requests: list[tuple[int, tuple[float, float]]] = []
        for runtime_index, runtime in enumerate(runtimes):
            if runtime.finished:
                continue
            remaining = MAX_EVALUATIONS - runtime.evaluations
            if remaining < 2:
                runtime.finished = True
                continue
            state = runtime.state
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
                points, dimensions = _canonical_probes(state.rectangles[rid])
                selections.append((runtime_index, rid, points, dimensions))
                remaining -= len(points)
                selected_any = True
                if remaining < 2:
                    break
            if not selected_any and remaining >= 2:
                pair = _fallback_boundary_pair(state)
                if pair is not None:
                    fallback_requests.extend((runtime_index, point) for point in pair)
                else:
                    runtime.finished = True

        if not selections and not fallback_requests:
            break

        if selections:
            flat_points: list[tuple[float, float]] = []
            flat_indices: list[int] = []
            for runtime_index, _, points, _ in selections:
                flat_points.extend(points)
                flat_indices.extend([runtime_index] * len(points))
            theta = torch.tensor(flat_points, dtype=DTYPE)
            selected_positions = positions[
                torch.tensor(flat_indices, dtype=torch.long)
            ]
            evaluated = _prefix_losses(theta, selected_positions).detach().cpu().tolist()
            cursor = 0
            boundary_candidates: list[
                tuple[int, DirectRectangle, list[tuple[float, float]], list[int], list[float]]
            ] = []
            for runtime_index, rid, points, dimensions in selections:
                runtime = runtimes[runtime_index]
                local_losses = [float(value) for value in evaluated[cursor : cursor + len(points)]]
                cursor += len(points)
                runtime.evaluations += len(points)
                runtime.state.adaptive_evaluations += len(points)
                runtime.state.sample_count += len(points)
                for point in points:
                    runtime.state.sample_keys.add(_point_key(*point))
                _update_best(runtime, points, local_losses)
                parent = runtime.state.rectangles[rid]
                parent_copy = DirectRectangle(
                    parent.rid,
                    parent.center_a,
                    parent.center_b,
                    parent.exponent_a,
                    parent.exponent_b,
                    parent.loss,
                )
                _split_canonical(runtime.state, rid, points, dimensions, local_losses)
                boundary_candidates.append(
                    (runtime_index, parent_copy, points, dimensions, local_losses)
                )

            for runtime in runtimes:
                if not runtime.finished and runtime.best_loss <= HANDOFF_GATE:
                    runtime.success = True
                    runtime.finished = True

            boundary_requests: list[tuple[int, tuple[float, float]]] = []
            pending = [0] * len(runtimes)
            for runtime_index, parent, points, dimensions, local_losses in boundary_candidates:
                runtime = runtimes[runtime_index]
                if runtime.finished:
                    continue
                remaining = MAX_EVALUATIONS - runtime.evaluations - pending[runtime_index]
                if remaining < 2:
                    continue
                pair = _boundary_pair_after_split(
                    runtime.state, parent, points, dimensions, local_losses
                )
                if pair is not None:
                    boundary_requests.extend((runtime_index, point) for point in pair)
                    pending[runtime_index] += 2
            if boundary_requests:
                theta = torch.tensor([point for _, point in boundary_requests], dtype=DTYPE)
                boundary_positions = positions[
                    torch.tensor(
                        [runtime_index for runtime_index, _ in boundary_requests],
                        dtype=torch.long,
                    )
                ]
                boundary_losses = _prefix_losses(theta, boundary_positions).detach().cpu().tolist()
                by_runtime: dict[int, tuple[list[tuple[float, float]], list[float]]] = {}
                for (runtime_index, point), loss in zip(boundary_requests, boundary_losses):
                    points_for_runtime, losses_for_runtime = by_runtime.setdefault(
                        runtime_index, ([], [])
                    )
                    points_for_runtime.append(point)
                    losses_for_runtime.append(float(loss))
                for runtime_index, (points_for_runtime, losses_for_runtime) in by_runtime.items():
                    runtime = runtimes[runtime_index]
                    runtime.evaluations += len(points_for_runtime)
                    runtime.state.adaptive_evaluations += len(points_for_runtime)
                    runtime.state.sample_count += len(points_for_runtime)
                    runtime.state.edge_probe_count += len(points_for_runtime)
                    for point in points_for_runtime:
                        runtime.state.sample_keys.add(_point_key(*point))
                    _update_best(runtime, points_for_runtime, losses_for_runtime)
                    if runtime.best_loss <= HANDOFF_GATE:
                        runtime.success = True
                        runtime.finished = True

        if fallback_requests:
            theta = torch.tensor([point for _, point in fallback_requests], dtype=DTYPE)
            fallback_positions = positions[
                torch.tensor(
                    [runtime_index for runtime_index, _ in fallback_requests],
                    dtype=torch.long,
                )
            ]
            evaluated = _prefix_losses(theta, fallback_positions).detach().cpu().tolist()
            by_runtime: dict[int, tuple[list[tuple[float, float]], list[float]]] = {}
            for (runtime_index, point), loss in zip(fallback_requests, evaluated):
                points_for_runtime, losses_for_runtime = by_runtime.setdefault(
                    runtime_index, ([], [])
                )
                points_for_runtime.append(point)
                losses_for_runtime.append(float(loss))
            for runtime_index, (points_for_runtime, losses_for_runtime) in by_runtime.items():
                runtime = runtimes[runtime_index]
                runtime.evaluations += len(points_for_runtime)
                runtime.state.adaptive_evaluations += len(points_for_runtime)
                runtime.state.sample_count += len(points_for_runtime)
                runtime.state.edge_probe_count += len(points_for_runtime)
                for point in points_for_runtime:
                    runtime.state.sample_keys.add(_point_key(*point))
                _update_best(runtime, points_for_runtime, losses_for_runtime)
                if runtime.best_loss <= HANDOFF_GATE:
                    runtime.success = True
                    runtime.finished = True

        for runtime in runtimes:
            if not runtime.finished and runtime.evaluations >= MAX_EVALUATIONS - 1:
                runtime.finished = True

    end = time.perf_counter()
    return [
        SearchOutcome(
            trajectory_id=index,
            success=runtime.success,
            evaluations=runtime.evaluations,
            best_loss=runtime.best_loss,
            best_a=runtime.best_a,
            best_b=runtime.best_b,
            runtime_seconds=end - runtime.start_time,
        )
        for index, runtime in enumerate(runtimes)
    ]


def _birect_add(state: BirectState, rectangle: BirectRectangle) -> None:
    state.rectangles[rectangle.rid] = rectangle
    group = state.groups.setdefault(rectangle.key, BirectGroup())
    group.ids.add(rectangle.rid)
    heapq.heappush(group.heap, (rectangle.score, rectangle.rid))


def _birect_remove(state: BirectState, rid: int) -> BirectRectangle:
    rectangle = state.rectangles.pop(rid)
    state.groups[rectangle.key].ids.discard(rid)
    return rectangle


def _birect_group_minima(
    state: BirectState, key: tuple[int, int], group: BirectGroup
) -> tuple[float, list[int]] | None:
    while group.heap:
        score, rid = group.heap[0]
        rectangle = state.rectangles.get(rid)
        if rectangle is None or rectangle.key != key or rectangle.score != score:
            heapq.heappop(group.heap)
            continue
        tolerance = 5.0e-14 * max(1.0, abs(score))
        tied = [
            active_rid
            for active_rid in group.ids
            if abs(state.rectangles[active_rid].score - score) <= tolerance
        ]
        return score, sorted(tied)
    return None


def _birect_potentially_optimal(state: BirectState) -> list[int]:
    size_candidates: list[tuple[float, float, list[int]]] = []
    empty_keys: list[tuple[int, int]] = []
    for key, group in state.groups.items():
        minimum = _birect_group_minima(state, key, group)
        if minimum is None:
            empty_keys.append(key)
            continue
        score, ids = minimum
        size_candidates.append((state.rectangles[ids[0]].size, score, ids))
    for key in empty_keys:
        del state.groups[key]
    size_candidates.sort(key=lambda item: (item[0], item[1], item[2][0]))

    groups: list[tuple[float, float, list[int]]] = []
    for size, score, ids in size_candidates:
        if not groups or abs(size - groups[-1][0]) > 1.0e-14:
            groups.append((size, score, list(ids)))
            continue
        old_size, old_score, old_ids = groups[-1]
        tolerance = 5.0e-14 * max(1.0, abs(old_score), abs(score))
        if score < old_score - tolerance:
            groups[-1] = (old_size, score, list(ids))
        elif abs(score - old_score) <= tolerance:
            groups[-1] = (old_size, old_score, sorted(old_ids + ids))

    hull: list[int] = []
    for group_index, (size, score, _) in enumerate(groups):
        while len(hull) >= 2:
            first = groups[hull[-2]]
            second = groups[hull[-1]]
            cross = (second[0] - first[0]) * (score - first[1]) - (
                second[1] - first[1]
            ) * (size - first[0])
            scale = max(1.0, abs(first[1]), abs(second[1]), abs(score))
            if cross < -5.0e-14 * scale:
                hull.pop()
            else:
                break
        hull.append(group_index)

    eligible: list[int] = []
    for position, group_index in enumerate(hull):
        current = groups[group_index]
        lower_slope = -math.inf
        upper_slope = math.inf
        if position > 0:
            previous = groups[hull[position - 1]]
            lower_slope = (current[1] - previous[1]) / (current[0] - previous[0])
        if position < len(hull) - 1:
            following = groups[hull[position + 1]]
            upper_slope = (following[1] - current[1]) / (
                following[0] - current[0]
            )
        if upper_slope > 0.0 and max(lower_slope, 0.0) <= upper_slope + 5.0e-14:
            eligible.extend(current[2])
    return sorted(
        eligible,
        key=lambda rid: (
            -state.rectangles[rid].size,
            state.rectangles[rid].score,
            rid,
        ),
    )


def _birect_new_points(
    rectangle: BirectRectangle,
) -> tuple[int, tuple[float, float], tuple[float, float]]:
    if rectangle.side_a >= rectangle.side_b - 1.0e-15:
        branch = 0
        side = rectangle.side_a
        l_coordinate = rectangle.l_a
        u_coordinate = rectangle.u_a
    else:
        branch = 1
        side = rectangle.side_b
        l_coordinate = rectangle.l_b
        u_coordinate = rectangle.u_b
    l_point = [rectangle.l_a, rectangle.l_b]
    u_point = [rectangle.u_a, rectangle.u_b]
    if l_coordinate < u_coordinate:
        upper_child_new = list(l_point)
        upper_child_new[branch] += side / 2.0
        lower_child_new = list(u_point)
        lower_child_new[branch] -= side / 2.0
    else:
        lower_child_new = list(l_point)
        lower_child_new[branch] -= side / 2.0
        upper_child_new = list(u_point)
        upper_child_new[branch] += side / 2.0
    return branch, tuple(lower_child_new), tuple(upper_child_new)


def _birect_split(
    state: BirectState,
    rid: int,
    branch: int,
    lower_new: tuple[float, float],
    upper_new: tuple[float, float],
    lower_new_loss: float,
    upper_new_loss: float,
) -> None:
    parent = _birect_remove(state, rid)
    middle = (
        0.5 * (parent.lower_a + parent.upper_a)
        if branch == 0
        else 0.5 * (parent.lower_b + parent.upper_b)
    )
    l_less = (
        parent.l_a < parent.u_a if branch == 0 else parent.l_b < parent.u_b
    )
    if branch == 0:
        lower_bounds = (parent.lower_a, middle, parent.lower_b, parent.upper_b)
        upper_bounds = (middle, parent.upper_a, parent.lower_b, parent.upper_b)
    else:
        lower_bounds = (parent.lower_a, parent.upper_a, parent.lower_b, middle)
        upper_bounds = (parent.lower_a, parent.upper_a, middle, parent.upper_b)
    child_exponent_a = parent.exponent_a + int(branch == 0)
    child_exponent_b = parent.exponent_b + int(branch == 1)

    if l_less:
        lower_l = (parent.l_a, parent.l_b, parent.l_loss)
        lower_u = (lower_new[0], lower_new[1], lower_new_loss)
        upper_l = (upper_new[0], upper_new[1], upper_new_loss)
        upper_u = (parent.u_a, parent.u_b, parent.u_loss)
    else:
        lower_l = (lower_new[0], lower_new[1], lower_new_loss)
        lower_u = (parent.u_a, parent.u_b, parent.u_loss)
        upper_l = (parent.l_a, parent.l_b, parent.l_loss)
        upper_u = (upper_new[0], upper_new[1], upper_new_loss)

    for bounds, l_sample, u_sample in (
        (lower_bounds, lower_l, lower_u),
        (upper_bounds, upper_l, upper_u),
    ):
        child = BirectRectangle(
            rid=state.next_rid,
            lower_a=bounds[0],
            upper_a=bounds[1],
            lower_b=bounds[2],
            upper_b=bounds[3],
            exponent_a=child_exponent_a,
            exponent_b=child_exponent_b,
            l_a=l_sample[0],
            l_b=l_sample[1],
            u_a=u_sample[0],
            u_b=u_sample[1],
            l_loss=l_sample[2],
            u_loss=u_sample[2],
        )
        state.next_rid += 1
        _birect_add(state, child)


def _assert_birect_geometry(rectangle: BirectRectangle) -> None:
    for lower, upper, first, second in (
        (rectangle.lower_a, rectangle.upper_a, rectangle.l_a, rectangle.u_a),
        (rectangle.lower_b, rectangle.upper_b, rectangle.l_b, rectangle.u_b),
    ):
        side = upper - lower
        targets = sorted((lower + side / 3.0, lower + 2.0 * side / 3.0))
        actual = sorted((first, second))
        if not np.allclose(targets, actual, rtol=0.0, atol=2.0e-12):
            raise RuntimeError("BIRECT child samples violated diagonal 1/3-2/3 geometry")


def _birect_outcomes(positions: torch.Tensor) -> list[SearchOutcome]:
    started = time.perf_counter()
    one_third = LOWER + DOMAIN_WIDTH / 3.0
    two_thirds = LOWER + 2.0 * DOMAIN_WIDTH / 3.0
    initial_theta = torch.tensor(
        [[one_third, one_third], [two_thirds, two_thirds]], dtype=DTYPE
    ).repeat(TRAJECTORY_COUNT, 1)
    trajectory_indices = torch.arange(TRAJECTORY_COUNT).repeat_interleave(2)
    initial_losses = _prefix_losses(
        initial_theta, positions[trajectory_indices]
    ).detach().cpu().reshape(TRAJECTORY_COUNT, 2)
    states: list[BirectState] = []
    for trajectory_id in range(TRAJECTORY_COUNT):
        losses = initial_losses[trajectory_id].tolist()
        best_index = int(np.argmin(losses))
        samples = ((one_third, one_third), (two_thirds, two_thirds))
        root = BirectRectangle(
            0,
            LOWER,
            UPPER,
            LOWER,
            UPPER,
            0,
            0,
            one_third,
            one_third,
            two_thirds,
            two_thirds,
            float(losses[0]),
            float(losses[1]),
        )
        state = BirectState(
            trajectory_id=trajectory_id,
            rectangles={},
            groups={},
            next_rid=1,
            queue=[],
            best_loss=float(losses[best_index]),
            best_a=samples[best_index][0],
            best_b=samples[best_index][1],
            evaluations=2,
            start_time=started,
        )
        _birect_add(state, root)
        if state.best_loss <= HANDOFF_GATE:
            state.finished = True
            state.success = True
        states.append(state)

    while any(not state.finished for state in states):
        selections: list[
            tuple[int, int, int, tuple[float, float], tuple[float, float]]
        ] = []
        for state_index, state in enumerate(states):
            if state.finished:
                continue
            remaining = MAX_EVALUATIONS - state.evaluations
            if remaining < 2:
                state.finished = True
                continue
            if not state.queue:
                state.queue = _birect_potentially_optimal(state)
            while state.queue and remaining >= 2:
                rid = state.queue.pop(0)
                if rid not in state.rectangles:
                    continue
                branch, lower_new, upper_new = _birect_new_points(
                    state.rectangles[rid]
                )
                selections.append(
                    (state_index, rid, branch, lower_new, upper_new)
                )
                remaining -= 2
            if not state.queue and not any(item[0] == state_index for item in selections):
                state.finished = True

        if not selections:
            break
        flat_points: list[tuple[float, float]] = []
        flat_indices: list[int] = []
        for state_index, _, _, lower_new, upper_new in selections:
            flat_points.extend((lower_new, upper_new))
            flat_indices.extend((state_index, state_index))
        theta = torch.tensor(flat_points, dtype=DTYPE)
        selected_positions = positions[
            torch.tensor(flat_indices, dtype=torch.long)
        ]
        evaluated = _prefix_losses(theta, selected_positions).detach().cpu().tolist()
        cursor = 0
        for state_index, rid, branch, lower_new, upper_new in selections:
            state = states[state_index]
            lower_loss = float(evaluated[cursor])
            upper_loss = float(evaluated[cursor + 1])
            cursor += 2
            state.evaluations += 2
            _update_best(
                state,
                [lower_new, upper_new],
                [lower_loss, upper_loss],
            )
            _birect_split(
                state,
                rid,
                branch,
                lower_new,
                upper_new,
                lower_loss,
                upper_loss,
            )
        for state in states:
            if state.finished:
                continue
            if state.best_loss <= HANDOFF_GATE:
                state.success = True
                state.finished = True
            elif state.evaluations >= MAX_EVALUATIONS:
                state.finished = True

    for state in states:
        for rectangle in state.rectangles.values():
            _assert_birect_geometry(rectangle)
    end = time.perf_counter()
    return [
        SearchOutcome(
            trajectory_id=state.trajectory_id,
            success=state.success,
            evaluations=state.evaluations,
            best_loss=state.best_loss,
            best_a=state.best_a,
            best_b=state.best_b,
            runtime_seconds=end - state.start_time,
        )
        for state in states
    ]


def _source_failure_sets() -> dict[str, set[int]]:
    rows = [
        row
        for row in _read_csv(SOURCE_RESULTS)
        if int(row["adaptive_evaluation_budget"]) == 5000
    ]
    failed = {
        int(row["trajectory_id"])
        for row in rows
        if float(row["best_observed_prefix_loss"]) > HANDOFF_GATE
    }
    near = {
        int(row["trajectory_id"])
        for row in rows
        if int(row["trajectory_id"]) in failed
        and float(row["distance_to_global_boundary_posthoc"]) <= 0.2
    }
    very = {
        int(row["trajectory_id"])
        for row in rows
        if int(row["trajectory_id"]) in failed
        and float(row["distance_to_global_boundary_posthoc"]) <= 0.05
    }
    interior = failed - near
    if (len(near), len(very), len(interior)) != (26, 6, 29):
        raise RuntimeError("Previous DIRECT failure cohorts do not match 26/6/29")
    return {
        "previous_direct_failed_near_edge": near,
        "previous_direct_failed_very_near_edge": very,
        "previous_direct_failed_interior": interior,
    }


def _comparison_rows(
    outcomes: dict[str, list[SearchOutcome]], references: np.ndarray
) -> list[dict[str, object]]:
    previous = _source_failure_sets()
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
                        outcome.trajectory_id
                        in previous["previous_direct_failed_near_edge"]
                    ),
                    "was_previous_direct_failed_very_near_edge": int(
                        outcome.trajectory_id
                        in previous["previous_direct_failed_very_near_edge"]
                    ),
                    "was_previous_direct_failed_interior": int(
                        outcome.trajectory_id
                        in previous["previous_direct_failed_interior"]
                    ),
                    "birect_recovered_previous_failure": int(
                        method == "birect"
                        and outcome.success
                        and outcome.trajectory_id
                        in (
                            previous["previous_direct_failed_near_edge"]
                            | previous["previous_direct_failed_interior"]
                        )
                    ),
                    "birect_rule": BIRECT_RULE if method == "birect" else "",
                    "potentially_optimal_selection_rule": SELECTION_RULE,
                }
            )
    return rows


def _summary_rows(comparison: list[dict[str, object]]) -> list[dict[str, object]]:
    previous_groups = {
        "previous_direct_failed_near_edge": "was_previous_direct_failed_near_edge",
        "previous_direct_failed_very_near_edge": "was_previous_direct_failed_very_near_edge",
        "previous_direct_failed_interior": "was_previous_direct_failed_interior",
    }
    ordinary_groups = {
        "all": None,
        "interior": "is_interior_gt_0p2",
        "near_edge": "is_near_edge_le_0p2",
        "very_near_edge": "is_very_near_edge_le_0p05",
    }
    summaries: list[dict[str, object]] = []
    for method in ("standard_direct", "birect"):
        method_rows = [row for row in comparison if row["method"] == method]
        group_map = dict(ordinary_groups)
        if method == "birect":
            group_map.update(previous_groups)
        for group, key in group_map.items():
            rows = method_rows if key is None else [row for row in method_rows if int(row[key]) == 1]
            successes = [row for row in rows if int(row["success_loss_le_1e-4"]) == 1]
            failures = [row for row in rows if int(row["success_loss_le_1e-4"]) == 0]
            evaluations = [float(row["objective_evaluations"]) for row in successes]
            errors = [float(row["handoff_parameter_error_posthoc"]) for row in successes]
            summaries.append(
                {
                    "method": method,
                    "subgroup": group,
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
                    "failure_count_at_5000": len(failures),
                    "failure_trajectory_ids": ";".join(
                        str(row["trajectory_id"]) for row in failures
                    ),
                    "gate": HANDOFF_GATE,
                    "cap": MAX_EVALUATIONS,
                    "birect_rule": BIRECT_RULE if method == "birect" else "",
                    "potentially_optimal_selection_rule": SELECTION_RULE,
                }
            )
    return summaries


def _plot(path: Path, comparison: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    groups = (
        ("all", lambda row: True),
        ("interior", lambda row: int(row["is_interior_gt_0p2"]) == 1),
        ("near edge", lambda row: int(row["is_near_edge_le_0p2"]) == 1),
        ("very near edge", lambda row: int(row["is_very_near_edge_le_0p05"]) == 1),
    )
    for axis, (group_name, predicate) in zip(axes.flat, groups):
        for method, color in (("standard_direct", "tab:orange"), ("birect", "tab:blue")):
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
        axis.set_title(group_name)
        axis.set_xlabel("objective evaluations")
        axis.set_ylabel("fraction reaching loss <= 1e-4")
        axis.set_xlim(0, MAX_EVALUATIONS)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    if not SOURCE_RESULTS.exists():
        raise FileNotFoundError(SOURCE_RESULTS)
    positions, _, hidden_c = _make_exact_labeled_split(
        TRAJECTORY_COUNT, FRESH_TEST_SEED
    )
    positions = positions.to(dtype=DTYPE)
    references = _posthoc_reference(hidden_c).cpu().numpy()
    source_final = sorted(
        (
            row
            for row in _read_csv(SOURCE_RESULTS)
            if int(row["adaptive_evaluation_budget"]) == 5000
        ),
        key=lambda row: int(row["trajectory_id"]),
    )
    source_references = np.asarray(
        [
            [float(row["reference_a_posthoc"]), float(row["reference_b_posthoc"])]
            for row in source_final
        ]
    )
    if not np.array_equal(references, source_references):
        raise RuntimeError("Generated trajectories do not match the previous DIRECT diagnostic")

    print("running standard DIRECT baseline", flush=True)
    direct_outcomes = _standard_outcomes(positions)
    print(
        f"standard DIRECT success={sum(outcome.success for outcome in direct_outcomes)}/256",
        flush=True,
    )
    print("running published-geometry BIRECT", flush=True)
    birect_outcomes = _birect_outcomes(positions)
    print(
        f"BIRECT success={sum(outcome.success for outcome in birect_outcomes)}/256",
        flush=True,
    )
    comparison = _comparison_rows(
        {"standard_direct": direct_outcomes, "birect": birect_outcomes},
        references,
    )
    summaries = _summary_rows(comparison)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "birect_vs_direct_summary.csv", summaries)
    _write_csv(OUTPUT_DIR / "per_trajectory_comparison.csv", comparison)
    _plot(OUTPUT_DIR / "handoff_success_vs_evaluations.png", comparison)
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
