#!/usr/bin/env python3
"""Stage 1B: learned prioritization inside a faithful 2-D DIRECT scaffold.

The learned policy never creates or deletes rectangles.  It may only prioritize
rectangles that are potentially optimal under the exact lower-right-hull test.
Persistently skipped potentially-optimal rectangles are subdivided by a
deterministic anti-starvation rule.  A pure DIRECT control uses identical
geometry, loss evaluations, and evidence-driven boundary probes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic import (
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    P,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    UPPER,
    VALIDATION_DATA_SEED,
    _posthoc_reference,
    _prefix_losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


DEVICE = torch.device("cpu")
DIM = 2
DOMAIN_DIAGONAL = math.sqrt(2.0) * DOMAIN_WIDTH
EPS_LOSS = 1.0e-30
STARVATION_LIMIT = 3
BUDGETS = (40, 80, 160)
SCALE_LABELS = ("coarse", "medium", "fine", "ultrafine")


@dataclass
class Rectangle:
    rid: int
    center: torch.Tensor
    half: torch.Tensor
    loss: float
    improvement: float = 0.0
    skipped: int = 0

    @property
    def sides(self) -> torch.Tensor:
        return 2.0 * self.half

    @property
    def diagonal(self) -> float:
        return float(torch.linalg.vector_norm(self.half).item())

    @property
    def lower(self) -> torch.Tensor:
        return self.center - self.half

    @property
    def upper(self) -> torch.Tensor:
        return self.center + self.half

    @property
    def canonical_cost(self) -> int:
        sides = self.sides
        longest = torch.isclose(sides, sides.max(), rtol=0.0, atol=1.0e-13)
        return 2 * int(longest.sum().item())


@dataclass
class SearchSnapshot:
    adaptive_evaluations: int
    active_bounds: list[tuple[np.ndarray, np.ndarray]]
    po_bounds: list[tuple[np.ndarray, np.ndarray]]
    po_count: int
    learned_selections: int
    forced_selections: int
    boundary_probes: int
    scale_counts: dict[str, int]
    large_scale_selections: int
    total_selections: int
    max_active_diagonal: float


@dataclass
class SearchState:
    trajectory_index: int
    rectangles: dict[int, Rectangle]
    next_rid: int
    sample_points: list[torch.Tensor]
    sample_losses: list[float]
    sample_kinds: list[str]
    adaptive_evaluations: int = 0
    learned_selections: int = 0
    forced_selections: int = 0
    boundary_probes: int = 0
    scale_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SCALE_LABELS}
    )
    large_scale_selections: int = 0
    total_selections: int = 0
    action_records: list[tuple[torch.Tensor, torch.Tensor, float]] = field(
        default_factory=list
    )
    direct_queue: list[int] = field(default_factory=list)
    snapshots: dict[int, SearchSnapshot] = field(default_factory=dict)

    @property
    def best_loss(self) -> float:
        return min(self.sample_losses)


class PriorityNetwork(nn.Module):
    """Shared rectangle encoder with pooled context and one score per rectangle."""

    def __init__(self, feature_dim: int = 15, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        mask_f = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        neg_inf = torch.full_like(encoded, -1.0e30)
        maximum = torch.where(mask_f.bool(), encoded, neg_inf).max(dim=1).values
        context = torch.cat((mean, maximum), dim=-1)
        expanded = context.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        scores = self.scorer(torch.cat((encoded, expanded), dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, -1.0e30)


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


def _potentially_optimal(
    rectangles: Iterable[Rectangle] | dict[int, Rectangle],
) -> list[int]:
    """Return every rectangle minimizing f-Kd for at least one K>0.

    This feasibility test is the exact lower-right hull definition, including
    same-size ties, and never selects or estimates a Lipschitz constant.
    """

    rects = list(rectangles.values()) if isinstance(rectangles, dict) else list(rectangles)
    by_id = {rectangle.rid: rectangle for rectangle in rects}
    ordered = sorted(rects, key=lambda rectangle: (rectangle.diagonal, rectangle.loss, rectangle.rid))

    # At a fixed d only the minimum-f rectangles can lie on the lower-right
    # hull. Preserve exact loss ties because all are potentially optimal.
    groups: list[tuple[float, float, list[int]]] = []
    for rectangle in ordered:
        d_value = rectangle.diagonal
        if not groups or abs(d_value - groups[-1][0]) > 1.0e-13:
            groups.append((d_value, rectangle.loss, [rectangle.rid]))
            continue
        d_group, minimum_loss, ids = groups[-1]
        loss_tolerance = 5.0e-13 * max(1.0, abs(minimum_loss), abs(rectangle.loss))
        if rectangle.loss < minimum_loss - loss_tolerance:
            groups[-1] = (d_group, rectangle.loss, [rectangle.rid])
        elif abs(rectangle.loss - minimum_loss) <= loss_tolerance:
            ids.append(rectangle.rid)

    # Lower convex hull in (d,f). Collinear groups are retained: at their
    # common positive slope they genuinely tie for the minimum f-Kd.
    hull: list[int] = []
    for group_index, (d_value, f_value, _) in enumerate(groups):
        while len(hull) >= 2:
            first = groups[hull[-2]]
            second = groups[hull[-1]]
            cross = (second[0] - first[0]) * (f_value - first[1]) - (
                second[1] - first[1]
            ) * (d_value - first[0])
            scale = max(1.0, abs(first[1]), abs(second[1]), abs(f_value))
            if cross < -5.0e-13 * scale:
                hull.pop()
            else:
                break
        hull.append(group_index)

    eligible_groups: list[int] = []
    for hull_position, group_index in enumerate(hull):
        current = groups[group_index]
        if hull_position == 0:
            lower_slope = -math.inf
        else:
            previous = groups[hull[hull_position - 1]]
            lower_slope = (current[1] - previous[1]) / (current[0] - previous[0])
        if hull_position == len(hull) - 1:
            upper_slope = math.inf
        else:
            following = groups[hull[hull_position + 1]]
            upper_slope = (following[1] - current[1]) / (following[0] - current[0])
        if upper_slope > 0.0 and max(lower_slope, 0.0) <= upper_slope + 5.0e-13:
            eligible_groups.append(group_index)

    result = [rid for group_index in eligible_groups for rid in groups[group_index][2]]
    return sorted(result, key=lambda rid: (by_id[rid].diagonal, by_id[rid].loss, rid))


def _scale_label(rectangle: Rectangle) -> str:
    maximum_side = float(rectangle.sides.max().item())
    if maximum_side > DOMAIN_WIDTH / 3.0 + 1.0e-13:
        return "coarse"
    if maximum_side > DOMAIN_WIDTH / 9.0 + 1.0e-13:
        return "medium"
    if maximum_side > DOMAIN_WIDTH / 27.0 + 1.0e-13:
        return "fine"
    return "ultrafine"


def _history_tensors(state: SearchState) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.stack(state.sample_points), torch.tensor(
        state.sample_losses, dtype=DTYPE, device=DEVICE
    )


def _rectangle_features(
    state: SearchState, candidate_ids: list[int], all_po_ids: list[int]
) -> torch.Tensor:
    points, losses = _history_tensors(state)
    best_log = math.log(state.best_loss + EPS_LOSS)
    all_diagonals = torch.tensor(
        [state.rectangles[rid].diagonal for rid in all_po_ids], dtype=DTYPE
    )
    ordered = sorted(all_po_ids, key=lambda rid: state.rectangles[rid].diagonal)
    rank = {rid: index / max(1, len(ordered) - 1) for index, rid in enumerate(ordered)}
    features: list[torch.Tensor] = []
    for rid in candidate_ids:
        rectangle = state.rectangles[rid]
        distances = torch.linalg.vector_norm(points - rectangle.center.unsqueeze(0), dim=1)
        local_radius = max(rectangle.diagonal, DOMAIN_WIDTH / 81.0)
        local = distances <= local_radius
        local_density = float(local.to(DTYPE).mean().item())
        nearby_best = float(losses[local].min().item()) if bool(local.any()) else max(losses).item()
        touches = bool(
            ((rectangle.lower <= LOWER + 1.0e-12) | (rectangle.upper >= UPPER - 1.0e-12)).any()
        )
        feature = torch.tensor(
            [
                float((rectangle.center[0] - LOWER) / DOMAIN_WIDTH),
                float((rectangle.center[1] - LOWER) / DOMAIN_WIDTH),
                float(rectangle.sides[0] / DOMAIN_WIDTH),
                float(rectangle.sides[1] / DOMAIN_WIDTH),
                rectangle.diagonal / DOMAIN_DIAGONAL,
                math.log(rectangle.loss + EPS_LOSS) - best_log,
                rank[rid],
                local_density,
                math.log(nearby_best + EPS_LOSS) - best_log,
                math.tanh(rectangle.improvement),
                rectangle.skipped / STARVATION_LIMIT,
                float(touches),
                float(rectangle.sides.max().item() / DOMAIN_WIDTH),
                float(torch.prod(rectangle.sides).item() / (DOMAIN_WIDTH**2)),
                float((all_diagonals >= rectangle.diagonal - 1.0e-14).to(DTYPE).mean().item()),
            ],
            dtype=DTYPE,
            device=DEVICE,
        )
        features.append(feature)
    return torch.stack(features)


def _initialize_states(positions: torch.Tensor) -> list[SearchState]:
    center = torch.zeros(2, dtype=DTYPE, device=DEVICE)
    root_points = center.unsqueeze(0).repeat(positions.shape[0], 1)
    root_losses = _prefix_losses(root_points, positions).detach().cpu()
    states: list[SearchState] = []
    for index, loss in enumerate(root_losses.tolist()):
        root = Rectangle(
            rid=0,
            center=center.clone(),
            half=torch.full((2,), DOMAIN_WIDTH / 2.0, dtype=DTYPE),
            loss=float(loss),
        )
        states.append(
            SearchState(
                trajectory_index=index,
                rectangles={0: root},
                next_rid=1,
                sample_points=[center.clone()],
                sample_losses=[float(loss)],
                sample_kinds=["root_center"],
            )
        )
    return states


def _longest_dimensions(rectangle: Rectangle) -> list[int]:
    sides = rectangle.sides
    maximum = sides.max()
    return [
        dimension
        for dimension in range(DIM)
        if bool(torch.isclose(sides[dimension], maximum, rtol=0.0, atol=1.0e-13))
    ]


def _canonical_probe_points(rectangle: Rectangle) -> tuple[list[torch.Tensor], list[int]]:
    points: list[torch.Tensor] = []
    dimensions: list[int] = []
    for dimension in _longest_dimensions(rectangle):
        displacement = float(rectangle.sides[dimension].item()) / 3.0
        for sign in (-1.0, 1.0):
            point = rectangle.center.clone()
            point[dimension] += sign * displacement
            points.append(point)
            dimensions.append(dimension)
    return points, dimensions


def _new_rectangle(
    state: SearchState,
    center: torch.Tensor,
    half: torch.Tensor,
    loss: float,
    parent_loss: float,
) -> Rectangle:
    rid = state.next_rid
    state.next_rid += 1
    improvement = max(
        0.0,
        math.log(parent_loss + EPS_LOSS) - math.log(float(loss) + EPS_LOSS),
    )
    return Rectangle(rid, center.clone(), half.clone(), float(loss), improvement)


def _split_canonical(
    state: SearchState,
    selected_id: int,
    probe_points: list[torch.Tensor],
    probe_dimensions: list[int],
    probe_losses: list[float],
) -> None:
    parent = state.rectangles.pop(selected_id)
    loss_by_dimension: dict[int, list[tuple[torch.Tensor, float]]] = {}
    for point, dimension, loss in zip(probe_points, probe_dimensions, probe_losses):
        loss_by_dimension.setdefault(dimension, []).append((point, float(loss)))
        state.sample_points.append(point.clone())
        state.sample_losses.append(float(loss))
        state.sample_kinds.append("direct_probe")
        state.adaptive_evaluations += 1

    order = sorted(
        loss_by_dimension,
        key=lambda dimension: min(loss for _, loss in loss_by_dimension[dimension]),
    )
    current_half = parent.half.clone()
    children: list[Rectangle] = []
    for dimension in order:
        new_half = current_half.clone()
        new_half[dimension] /= 3.0
        for point, loss in loss_by_dimension[dimension]:
            children.append(
                _new_rectangle(state, point, new_half, loss, parent.loss)
            )
        current_half[dimension] /= 3.0
    children.append(
        _new_rectangle(state, parent.center, current_half, parent.loss, parent.loss)
    )
    for child in children:
        state.rectangles[child.rid] = child


def _candidate_boundary_pairs(
    state: SearchState,
    rectangle: Rectangle,
    probe_points: list[torch.Tensor],
    probe_dimensions: list[int],
    probe_losses: list[float],
) -> list[list[torch.Tensor]]:
    """Create paired edge/near-edge probes only when geometric/loss evidence warrants."""

    by_dimension: dict[int, list[tuple[torch.Tensor, float]]] = {}
    for point, dimension, loss in zip(probe_points, probe_dimensions, probe_losses):
        by_dimension.setdefault(dimension, []).append((point, float(loss)))

    warranted: list[tuple[int, float, torch.Tensor]] = []
    for dimension in range(DIM):
        entries = sorted(by_dimension.get(dimension, []), key=lambda pair: pair[0][dimension])
        for value, direction_index in ((LOWER, 0), (UPPER, -1)):
            touches = (
                float(rectangle.lower[dimension].item()) <= LOWER + 1.0e-12
                if value == LOWER
                else float(rectangle.upper[dimension].item()) >= UPPER - 1.0e-12
            )
            improving = False
            anchor = rectangle.center.clone()
            if len(entries) == 2:
                directed = entries[direction_index]
                opposite = entries[-1 - direction_index]
                improving = directed[1] < min(rectangle.loss, opposite[1])
                anchor = directed[0].clone()
            near_boundary = abs(float(rectangle.center[dimension].item()) - value) <= max(
                float(rectangle.sides[dimension].item()), DOMAIN_WIDTH / 9.0
            )
            if touches or (improving and near_boundary):
                warranted.append((dimension, value, anchor))

    pairs: list[list[torch.Tensor]] = []
    # Corners are checked first when two independent boundary directions agree.
    for first_index, first in enumerate(warranted):
        for second in warranted[first_index + 1 :]:
            if first[0] == second[0]:
                continue
            corner = rectangle.center.clone()
            corner[first[0]] = first[1]
            corner[second[0]] = second[1]
            near = corner.clone()
            offset = min(DOMAIN_WIDTH / 54.0, float(rectangle.sides.max().item()) / 6.0)
            near[first[0]] += offset if first[1] == LOWER else -offset
            near[second[0]] += offset if second[1] == LOWER else -offset
            pairs.append([corner, near])
            break
        if pairs:
            break

    for dimension, value, anchor in warranted:
        edge = anchor.clone()
        edge[dimension] = value
        near = edge.clone()
        offset = min(DOMAIN_WIDTH / 54.0, float(rectangle.sides[dimension].item()) / 6.0)
        near[dimension] += offset if value == LOWER else -offset
        pairs.append([edge, near])

    existing = torch.stack(state.sample_points)
    unique_pairs: list[list[torch.Tensor]] = []
    for pair in pairs:
        if any(
            bool((torch.linalg.vector_norm(existing - point.unsqueeze(0), dim=1) < 1.0e-13).any())
            for point in pair
        ):
            continue
        if any(
            all(torch.linalg.vector_norm(point - prior) < 1.0e-13 for point, prior in zip(pair, old))
            for old in unique_pairs
        ):
            continue
        unique_pairs.append(pair)
    return unique_pairs


def _fallback_boundary_pair(state: SearchState) -> list[torch.Tensor] | None:
    """Use a warranted edge pair when two evaluations remain but no PO split fits."""

    existing = torch.stack(state.sample_points)
    po_ids = _potentially_optimal(state.rectangles)
    # A hull rectangle touching an edge is direct evidence. If those pairs are
    # exhausted, an unsampled gap in the surviving full-domain edge partition
    # is coverage evidence. This fallback is used only to complete a two-query
    # budget remainder and never changes the DIRECT partition or hull.
    candidates = sorted(po_ids, key=lambda rid: (state.rectangles[rid].loss, rid))
    candidates += [
        rectangle.rid
        for rectangle in sorted(
            state.rectangles.values(),
            key=lambda item: (-item.diagonal, item.loss, item.rid),
        )
        if rectangle.rid not in set(po_ids)
    ]
    for rid in candidates:
        rectangle = state.rectangles[rid]
        for dimension in range(DIM):
            for value in (LOWER, UPPER):
                touches = (
                    float(rectangle.lower[dimension].item()) <= LOWER + 1.0e-12
                    if value == LOWER
                    else float(rectangle.upper[dimension].item()) >= UPPER - 1.0e-12
                )
                if not touches:
                    continue
                other_dimension = 1 - dimension
                for along_fraction in (0.0, 1.0 / 6.0, -1.0 / 6.0, 1.0 / 3.0, -1.0 / 3.0):
                    edge = rectangle.center.clone()
                    edge[dimension] = value
                    edge[other_dimension] = torch.clamp(
                        edge[other_dimension]
                        + along_fraction * rectangle.sides[other_dimension],
                        LOWER,
                        UPPER,
                    )
                    for divisor in (54.0, 81.0, 162.0, 486.0):
                        near = edge.clone()
                        offset = min(
                            DOMAIN_WIDTH / divisor,
                            max(
                                float(rectangle.sides[dimension].item()) / 6.0,
                                DOMAIN_WIDTH / 1458.0,
                            ),
                        )
                        near[dimension] += offset if value == LOWER else -offset
                        if any(
                            bool(
                                (
                                    torch.linalg.vector_norm(
                                        existing - point.unsqueeze(0), dim=1
                                    )
                                    < 1.0e-13
                                ).any()
                            )
                            for point in (edge, near)
                        ):
                            continue
                        return [edge, near]
    return None


def _snapshot(state: SearchState) -> SearchSnapshot:
    po_ids = _potentially_optimal(state.rectangles)
    return SearchSnapshot(
        adaptive_evaluations=state.adaptive_evaluations,
        active_bounds=[
            (rectangle.lower.cpu().numpy(), rectangle.upper.cpu().numpy())
            for rectangle in state.rectangles.values()
        ],
        po_bounds=[
            (
                state.rectangles[rid].lower.cpu().numpy(),
                state.rectangles[rid].upper.cpu().numpy(),
            )
            for rid in po_ids
        ],
        po_count=len(po_ids),
        learned_selections=state.learned_selections,
        forced_selections=state.forced_selections,
        boundary_probes=state.boundary_probes,
        scale_counts=dict(state.scale_counts),
        large_scale_selections=state.large_scale_selections,
        total_selections=state.total_selections,
        max_active_diagonal=max(rectangle.diagonal for rectangle in state.rectangles.values()),
    )


def _choose_learned(
    states: list[SearchState],
    network: PriorityNetwork,
    remaining_by_state: list[int],
    greedy: bool,
) -> list[tuple[int, torch.Tensor, torch.Tensor, bool] | None]:
    descriptions: list[tuple[int, list[int], list[int], bool]] = []
    feature_batches: list[torch.Tensor] = []
    max_candidates = 0
    for state_index, (state, remaining) in enumerate(zip(states, remaining_by_state)):
        if remaining < 2:
            continue
        po_ids = _potentially_optimal(state.rectangles)
        po_set = set(po_ids)
        for rectangle in state.rectangles.values():
            if rectangle.rid not in po_set:
                rectangle.skipped = 0
        affordable = [
            rid for rid in po_ids if state.rectangles[rid].canonical_cost <= remaining
        ]
        if not affordable:
            continue
        forced_ids = [
            rid for rid in affordable if state.rectangles[rid].skipped >= STARVATION_LIMIT
        ]
        is_forced = bool(forced_ids)
        candidates = forced_ids if is_forced else affordable
        feature = _rectangle_features(state, candidates, po_ids)
        descriptions.append((state_index, candidates, po_ids, is_forced))
        feature_batches.append(feature)
        max_candidates = max(max_candidates, feature.shape[0])

    output: list[tuple[int, torch.Tensor, torch.Tensor, bool] | None] = [None] * len(states)
    if not descriptions:
        return output
    feature_dim = feature_batches[0].shape[1]
    padded = torch.zeros((len(descriptions), max_candidates, feature_dim), dtype=DTYPE)
    mask = torch.zeros((len(descriptions), max_candidates), dtype=torch.bool)
    for row, feature in enumerate(feature_batches):
        padded[row, : feature.shape[0]] = feature
        mask[row, : feature.shape[0]] = True
    scores = network(padded, mask)
    for row, (state_index, candidate_ids, all_po_ids, is_forced) in enumerate(descriptions):
        valid_scores = scores[row, : len(candidate_ids)]
        distribution = torch.distributions.Categorical(logits=valid_scores)
        if is_forced:
            local_index = min(
                range(len(candidate_ids)),
                key=lambda index: (-states[state_index].rectangles[candidate_ids[index]].skipped, candidate_ids[index]),
            )
            action = torch.tensor(local_index, dtype=torch.long)
            log_probability = valid_scores.sum() * 0.0
            entropy = valid_scores.sum() * 0.0
        else:
            action = torch.argmax(valid_scores) if greedy else distribution.sample()
            log_probability = distribution.log_prob(action)
            entropy = distribution.entropy()
        chosen_id = candidate_ids[int(action.item())]
        for rid in all_po_ids:
            rectangle = states[state_index].rectangles[rid]
            rectangle.skipped = 0 if rid == chosen_id else rectangle.skipped + 1
        output[state_index] = (chosen_id, log_probability, entropy, is_forced)
    return output


def _choose_pure(
    states: list[SearchState], remaining_by_state: list[int]
) -> list[tuple[int, torch.Tensor, torch.Tensor, bool] | None]:
    output: list[tuple[int, torch.Tensor, torch.Tensor, bool] | None] = [None] * len(states)
    zero = torch.tensor(0.0, dtype=DTYPE)
    for state_index, (state, remaining) in enumerate(zip(states, remaining_by_state)):
        if remaining < 2:
            continue
        while state.direct_queue:
            rid = state.direct_queue[0]
            if rid not in state.rectangles:
                state.direct_queue.pop(0)
            elif state.rectangles[rid].canonical_cost > remaining:
                break
            else:
                state.direct_queue.pop(0)
                output[state_index] = (rid, zero, zero, False)
                break
        if output[state_index] is not None:
            continue
        po_ids = _potentially_optimal(state.rectangles)
        state.direct_queue = sorted(
            po_ids,
            key=lambda rid: (
                -state.rectangles[rid].diagonal,
                state.rectangles[rid].loss,
                rid,
            ),
        )
        affordable = [
            rid for rid in state.direct_queue if state.rectangles[rid].canonical_cost <= remaining
        ]
        if affordable:
            rid = affordable[0]
            state.direct_queue.remove(rid)
            output[state_index] = (rid, zero, zero, False)
    return output


def _run_search(
    positions: torch.Tensor,
    network: PriorityNetwork | None,
    budget: int,
    method: str,
    greedy: bool,
    capture_budgets: tuple[int, ...] = (),
) -> list[SearchState]:
    states = _initialize_states(positions)
    captures = set(capture_budgets)
    milestones = sorted(captures | {budget})
    while any(state.adaptive_evaluations < budget for state in states):
        local_limits = [
            next(
                (milestone for milestone in milestones if milestone > state.adaptive_evaluations),
                budget,
            )
            for state in states
        ]
        remaining = [
            local_limit - state.adaptive_evaluations
            for state, local_limit in zip(states, local_limits)
        ]
        if method == "learned":
            assert network is not None
            choices = _choose_learned(states, network, remaining, greedy)
        else:
            choices = _choose_pure(states, remaining)

        batch_points: list[torch.Tensor] = []
        batch_positions: list[torch.Tensor] = []
        choice_payloads: dict[int, tuple[int, list[torch.Tensor], list[int], torch.Tensor, torch.Tensor, bool, float, float]] = {}
        for state_index, choice in enumerate(choices):
            if choice is None:
                continue
            rid, log_probability, entropy, forced = choice
            rectangle = states[state_index].rectangles[rid]
            points, dimensions = _canonical_probe_points(rectangle)
            if len(points) > remaining[state_index]:
                continue
            best_before = states[state_index].best_loss
            median_diagonal = float(
                np.median([item.diagonal for item in states[state_index].rectangles.values()])
            )
            choice_payloads[state_index] = (
                rid,
                points,
                dimensions,
                log_probability,
                entropy,
                forced,
                best_before,
                median_diagonal,
            )
            for point in points:
                batch_points.append(point)
                batch_positions.append(positions[state_index])
        if not batch_points:
            fallback_requests: list[tuple[int, torch.Tensor]] = []
            for state_index, state in enumerate(states):
                if remaining[state_index] < 2:
                    continue
                pair = _fallback_boundary_pair(state)
                if pair is not None:
                    fallback_requests.extend((state_index, point) for point in pair)
            if not fallback_requests:
                break
            fallback_points = torch.stack([point for _, point in fallback_requests])
            fallback_positions = torch.stack(
                [positions[state_index] for state_index, _ in fallback_requests]
            )
            fallback_losses = _prefix_losses(
                fallback_points, fallback_positions
            ).detach().cpu().tolist()
            for (state_index, point), loss in zip(fallback_requests, fallback_losses):
                state = states[state_index]
                state.sample_points.append(point.clone())
                state.sample_losses.append(float(loss))
                state.sample_kinds.append("global_boundary_probe")
                state.adaptive_evaluations += 1
                state.boundary_probes += 1
            for state in states:
                if (
                    state.adaptive_evaluations in captures
                    and state.adaptive_evaluations not in state.snapshots
                ):
                    state.snapshots[state.adaptive_evaluations] = _snapshot(state)
            continue
        losses = _prefix_losses(
            torch.stack(batch_points), torch.stack(batch_positions)
        ).detach().cpu().tolist()
        cursor = 0
        edge_requests: list[tuple[int, torch.Tensor]] = []
        for state_index, payload in choice_payloads.items():
            (
                rid,
                points,
                dimensions,
                log_probability,
                entropy,
                forced,
                best_before,
                median_diagonal,
            ) = payload
            rectangle = states[state_index].rectangles[rid]
            local_losses = [float(value) for value in losses[cursor : cursor + len(points)]]
            cursor += len(points)
            scale = _scale_label(rectangle)
            state = states[state_index]
            # Count actual prefix-loss evaluations spent at this rectangle
            # scale, not merely the number of subdivision decisions.
            state.scale_counts[scale] += len(points)
            state.large_scale_selections += int(rectangle.diagonal >= median_diagonal - 1.0e-14)
            state.total_selections += 1
            if method == "learned":
                if forced:
                    state.forced_selections += 1
                else:
                    state.learned_selections += 1
            original = copy.deepcopy(rectangle)
            _split_canonical(state, rid, points, dimensions, local_losses)
            room = local_limits[state_index] - state.adaptive_evaluations
            for pair in _candidate_boundary_pairs(
                state, original, points, dimensions, local_losses
            ):
                if room < 2:
                    break
                for point in pair:
                    edge_requests.append((state_index, point))
                room -= 2
                # At most one evidence-driven pair per subdivision.
                break
            best_after_direct = state.best_loss
            improvement = max(
                0.0,
                math.log(best_before + EPS_LOSS) - math.log(best_after_direct + EPS_LOSS),
            )
            novelty = original.diagonal / DOMAIN_DIAGONAL
            reward = improvement + 0.02 * novelty
            if method == "learned" and not forced:
                state.action_records.append((log_probability, entropy, reward))

        if edge_requests:
            edge_points = torch.stack([point for _, point in edge_requests])
            edge_positions = torch.stack(
                [positions[state_index] for state_index, _ in edge_requests]
            )
            edge_losses = _prefix_losses(edge_points, edge_positions).detach().cpu().tolist()
            for (state_index, point), loss in zip(edge_requests, edge_losses):
                state = states[state_index]
                state.sample_points.append(point.clone())
                state.sample_losses.append(float(loss))
                state.sample_kinds.append("global_boundary_probe")
                state.adaptive_evaluations += 1
                state.boundary_probes += 1

        for state in states:
            if state.adaptive_evaluations in captures and state.adaptive_evaluations not in state.snapshots:
                state.snapshots[state.adaptive_evaluations] = _snapshot(state)
    for state in states:
        if state.adaptive_evaluations in captures and state.adaptive_evaluations not in state.snapshots:
            state.snapshots[state.adaptive_evaluations] = _snapshot(state)
    return states


def _policy_loss(states: list[SearchState], entropy_weight: float = 2.0e-3) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for state in states:
        records = state.action_records
        if not records:
            continue
        returns: list[float] = []
        running = 0.0
        for _, _, reward in reversed(records):
            running = reward + 0.97 * running
            returns.append(running)
        returns.reverse()
        return_tensor = torch.tensor(returns, dtype=DTYPE)
        if len(returns) > 1:
            return_tensor = (return_tensor - return_tensor.mean()) / return_tensor.std().clamp_min(1.0e-8)
        for (log_probability, entropy, _), value in zip(records, return_tensor):
            terms.append(-log_probability * value - entropy_weight * entropy)
    if not terms:
        return torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    return torch.stack(terms).mean()


def _validation_score(
    network: PriorityNetwork, positions: torch.Tensor, budget: int
) -> tuple[float, float, float]:
    network.eval()
    with torch.no_grad():
        states = _run_search(positions, network, budget, "learned", True)
    log_losses = np.log10(np.array([state.best_loss for state in states]) + EPS_LOSS)
    return float(np.mean(log_losses)), float(np.median(log_losses)), float(np.max(log_losses))


def _train_priority_network(
    train_positions: torch.Tensor,
    validation_positions: torch.Tensor,
    seed: int,
) -> tuple[PriorityNetwork, list[dict[str, object]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    network = PriorityNetwork().to(dtype=DTYPE, device=DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=1.0e-4)
    history: list[dict[str, object]] = []
    curriculum = ((40, 140), (80, 100), (160, 100))
    generator = torch.Generator(device="cpu").manual_seed(seed + 991)
    for stage_index, (budget, updates) in enumerate(curriculum, start=1):
        best_state = copy.deepcopy(network.state_dict())
        best_score = math.inf
        stale = 0
        started = time.perf_counter()
        for update in range(1, updates + 1):
            indices = torch.randint(
                0, train_positions.shape[0], (16,), generator=generator
            )
            batch = train_positions[indices]
            network.train()
            states = _run_search(batch, network, budget, "learned", False)
            loss = _policy_loss(states)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            if update == 1 or update % 20 == 0 or update == updates:
                score, median_log_loss, worst_log_loss = _validation_score(
                    network, validation_positions, budget
                )
                improved = score < best_score - 1.0e-4
                if improved:
                    best_score = score
                    best_state = copy.deepcopy(network.state_dict())
                    stale = 0
                else:
                    stale += 20 if update > 1 else 1
                row = {
                    "row_type": "training_validation",
                    "stage": stage_index,
                    "budget": budget,
                    "update": update,
                    "policy_loss": float(loss.detach().item()),
                    "validation_mean_log10_best_loss": score,
                    "validation_median_log10_best_loss": median_log_loss,
                    "validation_worst_log10_best_loss": worst_log_loss,
                    "is_best": int(improved),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(row)
                print(
                    f"stage={stage_index} budget={budget} update={update}/{updates} "
                    f"policy={float(loss.detach()):.5g} val_log10={score:.4f} "
                    f"best={best_score:.4f}",
                    flush=True,
                )
                if stale >= 60:
                    break
        network.load_state_dict(best_state)
    return network, history


def _contains(bounds: list[tuple[np.ndarray, np.ndarray]], reference: np.ndarray) -> bool:
    return any(np.all(reference >= lower - 1.0e-12) and np.all(reference <= upper + 1.0e-12) for lower, upper in bounds)


def _best_curves(
    states: list[SearchState], references: np.ndarray, budget: int
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for evaluation in range(0, budget + 1, 2):
        best_losses: list[float] = []
        errors: list[float] = []
        for state, reference in zip(states, references):
            count = min(1 + evaluation, len(state.sample_losses))
            losses = np.asarray(state.sample_losses[:count])
            points = torch.stack(state.sample_points[:count]).cpu().numpy()
            best_losses.append(float(np.min(losses)))
            errors.append(float(np.min(np.linalg.norm(points - reference[None, :], axis=1))))
        rows.append(
            {
                "adaptive_evaluations": evaluation,
                "median_best_prefix_loss": float(np.median(best_losses)),
                "worst_best_prefix_loss": float(np.max(best_losses)),
                "median_nearest_parameter_error": float(np.median(errors)),
                "worst_nearest_parameter_error": float(np.max(errors)),
            }
        )
    return rows


def _summary_rows(
    method: str,
    states: list[SearchState],
    references: np.ndarray,
    hidden_c: np.ndarray,
    budgets: tuple[int, ...],
    runtime_seconds: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    boundary_mask = np.min(
        np.stack(
            [
                references[:, 0] - LOWER,
                UPPER - references[:, 0],
                references[:, 1] - LOWER,
                UPPER - references[:, 1],
            ],
            axis=1,
        ),
        axis=1,
    ) <= 0.25
    for budget in budgets:
        losses: list[float] = []
        errors: list[float] = []
        active_coverage: list[bool] = []
        po_coverage: list[bool] = []
        po_counts: list[int] = []
        learned_counts: list[int] = []
        forced_counts: list[int] = []
        boundary_counts: list[int] = []
        scale_counts = {name: 0 for name in SCALE_LABELS}
        large_counts = 0
        total_selections = 0
        max_diagonals: list[float] = []
        for state, reference in zip(states, references):
            snapshot = state.snapshots[budget]
            sample_count = 1 + budget
            sampled_losses = np.asarray(state.sample_losses[:sample_count])
            sampled_points = torch.stack(state.sample_points[:sample_count]).cpu().numpy()
            losses.append(float(np.min(sampled_losses)))
            errors.append(float(np.min(np.linalg.norm(sampled_points - reference[None, :], axis=1))))
            active_coverage.append(_contains(snapshot.active_bounds, reference))
            po_coverage.append(_contains(snapshot.po_bounds, reference))
            po_counts.append(snapshot.po_count)
            learned_counts.append(snapshot.learned_selections)
            forced_counts.append(snapshot.forced_selections)
            boundary_counts.append(snapshot.boundary_probes)
            for name in SCALE_LABELS:
                scale_counts[name] += snapshot.scale_counts[name]
            large_counts += snapshot.large_scale_selections
            total_selections += snapshot.total_selections
            max_diagonals.append(snapshot.max_active_diagonal)
        losses_array = np.asarray(losses)
        errors_array = np.asarray(errors)
        denominator = max(1, total_selections)
        scale_evaluation_denominator = max(1, sum(scale_counts.values()))
        row: dict[str, object] = {
            "method": method,
            "adaptive_evaluation_budget": budget,
            "total_prefix_loss_evaluations_including_root": budget + 1,
            "trajectories": len(states),
            "active_partition_coverage": float(np.mean(active_coverage)),
            "potentially_optimal_coverage": float(np.mean(po_coverage)),
            "boundary_subgroup_potentially_optimal_coverage": (
                float(np.mean(np.asarray(po_coverage)[boundary_mask]))
                if bool(boundary_mask.any())
                else float("nan")
            ),
            "median_nearest_parameter_error": float(np.median(errors_array)),
            "worst_nearest_parameter_error": float(np.max(errors_array)),
            "median_best_prefix_loss": float(np.median(losses_array)),
            "worst_best_prefix_loss": float(np.max(losses_array)),
            "mean_potentially_optimal_rectangles": float(np.mean(po_counts)),
            "median_potentially_optimal_rectangles": float(np.median(po_counts)),
            "worst_potentially_optimal_rectangles": int(np.max(po_counts)),
            "learned_selections": int(np.sum(learned_counts)),
            "forced_anti_starvation_selections": int(np.sum(forced_counts)),
            "total_rectangle_subdivisions": int(total_selections),
            "forced_selection_fraction": float(np.sum(forced_counts) / denominator),
            "global_boundary_probes": int(np.sum(boundary_counts)),
            "global_boundary_probe_fraction_of_evaluations": float(
                np.sum(boundary_counts) / (len(states) * budget)
            ),
            "coarse_scale_evaluation_fraction": scale_counts["coarse"]
            / scale_evaluation_denominator,
            "medium_scale_evaluation_fraction": scale_counts["medium"]
            / scale_evaluation_denominator,
            "fine_scale_evaluation_fraction": scale_counts["fine"]
            / scale_evaluation_denominator,
            "ultrafine_scale_evaluation_fraction": scale_counts["ultrafine"]
            / scale_evaluation_denominator,
            "large_unexplored_rectangle_selection_fraction": large_counts / denominator,
            "mean_max_active_half_diagonal": float(np.mean(max_diagonals)),
            "runtime_seconds_for_full_160_budget": runtime_seconds,
            "reference_a_note": "posthoc_only",
            "reference_b_note": "posthoc_only",
            "hidden_c_min_posthoc": float(np.min(hidden_c)),
            "hidden_c_max_posthoc": float(np.max(hidden_c)),
        }
        rows.append(row)
    return rows


def _plot_curves(path: Path, learned_rows: list[dict[str, float]], pure_rows: list[dict[str, float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for name, rows, color in (
        ("learned-priority DIRECT", learned_rows, "tab:blue"),
        ("pure DIRECT", pure_rows, "tab:orange"),
    ):
        evaluations = [row["adaptive_evaluations"] for row in rows]
        losses = [max(row["median_best_prefix_loss"], 1.0e-30) for row in rows]
        errors = [max(row["median_nearest_parameter_error"], 1.0e-16) for row in rows]
        axes[0].plot(evaluations, losses, linewidth=2.0, label=name, color=color)
        axes[1].plot(evaluations, errors, linewidth=2.0, label=name, color=color)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("adaptive prefix-loss evaluations")
    axes[1].set_xlabel("adaptive prefix-loss evaluations")
    axes[0].set_ylabel("median best prefix loss")
    axes[1].set_ylabel("median nearest parameter error")
    axes[0].set_title("Best observed loss")
    axes[1].set_title("Nearest sampled (a,b)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sho/stage1B/direct_scaffold_2d_search"),
    )
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED + 271)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_size = 96 if args.smoke else args.train_size
    validation_size = 16 if args.smoke else args.validation_size
    test_size = 20 if args.smoke else args.test_size
    budgets = (8, 12, 16) if args.smoke else BUDGETS

    train_positions, _, _ = _make_exact_labeled_split(train_size, TRAIN_DATA_SEED)
    validation_positions, _, _ = _make_exact_labeled_split(
        validation_size, VALIDATION_DATA_SEED
    )
    test_positions, _, test_hidden_c = _make_exact_labeled_split(
        test_size, TEST_DATA_SEED + 44003
    )
    train_positions = train_positions.to(dtype=DTYPE, device=DEVICE)
    validation_positions = validation_positions.to(dtype=DTYPE, device=DEVICE)
    test_positions = test_positions.to(dtype=DTYPE, device=DEVICE)

    if args.smoke:
        network = PriorityNetwork().to(dtype=DTYPE)
        training_history: list[dict[str, object]] = []
    else:
        network, training_history = _train_priority_network(
            train_positions, validation_positions, args.seed
        )

    references = _posthoc_reference(test_hidden_c).cpu().numpy()
    hidden_c = test_hidden_c.cpu().numpy()
    learned_started = time.perf_counter()
    with torch.no_grad():
        learned_states = _run_search(
            test_positions,
            network,
            budgets[-1],
            "learned",
            True,
            capture_budgets=budgets,
        )
    learned_runtime = time.perf_counter() - learned_started
    pure_started = time.perf_counter()
    with torch.no_grad():
        pure_states = _run_search(
            test_positions,
            None,
            budgets[-1],
            "pure_direct",
            True,
            capture_budgets=budgets,
        )
    pure_runtime = time.perf_counter() - pure_started

    summary = _summary_rows(
        "learned_priority_direct",
        learned_states,
        references,
        hidden_c,
        budgets,
        learned_runtime,
    ) + _summary_rows(
        "pure_direct_control",
        pure_states,
        references,
        hidden_c,
        budgets,
        pure_runtime,
    )

    learned_curves = _best_curves(learned_states, references, budgets[-1])
    pure_curves = _best_curves(pure_states, references, budgets[-1])
    history_rows = list(training_history)
    for method, rows in (
        ("learned_priority_direct", learned_curves),
        ("pure_direct_control", pure_curves),
    ):
        for row in rows:
            history_rows.append({"row_type": "test_search_curve", "method": method, **row})

    _write_csv(args.output_dir / "direct_scaffold_summary.csv", summary)
    _write_csv(args.output_dir / "prioritization_history.csv", history_rows)
    _plot_curves(
        args.output_dir / "best_loss_and_parameter_error_vs_evaluations.png",
        learned_curves,
        pure_curves,
    )
    for row in summary:
        print(
            f"{row['method']} budget={row['adaptive_evaluation_budget']} "
            f"po_coverage={row['potentially_optimal_coverage']:.4f} "
            f"median_error={row['median_nearest_parameter_error']:.4g} "
            f"median_loss={row['median_best_prefix_loss']:.4g}",
            flush=True,
        )
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
