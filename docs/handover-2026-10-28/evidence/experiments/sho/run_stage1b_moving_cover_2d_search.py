"""Moving, overlapping-cover Network 1 search for unconstrained K(a,b)."""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch
from torch import nn

import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as data_source
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    UPPER,
    VALIDATION_DATA_SEED,
    _posthoc_reference,
    _prefix_losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    os.environ.get(
        "MOVING2D_OUTPUT_DIR",
        "outputs/sho/stage1B/moving_cover_2d_search",
    )
)
TRAIN_N = int(os.environ.get("MOVING2D_TRAIN_N", "4096"))
VALIDATION_N = int(os.environ.get("MOVING2D_VALIDATION_N", "512"))
TEST_N = int(os.environ.get("MOVING2D_TEST_N", "512"))
HORIZONS = tuple(
    int(value)
    for value in os.environ.get("MOVING2D_HORIZONS", "20,40,80").split(",")
)
UPDATE_BUDGETS = tuple(
    int(value)
    for value in os.environ.get("MOVING2D_UPDATES", "800,600,400").split(",")
)
if len(HORIZONS) != len(UPDATE_BUDGETS):
    raise ValueError("Each moving-cover horizon requires an update budget")
BATCH_SIZE = int(os.environ.get("MOVING2D_BATCH_SIZE", "32"))
EVAL_INTERVAL = int(os.environ.get("MOVING2D_EVAL_INTERVAL", "50"))
PATIENCE = int(os.environ.get("MOVING2D_PATIENCE", "250"))
PER_QUERY_GRID_RESOLUTION = int(os.environ.get("MOVING2D_QUERY_GRID", "24"))
SUMMARY_GRID_RESOLUTION = int(os.environ.get("MOVING2D_SUMMARY_GRID", "64"))
LEARNING_RATE = 1e-4
DISCOUNT = 0.97
ENTROPY_WEIGHT = 0.01
NOVELTY_WEIGHT = 0.025
SUPPORT_THRESHOLD = 0.10
AGREEMENT_BONUS = 0.25
MODEL_SEED = SEED + 2_470_000
LOG_EPSILON = 1e-30
DOMAIN_AREA = DOMAIN_WIDTH**2
DOMAIN_DIAGONAL = math.sqrt(2.0) * DOMAIN_WIDTH
SHORTLIST_EXPLOIT = 32
SHORTLIST_EXPLORE = 32
SHORTLIST_BOUNDARY = 20
SHORTLIST_EDGE = 20
SHORTLIST_BALANCED = 16


@dataclass(frozen=True)
class CandidateCover:
    lower: torch.Tensor
    upper: torch.Tensor
    center: torch.Tensor
    cover_id: torch.Tensor
    previous_boundary: torch.Tensor
    global_edge: torch.Tensor
    primary_count: int
    scale: float
    scale_index: int


@dataclass
class CandidateStatistics:
    features: torch.Tensor
    nearest_squared: torch.Tensor
    relative_nearby_log_loss: torch.Tensor
    counts: torch.Tensor


@dataclass
class MovingCoverRollout:
    query_theta: torch.Tensor
    query_loss: torch.Tensor
    query_lower: torch.Tensor
    query_upper: torch.Tensor
    query_cover_id: torch.Tensor
    query_scale_index: torch.Tensor
    query_category: torch.Tensor
    best_theta: torch.Tensor
    best_loss: torch.Tensor
    entropy: torch.Tensor
    log_probabilities: list[torch.Tensor]
    rewards: list[torch.Tensor]


@dataclass
class PlausibilityDiagnostics:
    covered: torch.Tensor
    area: torch.Tensor
    nearest_parameter_error: torch.Tensor
    best_prefix_loss: torch.Tensor
    multi_cover_area: torch.Tensor
    multi_cover_fraction: torch.Tensor
    mean_supporting_covers: torch.Tensor
    miss_global_edge: torch.Tensor
    miss_poor_overlap: torch.Tensor
    miss_insufficient_exploration: torch.Tensor
    miss_controller_selection: torch.Tensor


class MovingCoverSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.region_encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        encoded = self.region_encoder(features)
        valid_float = valid.to(DTYPE)[:, :, None]
        mean_context = (encoded * valid_float).sum(dim=1, keepdim=True) / (
            valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        masked = encoded.masked_fill(~valid[:, :, None], -torch.inf)
        max_context = masked.max(dim=1, keepdim=True).values
        context = torch.cat((mean_context, max_context), dim=2).expand(
            -1, encoded.shape[1], -1
        )
        logits = self.score_head(torch.cat((encoded, context), dim=2)).squeeze(-1)
        return logits.masked_fill(~valid, -1e30)


def _axis_intervals(scale: float, shifted: bool) -> list[tuple[float, float]]:
    start = LOWER - (0.5 * scale if shifted else 0.0)
    intervals: list[tuple[float, float]] = []
    value = start
    while value < UPPER - 1e-13:
        left = max(value, LOWER)
        right = min(value + scale, UPPER)
        if right - left > 1e-13:
            intervals.append((left, right))
        value += scale
    return intervals


def _prior_boundary_values(scale_index: int) -> list[float]:
    values: set[float] = set()
    for prior_index in range(scale_index):
        prior_scale = DOMAIN_WIDTH / (2 ** (prior_index + 1))
        for shifted in (False, True):
            for left, right in _axis_intervals(prior_scale, shifted):
                for value in (left, right):
                    if LOWER + 1e-12 < value < UPPER - 1e-12:
                        values.add(round(value, 14))
    return sorted(values)


def _make_cover(scale_index: int) -> CandidateCover:
    scale = DOMAIN_WIDTH / (2 ** (scale_index + 1))
    lowers: list[tuple[float, float]] = []
    uppers: list[tuple[float, float]] = []
    cover_ids: list[int] = []
    previous_flags: list[bool] = []
    edge_flags: list[bool] = []
    prior_boundaries = _prior_boundary_values(scale_index)

    cover_id = 0
    for shift_a, shift_b in ((False, False), (True, False), (False, True), (True, True)):
        for a_left, a_right in _axis_intervals(scale, shift_a):
            for b_left, b_right in _axis_intervals(scale, shift_b):
                center_a = 0.5 * (a_left + a_right)
                center_b = 0.5 * (b_left + b_right)
                on_prior = any(
                    abs(center_a - boundary) <= 1e-12
                    or abs(center_b - boundary) <= 1e-12
                    for boundary in prior_boundaries
                )
                lowers.append((a_left, b_left))
                uppers.append((a_right, b_right))
                cover_ids.append(cover_id)
                previous_flags.append(on_prior)
                edge_flags.append(False)
        cover_id += 1
    primary_count = len(lowers)

    # Explicit centers on old cover lines make old boundaries valid targets.
    ordinary_axis = [0.5 * (left + right) for left, right in _axis_intervals(scale, False)]
    seen_special: set[tuple[float, float, int]] = set()
    for boundary in prior_boundaries:
        for other in ordinary_axis:
            for center_a, center_b in ((boundary, other), (other, boundary)):
                key = (round(center_a, 14), round(center_b, 14), 4)
                if key in seen_special:
                    continue
                seen_special.add(key)
                lowers.append((max(LOWER, center_a - 0.5 * scale), max(LOWER, center_b - 0.5 * scale)))
                uppers.append((min(UPPER, center_a + 0.5 * scale), min(UPPER, center_b + 0.5 * scale)))
                cover_ids.append(4)
                previous_flags.append(True)
                edge_flags.append(False)

    # Domain edge and corner centers are always available; low-loss evidence
    # near an edge enters their ordinary nearby-loss features.
    edge_axis = ordinary_axis + [LOWER, UPPER]
    for other in edge_axis:
        for center_a, center_b in (
            (LOWER, other), (UPPER, other), (other, LOWER), (other, UPPER)
        ):
            key = (round(center_a, 14), round(center_b, 14), 5)
            if key in seen_special:
                continue
            seen_special.add(key)
            lowers.append((max(LOWER, center_a - 0.5 * scale), max(LOWER, center_b - 0.5 * scale)))
            uppers.append((min(UPPER, center_a + 0.5 * scale), min(UPPER, center_b + 0.5 * scale)))
            cover_ids.append(5)
            previous_flags.append(False)
            edge_flags.append(True)

    lower = torch.tensor(lowers, dtype=DTYPE)
    upper = torch.tensor(uppers, dtype=DTYPE)
    return CandidateCover(
        lower=lower,
        upper=upper,
        center=0.5 * (lower + upper),
        cover_id=torch.tensor(cover_ids, dtype=torch.long),
        previous_boundary=torch.tensor(previous_flags, dtype=torch.bool),
        global_edge=torch.tensor(edge_flags, dtype=torch.bool),
        primary_count=primary_count,
        scale=scale,
        scale_index=scale_index,
    )


COVERS = tuple(_make_cover(index) for index in range(4))


def _scale_index_for_step(step: int) -> int:
    if step < 5:
        return 0
    if step < 20:
        return 1
    if step < 40:
        return 2
    return 3


def _candidate_statistics(
    cover: CandidateCover,
    observed_theta: torch.Tensor,
    observed_loss: torch.Tensor,
    observed_cover_id: torch.Tensor,
) -> CandidateStatistics:
    batch_size = observed_theta.shape[0]
    count = cover.center.shape[0]
    sides = cover.upper - cover.lower
    area = sides.prod(dim=1)
    boundary = cover.previous_boundary.to(DTYPE)[None, :].expand(batch_size, -1)
    edge = cover.global_edge.to(DTYPE)[None, :].expand(batch_size, -1)
    if observed_theta.shape[1] == 0:
        nearest_squared = torch.full((batch_size, count), DOMAIN_DIAGONAL**2, dtype=DTYPE)
        relative_nearby = torch.zeros_like(nearest_squared)
        counts = torch.zeros_like(nearest_squared, dtype=torch.long)
        relative_inside = torch.zeros_like(nearest_squared)
        agreement = torch.zeros_like(nearest_squared)
    else:
        squared_distance = torch.sum(
            (observed_theta[:, None, :, :] - cover.center[None, :, None, :]).square(),
            dim=3,
        )
        nearest_squared, nearest_index = squared_distance.min(dim=2)
        nearest_loss = observed_loss.gather(1, nearest_index)
        inside = (
            (observed_theta[:, None, :, :] >= cover.lower[None, :, None, :] - 1e-14)
            & (observed_theta[:, None, :, :] <= cover.upper[None, :, None, :] + 1e-14)
        ).all(dim=3)
        counts = inside.sum(dim=2)
        masked_inside_loss = torch.where(
            inside,
            observed_loss[:, None, :],
            torch.full_like(observed_loss[:, None, :], float("inf")),
        )
        best_inside = masked_inside_loss.min(dim=2).values
        global_best = observed_loss.min(dim=1).values[:, None]
        relative_nearby = torch.clamp(
            torch.log(nearest_loss + LOG_EPSILON)
            - torch.log(global_best + LOG_EPSILON),
            min=0.0,
            max=80.0,
        )
        relative_inside = torch.where(
            torch.isfinite(best_inside),
            torch.clamp(
                torch.log(best_inside + LOG_EPSILON)
                - torch.log(global_best + LOG_EPSILON),
                min=0.0,
                max=80.0,
            ),
            torch.zeros_like(best_inside),
        )
        # Count how many differently aligned primary covers have already
        # supplied observations inside the proposed locality.
        agreement = torch.zeros_like(relative_nearby)
        for primary_id in range(4):
            from_cover = observed_cover_id == primary_id
            agreement += (inside & from_cover[:, None, :]).any(dim=2).to(DTYPE)

    features = torch.stack(
        (
            cover.center[:, 0][None, :].expand(batch_size, -1) / UPPER,
            cover.center[:, 1][None, :].expand(batch_size, -1) / UPPER,
            sides[:, 0][None, :].expand(batch_size, -1) / DOMAIN_WIDTH,
            sides[:, 1][None, :].expand(batch_size, -1) / DOMAIN_WIDTH,
            area[None, :].expand(batch_size, -1) / DOMAIN_AREA,
            nearest_squared / DOMAIN_DIAGONAL**2,
            relative_nearby / 20.0,
            torch.log1p(counts.to(DTYPE)) / math.log(96.0),
            (counts == 0).to(DTYPE),
            relative_inside / 20.0,
            agreement / 4.0,
            boundary,
            edge,
            torch.full((batch_size, count), cover.scale / DOMAIN_WIDTH, dtype=DTYPE),
            torch.sqrt(nearest_squared.clamp_min(0.0)) / max(cover.scale, 1e-12),
        ),
        dim=2,
    )
    return CandidateStatistics(features, nearest_squared, relative_nearby, counts)


def _shortlist(
    cover: CandidateCover,
    statistics: CandidateStatistics,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = cover.center.shape[0]

    def top(metric: torch.Tensor, requested: int, largest: bool) -> torch.Tensor:
        return torch.topk(metric, k=min(requested, count), dim=1, largest=largest).indices

    exploit = top(statistics.relative_nearby_log_loss, SHORTLIST_EXPLOIT, False)
    explore = top(statistics.nearest_squared, SHORTLIST_EXPLORE, True)
    boundary_metric = statistics.nearest_squared / DOMAIN_DIAGONAL**2 + 10.0 * cover.previous_boundary.to(DTYPE)[None, :]
    boundary = top(boundary_metric, SHORTLIST_BOUNDARY, True)
    edge_metric = statistics.nearest_squared / DOMAIN_DIAGONAL**2 + 10.0 * cover.global_edge.to(DTYPE)[None, :]
    edge = top(edge_metric, SHORTLIST_EDGE, True)
    balanced_metric = (
        statistics.nearest_squared / DOMAIN_DIAGONAL**2
        - 0.25 * statistics.relative_nearby_log_loss
    )
    balanced = top(balanced_metric, SHORTLIST_BALANCED, True)
    indices = torch.cat((exploit, explore, boundary, edge, balanced), dim=1)
    equal = indices[:, :, None] == indices[:, None, :]
    earlier = torch.tril(torch.ones(indices.shape[1], indices.shape[1], dtype=torch.bool), diagonal=-1)
    duplicate = (equal & earlier[None, :, :]).any(dim=2)
    return indices, ~duplicate


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim == 2:
        return values[indices]
    return values.gather(1, indices[:, :, None].expand(-1, -1, values.shape[2]))


def _moving_cover_rollout(
    model: MovingCoverSelector,
    positions: torch.Tensor,
    horizon: int,
    greedy: bool,
) -> MovingCoverRollout:
    batch_size = positions.shape[0]
    observed_theta = torch.empty(batch_size, 0, 2, dtype=DTYPE)
    observed_loss = torch.empty(batch_size, 0, dtype=DTYPE)
    observed_cover_id = torch.empty(batch_size, 0, dtype=torch.long)
    query_theta_history: list[torch.Tensor] = []
    query_loss_history: list[torch.Tensor] = []
    lower_history: list[torch.Tensor] = []
    upper_history: list[torch.Tensor] = []
    cover_id_history: list[torch.Tensor] = []
    scale_history: list[torch.Tensor] = []
    category_history: list[torch.Tensor] = []
    best_theta_history: list[torch.Tensor] = []
    best_loss_history: list[torch.Tensor] = []
    entropy_history: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for step in range(horizon):
        scale_index = _scale_index_for_step(step)
        cover = COVERS[scale_index]
        statistics = _candidate_statistics(
            cover, observed_theta, observed_loss, observed_cover_id
        )
        shortlist, valid = _shortlist(cover, statistics)
        shortlisted_features = _gather_candidates(statistics.features, shortlist)
        shortlisted_nearest = statistics.nearest_squared.gather(1, shortlist)
        valid = valid & (shortlisted_nearest > 1e-28)
        if not bool(valid.any(dim=1).all()):
            raise RuntimeError("No unevaluated moving-cover candidate remained")
        logits = model(shortlisted_features, valid)
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=1) if greedy else distribution.sample()
        chosen_global = shortlist.gather(1, action[:, None]).squeeze(1)
        query_theta = cover.center[chosen_global]
        query_lower = cover.lower[chosen_global]
        query_upper = cover.upper[chosen_global]
        query_cover_id = cover.cover_id[chosen_global]
        boundary_action = cover.previous_boundary[chosen_global]
        edge_action = cover.global_edge[chosen_global]
        nearby_relative = statistics.relative_nearby_log_loss.gather(
            1, chosen_global[:, None]
        ).squeeze(1)
        exploit_action = (
            torch.zeros(batch_size, dtype=torch.bool)
            if observed_loss.shape[1] == 0
            else nearby_relative <= math.log(4.0)
        )
        # Mutually exclusive query accounting: edge, old boundary, low-loss
        # exploit, then ordinary unexplored space.
        category = torch.full((batch_size,), 1, dtype=torch.long)
        category = torch.where(exploit_action, torch.zeros_like(category), category)
        category = torch.where(boundary_action, torch.full_like(category, 2), category)
        category = torch.where(edge_action, torch.full_like(category, 3), category)

        with torch.no_grad():
            query_loss = _prefix_losses(query_theta, positions)
            if observed_loss.shape[1] == 0:
                prior_best = query_loss
                improvement = torch.zeros_like(query_loss)
            else:
                prior_best = observed_loss.min(dim=1).values
                next_best = torch.minimum(prior_best, query_loss)
                improvement = torch.clamp(
                    torch.log(prior_best + LOG_EPSILON)
                    - torch.log(next_best + LOG_EPSILON),
                    min=0.0,
                    max=10.0,
                )
            novelty = torch.sqrt(
                statistics.nearest_squared.gather(1, chosen_global[:, None]).squeeze(1)
            ) / DOMAIN_DIAGONAL
            exploration_bonus = (
                novelty
                + 0.20 * (category == 1).to(DTYPE)
                + 0.25 * (category == 2).to(DTYPE)
                + 0.20 * (category == 3).to(DTYPE)
            )
            reward = improvement + NOVELTY_WEIGHT * exploration_bonus

        observed_theta = torch.cat((observed_theta, query_theta[:, None, :]), dim=1)
        observed_loss = torch.cat((observed_loss, query_loss[:, None]), dim=1)
        observed_cover_id = torch.cat(
            (observed_cover_id, query_cover_id[:, None]), dim=1
        )
        best_index = torch.argmin(observed_loss, dim=1)
        best_theta = observed_theta.gather(
            1, best_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        best_loss = observed_loss.gather(1, best_index[:, None]).squeeze(1)

        query_theta_history.append(query_theta.detach())
        query_loss_history.append(query_loss.detach())
        lower_history.append(query_lower.detach())
        upper_history.append(query_upper.detach())
        cover_id_history.append(query_cover_id.detach())
        scale_history.append(torch.full((batch_size,), scale_index, dtype=torch.long))
        category_history.append(category.detach())
        best_theta_history.append(best_theta)
        best_loss_history.append(best_loss)
        entropy_history.append(distribution.entropy())
        log_probabilities.append(distribution.log_prob(action))
        rewards.append(reward)

    return MovingCoverRollout(
        query_theta=torch.stack(query_theta_history, dim=1),
        query_loss=torch.stack(query_loss_history, dim=1),
        query_lower=torch.stack(lower_history, dim=1),
        query_upper=torch.stack(upper_history, dim=1),
        query_cover_id=torch.stack(cover_id_history, dim=1),
        query_scale_index=torch.stack(scale_history, dim=1),
        query_category=torch.stack(category_history, dim=1),
        best_theta=torch.stack(best_theta_history, dim=1),
        best_loss=torch.stack(best_loss_history, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        log_probabilities=log_probabilities,
        rewards=rewards,
    )


def _policy_loss(rollout: MovingCoverRollout) -> torch.Tensor:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rollout.rewards[0])
    for reward in reversed(rollout.rewards):
        running = reward + DISCOUNT * running
        returns.append(running)
    returns.reverse()
    terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    for value, log_probability, entropy in zip(
        returns, rollout.log_probabilities, rollout.entropy.unbind(dim=1)
    ):
        advantage = value - value.mean()
        advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-6)
        terms.append(-(advantage.detach() * log_probability).mean())
        entropy_terms.append(entropy.mean())
    return torch.stack(terms).mean() - ENTROPY_WEIGHT * torch.stack(entropy_terms).mean()


@torch.no_grad()
def _evaluate(
    model: MovingCoverSelector,
    positions: torch.Tensor,
    horizon: int,
) -> MovingCoverRollout:
    model.eval()
    parts = [
        _moving_cover_rollout(model, positions[start : start + BATCH_SIZE], horizon, True)
        for start in range(0, positions.shape[0], BATCH_SIZE)
    ]
    tensor_fields = (
        "query_theta", "query_loss", "query_lower", "query_upper",
        "query_cover_id", "query_scale_index", "query_category",
        "best_theta", "best_loss", "entropy",
    )
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in tensor_fields
    }
    return MovingCoverRollout(**combined, log_probabilities=[], rewards=[])


def _validation_score(rollout: MovingCoverRollout) -> float:
    return float(torch.log(rollout.best_loss[:, -1] + LOG_EPSILON).mean().item())


def _train_stage(
    model: MovingCoverSelector,
    train_positions: torch.Tensor,
    validation_positions: torch.Tensor,
    horizon: int,
    update_budget: int,
    generator: torch.Generator,
    cumulative_updates: int,
) -> tuple[dict[str, Any], int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    validation = _evaluate(model, validation_positions, horizon)
    best_score = _validation_score(validation)
    best_state = copy.deepcopy(model.state_dict())
    best_update = 0
    stale = 0
    updates_run = 0
    stop_reason = "maximum_updates"
    start = time.perf_counter()
    print(f"moving2d horizon={horizon} update=0000 validation={best_score:.6e}", flush=True)
    for update in range(1, update_budget + 1):
        model.train()
        indices = torch.randint(0, train_positions.shape[0], (BATCH_SIZE,), generator=generator)
        rollout = _moving_cover_rollout(model, train_positions[indices], horizon, False)
        loss = _policy_loss(rollout)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        cumulative_updates += 1
        updates_run = update
        if update % EVAL_INTERVAL == 0 or update == update_budget:
            validation = _evaluate(model, validation_positions, horizon)
            score = _validation_score(validation)
            if score < best_score - 1e-8:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_update = update
                stale = 0
            else:
                stale += EVAL_INTERVAL
            print(
                f"moving2d horizon={horizon} update={update:04d} "
                f"policy={float(loss.detach().item()):.6e} validation={score:.6e} "
                f"best_update={best_update} stale={stale}",
                flush=True,
            )
            if stale >= PATIENCE:
                stop_reason = "validation_plateau"
                break
    model.load_state_dict(best_state)
    return {
        "trained_horizon": horizon,
        "updates_run": updates_run,
        "best_update": best_update,
        "best_validation_score": best_score,
        "stop_reason": stop_reason,
        "runtime_seconds": time.perf_counter() - start,
        "cumulative_updates": cumulative_updates,
    }, cumulative_updates


def _domain_grid(resolution: int) -> torch.Tensor:
    step = DOMAIN_WIDTH / resolution
    axis = torch.linspace(LOWER + 0.5 * step, UPPER - 0.5 * step, resolution, dtype=DTYPE)
    aa, bb = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((aa.reshape(-1), bb.reshape(-1)), dim=1)


def _primary_cover_weights(
    cover: CandidateCover,
    theta: torch.Tensor,
    loss: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    global_best = loss.min(dim=1).values
    relative = torch.log(loss + LOG_EPSILON) - torch.log(global_best[:, None] + LOG_EPSILON)
    temperature = torch.quantile(relative, 0.5, dim=1).clamp_min(0.5)
    weights: list[torch.Tensor] = []
    evidence_any = torch.zeros(theta.shape[0], cover.primary_count, dtype=torch.bool)
    for cover_id in range(4):
        mask = cover.cover_id[: cover.primary_count] == cover_id
        lower = cover.lower[: cover.primary_count][mask]
        upper = cover.upper[: cover.primary_count][mask]
        inside = (
            (theta[:, :, None, :] >= lower[None, None, :, :] - 1e-14)
            & (theta[:, :, None, :] <= upper[None, None, :, :] + 1e-14)
        ).all(dim=3)
        cell_loss = torch.where(
            inside,
            loss[:, :, None],
            torch.full_like(loss[:, :, None], float("inf")),
        ).min(dim=1).values
        cell_weight = torch.where(
            torch.isfinite(cell_loss),
            torch.exp(
                -(
                    torch.log(cell_loss + LOG_EPSILON)
                    - torch.log(global_best[:, None] + LOG_EPSILON)
                )
                / temperature[:, None]
            ),
            torch.zeros_like(cell_loss),
        )
        weights.append(cell_weight)
        evidence_any[:, mask] = inside.any(dim=1)
    return weights, evidence_any


def _point_cover_support(
    cover: CandidateCover,
    weights: list[torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    support: list[torch.Tensor] = []
    for cover_id in range(4):
        mask = cover.cover_id[: cover.primary_count] == cover_id
        lower = cover.lower[: cover.primary_count][mask]
        upper = cover.upper[: cover.primary_count][mask]
        inside = (
            (points[:, None, :] >= lower[None, :, :] - 1e-14)
            & (points[:, None, :] <= upper[None, :, :] + 1e-14)
        ).all(dim=2)
        # Grid centers have one owner per cover. At a boundary, max support
        # from both adjacent cells prevents an individual cover from excluding it.
        support.append(
            torch.where(
                inside[None, :, :],
                weights[cover_id][:, None, :],
                torch.zeros_like(weights[cover_id][:, None, :]),
            ).max(dim=2).values
        )
    return torch.stack(support, dim=2)


@torch.no_grad()
def _plausibility_diagnostics(
    rollout: MovingCoverRollout,
    reference: torch.Tensor,
    query_number: int,
    grid_resolution: int,
) -> PlausibilityDiagnostics:
    scale_index = _scale_index_for_step(query_number - 1)
    cover = COVERS[scale_index]
    theta = rollout.query_theta[:, :query_number]
    loss = rollout.query_loss[:, :query_number]
    weights, _ = _primary_cover_weights(cover, theta, loss)

    reference_support_parts: list[torch.Tensor] = []
    for start in range(0, reference.shape[0], BATCH_SIZE):
        batch_weights = [value[start : start + BATCH_SIZE] for value in weights]
        batch_reference = reference[start : start + BATCH_SIZE]
        part: list[torch.Tensor] = []
        for cover_id in range(4):
            mask = cover.cover_id[: cover.primary_count] == cover_id
            lower = cover.lower[: cover.primary_count][mask]
            upper = cover.upper[: cover.primary_count][mask]
            inside = (
                (batch_reference[:, None, :] >= lower[None, :, :] - 1e-14)
                & (batch_reference[:, None, :] <= upper[None, :, :] + 1e-14)
            ).all(dim=2)
            part.append(
                torch.where(inside, batch_weights[cover_id], torch.zeros_like(batch_weights[cover_id])).max(dim=1).values
            )
        reference_support_parts.append(torch.stack(part, dim=1))
    reference_support = torch.cat(reference_support_parts, dim=0)
    reference_strength = reference_support.max(dim=1).values + AGREEMENT_BONUS * (
        reference_support.sum(dim=1) - reference_support.max(dim=1).values
    )
    covered = reference_strength >= SUPPORT_THRESHOLD

    grid = _domain_grid(grid_resolution)
    area_parts: list[torch.Tensor] = []
    multi_area_parts: list[torch.Tensor] = []
    multi_fraction_parts: list[torch.Tensor] = []
    supporting_parts: list[torch.Tensor] = []
    for start in range(0, theta.shape[0], 8):
        batch_weights = [value[start : start + 8] for value in weights]
        grid_support = _point_cover_support(cover, batch_weights, grid)
        strongest = grid_support.max(dim=2).values
        strength = strongest + AGREEMENT_BONUS * (grid_support.sum(dim=2) - strongest)
        plausible = strength >= SUPPORT_THRESHOLD
        cover_count = (grid_support >= SUPPORT_THRESHOLD).sum(dim=2)
        multiple = plausible & (cover_count >= 2)
        area = plausible.to(DTYPE).mean(dim=1) * DOMAIN_AREA
        multi_area = multiple.to(DTYPE).mean(dim=1) * DOMAIN_AREA
        area_parts.append(area)
        multi_area_parts.append(multi_area)
        multi_fraction_parts.append(multi_area / area.clamp_min(1e-15))
        supporting_parts.append(
            torch.where(plausible, cover_count.to(DTYPE), torch.zeros_like(strength)).sum(dim=1)
            / plausible.to(DTYPE).sum(dim=1).clamp_min(1.0)
        )

    squared_distance = torch.sum((reference[:, None, :] - theta).square(), dim=2)
    nearest_error = torch.sqrt(squared_distance.min(dim=1).values)
    missed = ~covered
    near_global_edge = (
        torch.minimum(reference - LOWER, UPPER - reference).min(dim=1).values
        <= 0.5 * cover.scale
    )
    any_local_evidence = (reference_support > 0).any(dim=1)
    miss_edge = missed & near_global_edge
    miss_overlap = missed & (~miss_edge) & any_local_evidence
    miss_insufficient = (
        missed
        & (~miss_edge)
        & (~miss_overlap)
        & (nearest_error > math.sqrt(2.0) * cover.scale)
    )
    miss_controller = missed & (~miss_edge) & (~miss_overlap) & (~miss_insufficient)
    return PlausibilityDiagnostics(
        covered=covered,
        area=torch.cat(area_parts),
        nearest_parameter_error=nearest_error,
        best_prefix_loss=loss.min(dim=1).values,
        multi_cover_area=torch.cat(multi_area_parts),
        multi_cover_fraction=torch.cat(multi_fraction_parts),
        mean_supporting_covers=torch.cat(supporting_parts),
        miss_global_edge=miss_edge,
        miss_poor_overlap=miss_overlap,
        miss_insufficient_exploration=miss_insufficient,
        miss_controller_selection=miss_controller,
    )


def _metrics_row(
    trained_horizon: int,
    query_number: int,
    diagnostics: PlausibilityDiagnostics,
    rollout: MovingCoverRollout,
    grid_resolution: int,
) -> dict[str, Any]:
    categories = rollout.query_category[:, :query_number]
    return {
        "trained_horizon": trained_horizon,
        "evaluations": query_number,
        "coverage": float(diagnostics.covered.to(DTYPE).mean().item()),
        "covered_count": int(diagnostics.covered.sum().item()),
        "median_plausible_area": float(torch.median(diagnostics.area).item()),
        "mean_plausible_area": float(diagnostics.area.mean().item()),
        "worst_plausible_area": float(diagnostics.area.max().item()),
        "median_nearest_parameter_error_posthoc": float(torch.median(diagnostics.nearest_parameter_error).item()),
        "worst_nearest_parameter_error_posthoc": float(diagnostics.nearest_parameter_error.max().item()),
        "median_best_prefix_loss": float(torch.median(diagnostics.best_prefix_loss).item()),
        "worst_best_prefix_loss": float(diagnostics.best_prefix_loss.max().item()),
        "median_multi_cover_support_area": float(torch.median(diagnostics.multi_cover_area).item()),
        "median_multi_cover_support_fraction": float(torch.median(diagnostics.multi_cover_fraction).item()),
        "mean_supporting_cover_count": float(diagnostics.mean_supporting_covers.mean().item()),
        "low_loss_exploitation_fraction": float((categories == 0).to(DTYPE).mean().item()),
        "ordinary_unexplored_fraction": float((categories == 1).to(DTYPE).mean().item()),
        "previous_boundary_fraction": float((categories == 2).to(DTYPE).mean().item()),
        "global_edge_fraction": float((categories == 3).to(DTYPE).mean().item()),
        "miss_global_edge_count": int(diagnostics.miss_global_edge.sum().item()),
        "miss_poor_shifted_overlap_count": int(diagnostics.miss_poor_overlap.sum().item()),
        "miss_insufficient_exploration_count": int(diagnostics.miss_insufficient_exploration.sum().item()),
        "miss_controller_selection_count": int(diagnostics.miss_controller_selection.sum().item()),
        "cover_scale": COVERS[_scale_index_for_step(query_number - 1)].scale,
        "area_grid_resolution": grid_resolution,
    }


def _per_query_rows(
    horizon: int,
    rollout: MovingCoverRollout,
    reference: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_number in range(1, horizon + 1):
        diagnostics = _plausibility_diagnostics(
            rollout, reference, query_number, PER_QUERY_GRID_RESOLUTION
        )
        rows.append(
            _metrics_row(
                horizon, query_number, diagnostics, rollout, PER_QUERY_GRID_RESOLUTION
            )
        )
    return rows


def _plot(rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for horizon in sorted({int(row["trained_horizon"]) for row in rows}):
        subset = [row for row in rows if int(row["trained_horizon"]) == horizon]
        x = [int(row["evaluations"]) for row in subset]
        axes[0].plot(x, [float(row["coverage"]) for row in subset], label=f"trained for {horizon}")
        axes[1].plot(x, [float(row["median_plausible_area"]) for row in subset], label=f"median, trained for {horizon}")
        axes[1].plot(x, [float(row["worst_plausible_area"]) for row in subset], linestyle="--", alpha=0.7, label=f"worst, trained for {horizon}")
    axes[0].set_ylabel("moving-cover coverage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("prefix-loss evaluations")
    axes[1].set_ylabel("plausible union area")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "coverage_and_plausible_area_vs_queries.png", dpi=180)
    plt.close(figure)


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, train_hidden = data_source._make_exact_labeled_split(TRAIN_N, TRAIN_DATA_SEED)
    validation_positions, _, validation_hidden = data_source._make_exact_labeled_split(VALIDATION_N, VALIDATION_DATA_SEED)
    test_positions, _, test_hidden = data_source._make_exact_labeled_split(TEST_N, TEST_DATA_SEED)
    del train_hidden, validation_hidden

    torch.manual_seed(MODEL_SEED)
    model = MovingCoverSelector().to(dtype=DTYPE)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    summary_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    cumulative_updates = 0
    prior_coverage: float | None = None
    stop_reason = "completed_all_horizons"

    for horizon, budget in zip(HORIZONS, UPDATE_BUDGETS):
        training, cumulative_updates = _train_stage(
            model, train_positions, validation_positions, horizon, budget, generator, cumulative_updates
        )
        rollout = _evaluate(model, test_positions, horizon)
        reference = _posthoc_reference(test_hidden)
        stage_rows = _per_query_rows(horizon, rollout, reference)
        query_rows.extend(stage_rows)
        diagnostics = _plausibility_diagnostics(
            rollout, reference, horizon, SUMMARY_GRID_RESOLUTION
        )
        final = _metrics_row(
            horizon, horizon, diagnostics, rollout, SUMMARY_GRID_RESOLUTION
        )
        del reference
        final.update({"row_type": "horizon_summary", **training})
        summary_rows.append(final)
        coverage = float(final["coverage"])
        if prior_coverage is not None and coverage < prior_coverage - 1e-12:
            stop_reason = f"coverage_degraded_at_{horizon}"
            summary_rows.append(
                {
                    "row_type": "stop",
                    "trained_horizon": horizon,
                    "coverage": coverage,
                    "prior_coverage": prior_coverage,
                    "stop_reason": stop_reason,
                    "runtime_seconds_total": time.perf_counter() - start,
                }
            )
            print({"status": stop_reason, "coverage": coverage, "prior_coverage": prior_coverage}, flush=True)
            break
        prior_coverage = coverage

    if not summary_rows or summary_rows[-1].get("row_type") != "stop":
        summary_rows.append(
            {
                "row_type": "stop",
                "trained_horizon": summary_rows[-1]["trained_horizon"],
                "coverage": summary_rows[-1]["coverage"],
                "prior_coverage": prior_coverage,
                "stop_reason": stop_reason,
                "runtime_seconds_total": time.perf_counter() - start,
            }
        )
    _write_csv(OUTPUT_DIR / "moving_cover_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_query_results.csv", query_rows)
    _plot(query_rows)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "stop_reason": stop_reason,
            "completed_horizons": sorted({int(row["trained_horizon"]) for row in query_rows}),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
