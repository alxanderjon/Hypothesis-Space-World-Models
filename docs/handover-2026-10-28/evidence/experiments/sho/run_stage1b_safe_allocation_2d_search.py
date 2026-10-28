"""Safe explore/exploit allocation over moving 2D covers for K(a,b)."""

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

import experiments.sho.run_stage1b_moving_cover_2d_search as moving
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
        "ALLOC2D_OUTPUT_DIR",
        "outputs/sho/stage1B/safe_allocation_2d_search",
    )
)
TRAIN_N = int(os.environ.get("ALLOC2D_TRAIN_N", "4096"))
VALIDATION_N = int(os.environ.get("ALLOC2D_VALIDATION_N", "512"))
TEST_N = int(os.environ.get("ALLOC2D_TEST_N", "512"))
HORIZONS = tuple(
    int(value) for value in os.environ.get("ALLOC2D_HORIZONS", "20,40,80").split(",")
)
UPDATE_BUDGETS = tuple(
    int(value) for value in os.environ.get("ALLOC2D_UPDATES", "800,600,400").split(",")
)
if len(HORIZONS) != len(UPDATE_BUDGETS):
    raise ValueError("Each allocation horizon requires an update budget")
BATCH_SIZE = int(os.environ.get("ALLOC2D_BATCH_SIZE", "32"))
EVAL_INTERVAL = int(os.environ.get("ALLOC2D_EVAL_INTERVAL", "50"))
PATIENCE = int(os.environ.get("ALLOC2D_PATIENCE", "250"))
PER_QUERY_GRID_RESOLUTION = int(os.environ.get("ALLOC2D_QUERY_GRID", "24"))
SUMMARY_GRID_RESOLUTION = int(os.environ.get("ALLOC2D_SUMMARY_GRID", "64"))
DENSITY_GRID_RESOLUTION = int(os.environ.get("ALLOC2D_DENSITY_GRID", "24"))
LEARNING_RATE = 1e-4
DISCOUNT = 0.97
ENTROPY_WEIGHT = 0.01
NOVELTY_WEIGHT = 0.025
EXPLORATION_FLOOR = 0.25
EXPLORATION_LOSS_THRESHOLD = 1e-2
EDGE_STARVATION_STEPS = 8
SUPPORT_THRESHOLD = moving.SUPPORT_THRESHOLD
AGREEMENT_BONUS = moving.AGREEMENT_BONUS
MATERIAL_COVERAGE_DROP = 0.01
MODEL_SEED = SEED + 2_580_000
LOG_EPSILON = 1e-30
DOMAIN_AREA = DOMAIN_WIDTH**2
DOMAIN_DIAGONAL = math.sqrt(2.0) * DOMAIN_WIDTH


@dataclass(frozen=True)
class AllocationCover:
    lower: torch.Tensor
    upper: torch.Tensor
    center: torch.Tensor
    cover_id: torch.Tensor
    global_edge: torch.Tensor
    primary_count: int
    scale: float
    scale_index: int


@dataclass
class AllocationStatistics:
    features: torch.Tensor
    nearest_squared: torch.Tensor
    relative_nearby_log_loss: torch.Tensor
    exploit_metric: torch.Tensor
    explore_metric: torch.Tensor
    edge_warranted: torch.Tensor


@dataclass
class CandidateState:
    nearest_squared: torch.Tensor
    nearest_loss: torch.Tensor
    nearest_improvement: torch.Tensor
    local_count: torch.Tensor


@dataclass
class AllocationRollout:
    query_theta: torch.Tensor
    query_loss: torch.Tensor
    query_explore: torch.Tensor
    query_edge: torch.Tensor
    forced_explore: torch.Tensor
    best_theta: torch.Tensor
    best_loss: torch.Tensor
    entropy: torch.Tensor
    log_probabilities: list[torch.Tensor]
    rewards: list[torch.Tensor]


@dataclass
class AllocationDiagnostics:
    covered: torch.Tensor
    area: torch.Tensor
    nearest_parameter_error: torch.Tensor
    best_prefix_loss: torch.Tensor
    multi_cover_fraction: torch.Tensor
    median_grid_distance: torch.Tensor
    p90_grid_distance: torch.Tensor
    worst_grid_distance: torch.Tensor
    underexplored_grid_fraction: torch.Tensor
    miss_poor_overlap: torch.Tensor
    miss_search_allocation: torch.Tensor
    miss_global_edge: torch.Tensor


class ExploreExploitSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.exploit_head = nn.Sequential(
            nn.Linear(64 * 3, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.explore_head = nn.Sequential(
            nn.Linear(64 * 3, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.mode_head = nn.Sequential(
            nn.Linear(64 * 2 + 4, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        valid_float = valid.to(DTYPE)[:, :, None]
        mean_context = (encoded * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        max_context = encoded.masked_fill(~valid[:, :, None], -torch.inf).max(dim=1).values
        context = torch.cat((mean_context, max_context), dim=1)
        region_context = context[:, None, :].expand(-1, encoded.shape[1], -1)
        scored = torch.cat((encoded, region_context), dim=2)
        exploit_logits = self.exploit_head(scored).squeeze(-1)
        explore_logits = self.explore_head(scored).squeeze(-1)
        mode_logit = self.mode_head(torch.cat((context, global_features), dim=1)).squeeze(-1)
        return exploit_logits, explore_logits, mode_logit


def _allocation_cover(base: moving.CandidateCover) -> AllocationCover:
    primary_indices = torch.arange(base.primary_count)
    edge_indices = torch.nonzero(base.global_edge, as_tuple=False).flatten()
    indices = torch.cat((primary_indices, edge_indices))
    return AllocationCover(
        lower=base.lower[indices],
        upper=base.upper[indices],
        center=base.center[indices],
        cover_id=base.cover_id[indices],
        global_edge=base.global_edge[indices],
        primary_count=base.primary_count,
        scale=base.scale,
        scale_index=base.scale_index,
    )


COVERS = tuple(_allocation_cover(base) for base in moving.COVERS)


def _initialize_candidate_state(
    cover: AllocationCover,
    observed_theta: torch.Tensor,
    observed_loss: torch.Tensor,
    observed_improvement: torch.Tensor,
) -> CandidateState:
    batch_size = observed_theta.shape[0]
    candidate_count = cover.center.shape[0]
    if observed_theta.shape[1] == 0:
        nearest_squared = torch.full(
            (batch_size, candidate_count), DOMAIN_DIAGONAL**2, dtype=DTYPE
        )
        nearest_loss = torch.full_like(nearest_squared, float("inf"))
        nearest_improvement = torch.zeros_like(nearest_squared)
        local_count = torch.zeros_like(nearest_squared, dtype=torch.long)
    else:
        distance = torch.cdist(
            observed_theta,
            cover.center[None, :, :].expand(batch_size, -1, -1),
        )
        nearest_distance, nearest_index = distance.min(dim=1)
        nearest_squared = nearest_distance.square()
        nearest_loss = observed_loss.gather(1, nearest_index)
        nearest_improvement = observed_improvement.gather(1, nearest_index)
        local_count = (distance <= cover.scale).sum(dim=1)
    return CandidateState(
        nearest_squared=nearest_squared,
        nearest_loss=nearest_loss,
        nearest_improvement=nearest_improvement,
        local_count=local_count,
    )


def _update_candidate_state(
    state: CandidateState,
    cover: AllocationCover,
    query_theta: torch.Tensor,
    query_loss: torch.Tensor,
    improvement: torch.Tensor,
) -> None:
    squared_distance = torch.sum(
        (query_theta[:, None, :] - cover.center[None, :, :]).square(), dim=2
    )
    closer = squared_distance < state.nearest_squared
    state.nearest_squared = torch.where(
        closer, squared_distance, state.nearest_squared
    )
    state.nearest_loss = torch.where(
        closer, query_loss[:, None], state.nearest_loss
    )
    state.nearest_improvement = torch.where(
        closer, improvement[:, None], state.nearest_improvement
    )
    state.local_count = state.local_count + (
        squared_distance <= cover.scale**2
    ).to(torch.long)


def _candidate_statistics(
    cover: AllocationCover,
    state: CandidateState,
    observed_loss: torch.Tensor,
) -> AllocationStatistics:
    batch_size = state.nearest_squared.shape[0]
    candidate_count = cover.center.shape[0]
    sides = cover.upper - cover.lower
    area = sides.prod(dim=1)
    if observed_loss.shape[1] == 0:
        relative_nearby = torch.zeros_like(state.nearest_squared)
    else:
        global_best = observed_loss.min(dim=1).values[:, None]
        relative_nearby = torch.clamp(
            torch.log(state.nearest_loss + LOG_EPSILON)
            - torch.log(global_best + LOG_EPSILON),
            min=0.0,
            max=80.0,
        )

    nearest_distance = torch.sqrt(state.nearest_squared.clamp_min(0.0))
    density = torch.log1p(state.local_count.to(DTYPE)) / math.log(96.0)
    empty = (state.local_count == 0).to(DTYPE)
    basin_evidence = torch.exp(-relative_nearby) * torch.log1p(
        state.local_count.to(DTYPE)
    )
    unexplored_score = (
        nearest_distance / DOMAIN_DIAGONAL
        + empty * area[None, :] / DOMAIN_AREA
        + 0.25 / (1.0 + state.local_count.to(DTYPE))
    )
    exploit_metric = (
        -relative_nearby
        + 0.50 * basin_evidence
        + 0.25 * state.nearest_improvement
    )
    explore_metric = unexplored_score
    poorly_sampled_edge = nearest_distance >= 0.5 * cover.scale
    competitive_edge = relative_nearby <= math.log(10.0)
    edge_warranted = cover.global_edge[None, :] & (poorly_sampled_edge | competitive_edge)

    features = torch.stack(
        (
            sides[:, 0][None, :].expand(batch_size, -1) / DOMAIN_WIDTH,
            sides[:, 1][None, :].expand(batch_size, -1) / DOMAIN_WIDTH,
            area[None, :].expand(batch_size, -1) / DOMAIN_AREA,
            nearest_distance / DOMAIN_DIAGONAL,
            relative_nearby / 20.0,
            density,
            empty,
            state.nearest_improvement / 10.0,
            basin_evidence / math.log(96.0),
            unexplored_score,
            torch.full((batch_size, candidate_count), cover.scale / DOMAIN_WIDTH, dtype=DTYPE),
            1.0 / (1.0 + state.local_count.to(DTYPE)),
        ),
        dim=2,
    )
    return AllocationStatistics(
        features=features,
        nearest_squared=state.nearest_squared,
        relative_nearby_log_loss=relative_nearby,
        exploit_metric=exploit_metric,
        explore_metric=explore_metric,
        edge_warranted=edge_warranted,
    )


def _candidate_union(
    cover: AllocationCover,
    statistics: AllocationStatistics,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_count = cover.center.shape[0]

    def top(metric: torch.Tensor, requested: int) -> torch.Tensor:
        return torch.topk(metric, k=min(requested, candidate_count), dim=1).indices

    exploit_pool = top(statistics.exploit_metric, 56)
    explore_pool = top(statistics.explore_metric, 56)
    edge_metric = torch.where(
        statistics.edge_warranted,
        statistics.explore_metric + 10.0,
        torch.full_like(statistics.explore_metric, -torch.inf),
    )
    edge_pool = top(edge_metric, 8)
    indices = torch.cat((exploit_pool, explore_pool, edge_pool), dim=1)
    equal = indices[:, :, None] == indices[:, None, :]
    earlier = torch.tril(
        torch.ones(indices.shape[1], indices.shape[1], dtype=torch.bool), diagonal=-1
    )
    valid = ~(equal & earlier[None, :, :]).any(dim=2)
    exploit_membership = (indices[:, :, None] == exploit_pool[:, None, :]).any(dim=2)
    explore_membership = (indices[:, :, None] == explore_pool[:, None, :]).any(dim=2)
    edge_membership = (indices[:, :, None] == edge_pool[:, None, :]).any(dim=2)
    exploit_membership |= edge_membership
    explore_membership |= edge_membership
    return indices, valid, exploit_membership, explore_membership


def _gather_features(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices[:, :, None].expand(-1, -1, values.shape[2]))


def _allocation_rollout(
    model: ExploreExploitSelector,
    positions: torch.Tensor,
    horizon: int,
    greedy: bool,
) -> AllocationRollout:
    batch_size = positions.shape[0]
    observed_theta = torch.empty(batch_size, 0, 2, dtype=DTYPE)
    observed_loss = torch.empty(batch_size, 0, dtype=DTYPE)
    observed_improvement = torch.empty(batch_size, 0, dtype=DTYPE)
    explore_count = torch.zeros(batch_size, dtype=torch.long)
    steps_since_edge = torch.zeros(batch_size, dtype=torch.long)
    theta_history: list[torch.Tensor] = []
    loss_history: list[torch.Tensor] = []
    explore_history: list[torch.Tensor] = []
    edge_history: list[torch.Tensor] = []
    forced_history: list[torch.Tensor] = []
    best_theta_history: list[torch.Tensor] = []
    best_loss_history: list[torch.Tensor] = []
    entropy_history: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    candidate_state: CandidateState | None = None
    active_scale_index = -1

    for step in range(horizon):
        scale_index = moving._scale_index_for_step(step)
        cover = COVERS[scale_index]
        if candidate_state is None or scale_index != active_scale_index:
            candidate_state = _initialize_candidate_state(
                cover, observed_theta, observed_loss, observed_improvement
            )
            active_scale_index = scale_index
        statistics = _candidate_statistics(cover, candidate_state, observed_loss)
        union, valid, exploit_membership, explore_membership = _candidate_union(
            cover, statistics
        )
        nearest = statistics.nearest_squared.gather(1, union)
        valid &= nearest > 1e-28
        features = _gather_features(statistics.features, union)
        if observed_loss.shape[1] == 0:
            best_loss = torch.full((batch_size,), float("inf"), dtype=DTYPE)
            best_log_feature = torch.zeros(batch_size, dtype=DTYPE)
            latest_improvement = torch.zeros(batch_size, dtype=DTYPE)
        else:
            best_loss = observed_loss.min(dim=1).values
            best_log_feature = torch.clamp(
                torch.log(best_loss + LOG_EPSILON) / 20.0, -4.0, 4.0
            )
            latest_improvement = observed_improvement[:, -1]
        global_features = torch.stack(
            (
                best_log_feature,
                torch.full((batch_size,), step / max(horizon, 1), dtype=DTYPE),
                explore_count.to(DTYPE) / max(step, 1),
                latest_improvement / 10.0,
            ),
            dim=1,
        )
        exploit_logits, explore_logits, mode_logit = model(features, valid, global_features)
        mode_distribution = torch.distributions.Bernoulli(logits=mode_logit)
        proposed_explore = (
            mode_distribution.probs >= 0.5
            if greedy
            else mode_distribution.sample().to(torch.bool)
        )
        required_after_query = math.ceil(EXPLORATION_FLOOR * (step + 1))
        loss_still_large = best_loss >= EXPLORATION_LOSS_THRESHOLD
        forced_explore = loss_still_large & (explore_count < required_after_query)
        choose_explore = proposed_explore | forced_explore

        edge_warranted_union = statistics.edge_warranted.gather(1, union)
        edge_overdue = (
            choose_explore
            & (steps_since_edge >= EDGE_STARVATION_STEPS - 1)
            & (edge_warranted_union & valid).any(dim=1)
        )
        mode_mask = torch.where(
            choose_explore[:, None], explore_membership, exploit_membership
        ) & valid
        mode_mask = torch.where(
            edge_overdue[:, None], mode_mask & edge_warranted_union, mode_mask
        )
        if not bool(mode_mask.any(dim=1).all()):
            raise RuntimeError("Explore/exploit allocation produced an empty candidate pool")
        candidate_logits = torch.where(
            choose_explore[:, None], explore_logits, exploit_logits
        ).masked_fill(~mode_mask, -1e30)
        candidate_distribution = torch.distributions.Categorical(logits=candidate_logits)
        action = (
            torch.argmax(candidate_logits, dim=1)
            if greedy
            else candidate_distribution.sample()
        )
        chosen_global = union.gather(1, action[:, None]).squeeze(1)
        query_theta = cover.center[chosen_global]
        query_edge = cover.global_edge[chosen_global]

        with torch.no_grad():
            query_loss = _prefix_losses(query_theta, positions)
            if observed_loss.shape[1] == 0:
                improvement = torch.zeros_like(query_loss)
            else:
                prior_best = observed_loss.min(dim=1).values
                improvement = torch.clamp(
                    torch.log(prior_best + LOG_EPSILON)
                    - torch.log(torch.minimum(prior_best, query_loss) + LOG_EPSILON),
                    min=0.0,
                    max=10.0,
                )
            novelty = torch.sqrt(
                statistics.nearest_squared.gather(1, chosen_global[:, None]).squeeze(1)
            ) / DOMAIN_DIAGONAL
            reward = improvement + NOVELTY_WEIGHT * choose_explore.to(DTYPE) * novelty

        mode_log_probability = mode_distribution.log_prob(choose_explore.to(DTYPE))
        mode_log_probability = torch.where(
            forced_explore, torch.zeros_like(mode_log_probability), mode_log_probability
        )
        mode_entropy = torch.where(
            forced_explore,
            torch.zeros_like(mode_distribution.entropy()),
            mode_distribution.entropy(),
        )
        observed_theta = torch.cat((observed_theta, query_theta[:, None, :]), dim=1)
        observed_loss = torch.cat((observed_loss, query_loss[:, None]), dim=1)
        observed_improvement = torch.cat(
            (observed_improvement, improvement[:, None]), dim=1
        )
        _update_candidate_state(
            candidate_state, cover, query_theta, query_loss, improvement
        )
        explore_count += choose_explore.to(torch.long)
        steps_since_edge = torch.where(
            query_edge, torch.zeros_like(steps_since_edge), steps_since_edge + 1
        )
        best_index = torch.argmin(observed_loss, dim=1)
        current_best_theta = observed_theta.gather(
            1, best_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        current_best_loss = observed_loss.gather(1, best_index[:, None]).squeeze(1)

        theta_history.append(query_theta.detach())
        loss_history.append(query_loss.detach())
        explore_history.append(choose_explore.detach())
        edge_history.append(query_edge.detach())
        forced_history.append(forced_explore.detach())
        best_theta_history.append(current_best_theta)
        best_loss_history.append(current_best_loss)
        entropy_history.append(candidate_distribution.entropy() + mode_entropy)
        log_probabilities.append(
            candidate_distribution.log_prob(action) + mode_log_probability
        )
        rewards.append(reward)

    return AllocationRollout(
        query_theta=torch.stack(theta_history, dim=1),
        query_loss=torch.stack(loss_history, dim=1),
        query_explore=torch.stack(explore_history, dim=1),
        query_edge=torch.stack(edge_history, dim=1),
        forced_explore=torch.stack(forced_history, dim=1),
        best_theta=torch.stack(best_theta_history, dim=1),
        best_loss=torch.stack(best_loss_history, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        log_probabilities=log_probabilities,
        rewards=rewards,
    )


def _policy_loss(rollout: AllocationRollout) -> torch.Tensor:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rollout.rewards[0])
    for reward in reversed(rollout.rewards):
        running = reward + DISCOUNT * running
        returns.append(running)
    returns.reverse()
    terms: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for value, log_probability, entropy in zip(
        returns, rollout.log_probabilities, rollout.entropy.unbind(dim=1)
    ):
        advantage = value - value.mean()
        advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-6)
        terms.append(-(advantage.detach() * log_probability).mean())
        entropies.append(entropy.mean())
    return torch.stack(terms).mean() - ENTROPY_WEIGHT * torch.stack(entropies).mean()


@torch.no_grad()
def _evaluate(
    model: ExploreExploitSelector,
    positions: torch.Tensor,
    horizon: int,
) -> AllocationRollout:
    model.eval()
    parts = [
        _allocation_rollout(
            model, positions[start : start + BATCH_SIZE], horizon, True
        )
        for start in range(0, positions.shape[0], BATCH_SIZE)
    ]
    fields = (
        "query_theta", "query_loss", "query_explore", "query_edge",
        "forced_explore", "best_theta", "best_loss", "entropy",
    )
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in fields
    }
    return AllocationRollout(**combined, log_probabilities=[], rewards=[])


def _validation_score(rollout: AllocationRollout) -> float:
    return float(torch.log(rollout.best_loss[:, -1] + LOG_EPSILON).mean().item())


def _train_stage(
    model: ExploreExploitSelector,
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
    print(f"alloc2d horizon={horizon} update=0000 validation={best_score:.6e}", flush=True)
    for update in range(1, update_budget + 1):
        model.train()
        indices = torch.randint(
            0, train_positions.shape[0], (BATCH_SIZE,), generator=generator
        )
        rollout = _allocation_rollout(
            model, train_positions[indices], horizon, False
        )
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
                f"alloc2d horizon={horizon} update={update:04d} "
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


def _cover_support(
    cover: AllocationCover,
    theta: torch.Tensor,
    loss: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    weights, _ = moving._primary_cover_weights(cover, theta, loss)
    return moving._point_cover_support(cover, weights, points)


@torch.no_grad()
def _diagnostics(
    rollout: AllocationRollout,
    reference: torch.Tensor,
    query_number: int,
    grid_resolution: int,
) -> AllocationDiagnostics:
    cover = COVERS[moving._scale_index_for_step(query_number - 1)]
    theta = rollout.query_theta[:, :query_number]
    loss = rollout.query_loss[:, :query_number]
    weights, _ = moving._primary_cover_weights(cover, theta, loss)

    reference_support_parts: list[torch.Tensor] = []
    for start in range(0, reference.shape[0], BATCH_SIZE):
        part: list[torch.Tensor] = []
        batch_reference = reference[start : start + BATCH_SIZE]
        for cover_id in range(4):
            mask = cover.cover_id[: cover.primary_count] == cover_id
            lower = cover.lower[: cover.primary_count][mask]
            upper = cover.upper[: cover.primary_count][mask]
            inside = (
                (batch_reference[:, None, :] >= lower[None, :, :] - 1e-14)
                & (batch_reference[:, None, :] <= upper[None, :, :] + 1e-14)
            ).all(dim=2)
            batch_weight = weights[cover_id][start : start + BATCH_SIZE]
            part.append(
                torch.where(inside, batch_weight, torch.zeros_like(batch_weight)).max(dim=1).values
            )
        reference_support_parts.append(torch.stack(part, dim=1))
    reference_support = torch.cat(reference_support_parts)
    strongest_reference = reference_support.max(dim=1).values
    reference_strength = strongest_reference + AGREEMENT_BONUS * (
        reference_support.sum(dim=1) - strongest_reference
    )
    covered = reference_strength >= SUPPORT_THRESHOLD

    grid = moving._domain_grid(grid_resolution)
    density_grid = moving._domain_grid(DENSITY_GRID_RESOLUTION)
    area_parts: list[torch.Tensor] = []
    multi_fraction_parts: list[torch.Tensor] = []
    median_density_parts: list[torch.Tensor] = []
    p90_density_parts: list[torch.Tensor] = []
    worst_density_parts: list[torch.Tensor] = []
    underexplored_parts: list[torch.Tensor] = []
    for start in range(0, theta.shape[0], 8):
        batch_weights = [value[start : start + 8] for value in weights]
        support = moving._point_cover_support(cover, batch_weights, grid)
        strongest = support.max(dim=2).values
        strength = strongest + AGREEMENT_BONUS * (support.sum(dim=2) - strongest)
        plausible = strength >= SUPPORT_THRESHOLD
        multiple = plausible & ((support >= SUPPORT_THRESHOLD).sum(dim=2) >= 2)
        area = plausible.to(DTYPE).mean(dim=1) * DOMAIN_AREA
        multiple_area = multiple.to(DTYPE).mean(dim=1) * DOMAIN_AREA
        area_parts.append(area)
        multi_fraction_parts.append(multiple_area / area.clamp_min(1e-15))

        batch_theta = theta[start : start + 8]
        nearest_grid = torch.sqrt(
            torch.sum(
                (density_grid[None, :, None, :] - batch_theta[:, None, :, :]).square(),
                dim=3,
            ).min(dim=2).values
        )
        median_density_parts.append(torch.quantile(nearest_grid, 0.5, dim=1))
        p90_density_parts.append(torch.quantile(nearest_grid, 0.9, dim=1))
        worst_density_parts.append(nearest_grid.max(dim=1).values)
        underexplored_parts.append((nearest_grid > cover.scale).to(DTYPE).mean(dim=1))

    squared_error = torch.sum((reference[:, None, :] - theta).square(), dim=2)
    nearest_error = torch.sqrt(squared_error.min(dim=1).values)
    missed = ~covered
    any_local_evidence = (reference_support > 0).any(dim=1)
    near_edge = (
        torch.minimum(reference - LOWER, UPPER - reference).min(dim=1).values
        <= 0.5 * cover.scale
    )
    miss_edge = missed & near_edge
    miss_overlap = missed & (~miss_edge) & any_local_evidence
    miss_allocation = missed & (~miss_edge) & (~miss_overlap)
    return AllocationDiagnostics(
        covered=covered,
        area=torch.cat(area_parts),
        nearest_parameter_error=nearest_error,
        best_prefix_loss=loss.min(dim=1).values,
        multi_cover_fraction=torch.cat(multi_fraction_parts),
        median_grid_distance=torch.cat(median_density_parts),
        p90_grid_distance=torch.cat(p90_density_parts),
        worst_grid_distance=torch.cat(worst_density_parts),
        underexplored_grid_fraction=torch.cat(underexplored_parts),
        miss_poor_overlap=miss_overlap,
        miss_search_allocation=miss_allocation,
        miss_global_edge=miss_edge,
    )


def _metrics_row(
    horizon: int,
    query_number: int,
    diagnostics: AllocationDiagnostics,
    rollout: AllocationRollout,
    grid_resolution: int,
) -> dict[str, Any]:
    return {
        "trained_horizon": horizon,
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
        "exploration_fraction": float(rollout.query_explore[:, :query_number].to(DTYPE).mean().item()),
        "exploitation_fraction": float((~rollout.query_explore[:, :query_number]).to(DTYPE).mean().item()),
        "forced_exploration_fraction": float(rollout.forced_explore[:, :query_number].to(DTYPE).mean().item()),
        "global_edge_query_fraction": float(rollout.query_edge[:, :query_number].to(DTYPE).mean().item()),
        "median_multi_cover_support_fraction": float(torch.median(diagnostics.multi_cover_fraction).item()),
        "median_domain_grid_distance": float(torch.median(diagnostics.median_grid_distance).item()),
        "median_p90_domain_grid_distance": float(torch.median(diagnostics.p90_grid_distance).item()),
        "worst_domain_grid_distance": float(diagnostics.worst_grid_distance.max().item()),
        "median_underexplored_grid_fraction": float(torch.median(diagnostics.underexplored_grid_fraction).item()),
        "miss_global_edge_count": int(diagnostics.miss_global_edge.sum().item()),
        "miss_poor_moving_cover_overlap_count": int(diagnostics.miss_poor_overlap.sum().item()),
        "miss_search_allocation_count": int(diagnostics.miss_search_allocation.sum().item()),
        "cover_scale": COVERS[moving._scale_index_for_step(query_number - 1)].scale,
        "area_grid_resolution": grid_resolution,
    }


def _per_query_rows(
    horizon: int,
    rollout: AllocationRollout,
    reference: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_number in range(1, horizon + 1):
        diagnostics = _diagnostics(
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
    figure.savefig(OUTPUT_DIR / "coverage_area_vs_queries.png", dpi=180)
    plt.close(figure)


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, train_hidden = data_source._make_exact_labeled_split(TRAIN_N, TRAIN_DATA_SEED)
    validation_positions, _, validation_hidden = data_source._make_exact_labeled_split(VALIDATION_N, VALIDATION_DATA_SEED)
    test_positions, _, test_hidden = data_source._make_exact_labeled_split(TEST_N, TEST_DATA_SEED)
    del train_hidden, validation_hidden

    torch.manual_seed(MODEL_SEED)
    model = ExploreExploitSelector().to(dtype=DTYPE)
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
        diagnostics = _diagnostics(rollout, reference, horizon, SUMMARY_GRID_RESOLUTION)
        final = _metrics_row(
            horizon, horizon, diagnostics, rollout, SUMMARY_GRID_RESOLUTION
        )
        del reference
        final.update({"row_type": "horizon_summary", **training})
        summary_rows.append(final)
        coverage = float(final["coverage"])
        if (
            prior_coverage is not None
            and coverage < prior_coverage - MATERIAL_COVERAGE_DROP
        ):
            stop_reason = f"material_coverage_degradation_at_{horizon}"
            summary_rows.append(
                {
                    "row_type": "stop",
                    "trained_horizon": horizon,
                    "coverage": coverage,
                    "prior_coverage": prior_coverage,
                    "material_drop_threshold": MATERIAL_COVERAGE_DROP,
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
                "material_drop_threshold": MATERIAL_COVERAGE_DROP,
                "stop_reason": stop_reason,
                "runtime_seconds_total": time.perf_counter() - start,
            }
        )
    _write_csv(OUTPUT_DIR / "allocation_summary.csv", summary_rows)
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
