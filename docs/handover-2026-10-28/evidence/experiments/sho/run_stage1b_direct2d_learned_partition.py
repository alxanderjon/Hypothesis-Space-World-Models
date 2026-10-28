"""Learned-DIRECT-style full-partition search for unconstrained K(a,b)."""

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
    P,
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
        "DIRECT2D_OUTPUT_DIR",
        "outputs/sho/stage1B/direct2d_learned_partition",
    )
)
TRAIN_N = int(os.environ.get("DIRECT2D_TRAIN_N", "4096"))
VALIDATION_N = int(os.environ.get("DIRECT2D_VALIDATION_N", "512"))
TEST_N = int(os.environ.get("DIRECT2D_TEST_N", "512"))
HORIZONS = tuple(
    int(value)
    for value in os.environ.get("DIRECT2D_HORIZONS", "20,40,80").split(",")
)
UPDATE_BUDGETS = tuple(
    int(value)
    for value in os.environ.get("DIRECT2D_UPDATES", "800,600,400").split(",")
)
if len(HORIZONS) != len(UPDATE_BUDGETS):
    raise ValueError("Each learned-DIRECT horizon requires an update budget")
BATCH_SIZE = int(os.environ.get("DIRECT2D_BATCH_SIZE", "32"))
LEARNING_RATE = 1e-4
EVAL_INTERVAL = int(os.environ.get("DIRECT2D_EVAL_INTERVAL", "50"))
PATIENCE = int(os.environ.get("DIRECT2D_PATIENCE", "250"))
DISCOUNT = 0.97
NOVELTY_WEIGHT = 0.025
ENTROPY_WEIGHT = 0.01
COMPETITIVE_LOSS_FACTOR = 10.0
MINIMUM_COVERAGE_TO_CONTINUE = float(
    os.environ.get("DIRECT2D_MIN_COVERAGE", "0.90")
)
MAXIMUM_COVERAGE_DROP = float(
    os.environ.get("DIRECT2D_MAX_COVERAGE_DROP", "0.02")
)
MODEL_SEED = SEED + 2_350_000
LOG_EPSILON = 1e-30


class Direct2DSelector(nn.Module):
    """Scores active leaves; it never predicts or changes their boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self.rectangle_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.rectangle_encoder(features)
        mean_context = embedded.mean(dim=1, keepdim=True)
        max_context = embedded.max(dim=1, keepdim=True).values
        context = torch.cat((mean_context, max_context), dim=2).expand(
            -1, embedded.shape[1], -1
        )
        return self.score_head(torch.cat((embedded, context), dim=2)).squeeze(-1)


@dataclass
class Direct2DRollout:
    query_theta: torch.Tensor
    query_loss: torch.Tensor
    best_theta: torch.Tensor
    best_loss: torch.Tensor
    candidate_area: torch.Tensor
    active_unexplored: torch.Tensor
    exploration_action: torch.Tensor
    selected_area: torch.Tensor
    entropy: torch.Tensor
    partition_lower: list[torch.Tensor]
    partition_upper: list[torch.Tensor]
    candidate_mask: list[torch.Tensor]
    log_probabilities: list[torch.Tensor]
    rewards: list[torch.Tensor]


@dataclass
class LeafStatistics:
    features: torch.Tensor
    counts: torch.Tensor
    nearest_squared: torch.Tensor
    best_nearby_loss: torch.Tensor


def _leaf_statistics(
    lower: torch.Tensor,
    upper: torch.Tensor,
    observed_theta: torch.Tensor,
    observed_loss: torch.Tensor,
) -> LeafStatistics:
    centers = 0.5 * (lower + upper)
    sides = upper - lower
    area = sides.prod(dim=2)
    batch_size, rectangle_count, _ = lower.shape
    if observed_theta.shape[1] == 0:
        zeros = torch.zeros(batch_size, rectangle_count, dtype=DTYPE)
        ones = torch.ones_like(zeros)
        features = torch.cat(
            (
                centers / UPPER,
                sides / DOMAIN_WIDTH,
                (area / DOMAIN_WIDTH**2)[:, :, None],
                zeros[:, :, None],
                zeros[:, :, None],
                zeros[:, :, None],
                ones[:, :, None],
                zeros[:, :, None],
                ones[:, :, None],
            ),
            dim=2,
        )
        return LeafStatistics(
            features=features,
            counts=torch.zeros_like(area, dtype=torch.long),
            nearest_squared=torch.full_like(area, 2.0 * DOMAIN_WIDTH**2),
            best_nearby_loss=torch.full_like(area, float("inf")),
        )

    delta = observed_theta[:, None, :, :] - centers[:, :, None, :]
    squared_distance = torch.sum(delta.square(), dim=3)
    nearest_squared, nearest_index = squared_distance.min(dim=2)
    center_evaluated = nearest_squared <= 1e-28
    nearest_loss = observed_loss.gather(1, nearest_index)
    inside = (
        (observed_theta[:, None, :, :] >= lower[:, :, None, :] - 1e-14)
        & (observed_theta[:, None, :, :] <= upper[:, :, None, :] + 1e-14)
    ).all(dim=3)
    counts = inside.sum(dim=2)
    nearest_count = min(3, observed_theta.shape[1])
    nearest_indices = torch.topk(
        squared_distance, k=nearest_count, dim=2, largest=False
    ).indices
    nearby_losses = observed_loss[:, None, :].expand(
        -1, rectangle_count, -1
    ).gather(2, nearest_indices)
    best_nearby_loss = nearby_losses.min(dim=2).values
    global_best = observed_loss.min(dim=1).values[:, None]
    features = torch.cat(
        (
            centers / UPPER,
            sides / DOMAIN_WIDTH,
            (area / DOMAIN_WIDTH**2)[:, :, None],
            center_evaluated.to(DTYPE)[:, :, None],
            torch.where(
                center_evaluated,
                torch.log(nearest_loss + LOG_EPSILON) / 20.0,
                torch.zeros_like(nearest_loss),
            )[:, :, None],
            (torch.log1p(counts.to(DTYPE)) / math.log(96.0))[:, :, None],
            (nearest_squared / (2.0 * DOMAIN_WIDTH**2))[:, :, None],
            (torch.log(best_nearby_loss + LOG_EPSILON) / 20.0)[:, :, None],
            (counts == 0).to(DTYPE)[:, :, None],
        ),
        dim=2,
    )
    # Relative loss makes the feature stable across trajectory amplitudes.
    features[:, :, 9] = torch.clamp(
        (
            torch.log(best_nearby_loss + LOG_EPSILON)
            - torch.log(global_best + LOG_EPSILON)
        )
        / 20.0,
        0.0,
        4.0,
    )
    return LeafStatistics(features, counts, nearest_squared, best_nearby_loss)


def _split_longest_side(
    lower: torch.Tensor,
    upper: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, rectangle_count, _ = lower.shape
    gather_index = actions[:, None, None].expand(-1, 1, 2)
    selected_lower = lower.gather(1, gather_index).squeeze(1)
    selected_upper = upper.gather(1, gather_index).squeeze(1)
    sides = selected_upper - selected_lower
    split_dimension = (sides[:, 1] > sides[:, 0]).to(torch.long)
    middle = 0.5 * (selected_lower + selected_upper)
    first_lower = selected_lower.clone()
    first_upper = selected_upper.clone()
    second_lower = selected_lower.clone()
    second_upper = selected_upper.clone()
    rows = torch.arange(batch_size)
    first_upper[rows, split_dimension] = middle[rows, split_dimension]
    second_lower[rows, split_dimension] = middle[rows, split_dimension]
    children_lower = torch.stack((first_lower, second_lower), dim=1)
    children_upper = torch.stack((first_upper, second_upper), dim=1)
    if rectangle_count == 1:
        return children_lower, children_upper
    indices = torch.arange(rectangle_count)[None, :].expand(batch_size, -1)
    keep = indices != actions[:, None]
    kept_lower = lower[keep].reshape(batch_size, rectangle_count - 1, 2)
    kept_upper = upper[keep].reshape(batch_size, rectangle_count - 1, 2)
    return (
        torch.cat((kept_lower, children_lower), dim=1),
        torch.cat((kept_upper, children_upper), dim=1),
    )


def _candidate_union_mask(
    lower: torch.Tensor,
    upper: torch.Tensor,
    best_theta: torch.Tensor,
    best_loss: torch.Tensor,
    best_nearby_loss: torch.Tensor,
    nearest_squared: torch.Tensor,
) -> torch.Tensor:
    base = (
        (best_theta[:, None, :] >= lower - 1e-14)
        & (best_theta[:, None, :] <= upper + 1e-14)
    ).all(dim=2)
    overlap_or_touch = (
        (lower[:, :, None, :] <= upper[:, None, :, :] + 1e-14)
        & (upper[:, :, None, :] >= lower[:, None, :, :] - 1e-14)
    ).all(dim=3)
    adjacent_to_base = (overlap_or_touch & base[:, None, :]).any(dim=2)
    # An adjacent leaf whose own center has never been tested is still
    # plausible: nearby samples are evidence, but not grounds for a hard
    # exclusion. This is the boundary-safety rule that preserves information.
    unexplored = nearest_squared > 1e-28
    competitive = unexplored | (
        best_nearby_loss
        <= COMPETITIVE_LOSS_FACTOR * best_loss[:, None] + LOG_EPSILON
    )
    candidate = base | (adjacent_to_base & competitive)
    if not bool(torch.all(candidate.any(dim=1))):
        raise RuntimeError("Candidate union unexpectedly became empty")
    return candidate


def _direct2d_rollout(
    model: Direct2DSelector,
    positions: torch.Tensor,
    horizon: int,
    greedy: bool,
) -> Direct2DRollout:
    batch_size = positions.shape[0]
    lower = torch.full((batch_size, 1, 2), LOWER, dtype=DTYPE)
    upper = torch.full((batch_size, 1, 2), UPPER, dtype=DTYPE)
    observed_theta = torch.empty(batch_size, 0, 2, dtype=DTYPE)
    observed_loss = torch.empty(batch_size, 0, dtype=DTYPE)
    query_theta_history: list[torch.Tensor] = []
    query_loss_history: list[torch.Tensor] = []
    best_theta_history: list[torch.Tensor] = []
    best_loss_history: list[torch.Tensor] = []
    candidate_area_history: list[torch.Tensor] = []
    active_unexplored_history: list[torch.Tensor] = []
    exploration_history: list[torch.Tensor] = []
    selected_area_history: list[torch.Tensor] = []
    entropy_history: list[torch.Tensor] = []
    partition_lower_history: list[torch.Tensor] = []
    partition_upper_history: list[torch.Tensor] = []
    candidate_mask_history: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for _ in range(horizon):
        statistics = _leaf_statistics(
            lower, upper, observed_theta, observed_loss
        )
        logits = model(statistics.features)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=1) if greedy else distribution.sample()
        gather_index = actions[:, None, None].expand(-1, 1, 2)
        selected_lower = lower.gather(1, gather_index).squeeze(1)
        selected_upper = upper.gather(1, gather_index).squeeze(1)
        selected_area = (selected_upper - selected_lower).prod(dim=1)
        query_theta = 0.5 * (selected_lower + selected_upper)
        if observed_theta.shape[1] == 0:
            exploration = torch.ones(batch_size, dtype=torch.bool)
            prior_best = torch.full((batch_size,), float("inf"), dtype=DTYPE)
        else:
            best_before_index = torch.argmin(observed_loss, dim=1)
            best_before = observed_theta.gather(
                1, best_before_index[:, None, None].expand(-1, 1, 2)
            ).squeeze(1)
            selected_contains_best = (
                (best_before >= selected_lower - 1e-14)
                & (best_before <= selected_upper + 1e-14)
            ).all(dim=1)
            exploration = ~selected_contains_best
            prior_best = observed_loss.min(dim=1).values
        with torch.no_grad():
            query_loss = _prefix_losses(query_theta, positions)
            next_best = (
                query_loss
                if observed_loss.shape[1] == 0
                else torch.minimum(prior_best, query_loss)
            )
            improvement = (
                torch.zeros_like(query_loss)
                if observed_loss.shape[1] == 0
                else torch.clamp(
                    torch.log(prior_best + LOG_EPSILON)
                    - torch.log(next_best + LOG_EPSILON),
                    min=0.0,
                    max=10.0,
                )
            )
            selected_nearest = statistics.nearest_squared.gather(
                1, actions[:, None]
            ).squeeze(1)
            novelty = (
                0.4 * selected_area / DOMAIN_WIDTH**2
                + 0.4 * selected_nearest / (2.0 * DOMAIN_WIDTH**2)
                + 0.2 * exploration.to(DTYPE)
            )
            reward = improvement + NOVELTY_WEIGHT * novelty

        observed_theta = torch.cat(
            (observed_theta, query_theta.detach()[:, None, :]), dim=1
        )
        observed_loss = torch.cat(
            (observed_loss, query_loss.detach()[:, None]), dim=1
        )
        lower, upper = _split_longest_side(lower, upper, actions)
        best_index = torch.argmin(observed_loss, dim=1)
        best_theta = observed_theta.gather(
            1, best_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        best_loss = observed_loss.gather(1, best_index[:, None]).squeeze(1)
        next_statistics = _leaf_statistics(
            lower, upper, observed_theta, observed_loss
        )
        candidate_mask = _candidate_union_mask(
            lower,
            upper,
            best_theta,
            best_loss,
            next_statistics.best_nearby_loss,
            next_statistics.nearest_squared,
        )
        areas = (upper - lower).prod(dim=2)
        candidate_area = torch.where(
            candidate_mask, areas, torch.zeros_like(areas)
        ).sum(dim=1)

        query_theta_history.append(query_theta.detach())
        query_loss_history.append(query_loss.detach())
        best_theta_history.append(best_theta)
        best_loss_history.append(best_loss)
        candidate_area_history.append(candidate_area)
        # An active leaf is unexplored until its own center has been sampled.
        # Parent-center samples can lie on both child boundaries, so geometric
        # membership counts alone would incorrectly mark new children explored.
        active_unexplored_history.append(
            (next_statistics.nearest_squared > 1e-28).sum(dim=1)
        )
        exploration_history.append(exploration)
        selected_area_history.append(selected_area)
        entropy_history.append(distribution.entropy())
        partition_lower_history.append(lower.detach())
        partition_upper_history.append(upper.detach())
        candidate_mask_history.append(candidate_mask.detach())
        log_probabilities.append(distribution.log_prob(actions))
        rewards.append(reward)

    return Direct2DRollout(
        query_theta=torch.stack(query_theta_history, dim=1),
        query_loss=torch.stack(query_loss_history, dim=1),
        best_theta=torch.stack(best_theta_history, dim=1),
        best_loss=torch.stack(best_loss_history, dim=1),
        candidate_area=torch.stack(candidate_area_history, dim=1),
        active_unexplored=torch.stack(active_unexplored_history, dim=1),
        exploration_action=torch.stack(exploration_history, dim=1),
        selected_area=torch.stack(selected_area_history, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        partition_lower=partition_lower_history,
        partition_upper=partition_upper_history,
        candidate_mask=candidate_mask_history,
        log_probabilities=log_probabilities,
        rewards=rewards,
    )


def _policy_loss(rollout: Direct2DRollout) -> torch.Tensor:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rollout.rewards[0])
    for reward in reversed(rollout.rewards):
        running = reward + DISCOUNT * running
        returns.append(running)
    returns.reverse()
    terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    for return_value, log_probability, entropy in zip(
        returns, rollout.log_probabilities, rollout.entropy.unbind(dim=1)
    ):
        advantage = return_value - return_value.mean()
        advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-6)
        terms.append(-(advantage.detach() * log_probability).mean())
        entropy_terms.append(entropy.mean())
    return torch.stack(terms).mean() - ENTROPY_WEIGHT * torch.stack(
        entropy_terms
    ).mean()


@torch.no_grad()
def _evaluate(
    model: Direct2DSelector,
    positions: torch.Tensor,
    horizon: int,
) -> Direct2DRollout:
    model.eval()
    parts = [
        _direct2d_rollout(
            model, positions[start : start + BATCH_SIZE], horizon, True
        )
        for start in range(0, positions.shape[0], BATCH_SIZE)
    ]
    tensor_fields = (
        "query_theta",
        "query_loss",
        "best_theta",
        "best_loss",
        "candidate_area",
        "active_unexplored",
        "exploration_action",
        "selected_area",
        "entropy",
    )
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in tensor_fields
    }
    lower_history = [
        torch.cat([part.partition_lower[index] for part in parts], dim=0)
        for index in range(horizon)
    ]
    upper_history = [
        torch.cat([part.partition_upper[index] for part in parts], dim=0)
        for index in range(horizon)
    ]
    mask_history = [
        torch.cat([part.candidate_mask[index] for part in parts], dim=0)
        for index in range(horizon)
    ]
    return Direct2DRollout(
        **combined,
        partition_lower=lower_history,
        partition_upper=upper_history,
        candidate_mask=mask_history,
        log_probabilities=[],
        rewards=[],
    )


def _validation_score(rollout: Direct2DRollout) -> float:
    return float(
        torch.log(rollout.best_loss[:, -1] + LOG_EPSILON).mean().item()
    )


def _contains_reference(
    lower: torch.Tensor,
    upper: torch.Tensor,
    candidate_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    contains = (
        (reference[:, None, :] >= lower - 1e-14)
        & (reference[:, None, :] <= upper + 1e-14)
    ).all(dim=2)
    return (contains & candidate_mask).any(dim=1)


def _boundary_associated_failure(
    lower: torch.Tensor,
    upper: torch.Tensor,
    candidate_mask: torch.Tensor,
    reference: torch.Tensor,
    covered: torch.Tensor,
) -> torch.Tensor:
    reference_leaf = (
        (reference[:, None, :] >= lower - 1e-14)
        & (reference[:, None, :] <= upper + 1e-14)
    ).all(dim=2)
    overlap_or_touch = (
        (lower[:, :, None, :] <= upper[:, None, :, :] + 1e-14)
        & (upper[:, :, None, :] >= lower[:, None, :, :] - 1e-14)
    ).all(dim=3)
    reference_adjacent_to_candidate = (
        overlap_or_touch
        & reference_leaf[:, :, None]
        & candidate_mask[:, None, :]
    ).any(dim=(1, 2))
    return (~covered) & reference_adjacent_to_candidate


def _metrics_at_query(
    rollout: Direct2DRollout,
    reference: torch.Tensor,
    query_number: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    index = query_number - 1
    lower = rollout.partition_lower[index]
    upper = rollout.partition_upper[index]
    mask = rollout.candidate_mask[index]
    covered = _contains_reference(lower, upper, mask, reference)
    boundary_failure = _boundary_associated_failure(
        lower, upper, mask, reference, covered
    )
    parameter_error = torch.sqrt(
        torch.sum(
            (rollout.best_theta[:, index] - reference).square(), dim=1
        )
    )
    area = rollout.candidate_area[:, index]
    loss = rollout.best_loss[:, index]
    row = {
        "evaluations": query_number,
        "coverage": float(covered.to(DTYPE).mean().item()),
        "covered_count": int(covered.sum().item()),
        "median_candidate_area": float(torch.median(area).item()),
        "worst_candidate_area": float(area.max().item()),
        "median_best_parameter_error_posthoc": float(
            torch.median(parameter_error).item()
        ),
        "worst_best_parameter_error_posthoc": float(parameter_error.max().item()),
        "median_best_prefix_loss": float(torch.median(loss).item()),
        "worst_best_prefix_loss": float(loss.max().item()),
        "median_active_unexplored_rectangles": float(
            torch.median(rollout.active_unexplored[:, index].to(DTYPE)).item()
        ),
        "worst_active_unexplored_rectangles": int(
            rollout.active_unexplored[:, index].max().item()
        ),
        "exploration_fraction_cumulative": float(
            rollout.exploration_action[:, :query_number].to(DTYPE).mean().item()
        ),
        "exploitation_fraction_cumulative": float(
            (~rollout.exploration_action[:, :query_number]).to(DTYPE).mean().item()
        ),
        "boundary_associated_failure_count": int(boundary_failure.sum().item()),
        "boundary_associated_fraction_among_failures": float(
            boundary_failure.to(DTYPE).sum().item()
            / max(int((~covered).sum().item()), 1)
        ),
    }
    return row, covered, boundary_failure


def _per_query_rows(
    trained_horizon: int,
    rollout: Direct2DRollout,
    reference: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_number in range(1, trained_horizon + 1):
        metrics, _, _ = _metrics_at_query(rollout, reference, query_number)
        rows.append({"trained_horizon": trained_horizon, **metrics})
    return rows


def _train_stage(
    model: Direct2DSelector,
    train_positions: torch.Tensor,
    validation_positions: torch.Tensor,
    horizon: int,
    update_budget: int,
    batch_generator: torch.Generator,
    cumulative_updates: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    validation = _evaluate(model, validation_positions, horizon)
    best_score = _validation_score(validation)
    best_state = copy.deepcopy(model.state_dict())
    best_update = 0
    stale = 0
    updates_run = 0
    stop_reason = "maximum_updates"
    start = time.perf_counter()
    print(
        f"direct2d horizon={horizon} update=0000 "
        f"validation mean log best loss={best_score:.6e}",
        flush=True,
    )
    for update in range(1, update_budget + 1):
        model.train()
        indices = torch.randint(
            0,
            train_positions.shape[0],
            (BATCH_SIZE,),
            generator=batch_generator,
        )
        rollout = _direct2d_rollout(
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
                f"direct2d horizon={horizon} update={update:04d} "
                f"policy loss={float(loss.detach().item()):.6e} "
                f"validation mean log best loss={score:.6e} "
                f"best_update={best_update} stale={stale}",
                flush=True,
            )
            if stale >= PATIENCE:
                stop_reason = "validation_plateau"
                break
    model.load_state_dict(best_state)
    history = {
        "trained_horizon": horizon,
        "updates_run": updates_run,
        "best_update": best_update,
        "best_validation_score": best_score,
        "stop_reason": stop_reason,
        "runtime_seconds": time.perf_counter() - start,
        "cumulative_updates": cumulative_updates,
    }
    return best_state, history, cumulative_updates


def _plot(rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    horizons = sorted({int(row["trained_horizon"]) for row in rows})
    for horizon in horizons:
        subset = [row for row in rows if int(row["trained_horizon"]) == horizon]
        evaluations = [int(row["evaluations"]) for row in subset]
        axes[0].plot(
            evaluations,
            [float(row["coverage"]) for row in subset],
            label=f"trained for {horizon}",
        )
        axes[1].plot(
            evaluations,
            [float(row["median_candidate_area"]) for row in subset],
            label=f"median, trained for {horizon}",
        )
        axes[1].plot(
            evaluations,
            [float(row["worst_candidate_area"]) for row in subset],
            linestyle="--",
            alpha=0.7,
            label=f"worst, trained for {horizon}",
        )
    axes[0].set_ylabel("candidate-union coverage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("prefix-loss evaluations")
    axes[1].set_ylabel("candidate-union area")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "coverage_and_area_vs_evaluations.png", dpi=180
    )
    plt.close(figure)


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, train_hidden = data_source._make_exact_labeled_split(
        TRAIN_N, TRAIN_DATA_SEED
    )
    validation_positions, _, validation_hidden = data_source._make_exact_labeled_split(
        VALIDATION_N, VALIDATION_DATA_SEED
    )
    test_positions, _, test_hidden = data_source._make_exact_labeled_split(
        TEST_N, TEST_DATA_SEED
    )
    del train_hidden, validation_hidden

    torch.manual_seed(MODEL_SEED)
    model = Direct2DSelector().to(dtype=DTYPE)
    batch_generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    summary_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    cumulative_updates = 0
    prior_coverage: float | None = None
    stop_reason = "completed_all_horizons"

    for horizon, budget in zip(HORIZONS, UPDATE_BUDGETS):
        _, history, cumulative_updates = _train_stage(
            model,
            train_positions,
            validation_positions,
            horizon,
            budget,
            batch_generator,
            cumulative_updates,
        )
        test_rollout = _evaluate(model, test_positions, horizon)
        # Hidden parameters enter only after the search is complete and solely
        # determine diagnostic coverage/error and the requested stop gate.
        reference = _posthoc_reference(test_hidden)
        stage_query_rows = _per_query_rows(horizon, test_rollout, reference)
        del reference
        query_rows.extend(stage_query_rows)
        final_metrics = dict(stage_query_rows[-1])
        final_metrics.update(
            {
                "row_type": "horizon_summary",
                **history,
            }
        )
        summary_rows.append(final_metrics)
        coverage = float(final_metrics["coverage"])
        collapsed = coverage < MINIMUM_COVERAGE_TO_CONTINUE or (
            prior_coverage is not None
            and coverage < prior_coverage - MAXIMUM_COVERAGE_DROP
        )
        if collapsed:
            stop_reason = f"coverage_collapse_at_{horizon}"
            summary_rows.append(
                {
                    "row_type": "stop",
                    "trained_horizon": horizon,
                    "coverage": coverage,
                    "prior_coverage": prior_coverage if prior_coverage is not None else "",
                    "stop_reason": stop_reason,
                    "runtime_seconds_total": time.perf_counter() - start,
                }
            )
            print(
                {
                    "status": stop_reason,
                    "coverage": coverage,
                    "candidate_area_median": final_metrics["median_candidate_area"],
                },
                flush=True,
            )
            break
        prior_coverage = coverage

    if not summary_rows or summary_rows[-1].get("row_type") != "stop":
        summary_rows.append(
            {
                "row_type": "stop",
                "trained_horizon": summary_rows[-1]["trained_horizon"],
                "coverage": summary_rows[-1]["coverage"],
                "stop_reason": stop_reason,
                "runtime_seconds_total": time.perf_counter() - start,
            }
        )
    _write_csv(OUTPUT_DIR / "direct2d_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_query_results.csv", query_rows)
    _plot(query_rows)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "stop_reason": stop_reason,
            "completed_horizons": sorted(
                {int(row["trained_horizon"]) for row in query_rows}
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
