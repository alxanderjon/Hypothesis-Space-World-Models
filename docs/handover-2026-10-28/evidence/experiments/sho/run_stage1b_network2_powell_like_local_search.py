"""Learned explore/exploit local search using frozen Network 1 intervals."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import statistics
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

import experiments.sho.run_stage1b_learned_black_box_search as reference_search
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as data_source
from experiments.sho.run_stage1b_continuous_c_search_comparison import _losses
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

FROZEN_ROOT = Path(
    "outputs/sho/stage1B/frozen_network1_region_search/selected"
)
OUTPUT_DIR = Path(
    os.environ.get(
        "NETWORK2_LOCAL_SEARCH_OUTPUT_DIR",
        "outputs/sho/stage1B/network2_powell_like_local_search",
    )
)
CHECKPOINT_PATH = OUTPUT_DIR / "best_checkpoint.pt"
PREVIOUS_LOCAL_RESULTS = Path(
    "outputs/sho/stage1B/learned_local_refinement/per_query_results.csv"
)
PREVIOUS_SOFT_RESULTS = Path(
    "outputs/sho/stage1B/soft_continuous_local_search/per_trajectory_search.csv"
)
INITIAL_EVALUATIONS = 4
EMERGENCY_CAP = int(os.environ.get("NETWORK2_EVALUATION_CAP", "250"))
MAX_LEARNED_QUERIES = EMERGENCY_CAP - INITIAL_EVALUATIONS
TRAINING_HORIZONS = tuple(
    int(value)
    for value in os.environ.get(
        "NETWORK2_TRAINING_HORIZONS", "20,60,120,200"
    ).split(",")
)
DEFAULT_STAGE_UPDATES = (300, 250, 150, 100)
STAGE_UPDATES = tuple(
    int(value)
    for value in os.environ.get(
        "NETWORK2_STAGE_UPDATES",
        ",".join(str(value) for value in DEFAULT_STAGE_UPDATES),
    ).split(",")
)
if len(TRAINING_HORIZONS) != len(STAGE_UPDATES):
    raise ValueError("Each training horizon needs one update budget")
BATCH_SIZE = int(os.environ.get("NETWORK2_BATCH_SIZE", "32"))
LEARNING_RATE = 1e-4
EVAL_INTERVAL = int(os.environ.get("NETWORK2_EVAL_INTERVAL", "50"))
PLATEAU_PATIENCE = int(os.environ.get("NETWORK2_PATIENCE", "150"))
DISCOUNT = 0.97
NOVELTY_WEIGHT = 0.01
ENTROPY_WEIGHT = 0.002
EVALUATION_COST = 0.001
SUCCESS_BONUS = 5.0
LOG_EPSILON = 1e-30
SUCCESS_LOSS = 1e-20
LOSS_THRESHOLDS = (1e-4, 1e-8, 1e-12, 1e-16, 1e-20)
FIXED_LOSS_BANDS = (
    ("above_1e_4", 1e-4, float("inf")),
    ("1e_8_to_1e_4", 1e-8, 1e-4),
    ("1e_12_to_1e_8", 1e-12, 1e-8),
    ("1e_16_to_1e_12", 1e-16, 1e-12),
    ("1e_20_to_1e_16", 1e-20, 1e-16),
)
MODEL_SEED = SEED + 2_040_000
EVALUATION_SEED = SEED + 2_050_000
DIFFUSE_SCALE_THRESHOLD = 1e-6
REPLAY_TARGETS = {
    "success_count": 512,
    "median_success_evaluations": 59.0,
    "worst_success_evaluations": 105.0,
    "median_final_prefix_loss": 3.1754036857518787e-21,
    "worst_final_prefix_loss": 9.95136669625517e-21,
    "median_final_c_error_posthoc": 2.404521026733164e-12,
    "worst_final_c_error_posthoc": 7.906408860947067e-11,
}


class ExploreExploitController(nn.Module):
    """DeepSets policy with explicit explore/exploit query branches."""

    def __init__(self) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 2 + 14, 64),
            nn.SiLU(),
            nn.Linear(64, 6),
        )
        with torch.no_grad():
            final = self.head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Unexpected controller head")
            final.bias.zero_()
            final.bias[0] = math.log(0.35 / 0.65)  # initial explore probability
            final.bias[2] = -4.0  # initially broad exploitation
            final.bias[5] = 0.0

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.element(history_features)
        masked = embedded * history_mask[:, :, None].to(DTYPE)
        count = history_mask.sum(dim=1).clamp_min(1).to(DTYPE)
        mean_context = masked.sum(dim=1) / count[:, None]
        max_context = torch.where(
            history_mask[:, :, None],
            embedded,
            torch.full_like(embedded, -float("inf")),
        ).max(dim=1).values
        return self.head(
            torch.cat((mean_context, max_context, global_features), dim=1)
        )


@dataclass
class HistoryStatistics:
    features: torch.Tensor
    global_features: torch.Tensor
    largest_gap_midpoint: torch.Tensor
    largest_gap_width: torch.Tensor
    second_gap_midpoint: torch.Tensor
    second_gap_width: torch.Tensor
    best_left_gap: torch.Tensor
    best_right_gap: torch.Tensor


@dataclass
class SearchRollout:
    best_c: torch.Tensor
    best_loss: torch.Tensor
    best_c_history: torch.Tensor
    best_loss_history: torch.Tensor
    query_c: torch.Tensor
    query_mask: torch.Tensor
    explore_action: torch.Tensor
    proposal_scale: torch.Tensor
    probe_distance_from_best: torch.Tensor
    novelty_distance: torch.Tensor
    prior_best_loss: torch.Tensor
    improved_best: torch.Tensor
    first_threshold_evaluation: dict[float, torch.Tensor]
    success: torch.Tensor
    evaluations_consumed: torch.Tensor
    log_probabilities: list[torch.Tensor]
    rewards: list[torch.Tensor]
    entropies: list[torch.Tensor]
    entropy_factors: list[torch.Tensor]
    active_masks: list[torch.Tensor]


def _load_frozen_config() -> dict[str, Any]:
    required = [
        FROZEN_ROOT / "config.json",
        FROZEN_ROOT / "best_checkpoint.pt",
        FROZEN_ROOT / "train_safe_intervals.csv",
        FROZEN_ROOT / "validation_safe_intervals.csv",
        FROZEN_ROOT / "test_safe_intervals.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Network 1 artifacts: {missing}")
    config = json.loads((FROZEN_ROOT / "config.json").read_text())
    checkpoint = torch.load(
        FROZEN_ROOT / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        config.get("selected_or_stage_horizon") != 20
        or checkpoint.get("horizon") != 20
        or not config.get("frozen_network1", False)
    ):
        raise RuntimeError("Selected Network 1 artifact is not the frozen stage 20")
    return config


def _checkpoint_payload(
    state: dict[str, torch.Tensor],
    horizon: int,
    best_update: int,
    best_validation_score: float,
    training_history: list[dict[str, Any]],
    *,
    status: str,
    replay_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_class": "ExploreExploitController",
        "state_dict": {
            key: value.detach().cpu() for key, value in state.items()
        },
        "training_horizon": horizon,
        "best_update": best_update,
        "best_validation_score": best_validation_score,
        "training_history": [dict(row) for row in training_history],
        "status": status,
        "frozen_network1_root": str(FROZEN_ROOT),
        "frozen_network1_horizon": 20,
        "model_seed": MODEL_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "training_horizons": list(TRAINING_HORIZONS),
        "stage_updates": list(STAGE_UPDATES),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "evaluation_interval": EVAL_INTERVAL,
        "plateau_patience": PLATEAU_PATIENCE,
        "discount": DISCOUNT,
        "novelty_weight": NOVELTY_WEIGHT,
        "entropy_weight": ENTROPY_WEIGHT,
        "evaluation_cost": EVALUATION_COST,
        "success_bonus": SUCCESS_BONUS,
        "emergency_evaluation_cap": EMERGENCY_CAP,
        "replay_verification": replay_verification,
    }


def _save_checkpoint(
    state: dict[str, torch.Tensor],
    horizon: int,
    best_update: int,
    best_validation_score: float,
    training_history: list[dict[str, Any]],
    *,
    status: str,
    replay_verification: dict[str, Any] | None = None,
) -> None:
    torch.save(
        _checkpoint_payload(
            state,
            horizon,
            best_update,
            best_validation_score,
            training_history,
            status=status,
            replay_verification=replay_verification,
        ),
        CHECKPOINT_PATH,
    )


def _load_intervals(
    path: Path, expected_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise RuntimeError(f"Unexpected interval count in {path}: {len(rows)}")
    indices = [int(row["trajectory_index"]) for row in rows]
    if indices != list(range(expected_count)):
        raise RuntimeError(f"Non-canonical trajectory order in {path}")
    lower = torch.tensor(
        [float(row["safe_interval_lower"]) for row in rows], dtype=DTYPE
    )
    upper = torch.tensor(
        [float(row["safe_interval_upper"]) for row in rows], dtype=DTYPE
    )
    best_c = torch.tensor(
        [float(row["best_observed_c"]) for row in rows], dtype=DTYPE
    )
    best_loss = torch.tensor(
        [float(row["best_observed_prefix_loss"]) for row in rows], dtype=DTYPE
    )
    if not bool(torch.all((lower <= best_c) & (best_c <= upper))):
        raise RuntimeError(f"Cached best c falls outside an interval in {path}")
    return lower, upper, best_c, best_loss


@torch.no_grad()
def _initial_history(
    positions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    cached_best_c: torch.Tensor,
    cached_best_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = upper - lower
    cached_best_u = ((cached_best_c - lower) / width).clamp(0.0, 1.0)
    sampled_u = torch.stack(
        (
            torch.zeros_like(lower),
            torch.full_like(lower, 0.5),
            torch.ones_like(lower),
            cached_best_u,
        ),
        dim=1,
    )
    losses = []
    for index in range(INITIAL_EVALUATIONS):
        c_value = lower + sampled_u[:, index] * width
        losses.append(_losses(c_value, positions))
    sampled_loss = torch.stack(losses, dim=1)
    cached_difference = torch.abs(sampled_loss[:, 3] - cached_best_loss)
    tolerance = 1e-10 * torch.maximum(
        torch.ones_like(cached_best_loss), torch.abs(cached_best_loss)
    )
    if not bool(torch.all(cached_difference <= tolerance)):
        raise RuntimeError("Cached Network 1 best losses failed prefix-loss audit")
    return sampled_u, sampled_loss


def _history_statistics(
    sampled_u: torch.Tensor,
    sampled_loss: torch.Tensor,
    history_mask: torch.Tensor,
    best_u: torch.Tensor,
    best_loss: torch.Tensor,
    initial_best_loss: torch.Tensor,
    interval_width: torch.Tensor,
) -> HistoryStatistics:
    batch_size, count = sampled_u.shape
    masked_u = torch.where(history_mask, sampled_u, torch.full_like(sampled_u, 2.0))
    sorted_u, order = torch.sort(masked_u, dim=1)
    sorted_mask = history_mask.gather(1, order)
    pair_valid = sorted_mask[:, :-1] & sorted_mask[:, 1:]
    gaps = torch.where(
        pair_valid,
        sorted_u[:, 1:] - sorted_u[:, :-1],
        torch.full_like(sorted_u[:, 1:], -1.0),
    )
    top_count = min(2, gaps.shape[1])
    top_widths, top_indices = torch.topk(gaps, k=top_count, dim=1)
    top_midpoints = 0.5 * (
        sorted_u.gather(1, top_indices)
        + sorted_u.gather(1, top_indices + 1)
    )
    largest_width = top_widths[:, 0].clamp_min(0.0)
    largest_midpoint = top_midpoints[:, 0].clamp(0.0, 1.0)
    if top_count == 2:
        second_width = top_widths[:, 1].clamp_min(0.0)
        second_midpoint = top_midpoints[:, 1].clamp(0.0, 1.0)
    else:
        second_width = largest_width
        second_midpoint = largest_midpoint

    left_gap_sorted = torch.zeros_like(sorted_u)
    right_gap_sorted = torch.zeros_like(sorted_u)
    left_gap_sorted[:, 1:] = torch.where(
        pair_valid, sorted_u[:, 1:] - sorted_u[:, :-1], 0.0
    )
    right_gap_sorted[:, :-1] = torch.where(
        pair_valid, sorted_u[:, 1:] - sorted_u[:, :-1], 0.0
    )
    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(
        1,
        order,
        torch.arange(count, dtype=torch.long)[None, :].expand(batch_size, -1),
    )
    left_gaps = left_gap_sorted.gather(1, inverse_order)
    right_gaps = right_gap_sorted.gather(1, inverse_order)

    log_loss = torch.log(sampled_loss + LOG_EPSILON)
    log_best = torch.log(best_loss + LOG_EPSILON)
    distance_best = torch.abs(sampled_u - best_u[:, None])
    loss_difference = torch.clamp(
        log_loss - log_best[:, None], min=0.0, max=60.0
    )
    features = torch.stack(
        (
            sampled_u,
            log_loss / 20.0,
            distance_best,
            loss_difference / 20.0,
            left_gaps,
            right_gaps,
        ),
        dim=2,
    )
    features = torch.where(history_mask[:, :, None], features, 0.0)

    left_mask = history_mask & (sampled_u < best_u[:, None] - 1e-15)
    right_mask = history_mask & (sampled_u > best_u[:, None] + 1e-15)
    left_values = torch.where(left_mask, sampled_u, torch.full_like(sampled_u, -1.0))
    right_values = torch.where(right_mask, sampled_u, torch.full_like(sampled_u, 2.0))
    left_u, left_index = left_values.max(dim=1)
    right_u, right_index = right_values.min(dim=1)
    has_left = left_mask.any(dim=1)
    has_right = right_mask.any(dim=1)
    best_left_gap = torch.where(has_left, best_u - left_u, best_u)
    best_right_gap = torch.where(has_right, right_u - best_u, 1.0 - best_u)
    left_loss = log_loss.gather(1, left_index[:, None]).squeeze(1)
    right_loss = log_loss.gather(1, right_index[:, None]).squeeze(1)
    left_loss_difference = torch.where(
        has_left, torch.clamp(left_loss - log_best, 0.0, 60.0), 0.0
    )
    right_loss_difference = torch.where(
        has_right, torch.clamp(right_loss - log_best, 0.0, 60.0), 0.0
    )
    valid_count = history_mask.sum(dim=1).to(DTYPE)
    masked_values = sampled_u * history_mask.to(DTYPE)
    mean_u = masked_values.sum(dim=1) / valid_count
    variance_u = torch.clamp(
        (sampled_u.square() * history_mask.to(DTYPE)).sum(dim=1) / valid_count
        - mean_u.square(),
        min=0.0,
    )
    high_loss_factor = torch.sigmoid(
        (log_best - math.log(1e-8)) / 2.0
    )
    global_features = torch.stack(
        (
            best_u,
            log_best / 20.0,
            torch.log(interval_width + LOG_EPSILON) / 10.0,
            largest_midpoint,
            largest_width,
            second_midpoint,
            second_width,
            best_left_gap,
            best_right_gap,
            left_loss_difference / 20.0,
            right_loss_difference / 20.0,
            torch.sqrt(variance_u + 1e-30),
            torch.clamp(valid_count / float(EMERGENCY_CAP), max=1.0),
            high_loss_factor,
        ),
        dim=1,
    )
    return HistoryStatistics(
        features=features,
        global_features=global_features,
        largest_gap_midpoint=largest_midpoint,
        largest_gap_width=largest_width,
        second_gap_midpoint=second_midpoint,
        second_gap_width=second_width,
        best_left_gap=best_left_gap,
        best_right_gap=best_right_gap,
    )


def _sample_query(
    raw: torch.Tensor,
    statistics_: HistoryStatistics,
    best_u: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    explore_distribution = torch.distributions.Bernoulli(logits=raw[:, 0])
    explore_action = explore_distribution.sample()
    local_radius = torch.maximum(
        statistics_.best_left_gap, statistics_.best_right_gap
    ).clamp_min(1e-12)
    exploit_center = (
        best_u + torch.tanh(raw[:, 1]) * local_radius
    ).clamp(0.0, 1.0)
    exploit_log_scale_factor = -1.5 - 30.0 * torch.sigmoid(raw[:, 2])
    exploit_scale = (
        local_radius * torch.exp(exploit_log_scale_factor)
    ).clamp_min(1e-15)

    gap_blend = torch.sigmoid(raw[:, 3])
    blended_midpoint = (
        gap_blend * statistics_.largest_gap_midpoint
        + (1.0 - gap_blend) * statistics_.second_gap_midpoint
    )
    blended_width = (
        gap_blend * statistics_.largest_gap_width
        + (1.0 - gap_blend) * statistics_.second_gap_width
    ).clamp_min(1e-12)
    explore_center = (
        blended_midpoint
        + 0.5 * torch.tanh(raw[:, 4]) * blended_width
    ).clamp(0.0, 1.0)
    explore_scale = blended_width * (
        0.05 + 0.45 * torch.sigmoid(raw[:, 5])
    )
    center = torch.where(explore_action.bool(), explore_center, exploit_center)
    scale = torch.where(explore_action.bool(), explore_scale, exploit_scale)
    normal = torch.distributions.Normal(center, scale)
    raw_query = normal.sample()
    query_u = raw_query.clamp(0.0, 1.0)
    log_probability = (
        explore_distribution.log_prob(explore_action) + normal.log_prob(raw_query)
    )
    entropy = explore_distribution.entropy() + normal.entropy()
    return query_u, explore_action.bool(), scale, log_probability, entropy


def _rollout(
    model: ExploreExploitController,
    positions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    initial_u: torch.Tensor,
    initial_loss: torch.Tensor,
    learned_queries: int,
) -> SearchRollout:
    batch_size = positions.shape[0]
    width = upper - lower
    sampled_u = initial_u
    sampled_loss = initial_loss
    history_mask = torch.ones_like(sampled_u, dtype=torch.bool)
    best_indices = torch.argmin(sampled_loss, dim=1)
    best_loss = sampled_loss.gather(1, best_indices[:, None]).squeeze(1)
    best_u = sampled_u.gather(1, best_indices[:, None]).squeeze(1)
    initial_best_loss = best_loss.clone()
    active = best_loss >= SUCCESS_LOSS
    evaluations_consumed = torch.full(
        (batch_size,), EMERGENCY_CAP, dtype=torch.long
    )
    evaluations_consumed = torch.where(
        active,
        evaluations_consumed,
        torch.full_like(evaluations_consumed, INITIAL_EVALUATIONS),
    )
    first_threshold_evaluation = {
        threshold: torch.where(
            best_loss < threshold,
            torch.full((batch_size,), INITIAL_EVALUATIONS, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long),
        )
        for threshold in LOSS_THRESHOLDS
    }

    best_loss_history = [best_loss]
    best_c_history = [lower + best_u * width]
    query_c_history: list[torch.Tensor] = []
    query_mask_history: list[torch.Tensor] = []
    explore_history: list[torch.Tensor] = []
    scale_history: list[torch.Tensor] = []
    distance_history: list[torch.Tensor] = []
    novelty_history: list[torch.Tensor] = []
    prior_best_history: list[torch.Tensor] = []
    improved_history: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    entropy_factors: list[torch.Tensor] = []

    for query_index in range(learned_queries):
        statistics_ = _history_statistics(
            sampled_u,
            sampled_loss,
            history_mask,
            best_u,
            best_loss,
            initial_best_loss,
            width,
        )
        raw = model(
            statistics_.features, history_mask, statistics_.global_features
        )
        query_u, explore_action, proposal_scale_u, log_probability, entropy = (
            _sample_query(raw, statistics_, best_u)
        )
        query_mask = active.clone()
        prior_best_u = best_u.clone()
        query_c = lower + query_u * width
        with torch.no_grad():
            query_loss = _losses(query_c.detach(), positions)
            prior_best = best_loss.clone()
            improved = query_mask & (query_loss < best_loss)
            next_best = torch.where(improved, query_loss, best_loss)
            log_improvement = torch.clamp(
                torch.log(prior_best + LOG_EPSILON)
                - torch.log(next_best + LOG_EPSILON),
                min=0.0,
                max=40.0,
            )
            nearest_distance = torch.where(
                history_mask,
                torch.abs(query_u.detach()[:, None] - sampled_u),
                torch.full_like(sampled_u, float("inf")),
            ).min(dim=1).values
            high_loss_factor = torch.sigmoid(
                (torch.log(prior_best + LOG_EPSILON) - math.log(1e-8)) / 2.0
            )
            concentration_cost = (
                (1.0 - high_loss_factor)
                * torch.abs(query_u.detach() - best_u)
                * 0.01
            )
            newly_successful = query_mask & (next_best < SUCCESS_LOSS)
            reward = (
                log_improvement
                + NOVELTY_WEIGHT * high_loss_factor * nearest_distance
                + SUCCESS_BONUS * newly_successful.to(DTYPE)
                - concentration_cost
                - EVALUATION_COST
            )
            best_loss = next_best
            best_u = torch.where(improved, query_u.detach(), best_u)
            total_evaluations = INITIAL_EVALUATIONS + query_index + 1
            for threshold, first_hit in first_threshold_evaluation.items():
                newly_reached = (
                    query_mask & (first_hit == 0) & (best_loss < threshold)
                )
                first_threshold_evaluation[threshold] = torch.where(
                    newly_reached,
                    torch.full_like(first_hit, total_evaluations),
                    first_hit,
                )
            evaluations_consumed = torch.where(
                newly_successful,
                torch.full_like(evaluations_consumed, total_evaluations),
                evaluations_consumed,
            )
            active = query_mask & ~newly_successful

        sampled_u = torch.cat((sampled_u, query_u.detach()[:, None]), dim=1)
        sampled_loss = torch.cat(
            (sampled_loss, query_loss.detach()[:, None]), dim=1
        )
        history_mask = torch.cat((history_mask, query_mask[:, None]), dim=1)
        query_c_history.append(query_c.detach())
        query_mask_history.append(query_mask)
        explore_history.append(explore_action.detach())
        scale_history.append((proposal_scale_u.detach() * width))
        distance_history.append(
            (torch.abs(query_u.detach() - prior_best_u) * width).detach()
        )
        novelty_history.append(nearest_distance.detach())
        prior_best_history.append(prior_best.detach())
        improved_history.append(improved.detach())
        log_probabilities.append(log_probability)
        rewards.append(torch.where(query_mask, reward, torch.zeros_like(reward)))
        entropies.append(entropy)
        entropy_factors.append(high_loss_factor)
        best_loss_history.append(best_loss.detach())
        best_c_history.append((lower + best_u * width).detach())

    return SearchRollout(
        best_c=(lower + best_u * width).detach(),
        best_loss=best_loss.detach(),
        best_c_history=torch.stack(best_c_history, dim=1),
        best_loss_history=torch.stack(best_loss_history, dim=1),
        query_c=torch.stack(query_c_history, dim=1),
        query_mask=torch.stack(query_mask_history, dim=1),
        explore_action=torch.stack(explore_history, dim=1),
        proposal_scale=torch.stack(scale_history, dim=1),
        probe_distance_from_best=torch.stack(distance_history, dim=1),
        novelty_distance=torch.stack(novelty_history, dim=1),
        prior_best_loss=torch.stack(prior_best_history, dim=1),
        improved_best=torch.stack(improved_history, dim=1),
        first_threshold_evaluation=first_threshold_evaluation,
        success=best_loss < SUCCESS_LOSS,
        evaluations_consumed=evaluations_consumed,
        log_probabilities=log_probabilities,
        rewards=rewards,
        entropies=entropies,
        entropy_factors=entropy_factors,
        active_masks=query_mask_history,
    )


def _policy_loss(rollout: SearchRollout) -> torch.Tensor:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rollout.rewards[0])
    for reward in reversed(rollout.rewards):
        running = reward + DISCOUNT * running
        returns.append(running)
    returns.reverse()
    terms: list[torch.Tensor] = []
    for return_value, log_probability, entropy, factor, mask in zip(
        returns,
        rollout.log_probabilities,
        rollout.entropies,
        rollout.entropy_factors,
        rollout.active_masks,
    ):
        if not bool(torch.any(mask)):
            continue
        values = return_value[mask]
        advantage = values - values.mean()
        advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-6)
        reinforce = -(advantage.detach() * log_probability[mask]).mean()
        entropy_term = (factor[mask].detach() * entropy[mask]).mean()
        terms.append(reinforce - ENTROPY_WEIGHT * entropy_term)
    if not terms:
        raise RuntimeError("No active policy-gradient terms")
    return torch.stack(terms).mean()


@torch.no_grad()
def _evaluate(
    model: ExploreExploitController,
    positions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    initial_u: torch.Tensor,
    initial_loss: torch.Tensor,
    learned_queries: int,
    seed: int,
) -> SearchRollout:
    model.eval()
    parts: list[SearchRollout] = []
    saved_state = torch.random.get_rng_state()
    try:
        for batch_number, start in enumerate(range(0, positions.shape[0], BATCH_SIZE)):
            stop = min(start + BATCH_SIZE, positions.shape[0])
            torch.manual_seed(seed + batch_number)
            parts.append(
                _rollout(
                    model,
                    positions[start:stop],
                    lower[start:stop],
                    upper[start:stop],
                    initial_u[start:stop],
                    initial_loss[start:stop],
                    learned_queries,
                )
            )
    finally:
        torch.random.set_rng_state(saved_state)
    tensor_fields = [
        "best_c",
        "best_loss",
        "best_c_history",
        "best_loss_history",
        "query_c",
        "query_mask",
        "explore_action",
        "proposal_scale",
        "probe_distance_from_best",
        "novelty_distance",
        "prior_best_loss",
        "improved_best",
        "success",
        "evaluations_consumed",
    ]
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in tensor_fields
    }
    thresholds = {
        threshold: torch.cat(
            [part.first_threshold_evaluation[threshold] for part in parts]
        )
        for threshold in LOSS_THRESHOLDS
    }
    return SearchRollout(
        **combined,
        first_threshold_evaluation=thresholds,
        log_probabilities=[],
        rewards=[],
        entropies=[],
        entropy_factors=[],
        active_masks=[],
    )


def _validation_score(rollout: SearchRollout) -> float:
    gate = float(rollout.success.to(DTYPE).mean().item())
    mean_log = float(
        torch.log(rollout.best_loss + LOG_EPSILON).mean().item()
    )
    return -100.0 * gate + mean_log


def _verify_replay(
    rollout: SearchRollout, c_star: torch.Tensor
) -> dict[str, Any]:
    success = rollout.success
    success_evaluations = rollout.evaluations_consumed[success].to(DTYPE)
    errors = torch.abs(rollout.best_c - c_star)
    actual = {
        "success_count": int(success.sum().item()),
        "median_success_evaluations": float(
            torch.median(success_evaluations).item()
        ),
        "worst_success_evaluations": float(success_evaluations.max().item()),
        "median_final_prefix_loss": float(torch.median(rollout.best_loss).item()),
        "worst_final_prefix_loss": float(rollout.best_loss.max().item()),
        "median_final_c_error_posthoc": float(torch.median(errors).item()),
        "worst_final_c_error_posthoc": float(errors.max().item()),
    }
    verification: dict[str, Any] = {}
    for name, target in REPLAY_TARGETS.items():
        value = actual[name]
        if name == "success_count":
            tolerance = 0.0
        elif "evaluations" in name:
            tolerance = 1.0
        else:
            tolerance = max(abs(float(target)) * 0.05, 1e-30)
        difference = abs(float(value) - float(target))
        verification[name] = {
            "target": target,
            "actual": value,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": difference <= tolerance,
        }
    verification["all_passed"] = all(
        bool(check["passed"])
        for check in verification.values()
        if isinstance(check, dict)
    )
    return verification


def _train(
    train: tuple[torch.Tensor, ...],
    validation: tuple[torch.Tensor, ...],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    train_positions, train_lower, train_upper, train_u, train_loss = train
    val_positions, val_lower, val_upper, val_u, val_loss = validation
    torch.manual_seed(MODEL_SEED)
    model = ExploreExploitController().to(dtype=DTYPE)
    batch_generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    history: list[dict[str, Any]] = []
    final_state = copy.deepcopy(model.state_dict())
    cumulative_updates = 0

    for horizon, budget in zip(TRAINING_HORIZONS, STAGE_UPDATES):
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        validation_rollout = _evaluate(
            model,
            val_positions,
            val_lower,
            val_upper,
            val_u,
            val_loss,
            horizon,
            EVALUATION_SEED + horizon,
        )
        best_score = _validation_score(validation_rollout)
        best_state = copy.deepcopy(model.state_dict())
        best_update = 0
        _save_checkpoint(
            best_state,
            horizon,
            best_update,
            best_score,
            history,
            status="training_stage_best",
        )
        stale = 0
        stop_reason = "maximum_updates"
        stage_start = time.perf_counter()
        updates_run = 0
        print(
            f"network2 horizon={horizon} update=0000 "
            f"validation gate={float(validation_rollout.success.to(DTYPE).mean().item()):.6f} "
            f"mean log best="
            f"{float(torch.log(validation_rollout.best_loss + LOG_EPSILON).mean().item()):.6e}",
            flush=True,
        )
        for update in range(1, budget + 1):
            model.train()
            indices = torch.randint(
                0,
                train_positions.shape[0],
                (BATCH_SIZE,),
                generator=batch_generator,
            )
            rollout = _rollout(
                model,
                train_positions[indices],
                train_lower[indices],
                train_upper[indices],
                train_u[indices],
                train_loss[indices],
                horizon,
            )
            loss = _policy_loss(rollout)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            updates_run = update
            cumulative_updates += 1
            if update % EVAL_INTERVAL == 0 or update == budget:
                validation_rollout = _evaluate(
                    model,
                    val_positions,
                    val_lower,
                    val_upper,
                    val_u,
                    val_loss,
                    horizon,
                    EVALUATION_SEED + horizon,
                )
                score = _validation_score(validation_rollout)
                if score < best_score - 1e-8:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    best_update = update
                    stale = 0
                    _save_checkpoint(
                        best_state,
                        horizon,
                        best_update,
                        best_score,
                        history,
                        status="training_stage_best",
                    )
                else:
                    stale += EVAL_INTERVAL
                print(
                    f"network2 horizon={horizon} update={update:04d} "
                    f"policy loss={float(loss.detach().item()):.6e} "
                    f"validation gate="
                    f"{float(validation_rollout.success.to(DTYPE).mean().item()):.6f} "
                    f"mean log best="
                    f"{float(torch.log(validation_rollout.best_loss + LOG_EPSILON).mean().item()):.6e} "
                    f"best_update={best_update} stale={stale}",
                    flush=True,
                )
                if stale >= PLATEAU_PATIENCE:
                    stop_reason = "validation_plateau"
                    break
        model.load_state_dict(best_state)
        final_state = copy.deepcopy(best_state)
        history.append(
            {
                "row_type": "training_stage",
                "training_horizon": horizon,
                "updates_run": updates_run,
                "best_update": best_update,
                "best_validation_score": best_score,
                "stop_reason": stop_reason,
                "runtime_seconds": time.perf_counter() - stage_start,
                "cumulative_updates": cumulative_updates,
            }
        )
        _save_checkpoint(
            best_state,
            horizon,
            best_update,
            best_score,
            history,
            status="training_stage_complete",
        )
    return final_state, history


def _previous_difficult_masks() -> dict[str, torch.Tensor]:
    by_index: dict[int, dict[int, float]] = {}
    with PREVIOUS_LOCAL_RESULTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            query = int(row["adaptive_query_number"])
            if query in (1, 15):
                by_index.setdefault(int(row["trajectory_index"]), {})[query] = float(
                    row["best_observed_prefix_loss"]
                )
    local_stalled = torch.tensor(
        [
            abs(by_index[index][1] - by_index[index][15]) <= 1e-15
            for index in range(len(by_index))
        ],
        dtype=torch.bool,
    )
    with PREVIOUS_SOFT_RESULTS.open(newline="") as handle:
        soft_rows = list(csv.DictReader(handle))
    soft_rows.sort(key=lambda row: int(row["trajectory_index"]))
    soft_high_loss = torch.tensor(
        [float(row["best_observed_prefix_loss"]) >= 1e-4 for row in soft_rows],
        dtype=torch.bool,
    )
    return {
        "previous_local_stalled": local_stalled,
        "previous_soft_high_loss": soft_high_loss,
        "previous_network2_difficult_union": local_stalled | soft_high_loss,
    }


def _trajectory_rows(
    rollout: SearchRollout,
    c_star: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    difficult_masks: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    strict_one_side = torch.zeros_like(rollout.success)
    dominant_one_side = torch.zeros_like(rollout.success)
    too_diffuse = torch.zeros_like(rollout.success)
    for trajectory in range(c_star.shape[0]):
        valid = rollout.query_mask[trajectory]
        queries = rollout.query_c[trajectory, valid]
        reference = c_star[trajectory]
        left = int((queries < reference).sum().item())
        right = int((queries > reference).sum().item())
        side_total = max(left + right, 1)
        strict_one_side[trajectory] = left == 0 or right == 0
        dominant_one_side[trajectory] = max(left, right) / side_total >= 0.9
        final_scale = rollout.proposal_scale[trajectory, valid][-1]
        too_diffuse[trajectory] = final_scale > DIFFUSE_SCALE_THRESHOLD
        row: dict[str, Any] = {
            "trajectory_index": trajectory,
            **{
                name: bool(mask[trajectory])
                for name, mask in difficult_masks.items()
            },
            "network1_safe_lower": float(lower[trajectory].item()),
            "network1_safe_upper": float(upper[trajectory].item()),
            "network1_safe_width": float((upper[trajectory] - lower[trajectory]).item()),
            "reference_c_star_posthoc": float(reference.item()),
            "success_1e_20": bool(rollout.success[trajectory]),
            "evaluations_consumed": int(rollout.evaluations_consumed[trajectory].item()),
            "hit_emergency_cap": not bool(rollout.success[trajectory]),
            "best_observed_c": float(rollout.best_c[trajectory].item()),
            "best_observed_prefix_loss": float(rollout.best_loss[trajectory].item()),
            "absolute_best_c_error_posthoc": float(
                torch.abs(rollout.best_c[trajectory] - reference).item()
            ),
            "learned_queries_evaluated": int(valid.sum().item()),
            "exploration_fraction": float(
                rollout.explore_action[trajectory, valid].to(DTYPE).mean().item()
            ),
            "mean_proposal_scale": float(
                rollout.proposal_scale[trajectory, valid].mean().item()
            ),
            "final_proposal_scale": float(final_scale.item()),
            "mean_probe_distance_from_best": float(
                rollout.probe_distance_from_best[trajectory, valid].mean().item()
            ),
            "learned_queries_left_of_optimum_posthoc": left,
            "learned_queries_right_of_optimum_posthoc": right,
            "strictly_one_sided_failure_pattern": bool(strict_one_side[trajectory]),
            "dominant_one_sided_failure_pattern": bool(dominant_one_side[trajectory]),
            "failure_remained_too_diffuse": bool(too_diffuse[trajectory]),
        }
        for threshold in LOSS_THRESHOLDS:
            exponent = int(round(-math.log10(threshold)))
            first = int(
                rollout.first_threshold_evaluation[threshold][trajectory].item()
            )
            row[f"reached_1e_{exponent}"] = first > 0
            row[f"evaluations_to_1e_{exponent}"] = first if first > 0 else ""
        rows.append(row)
    return rows, {
        "strict_one_side": strict_one_side,
        "dominant_one_side": dominant_one_side,
        "too_diffuse": too_diffuse,
    }


def _group_summary(
    name: str,
    mask: torch.Tensor,
    rollout: SearchRollout,
    c_star: torch.Tensor,
    failure_flags: dict[str, torch.Tensor],
) -> dict[str, Any]:
    success = rollout.success & mask
    failure = (~rollout.success) & mask
    errors = torch.abs(rollout.best_c - c_star)
    success_evaluations = rollout.evaluations_consumed[success].to(DTYPE)
    row: dict[str, Any] = {
        "row_type": "group_summary",
        "group": name,
        "trajectories": int(mask.sum().item()),
        "fraction_reaching_1e_20": float(rollout.success[mask].to(DTYPE).mean().item()),
        "successful_trajectories": int(success.sum().item()),
        "emergency_cap_failures": int(failure.sum().item()),
        "median_success_evaluations": float(torch.median(success_evaluations).item())
        if bool(torch.any(success))
        else "",
        "mean_success_evaluations": float(success_evaluations.mean().item())
        if bool(torch.any(success))
        else "",
        "worst_success_evaluations": float(success_evaluations.max().item())
        if bool(torch.any(success))
        else "",
        "median_final_prefix_loss": float(torch.median(rollout.best_loss[mask]).item()),
        "worst_final_prefix_loss": float(rollout.best_loss[mask].max().item()),
        "median_final_c_error_posthoc": float(torch.median(errors[mask]).item()),
        "worst_final_c_error_posthoc": float(errors[mask].max().item()),
        "strict_one_side_fraction_among_failures": float(
            failure_flags["strict_one_side"][failure].to(DTYPE).mean().item()
        )
        if bool(torch.any(failure))
        else "",
        "dominant_one_side_fraction_among_failures": float(
            failure_flags["dominant_one_side"][failure].to(DTYPE).mean().item()
        )
        if bool(torch.any(failure))
        else "",
        "too_diffuse_fraction_among_failures": float(
            failure_flags["too_diffuse"][failure].to(DTYPE).mean().item()
        )
        if bool(torch.any(failure))
        else "",
    }
    valid = rollout.query_mask & mask[:, None]
    row["exploration_fraction"] = float(
        rollout.explore_action[valid].to(DTYPE).mean().item()
    )
    row["mean_proposal_scale"] = float(rollout.proposal_scale[valid].mean().item())
    row["mean_probe_distance_from_best"] = float(
        rollout.probe_distance_from_best[valid].mean().item()
    )
    for threshold in LOSS_THRESHOLDS:
        exponent = int(round(-math.log10(threshold)))
        hits = rollout.first_threshold_evaluation[threshold]
        reached = (hits > 0) & mask
        values = hits[reached].to(DTYPE)
        row[f"fraction_reaching_1e_{exponent}"] = float(
            reached[mask].to(DTYPE).mean().item()
        )
        row[f"median_evaluations_to_1e_{exponent}"] = (
            float(torch.median(values).item()) if bool(torch.any(reached)) else ""
        )
        row[f"mean_evaluations_to_1e_{exponent}"] = (
            float(values.mean().item()) if bool(torch.any(reached)) else ""
        )
        row[f"worst_evaluations_to_1e_{exponent}"] = (
            float(values.max().item()) if bool(torch.any(reached)) else ""
        )
    return row


def _summary_rows(
    rollout: SearchRollout,
    c_star: torch.Tensor,
    difficult_masks: dict[str, torch.Tensor],
    failure_flags: dict[str, torch.Tensor],
    training_history: list[dict[str, Any]],
    runtime: float,
) -> list[dict[str, Any]]:
    groups = {"all_test": torch.ones_like(rollout.success), **difficult_masks}
    rows = [
        _group_summary(name, mask, rollout, c_star, failure_flags)
        for name, mask in groups.items()
    ]
    for band_name, lower_loss, upper_loss in FIXED_LOSS_BANDS:
        band = (
            rollout.query_mask
            & (rollout.prior_best_loss >= lower_loss)
            & (rollout.prior_best_loss < upper_loss)
        )
        rows.append(
            {
                "row_type": "behavior_by_loss_band",
                "group": "all_test",
                "loss_band": band_name,
                "queries_in_band": int(band.sum().item()),
                "exploration_fraction": float(
                    rollout.explore_action[band].to(DTYPE).mean().item()
                )
                if bool(torch.any(band))
                else "",
                "median_proposal_scale": float(
                    torch.median(rollout.proposal_scale[band]).item()
                )
                if bool(torch.any(band))
                else "",
                "mean_proposal_scale": float(
                    rollout.proposal_scale[band].mean().item()
                )
                if bool(torch.any(band))
                else "",
                "median_probe_distance_from_best": float(
                    torch.median(rollout.probe_distance_from_best[band]).item()
                )
                if bool(torch.any(band))
                else "",
                "new_probe_improved_fraction": float(
                    rollout.improved_best[band].to(DTYPE).mean().item()
                )
                if bool(torch.any(band))
                else "",
            }
        )
    for evaluation in (4, 10, 25, 50, 100, 150, 200, 250):
        if evaluation > EMERGENCY_CAP:
            continue
        losses = rollout.best_loss_history[:, evaluation - INITIAL_EVALUATIONS]
        rows.append(
            {
                "row_type": "evaluation_progress",
                "group": "all_test",
                "evaluations": evaluation,
                "median_best_prefix_loss": float(torch.median(losses).item()),
                "worst_best_prefix_loss": float(losses.max().item()),
                "fraction_below_1e_20": float(
                    (losses < SUCCESS_LOSS).to(DTYPE).mean().item()
                ),
            }
        )
    rows.extend(training_history)
    shared = {
        "frozen_network1_horizon": 20,
        "emergency_evaluation_cap": EMERGENCY_CAP,
        "total_runtime_seconds": runtime,
    }
    for row in rows:
        row.update(shared)
    return rows


def _plot(rollout: SearchRollout) -> None:
    evaluations = list(range(INITIAL_EVALUATIONS, EMERGENCY_CAP + 1))
    median = torch.median(rollout.best_loss_history, dim=0).values
    ninety = torch.quantile(rollout.best_loss_history, 0.9, dim=0)
    worst = rollout.best_loss_history.max(dim=0).values
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(evaluations, median, label="median")
    axes[0].plot(evaluations, ninety, label="90th percentile")
    axes[0].plot(evaluations, worst, label="worst")
    for threshold in LOSS_THRESHOLDS:
        axes[0].axhline(threshold, color="gray", alpha=0.25, linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("best observed prefix loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for threshold in LOSS_THRESHOLDS:
        axes[1].plot(
            evaluations,
            (rollout.best_loss_history < threshold).to(DTYPE).mean(dim=0),
            label=f"below {threshold:g}",
        )
    axes[1].set_xlabel("total prefix-loss evaluations")
    axes[1].set_ylabel("test fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "prefix_loss_vs_evaluations.png", dpi=180)
    plt.close(figure)


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = _load_frozen_config()
    train_count = int(config["train_trajectories"])
    validation_count = int(config["validation_trajectories"])
    test_count = int(config["test_trajectories"])
    train_positions, _, _ = data_source._make_exact_labeled_split(
        train_count, int(config["train_data_seed"])
    )
    validation_positions, _, _ = data_source._make_exact_labeled_split(
        validation_count, int(config["validation_data_seed"])
    )
    test_positions, test_labels, _test_hidden_c = data_source._make_exact_labeled_split(
        test_count, int(config["test_data_seed"])
    )
    del _test_hidden_c
    train_lower, train_upper, train_best_c, train_best_loss = _load_intervals(
        FROZEN_ROOT / "train_safe_intervals.csv", train_count
    )
    val_lower, val_upper, val_best_c, val_best_loss = _load_intervals(
        FROZEN_ROOT / "validation_safe_intervals.csv", validation_count
    )
    test_lower, test_upper, test_best_c, test_best_loss = _load_intervals(
        FROZEN_ROOT / "test_safe_intervals.csv", test_count
    )
    print("auditing frozen Network 1 cached losses; no Network 1 execution", flush=True)
    train_u, train_loss = _initial_history(
        train_positions, train_lower, train_upper, train_best_c, train_best_loss
    )
    val_u, val_loss = _initial_history(
        validation_positions, val_lower, val_upper, val_best_c, val_best_loss
    )
    test_u, test_loss = _initial_history(
        test_positions, test_lower, test_upper, test_best_c, test_best_loss
    )
    print("training Network 2 explore/exploit policy", flush=True)
    final_state, training_history = _train(
        (train_positions, train_lower, train_upper, train_u, train_loss),
        (validation_positions, val_lower, val_upper, val_u, val_loss),
    )
    model = ExploreExploitController().to(dtype=DTYPE)
    model.load_state_dict(final_state)
    print("evaluating frozen Network 2 through the emergency cap", flush=True)
    test_rollout = _evaluate(
        model,
        test_positions,
        test_lower,
        test_upper,
        test_u,
        test_loss,
        MAX_LEARNED_QUERIES,
        EVALUATION_SEED + 10_000,
    )

    # Post-hoc references and previous-difficulty labels remain unopened until
    # the full test search and all stopping decisions have been frozen.
    data_source._verify_test_labels(test_labels)
    c_star = reference_search._stored_test_c_star()
    replay_verification = _verify_replay(test_rollout, c_star)
    if not bool(replay_verification["all_passed"]):
        raise RuntimeError(
            "Network 2 deterministic replay materially disagreed with the "
            f"recorded successful run: {replay_verification}"
        )
    _save_checkpoint(
        final_state,
        TRAINING_HORIZONS[-1],
        int(training_history[-1]["best_update"]),
        float(training_history[-1]["best_validation_score"]),
        training_history,
        status="permanently_frozen_replay_verified",
        replay_verification=replay_verification,
    )
    difficult_masks = _previous_difficult_masks()
    trajectory_rows, failure_flags = _trajectory_rows(
        test_rollout,
        c_star,
        test_lower,
        test_upper,
        difficult_masks,
    )
    runtime = time.perf_counter() - start
    summary_rows = _summary_rows(
        test_rollout,
        c_star,
        difficult_masks,
        failure_flags,
        training_history,
        runtime,
    )
    _write_csv(OUTPUT_DIR / "network2_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_trajectory_search.csv", trajectory_rows)
    _plot(test_rollout)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "success_fraction": float(test_rollout.success.to(DTYPE).mean().item()),
            "median_best_loss": float(torch.median(test_rollout.best_loss).item()),
            "worst_best_loss": float(test_rollout.best_loss.max().item()),
            "runtime_seconds": runtime,
            "checkpoint": str(CHECKPOINT_PATH),
            "replay_verified": True,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
