#!/usr/bin/env python3
"""Learned two-parameter N2 refinement from persisted eDIRECT N1 outputs.

The successful original eDIRECT run persisted best points, losses, and search
evaluation counts but not its complete Voronoi histories.  For normal N1
successes, this diagnostic therefore constructs a local rectangle centered on
the persisted best point with half-width
``3.98/sqrt(original_evaluations)`` on each axis.  This is a search-geometry
spacing estimate and uses no reference parameter.  For all four trajectories
where adaptive-strip rescue was invoked, the exact saved strip and pre-nested-
1D rescue endpoint are used.  Fixed initial probes and every learned proposal
are evaluated with the actual prefix reconstruction loss.

One shared 2-D DeepSets controller explicitly selects explore versus exploit
and emits a continuous bivariate Normal proposal.  It is trained recurrently
with REINFORCE on actual loss improvement, novelty, decaying entropy, and an
evaluation cost.  Reference parameters are used only after the frozen policy
evaluation for reporting.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

from experiments.sho.run_stage1b_edirect_vs_direct import (
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
    _prefix_losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/nested1d_and_network2_2d")
EDIRECT_SOURCE = Path(
    "outputs/sho/stage1B/edirect_vs_direct/per_trajectory_comparison.csv"
)
RESCUE_SOURCE = Path(
    "outputs/sho/stage1B/adaptive_boundary_rescue/adaptive_boundary_rescue_cases.csv"
)
TRAJECTORY_COUNT = 256
INITIAL_EVALUATIONS = 10
EMERGENCY_CAP = 250
MAX_LEARNED_QUERIES = EMERGENCY_CAP - INITIAL_EVALUATIONS
TRAINING_HORIZONS = (20, 60, 120, 200)
STAGE_UPDATES = (250, 200, 150, 100)
BATCH_SIZE = 32
LEARNING_RATE = 1.0e-4
EVAL_INTERVAL = 50
DISCOUNT = 0.97
NOVELTY_WEIGHT = 0.01
ENTROPY_WEIGHT = 0.002
EVALUATION_COST = 0.001
SUCCESS_BONUS = 5.0
LOG_EPSILON = 1.0e-30
SUCCESS_LOSS = 1.0e-20
THRESHOLDS = (1.0e-4, 1.0e-8, 1.0e-12, 1.0e-16, 1.0e-20)
MODEL_SEED = SEED + 9_610_000
SPLIT_SEED = SEED + 9_620_000
EVALUATION_SEED = SEED + 9_630_000

DOMAIN_RULE = (
    "normal eDIRECT pass: rectangle centered on persisted best (a,b), each "
    "half-width=3.98/sqrt(saved objective evaluations), clipped to [-1.99,1.99]; "
    "adaptive-rescue cases: exact saved d+s strip; all use pre-Part-A endpoints"
)
POLICY_RULE = (
    "shared 2-D DeepSets explore/exploit controller over complete normalized "
    "((a,b),loss) history; poorly explored point from deterministic 128-point "
    "Sobol coverage set; continuous bivariate-Normal proposals; recurrent "
    "REINFORCE from real prefix-loss improvement, novelty, decaying entropy, "
    "and evaluation cost; no loss gradients or parameter supervision"
)


class ExploreExploitController2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(9, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(48 * 2 + 16, 96),
            nn.SiLU(),
            nn.Linear(96, 9),
        )
        with torch.no_grad():
            final = self.head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Unexpected controller head")
            final.bias.zero_()
            final.bias[0] = math.log(0.25 / 0.75)
            final.bias[3:5] = -4.0

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
        return self.head(torch.cat((mean_context, max_context, global_features), dim=1))


@dataclass(slots=True)
class N1Inputs:
    lower: torch.Tensor
    upper: torch.Tensor
    best_theta: torch.Tensor
    best_loss: torch.Tensor
    source: list[str]
    category: list[str]
    original_evaluations: torch.Tensor


@dataclass(slots=True)
class Statistics:
    features: torch.Tensor
    global_features: torch.Tensor
    farthest_point: torch.Tensor
    farthest_distance: torch.Tensor
    local_radius: torch.Tensor


@dataclass(slots=True)
class Rollout:
    best_theta: torch.Tensor
    best_loss: torch.Tensor
    success: torch.Tensor
    evaluations_consumed: torch.Tensor
    first_threshold_evaluation: dict[float, torch.Tensor]
    exploration_fraction: torch.Tensor
    mean_proposal_scale: torch.Tensor
    mean_probe_distance: torch.Tensor
    log_probabilities: list[torch.Tensor]
    rewards: list[torch.Tensor]
    entropies: list[torch.Tensor]
    entropy_factors: list[torch.Tensor]
    active_masks: list[torch.Tensor]


def _load_n1_inputs() -> N1Inputs:
    edirect = [row for row in _read_csv(EDIRECT_SOURCE) if row["method"] == "edirect"]
    edirect.sort(key=lambda row: int(row["trajectory_id"]))
    rescue = {int(row["trajectory_id"]): row for row in _read_csv(RESCUE_SOURCE)}
    if len(edirect) != TRAJECTORY_COUNT or set(rescue) != {36, 68, 116, 230}:
        raise RuntimeError("Persisted N1 inputs are incomplete")

    lower: list[list[float]] = []
    upper: list[list[float]] = []
    best_theta: list[list[float]] = []
    best_loss: list[float] = []
    source: list[str] = []
    category: list[str] = []
    original_evaluations: list[int] = []
    for trajectory_id, row in enumerate(edirect):
        evaluations = int(row["objective_evaluations"])
        original_evaluations.append(evaluations)
        if trajectory_id in rescue:
            saved = rescue[trajectory_id]
            lower.append(
                [float(saved["strip_a_lower"]), float(saved["strip_b_lower"])]
            )
            upper.append(
                [float(saved["strip_a_upper"]), float(saved["strip_b_upper"])]
            )
            best_theta.append(
                [float(saved["final_best_a"]), float(saved["final_best_b"])]
            )
            loss = float(saved["final_loss"])
            best_loss.append(loss)
            source.append("adaptive_strip_rescue_pre_part_a")
        else:
            point = np.asarray(
                (float(row["best_sample_a"]), float(row["best_sample_b"]))
            )
            spacing = DOMAIN_WIDTH / math.sqrt(evaluations)
            lower.append(np.maximum(LOWER, point - spacing).tolist())
            upper.append(np.minimum(UPPER, point + spacing).tolist())
            best_theta.append(point.tolist())
            loss = float(row["best_prefix_loss"])
            best_loss.append(loss)
            source.append("normal_edirect_nominal_local_spacing")
        if loss <= 1.0e-4:
            category.append("n1_passed_1e-4")
        elif loss < 1.0e-3:
            category.append("n1_failed_below_1e-3")
        else:
            category.append("n1_above_or_equal_1e-3")
    return N1Inputs(
        lower=torch.tensor(lower, dtype=DTYPE),
        upper=torch.tensor(upper, dtype=DTYPE),
        best_theta=torch.tensor(best_theta, dtype=DTYPE),
        best_loss=torch.tensor(best_loss, dtype=DTYPE),
        source=source,
        category=category,
        original_evaluations=torch.tensor(original_evaluations, dtype=torch.long),
    )


@torch.no_grad()
def _initial_history(
    positions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    cached_best_theta: torch.Tensor,
    cached_best_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = upper - lower
    best_u = ((cached_best_theta - lower) / width).clamp(0.0, 1.0)
    fixed = torch.tensor(
        (
            (0.0, 0.0),
            (0.0, 0.5),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.5, 1.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (1.0, 1.0),
        ),
        dtype=DTYPE,
    )
    sampled_u = torch.cat(
        (best_u[:, None, :], fixed[None, :, :].expand(len(best_u), -1, -1)),
        dim=1,
    )
    losses: list[torch.Tensor] = []
    for probe in range(INITIAL_EVALUATIONS):
        theta = lower + sampled_u[:, probe] * width
        losses.append(_prefix_losses(theta, positions))
    sampled_loss = torch.stack(losses, dim=1)
    tolerance = 2.0e-12 * torch.maximum(
        torch.ones_like(cached_best_loss), torch.abs(cached_best_loss)
    )
    if not bool(torch.all(torch.abs(sampled_loss[:, 0] - cached_best_loss) <= tolerance)):
        maximum = float(torch.max(torch.abs(sampled_loss[:, 0] - cached_best_loss)))
        raise RuntimeError(f"Persisted N1 best-loss audit failed: {maximum}")
    mask = torch.ones_like(sampled_loss, dtype=torch.bool)
    return sampled_u, sampled_loss, mask


def _coverage_points() -> torch.Tensor:
    sobol = qmc.Sobol(d=2, scramble=False).random_base2(7)
    return torch.tensor(sobol, dtype=DTYPE)


COVERAGE_POINTS = _coverage_points()


def _statistics(
    sampled_u: torch.Tensor,
    sampled_loss: torch.Tensor,
    mask: torch.Tensor,
    best_u: torch.Tensor,
    best_loss: torch.Tensor,
    width: torch.Tensor,
) -> Statistics:
    batch, count, _ = sampled_u.shape
    valid = mask.to(DTYPE)
    valid_count = mask.sum(dim=1).clamp_min(1).to(DTYPE)
    log_loss = torch.log(sampled_loss + LOG_EPSILON)
    log_best = torch.log(best_loss + LOG_EPSILON)
    delta = sampled_u - best_u[:, None, :]
    delta_norm = torch.linalg.norm(delta, dim=2)
    loss_difference = torch.clamp(log_loss - log_best[:, None], 0.0, 60.0)

    pair_distance = torch.cdist(sampled_u, sampled_u)
    diagonal = torch.eye(count, dtype=torch.bool)[None, :, :]
    invalid_pair = diagonal | ~mask[:, None, :] | ~mask[:, :, None]
    pair_distance = torch.where(
        invalid_pair, torch.full_like(pair_distance, float("inf")), pair_distance
    )
    nearest = pair_distance.min(dim=2).values
    nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
    best_indicator = (sampled_loss <= best_loss[:, None] * (1.0 + 1.0e-15)).to(DTYPE)
    features = torch.cat(
        (
            sampled_u,
            (log_loss / 20.0)[:, :, None],
            delta,
            delta_norm[:, :, None],
            (loss_difference / 20.0)[:, :, None],
            nearest[:, :, None],
            best_indicator[:, :, None],
        ),
        dim=2,
    )
    features = torch.where(mask[:, :, None], features, 0.0)

    candidates = COVERAGE_POINTS.to(sampled_u.device)
    candidate_delta = candidates[None, :, None, :] - sampled_u[:, None, :, :]
    candidate_distance = torch.linalg.norm(candidate_delta, dim=3)
    candidate_distance = torch.where(
        mask[:, None, :],
        candidate_distance,
        torch.full_like(candidate_distance, float("inf")),
    )
    nearest_candidate_distance = candidate_distance.min(dim=2).values
    farthest_distance, farthest_index = nearest_candidate_distance.max(dim=1)
    farthest_point = candidates[farthest_index]

    coordinate_distance = torch.abs(sampled_u - best_u[:, None, :])
    coordinate_distance = torch.where(
        mask[:, :, None] & (coordinate_distance > 1.0e-14),
        coordinate_distance,
        torch.full_like(coordinate_distance, float("inf")),
    )
    local_radius = coordinate_distance.min(dim=1).values
    local_radius = torch.where(
        torch.isfinite(local_radius), local_radius, torch.full_like(local_radius, 0.25)
    ).clamp_min(1.0e-12)

    mean_u = (sampled_u * valid[:, :, None]).sum(dim=1) / valid_count[:, None]
    variance = torch.clamp(
        (sampled_u.square() * valid[:, :, None]).sum(dim=1) / valid_count[:, None]
        - mean_u.square(),
        min=0.0,
    )
    high_loss_factor = torch.sigmoid((log_best - math.log(1.0e-8)) / 2.0)
    global_features = torch.cat(
        (
            best_u,
            (log_best / 20.0)[:, None],
            torch.log(width + LOG_EPSILON) / 10.0,
            farthest_point,
            farthest_distance[:, None],
            local_radius,
            mean_u,
            torch.sqrt(variance + 1.0e-30),
            torch.clamp(valid_count / EMERGENCY_CAP, max=1.0)[:, None],
            high_loss_factor[:, None],
        ),
        dim=1,
    )
    return Statistics(features, global_features, farthest_point, farthest_distance, local_radius)


def _sample_query(
    raw: torch.Tensor, statistics: Statistics, best_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    explore_distribution = torch.distributions.Bernoulli(logits=raw[:, 0])
    explore = explore_distribution.sample()
    exploit_center = (
        best_u + torch.tanh(raw[:, 1:3]) * statistics.local_radius
    ).clamp(0.0, 1.0)
    exploit_scale = (
        statistics.local_radius
        * torch.exp(-1.5 - 30.0 * torch.sigmoid(raw[:, 3:5]))
    ).clamp_min(1.0e-15)
    explore_center = (
        statistics.farthest_point
        + 0.5
        * torch.tanh(raw[:, 5:7])
        * statistics.farthest_distance[:, None]
    ).clamp(0.0, 1.0)
    explore_scale = (
        statistics.farthest_distance[:, None]
        * (0.05 + 0.30 * torch.sigmoid(raw[:, 7:9]))
    ).clamp_min(1.0e-12)
    center = torch.where(explore[:, None].bool(), explore_center, exploit_center)
    scale = torch.where(explore[:, None].bool(), explore_scale, exploit_scale)
    normal = torch.distributions.Normal(center, scale)
    raw_query = normal.sample()
    query_u = raw_query.clamp(0.0, 1.0)
    log_probability = explore_distribution.log_prob(explore) + normal.log_prob(raw_query).sum(dim=1)
    entropy = explore_distribution.entropy() + normal.entropy().sum(dim=1)
    return query_u, explore.bool(), scale, log_probability, entropy


def _rollout(
    model: ExploreExploitController2D,
    positions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    initial_u: torch.Tensor,
    initial_loss: torch.Tensor,
    initial_mask: torch.Tensor,
    learned_queries: int,
) -> Rollout:
    width = upper - lower
    sampled_u = initial_u
    sampled_loss = initial_loss
    mask = initial_mask
    best_indices = sampled_loss.argmin(dim=1)
    best_loss = sampled_loss.gather(1, best_indices[:, None]).squeeze(1)
    best_u = sampled_u.gather(
        1, best_indices[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    active = best_loss >= SUCCESS_LOSS
    evaluations_consumed = torch.where(
        active,
        torch.full_like(best_indices, EMERGENCY_CAP),
        torch.full_like(best_indices, INITIAL_EVALUATIONS),
    )
    first_threshold = {
        threshold: torch.where(
            best_loss < threshold,
            torch.full_like(best_indices, INITIAL_EVALUATIONS),
            torch.zeros_like(best_indices),
        )
        for threshold in THRESHOLDS
    }
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    entropy_factors: list[torch.Tensor] = []
    active_masks: list[torch.Tensor] = []
    explore_count = torch.zeros(len(positions), dtype=DTYPE)
    scale_sum = torch.zeros((len(positions), 2), dtype=DTYPE)
    distance_sum = torch.zeros(len(positions), dtype=DTYPE)
    query_count = torch.zeros(len(positions), dtype=DTYPE)

    for query_index in range(learned_queries):
        statistics = _statistics(sampled_u, sampled_loss, mask, best_u, best_loss, width)
        raw = model(statistics.features, mask, statistics.global_features)
        query_u, explore, proposal_scale, log_probability, entropy = _sample_query(
            raw, statistics, best_u
        )
        query_mask = active.clone()
        prior_best_u = best_u.clone()
        theta = lower + query_u * width
        with torch.no_grad():
            query_loss = _prefix_losses(theta.detach(), positions)
            prior_best = best_loss.clone()
            improved = query_mask & (query_loss < best_loss)
            next_best = torch.where(improved, query_loss, best_loss)
            log_improvement = torch.clamp(
                torch.log(prior_best + LOG_EPSILON)
                - torch.log(next_best + LOG_EPSILON),
                0.0,
                40.0,
            )
            novelty = torch.linalg.norm(
                query_u.detach()[:, None, :] - sampled_u, dim=2
            )
            novelty = torch.where(mask, novelty, torch.full_like(novelty, float("inf"))).min(dim=1).values
            high_loss = torch.sigmoid(
                (torch.log(prior_best + LOG_EPSILON) - math.log(1.0e-8)) / 2.0
            )
            concentration_cost = (
                (1.0 - high_loss)
                * torch.linalg.norm(query_u.detach() - best_u, dim=1)
                * 0.01
            )
            newly_successful = query_mask & (next_best < SUCCESS_LOSS)
            reward = (
                log_improvement
                + NOVELTY_WEIGHT * high_loss * novelty
                + SUCCESS_BONUS * newly_successful.to(DTYPE)
                - concentration_cost
                - EVALUATION_COST
            )
            best_loss = next_best
            best_u = torch.where(improved[:, None], query_u.detach(), best_u)
            total_evaluations = INITIAL_EVALUATIONS + query_index + 1
            for threshold, first in first_threshold.items():
                reached = query_mask & (first == 0) & (best_loss < threshold)
                first_threshold[threshold] = torch.where(
                    reached, torch.full_like(first, total_evaluations), first
                )
            evaluations_consumed = torch.where(
                newly_successful,
                torch.full_like(evaluations_consumed, total_evaluations),
                evaluations_consumed,
            )
            active = query_mask & ~newly_successful
            explore_count += (query_mask & explore).to(DTYPE)
            scale_sum += query_mask[:, None].to(DTYPE) * proposal_scale.detach() * width
            distance_sum += query_mask.to(DTYPE) * torch.linalg.norm(
                (query_u.detach() - prior_best_u) * width, dim=1
            )
            query_count += query_mask.to(DTYPE)

        sampled_u = torch.cat((sampled_u, query_u.detach()[:, None, :]), dim=1)
        sampled_loss = torch.cat((sampled_loss, query_loss.detach()[:, None]), dim=1)
        mask = torch.cat((mask, query_mask[:, None]), dim=1)
        log_probabilities.append(log_probability)
        rewards.append(torch.where(query_mask, reward, torch.zeros_like(reward)))
        entropies.append(entropy)
        entropy_factors.append(high_loss)
        active_masks.append(query_mask)

    safe_count = query_count.clamp_min(1.0)
    return Rollout(
        best_theta=(lower + best_u * width).detach(),
        best_loss=best_loss.detach(),
        success=best_loss < SUCCESS_LOSS,
        evaluations_consumed=evaluations_consumed,
        first_threshold_evaluation=first_threshold,
        exploration_fraction=explore_count / safe_count,
        mean_proposal_scale=scale_sum.mean(dim=1) / safe_count,
        mean_probe_distance=distance_sum / safe_count,
        log_probabilities=log_probabilities,
        rewards=rewards,
        entropies=entropies,
        entropy_factors=entropy_factors,
        active_masks=active_masks,
    )


def _policy_loss(rollout: Rollout) -> torch.Tensor:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rollout.rewards[0])
    for reward in reversed(rollout.rewards):
        running = reward + DISCOUNT * running
        returns.append(running)
    returns.reverse()
    terms: list[torch.Tensor] = []
    for value, log_probability, entropy, factor, mask in zip(
        returns,
        rollout.log_probabilities,
        rollout.entropies,
        rollout.entropy_factors,
        rollout.active_masks,
    ):
        if not bool(torch.any(mask)):
            continue
        selected = value[mask]
        advantage = (selected - selected.mean()) / selected.std(unbiased=False).clamp_min(1.0e-6)
        terms.append(
            -(advantage.detach() * log_probability[mask]).mean()
            - ENTROPY_WEIGHT * (factor[mask].detach() * entropy[mask]).mean()
        )
    if not terms:
        raise RuntimeError("No active policy-gradient terms")
    return torch.stack(terms).mean()


@torch.no_grad()
def _evaluate(
    model: ExploreExploitController2D,
    positions: torch.Tensor,
    n1: N1Inputs,
    initial_u: torch.Tensor,
    initial_loss: torch.Tensor,
    initial_mask: torch.Tensor,
    indices: torch.Tensor,
    learned_queries: int,
    seed: int,
) -> Rollout:
    parts: list[Rollout] = []
    saved_rng = torch.random.get_rng_state()
    model.eval()
    try:
        for batch_number, start in enumerate(range(0, len(indices), BATCH_SIZE)):
            chosen = indices[start : start + BATCH_SIZE]
            torch.manual_seed(seed + batch_number)
            parts.append(
                _rollout(
                    model,
                    positions[chosen],
                    n1.lower[chosen],
                    n1.upper[chosen],
                    initial_u[chosen],
                    initial_loss[chosen],
                    initial_mask[chosen],
                    learned_queries,
                )
            )
    finally:
        torch.random.set_rng_state(saved_rng)

    return Rollout(
        best_theta=torch.cat([part.best_theta for part in parts]),
        best_loss=torch.cat([part.best_loss for part in parts]),
        success=torch.cat([part.success for part in parts]),
        evaluations_consumed=torch.cat([part.evaluations_consumed for part in parts]),
        first_threshold_evaluation={
            threshold: torch.cat(
                [part.first_threshold_evaluation[threshold] for part in parts]
            )
            for threshold in THRESHOLDS
        },
        exploration_fraction=torch.cat([part.exploration_fraction for part in parts]),
        mean_proposal_scale=torch.cat([part.mean_proposal_scale for part in parts]),
        mean_probe_distance=torch.cat([part.mean_probe_distance for part in parts]),
        log_probabilities=[],
        rewards=[],
        entropies=[],
        entropy_factors=[],
        active_masks=[],
    )


def _validation_score(rollout: Rollout) -> float:
    success = float(rollout.success.to(DTYPE).mean())
    quality = float(
        -torch.log10(rollout.best_loss.clamp_min(1.0e-30)).clamp(-20.0, 20.0).mean()
    )
    return 100.0 * success + quality


def _train(
    positions: torch.Tensor,
    n1: N1Inputs,
    initial_u: torch.Tensor,
    initial_loss: torch.Tensor,
    initial_mask: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
) -> tuple[ExploreExploitController2D, list[dict[str, object]]]:
    torch.manual_seed(MODEL_SEED)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    model = ExploreExploitController2D()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    stage_rows: list[dict[str, object]] = []
    cumulative = 0
    for stage, (horizon, updates) in enumerate(zip(TRAINING_HORIZONS, STAGE_UPDATES), start=1):
        best_state = copy.deepcopy(model.state_dict())
        best_score = -math.inf
        best_update = 0
        for update in range(1, updates + 1):
            choice = train_indices[
                torch.randint(0, len(train_indices), (BATCH_SIZE,), generator=generator)
            ]
            torch.manual_seed(MODEL_SEED + 10_000 * stage + update)
            model.train()
            rollout = _rollout(
                model,
                positions[choice],
                n1.lower[choice],
                n1.upper[choice],
                initial_u[choice],
                initial_loss[choice],
                initial_mask[choice],
                horizon,
            )
            loss = _policy_loss(rollout)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            cumulative += 1
            if update % EVAL_INTERVAL == 0 or update == updates:
                validation = _evaluate(
                    model,
                    positions,
                    n1,
                    initial_u,
                    initial_loss,
                    initial_mask,
                    validation_indices,
                    horizon,
                    EVALUATION_SEED + stage * 1000 + update,
                )
                score = _validation_score(validation)
                if score > best_score:
                    best_score = score
                    best_update = update
                    best_state = copy.deepcopy(model.state_dict())
                print(
                    f"N2-2D stage={horizon} update={update}/{updates} "
                    f"val_success={int(validation.success.sum())}/{len(validation_indices)} "
                    f"median_loss={float(validation.best_loss.median()):.3e}",
                    flush=True,
                )
        model.load_state_dict(best_state)
        stage_rows.append(
            {
                "training_horizon": horizon,
                "updates": updates,
                "best_update": best_update,
                "best_validation_score": best_score,
                "cumulative_updates": cumulative,
            }
        )
    return model, stage_rows


def _per_trajectory_rows(
    rollout: Rollout,
    n1: N1Inputs,
    references: torch.Tensor,
    split_labels: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    errors = torch.linalg.norm(rollout.best_theta - references, dim=1)
    inside = torch.all(
        (references >= n1.lower - 1.0e-14) & (references <= n1.upper + 1.0e-14),
        dim=1,
    )
    for index in range(TRAJECTORY_COUNT):
        row: dict[str, object] = {
            "trajectory_id": index,
            "n2_split": split_labels[index],
            "n1_category": n1.category[index],
            "n1_source": n1.source[index],
            "n1_starting_loss": float(n1.best_loss[index]),
            "n1_best_a": float(n1.best_theta[index, 0]),
            "n1_best_b": float(n1.best_theta[index, 1]),
            "n2_domain_a_lower": float(n1.lower[index, 0]),
            "n2_domain_a_upper": float(n1.upper[index, 0]),
            "n2_domain_b_lower": float(n1.lower[index, 1]),
            "n2_domain_b_upper": float(n1.upper[index, 1]),
            "reference_inside_n2_domain_posthoc": int(inside[index]),
            "success_1e-20": int(rollout.success[index]),
            "evaluations_consumed": int(rollout.evaluations_consumed[index]),
            "hit_emergency_cap": int(not bool(rollout.success[index])),
            "final_best_loss": float(rollout.best_loss[index]),
            "final_best_a": float(rollout.best_theta[index, 0]),
            "final_best_b": float(rollout.best_theta[index, 1]),
            "final_parameter_error_posthoc": float(errors[index]),
            "exploration_fraction": float(rollout.exploration_fraction[index]),
            "mean_proposal_scale": float(rollout.mean_proposal_scale[index]),
            "mean_probe_distance": float(rollout.mean_probe_distance[index]),
            "domain_rule": DOMAIN_RULE,
            "policy_rule": POLICY_RULE,
        }
        for threshold in THRESHOLDS:
            key = f"{threshold:.0e}".replace("-0", "-")
            evaluation = int(rollout.first_threshold_evaluation[threshold][index])
            row[f"reached_{key}"] = int(evaluation > 0)
            row[f"evaluations_to_{key}"] = evaluation if evaluation > 0 else ""
        rows.append(row)
    return rows


def _summary_rows(
    rows: list[dict[str, object]], stage_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    groups: list[tuple[str, list[dict[str, object]]]] = [
        ("all", rows),
        ("n1_already_passed_1e-4", [row for row in rows if row["n1_category"] == "n1_passed_1e-4"]),
        ("n1_failed_1e-4_but_below_1e-3", [row for row in rows if row["n1_category"] == "n1_failed_below_1e-3"]),
        ("n1_remained_at_or_above_1e-3", [row for row in rows if row["n1_category"] == "n1_above_or_equal_1e-3"]),
        ("trajectory_36", [row for row in rows if int(row["trajectory_id"]) == 36]),
        ("trajectory_230", [row for row in rows if int(row["trajectory_id"]) == 230]),
        ("n2_train_split", [row for row in rows if row["n2_split"] == "train"]),
        ("n2_validation_split", [row for row in rows if row["n2_split"] == "validation"]),
    ]
    output: list[dict[str, object]] = []
    training_metadata = ";".join(
        f"h{row['training_horizon']}:updates={row['updates']}:best={row['best_update']}"
        for row in stage_rows
    )
    for name, selected in groups:
        summary: dict[str, object] = {
            "group": name,
            "trajectories": len(selected),
            "n1_starting_loss_median": float(np.median([float(row["n1_starting_loss"]) for row in selected])) if selected else "",
            "n1_starting_loss_worst": max([float(row["n1_starting_loss"]) for row in selected], default=""),
            "n2_success_count_1e-20": sum(int(row["success_1e-20"]) for row in selected),
            "n2_success_fraction_1e-20": sum(int(row["success_1e-20"]) for row in selected) / len(selected) if selected else "",
            "n2_failure_count": sum(1 - int(row["success_1e-20"]) for row in selected),
            "domain_posthoc_coverage": sum(int(row["reference_inside_n2_domain_posthoc"]) for row in selected) / len(selected) if selected else "",
            "median_final_parameter_error_posthoc": float(np.median([float(row["final_parameter_error_posthoc"]) for row in selected])) if selected else "",
            "worst_final_parameter_error_posthoc": max([float(row["final_parameter_error_posthoc"]) for row in selected], default=""),
            "median_evaluations_among_1e-20_successes": "",
            "mean_evaluations_among_1e-20_successes": "",
            "worst_evaluations_among_1e-20_successes": "",
            "emergency_cap": EMERGENCY_CAP,
            "training_metadata": training_metadata,
            "domain_rule": DOMAIN_RULE,
            "policy_rule": POLICY_RULE,
        }
        successful_evaluations = [
            int(row["evaluations_consumed"])
            for row in selected
            if int(row["success_1e-20"]) == 1
        ]
        if successful_evaluations:
            summary["median_evaluations_among_1e-20_successes"] = float(np.median(successful_evaluations))
            summary["mean_evaluations_among_1e-20_successes"] = float(np.mean(successful_evaluations))
            summary["worst_evaluations_among_1e-20_successes"] = max(successful_evaluations)
        for threshold in THRESHOLDS:
            key = f"{threshold:.0e}".replace("-0", "-")
            reached = [row for row in selected if int(row[f"reached_{key}"]) == 1]
            evaluations = [int(row[f"evaluations_to_{key}"]) for row in reached]
            summary[f"fraction_reaching_{key}"] = len(reached) / len(selected) if selected else ""
            summary[f"median_evaluations_to_{key}"] = float(np.median(evaluations)) if evaluations else ""
            summary[f"mean_evaluations_to_{key}"] = float(np.mean(evaluations)) if evaluations else ""
            summary[f"worst_evaluations_to_{key}"] = max(evaluations, default="")
        output.append(summary)
    return output


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    starting = np.asarray([float(row["n1_starting_loss"]) for row in rows])
    final = np.asarray([float(row["final_best_loss"]) for row in rows])
    success = np.asarray([bool(int(row["success_1e-20"])) for row in rows])
    axes[0].scatter(starting[~success], final[~success], s=24, alpha=0.8, color="tab:red", label="N2 failed")
    axes[0].scatter(starting[success], final[success], s=20, alpha=0.65, color="tab:blue", label="N2 reached 1e-20")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].axhline(SUCCESS_LOSS, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("N1 starting prefix loss")
    axes[0].set_ylabel("N2 final prefix loss")
    axes[0].grid(alpha=0.23)
    axes[0].legend(frameon=False)

    labels = ("N1 <=1e-4", "1e-4..1e-3", "N1 >=1e-3")
    categories = ("n1_passed_1e-4", "n1_failed_below_1e-3", "n1_above_or_equal_1e-3")
    fractions = []
    counts = []
    for category in categories:
        selected = [row for row in rows if row["n1_category"] == category]
        counts.append(len(selected))
        fractions.append(sum(int(row["success_1e-20"]) for row in selected) / len(selected) if selected else 0.0)
    bars = axes[1].bar(labels, fractions, color=("tab:blue", "0.65", "tab:orange"))
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={count}", ha="center")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("fraction reaching 1e-20")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.23)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    if not EDIRECT_SOURCE.exists():
        raise FileNotFoundError(EDIRECT_SOURCE)
    if not RESCUE_SOURCE.exists():
        raise FileNotFoundError(RESCUE_SOURCE)
    positions, _, hidden_c = _make_exact_labeled_split(TRAJECTORY_COUNT, FRESH_TEST_SEED)
    positions = positions.to(dtype=DTYPE)
    references = _posthoc_reference(hidden_c).to(dtype=DTYPE)
    n1 = _load_n1_inputs()
    initial_u, initial_loss, initial_mask = _initial_history(
        positions, n1.lower, n1.upper, n1.best_theta, n1.best_loss
    )

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(TRAJECTORY_COUNT, generator=generator)
    validation_indices = torch.sort(permutation[:64]).values
    train_indices = torch.sort(permutation[64:]).values
    split_labels = ["train"] * TRAJECTORY_COUNT
    for index in validation_indices.tolist():
        split_labels[index] = "validation"

    model, stage_rows = _train(
        positions,
        n1,
        initial_u,
        initial_loss,
        initial_mask,
        train_indices,
        validation_indices,
    )
    all_indices = torch.arange(TRAJECTORY_COUNT)
    evaluation = _evaluate(
        model,
        positions,
        n1,
        initial_u,
        initial_loss,
        initial_mask,
        all_indices,
        MAX_LEARNED_QUERIES,
        EVALUATION_SEED + 100_000,
    )
    rows = _per_trajectory_rows(evaluation, n1, references, split_labels)
    summary = _summary_rows(rows, stage_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "network2_2d_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "network2_2d_per_trajectory.csv", rows)
    _plot(OUTPUT_DIR / "n2_success_vs_n1_starting_loss.png", rows)
    print(
        f"N2-2D final success={int(evaluation.success.sum())}/{TRAJECTORY_COUNT}; "
        f"wrote {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
