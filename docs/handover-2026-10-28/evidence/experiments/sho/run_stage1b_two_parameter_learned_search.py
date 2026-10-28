"""Learned global-to-local search for K(a,b) without coefficient supervision."""

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

import torch
from torch import nn

import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as data_source
from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    DRIFT_HEAVY,
    J_ALL,
    P,
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
        "TWO_PARAMETER_SEARCH_OUTPUT_DIR",
        "outputs/sho/stage1B/two_parameter_learned_search",
    )
)
LOWER = -1.99
UPPER = 1.99
DOMAIN_WIDTH = UPPER - LOWER
TRAIN_N = int(os.environ.get("TWO_PARAMETER_TRAIN_N", "4096"))
VALIDATION_N = int(os.environ.get("TWO_PARAMETER_VALIDATION_N", "512"))
TEST_N = int(os.environ.get("TWO_PARAMETER_TEST_N", "512"))
TRAIN_DATA_SEED = SEED + 1_810_100
VALIDATION_DATA_SEED = SEED + 1_810_200
TEST_DATA_SEED = SEED + 1_810_300
NETWORK1_MODEL_SEED = SEED + 2_300_000
NETWORK1_HORIZONS = tuple(
    int(value)
    for value in os.environ.get("TWO_PARAMETER_N1_HORIZONS", "10,20,30").split(",")
)
NETWORK1_STAGE_UPDATES = tuple(
    int(value)
    for value in os.environ.get(
        "TWO_PARAMETER_N1_UPDATES", "1000,1000,1000"
    ).split(",")
)
if len(NETWORK1_HORIZONS) != len(NETWORK1_STAGE_UPDATES):
    raise ValueError("Each Network 1 horizon requires an update budget")
NETWORK1_BATCH_SIZE = int(os.environ.get("TWO_PARAMETER_N1_BATCH_SIZE", "64"))
NETWORK1_LEARNING_RATE = 1e-4
NETWORK1_EVAL_INTERVAL = int(os.environ.get("TWO_PARAMETER_N1_EVAL_INTERVAL", "50"))
NETWORK1_PATIENCE = int(os.environ.get("TWO_PARAMETER_N1_PATIENCE", "300"))
NETWORK1_DISCOUNT = 0.97
NETWORK1_NOVELTY_WEIGHT = 0.02
NETWORK1_ENTROPY_WEIGHT = 0.01
LOG_EPSILON = 1e-30


def _prefix_losses(theta: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Actual prefix reconstruction loss for unconstrained (a,b)."""
    prefix = positions[:, :P]
    parameter_a = theta[:, 0]
    parameter_b = theta[:, 1]
    losses = torch.zeros(theta.shape[0], dtype=DTYPE, device=theta.device)
    for horizon in J_ALL:
        starts = torch.arange(1, P - horizon, device=prefix.device)
        previous = prefix[:, starts - 1]
        current = prefix[:, starts]
        predictions: list[torch.Tensor] = []
        for _ in range(horizon):
            following = (
                parameter_a[:, None] * previous
                + parameter_b[:, None] * current
            )
            predictions.append(following)
            previous, current = current, following
        prediction = torch.stack(predictions, dim=-1)
        truth = torch.stack(
            [prefix[:, starts + step] for step in range(1, horizon + 1)],
            dim=-1,
        )
        losses = losses + DRIFT_HEAVY[horizon] * torch.mean(
            (prediction - truth).square(), dim=(1, 2)
        )
    return losses


class RectanglePolicy(nn.Module):
    """Permutation-equivariant scorer over all active 2D rectangles."""

    def __init__(self) -> None:
        super().__init__()
        self.rectangle_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, rectangle_features: torch.Tensor) -> torch.Tensor:
        embedded = self.rectangle_encoder(rectangle_features)
        mean_context = embedded.mean(dim=1, keepdim=True)
        max_context = embedded.max(dim=1, keepdim=True).values
        context = torch.cat((mean_context, max_context), dim=2).expand(
            -1, embedded.shape[1], -1
        )
        return self.score_head(
            torch.cat((embedded, context), dim=2)
        ).squeeze(-1)


@dataclass
class RectangleRollout:
    query_theta: torch.Tensor
    query_loss: torch.Tensor
    best_theta: torch.Tensor
    best_loss: torch.Tensor
    candidate_lower: torch.Tensor
    candidate_upper: torch.Tensor
    selected_side_lengths: torch.Tensor
    selected_area: torch.Tensor
    selected_count: torch.Tensor
    selected_underexplored: torch.Tensor
    selected_large: torch.Tensor
    entropy: torch.Tensor
    log_probabilities: list[torch.Tensor]
    rewards: list[torch.Tensor]


def _rectangle_statistics(
    rectangle_lower: torch.Tensor,
    rectangle_upper: torch.Tensor,
    observed_theta: torch.Tensor,
    observed_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = 0.5 * (rectangle_lower + rectangle_upper)
    sides = rectangle_upper - rectangle_lower
    area = sides.prod(dim=2)
    if observed_theta.shape[1] == 0:
        features = torch.cat(
            (
                centers / UPPER,
                sides / DOMAIN_WIDTH,
                (area / DOMAIN_WIDTH**2)[:, :, None],
                torch.zeros_like(area)[:, :, None],
                torch.zeros_like(area)[:, :, None],
                (
                    torch.sum((0.5 * sides).square(), dim=2)
                    / (2.0 * DOMAIN_WIDTH**2)
                )[:, :, None],
                torch.ones_like(area)[:, :, None],
            ),
            dim=2,
        )
        return features, torch.zeros_like(area, dtype=torch.long)

    inside = (
        (observed_theta[:, None, :, :] >= rectangle_lower[:, :, None, :] - 1e-14)
        & (observed_theta[:, None, :, :] <= rectangle_upper[:, :, None, :] + 1e-14)
    ).all(dim=3)
    counts = inside.sum(dim=2)
    masked_loss = torch.where(
        inside,
        observed_loss[:, None, :],
        torch.full_like(observed_loss[:, None, :], float("inf")),
    )
    best_inside = masked_loss.min(dim=2).values
    global_best = observed_loss.min(dim=1).values[:, None]
    best_inside = torch.where(torch.isfinite(best_inside), best_inside, global_best)
    squared_distances = torch.sum(
        (observed_theta[:, None, :, :] - centers[:, :, None, :]).square(),
        dim=3,
    )
    nearest_squared = squared_distances.min(dim=2).values
    empty = (counts == 0).to(DTYPE)
    features = torch.cat(
        (
            centers / UPPER,
            sides / DOMAIN_WIDTH,
            (area / DOMAIN_WIDTH**2)[:, :, None],
            (torch.log(best_inside + LOG_EPSILON) / 10.0)[:, :, None],
            (torch.log1p(counts.to(DTYPE)) / math.log(32.0))[:, :, None],
            (nearest_squared / (2.0 * DOMAIN_WIDTH**2))[:, :, None],
            empty[:, :, None],
        ),
        dim=2,
    )
    return features, counts


def _split_rectangles(
    rectangle_lower: torch.Tensor,
    rectangle_upper: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, rectangle_count, _ = rectangle_lower.shape
    selected_index = actions[:, None, None].expand(-1, 1, 2)
    lower = rectangle_lower.gather(1, selected_index).squeeze(1)
    upper = rectangle_upper.gather(1, selected_index).squeeze(1)
    middle = 0.5 * (lower + upper)
    lower_a, lower_b = lower.unbind(dim=1)
    middle_a, middle_b = middle.unbind(dim=1)
    upper_a, upper_b = upper.unbind(dim=1)
    children_lower = torch.stack(
        (
            lower,
            torch.stack((middle_a, lower_b), dim=1),
            torch.stack((lower_a, middle_b), dim=1),
            middle,
        ),
        dim=1,
    )
    children_upper = torch.stack(
        (
            middle,
            torch.stack((upper_a, middle_b), dim=1),
            torch.stack((middle_a, upper_b), dim=1),
            upper,
        ),
        dim=1,
    )
    if rectangle_count == 1:
        return children_lower, children_upper
    indices = torch.arange(rectangle_count)[None, :].expand(batch_size, -1)
    keep = indices != actions[:, None]
    kept_lower = rectangle_lower[keep].reshape(batch_size, rectangle_count - 1, 2)
    kept_upper = rectangle_upper[keep].reshape(batch_size, rectangle_count - 1, 2)
    return (
        torch.cat((kept_lower, children_lower), dim=1),
        torch.cat((kept_upper, children_upper), dim=1),
    )


def _candidate_hull(
    rectangle_lower: torch.Tensor,
    rectangle_upper: torch.Tensor,
    best_theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    touching = (
        (best_theta[:, None, :] >= rectangle_lower - 1e-14)
        & (best_theta[:, None, :] <= rectangle_upper + 1e-14)
    ).all(dim=2)
    lower = torch.where(
        touching[:, :, None],
        rectangle_lower,
        torch.full_like(rectangle_lower, float("inf")),
    ).min(dim=1).values
    upper = torch.where(
        touching[:, :, None],
        rectangle_upper,
        torch.full_like(rectangle_upper, -float("inf")),
    ).max(dim=1).values
    if not bool(torch.all(torch.isfinite(lower) & torch.isfinite(upper))):
        raise RuntimeError("Best sampled hypothesis is not covered by active rectangles")
    return lower, upper


def _network1_rollout(
    model: RectanglePolicy,
    positions: torch.Tensor,
    horizon: int,
    greedy: bool,
) -> RectangleRollout:
    batch_size = positions.shape[0]
    rectangle_lower = torch.full((batch_size, 1, 2), LOWER, dtype=DTYPE)
    rectangle_upper = torch.full((batch_size, 1, 2), UPPER, dtype=DTYPE)
    observed_theta = torch.empty(batch_size, 0, 2, dtype=DTYPE)
    observed_loss = torch.empty(batch_size, 0, dtype=DTYPE)
    query_theta_history: list[torch.Tensor] = []
    query_loss_history: list[torch.Tensor] = []
    best_theta_history: list[torch.Tensor] = []
    best_loss_history: list[torch.Tensor] = []
    candidate_lower_history: list[torch.Tensor] = []
    candidate_upper_history: list[torch.Tensor] = []
    selected_sides_history: list[torch.Tensor] = []
    selected_area_history: list[torch.Tensor] = []
    selected_count_history: list[torch.Tensor] = []
    underexplored_history: list[torch.Tensor] = []
    large_history: list[torch.Tensor] = []
    entropy_history: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for _ in range(horizon):
        features, counts = _rectangle_statistics(
            rectangle_lower, rectangle_upper, observed_theta, observed_loss
        )
        logits = model(features)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=1) if greedy else distribution.sample()
        gather_index = actions[:, None, None].expand(-1, 1, 2)
        selected_lower = rectangle_lower.gather(1, gather_index).squeeze(1)
        selected_upper = rectangle_upper.gather(1, gather_index).squeeze(1)
        selected_sides = selected_upper - selected_lower
        selected_area = selected_sides.prod(dim=1)
        selected_count = counts.gather(1, actions[:, None]).squeeze(1)
        minimum_count = counts.min(dim=1).values
        all_areas = (rectangle_upper - rectangle_lower).prod(dim=2)
        median_area = all_areas.median(dim=1).values
        underexplored = selected_count == minimum_count
        selected_large = selected_area >= median_area - 1e-14
        query_theta = 0.5 * (selected_lower + selected_upper)
        with torch.no_grad():
            query_loss = _prefix_losses(query_theta, positions)
            prior_best = (
                observed_loss.min(dim=1).values
                if observed_loss.shape[1] > 0
                else query_loss
            )
            next_best = torch.minimum(prior_best, query_loss)
            improvement = torch.clamp(
                torch.log(prior_best + LOG_EPSILON)
                - torch.log(next_best + LOG_EPSILON),
                min=0.0,
                max=10.0,
            )
            novelty = 0.5 * selected_area / DOMAIN_WIDTH**2 + 0.5 / (
                selected_count.to(DTYPE) + 1.0
            )
            reward = improvement + NETWORK1_NOVELTY_WEIGHT * novelty

        observed_theta = torch.cat(
            (observed_theta, query_theta.detach()[:, None, :]), dim=1
        )
        observed_loss = torch.cat(
            (observed_loss, query_loss.detach()[:, None]), dim=1
        )
        rectangle_lower, rectangle_upper = _split_rectangles(
            rectangle_lower, rectangle_upper, actions
        )
        best_indices = torch.argmin(observed_loss, dim=1)
        best_theta = observed_theta.gather(
            1, best_indices[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        best_loss = observed_loss.gather(1, best_indices[:, None]).squeeze(1)
        candidate_lower, candidate_upper = _candidate_hull(
            rectangle_lower, rectangle_upper, best_theta
        )

        query_theta_history.append(query_theta.detach())
        query_loss_history.append(query_loss.detach())
        best_theta_history.append(best_theta)
        best_loss_history.append(best_loss)
        candidate_lower_history.append(candidate_lower)
        candidate_upper_history.append(candidate_upper)
        selected_sides_history.append(selected_sides)
        selected_area_history.append(selected_area)
        selected_count_history.append(selected_count)
        underexplored_history.append(underexplored)
        large_history.append(selected_large)
        entropy_history.append(distribution.entropy())
        log_probabilities.append(distribution.log_prob(actions))
        rewards.append(reward)

    return RectangleRollout(
        query_theta=torch.stack(query_theta_history, dim=1),
        query_loss=torch.stack(query_loss_history, dim=1),
        best_theta=torch.stack(best_theta_history, dim=1),
        best_loss=torch.stack(best_loss_history, dim=1),
        candidate_lower=torch.stack(candidate_lower_history, dim=1),
        candidate_upper=torch.stack(candidate_upper_history, dim=1),
        selected_side_lengths=torch.stack(selected_sides_history, dim=1),
        selected_area=torch.stack(selected_area_history, dim=1),
        selected_count=torch.stack(selected_count_history, dim=1),
        selected_underexplored=torch.stack(underexplored_history, dim=1),
        selected_large=torch.stack(large_history, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        log_probabilities=log_probabilities,
        rewards=rewards,
    )


def _network1_policy_loss(rollout: RectangleRollout) -> torch.Tensor:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rollout.rewards[0])
    for reward in reversed(rollout.rewards):
        running = reward + NETWORK1_DISCOUNT * running
        returns.append(running)
    returns.reverse()
    reinforce_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    for return_value, log_probability, entropy in zip(
        returns, rollout.log_probabilities, rollout.entropy.unbind(dim=1)
    ):
        advantage = return_value - return_value.mean()
        advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-6)
        reinforce_terms.append(
            -(advantage.detach() * log_probability).mean()
        )
        entropy_terms.append(entropy.mean())
    return torch.stack(reinforce_terms).mean() - NETWORK1_ENTROPY_WEIGHT * torch.stack(
        entropy_terms
    ).mean()


@torch.no_grad()
def _evaluate_network1(
    model: RectanglePolicy,
    positions: torch.Tensor,
    horizon: int,
) -> RectangleRollout:
    model.eval()
    parts = [
        _network1_rollout(
            model,
            positions[start : start + NETWORK1_BATCH_SIZE],
            horizon,
            True,
        )
        for start in range(0, positions.shape[0], NETWORK1_BATCH_SIZE)
    ]
    fields = (
        "query_theta",
        "query_loss",
        "best_theta",
        "best_loss",
        "candidate_lower",
        "candidate_upper",
        "selected_side_lengths",
        "selected_area",
        "selected_count",
        "selected_underexplored",
        "selected_large",
        "entropy",
    )
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in fields
    }
    return RectangleRollout(**combined, log_probabilities=[], rewards=[])


def _network1_validation_score(rollout: RectangleRollout) -> float:
    return float(
        torch.log(rollout.best_loss[:, -1] + LOG_EPSILON).mean().item()
    )


def _network1_checkpoint_payload(
    state: dict[str, torch.Tensor],
    horizon: int,
    best_update: int,
    best_validation_score: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_class": "RectanglePolicy",
        "state_dict": {
            key: value.detach().cpu() for key, value in state.items()
        },
        "horizon": horizon,
        "best_update": best_update,
        "best_validation_score": best_validation_score,
        "training_history": [dict(row) for row in history],
        "parameter_domain": [[LOWER, UPPER], [LOWER, UPPER]],
        "position_prefix_length": P,
        "training_uses_reference_parameters": False,
        "model_seed": NETWORK1_MODEL_SEED,
    }


def _save_network1_checkpoint(
    state: dict[str, torch.Tensor],
    horizon: int,
    best_update: int,
    best_validation_score: float,
    history: list[dict[str, Any]],
) -> None:
    torch.save(
        _network1_checkpoint_payload(
            state, horizon, best_update, best_validation_score, history
        ),
        OUTPUT_DIR / f"network1_h{horizon}_best_checkpoint.pt",
    )


def _train_network1(
    train_positions: torch.Tensor,
    validation_positions: torch.Tensor,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    torch.manual_seed(NETWORK1_MODEL_SEED)
    model = RectanglePolicy().to(dtype=DTYPE)
    batch_generator = torch.Generator().manual_seed(NETWORK1_MODEL_SEED + 1)
    stage_states: dict[int, dict[str, torch.Tensor]] = {}
    history: list[dict[str, Any]] = []
    cumulative_updates = 0
    for horizon, budget in zip(NETWORK1_HORIZONS, NETWORK1_STAGE_UPDATES):
        optimizer = torch.optim.Adam(model.parameters(), lr=NETWORK1_LEARNING_RATE)
        validation = _evaluate_network1(model, validation_positions, horizon)
        best_score = _network1_validation_score(validation)
        best_state = copy.deepcopy(model.state_dict())
        best_update = 0
        _save_network1_checkpoint(
            best_state, horizon, best_update, best_score, history
        )
        stale = 0
        updates_run = 0
        stop_reason = "maximum_updates"
        stage_start = time.perf_counter()
        print(
            f"network1 horizon={horizon} update=0000 "
            f"validation mean log best loss={best_score:.6e}",
            flush=True,
        )
        for update in range(1, budget + 1):
            model.train()
            indices = torch.randint(
                0,
                train_positions.shape[0],
                (NETWORK1_BATCH_SIZE,),
                generator=batch_generator,
            )
            rollout = _network1_rollout(
                model, train_positions[indices], horizon, False
            )
            loss = _network1_policy_loss(rollout)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            cumulative_updates += 1
            updates_run = update
            if update % NETWORK1_EVAL_INTERVAL == 0 or update == budget:
                validation = _evaluate_network1(
                    model, validation_positions, horizon
                )
                score = _network1_validation_score(validation)
                if score < best_score - 1e-8:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    best_update = update
                    stale = 0
                    _save_network1_checkpoint(
                        best_state,
                        horizon,
                        best_update,
                        best_score,
                        history,
                    )
                else:
                    stale += NETWORK1_EVAL_INTERVAL
                print(
                    f"network1 horizon={horizon} update={update:04d} "
                    f"policy loss={float(loss.detach().item()):.6e} "
                    f"validation mean log best loss={score:.6e} "
                    f"best_update={best_update} stale={stale}",
                    flush=True,
                )
                if stale >= NETWORK1_PATIENCE:
                    stop_reason = "validation_plateau"
                    break
        model.load_state_dict(best_state)
        stage_states[horizon] = copy.deepcopy(best_state)
        history.append(
            {
                "stage": "network1_training",
                "horizon": horizon,
                "updates_run": updates_run,
                "best_update": best_update,
                "best_validation_score": best_score,
                "stop_reason": stop_reason,
                "runtime_seconds": time.perf_counter() - stage_start,
                "cumulative_updates": cumulative_updates,
            }
        )
        _save_network1_checkpoint(
            best_state, horizon, best_update, best_score, history
        )
    return stage_states, history


def _posthoc_reference(hidden_c: torch.Tensor) -> torch.Tensor:
    """Generating coefficients; never called from a fitting path."""
    return torch.stack((-torch.ones_like(hidden_c), hidden_c), dim=1)


def _network1_metrics(
    rollout: RectangleRollout,
    reference: torch.Tensor,
    horizon: int,
) -> dict[str, Any]:
    index = horizon - 1
    lower = rollout.candidate_lower[:, index]
    upper = rollout.candidate_upper[:, index]
    sides = upper - lower
    area = sides.prod(dim=1)
    covered = ((reference >= lower) & (reference <= upper)).all(dim=1)
    error = torch.sqrt(
        torch.sum((rollout.best_theta[:, index] - reference).square(), dim=1)
    )
    loss = rollout.best_loss[:, index]
    return {
        "stage": "network1_horizon",
        "horizon": horizon,
        "test_trajectories": reference.shape[0],
        "coverage": float(covered.to(DTYPE).mean().item()),
        "covered_count": int(covered.sum().item()),
        "median_side_a": float(torch.median(sides[:, 0]).item()),
        "mean_side_a": float(sides[:, 0].mean().item()),
        "worst_side_a": float(sides[:, 0].max().item()),
        "median_side_b": float(torch.median(sides[:, 1]).item()),
        "mean_side_b": float(sides[:, 1].mean().item()),
        "worst_side_b": float(sides[:, 1].max().item()),
        "median_area": float(torch.median(area).item()),
        "mean_area": float(area.mean().item()),
        "worst_area": float(area.max().item()),
        "median_best_parameter_error_posthoc": float(torch.median(error).item()),
        "worst_best_parameter_error_posthoc": float(error.max().item()),
        "median_best_prefix_loss": float(torch.median(loss).item()),
        "worst_best_prefix_loss": float(loss.max().item()),
    }


def _select_network1_stage(
    metrics: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    selected = metrics[0]
    decisions: list[dict[str, Any]] = []
    for candidate in metrics[1:]:
        coverage_not_worse = float(candidate["coverage"]) >= float(
            selected["coverage"]
        )
        area_tighter = float(candidate["median_area"]) < float(
            selected["median_area"]
        ) and float(candidate["mean_area"]) < float(selected["mean_area"])
        accepted = coverage_not_worse and area_tighter
        decisions.append(
            {
                "candidate_horizon": int(candidate["horizon"]),
                "compared_with_horizon": int(selected["horizon"]),
                "coverage_not_worse": coverage_not_worse,
                "median_and_mean_area_tighter": area_tighter,
                "accepted": accepted,
            }
        )
        if accepted:
            selected = candidate
    return int(selected["horizon"]), {
        "selected_horizon": int(selected["horizon"]),
        "decisions": decisions,
    }


@torch.no_grad()
def _cache_network1_regions(
    state: dict[str, torch.Tensor],
    horizon: int,
    train_positions: torch.Tensor,
    validation_positions: torch.Tensor,
    test_positions: torch.Tensor,
) -> None:
    model = RectanglePolicy().to(dtype=DTYPE)
    model.load_state_dict(state)
    cache: dict[str, Any] = {
        "format_version": 1,
        "horizon": horizon,
        "network1_checkpoint": str(
            OUTPUT_DIR / f"network1_h{horizon}_best_checkpoint.pt"
        ),
    }
    for name, positions in (
        ("train", train_positions),
        ("validation", validation_positions),
        ("test", test_positions),
    ):
        rollout = _evaluate_network1(model, positions, horizon)
        cache[name] = {
            "lower": rollout.candidate_lower[:, -1].cpu(),
            "upper": rollout.candidate_upper[:, -1].cpu(),
            "best_theta": rollout.best_theta[:, -1].cpu(),
            "best_loss": rollout.best_loss[:, -1].cpu(),
        }
    torch.save(cache, OUTPUT_DIR / "network1_cached_regions.pt")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _train_hidden = data_source._make_exact_labeled_split(
        TRAIN_N, TRAIN_DATA_SEED
    )
    validation_positions, _, _validation_hidden = data_source._make_exact_labeled_split(
        VALIDATION_N, VALIDATION_DATA_SEED
    )
    test_positions, _, test_hidden = data_source._make_exact_labeled_split(
        TEST_N, TEST_DATA_SEED
    )
    del _train_hidden, _validation_hidden
    print("training 2D rectangle-search Network 1", flush=True)
    stage_states, training_history = _train_network1(
        train_positions, validation_positions
    )

    # Generating coefficients are exposed only after every stage checkpoint and
    # test search is frozen; they do not affect controller training or queries.
    reference = _posthoc_reference(test_hidden)
    stage_rollouts: dict[int, RectangleRollout] = {}
    summary_rows: list[dict[str, Any]] = []
    for horizon in NETWORK1_HORIZONS:
        model = RectanglePolicy().to(dtype=DTYPE)
        model.load_state_dict(stage_states[horizon])
        rollout = _evaluate_network1(model, test_positions, horizon)
        stage_rollouts[horizon] = rollout
        summary_rows.append(_network1_metrics(rollout, reference, horizon))
    selected_horizon, decision = _select_network1_stage(summary_rows)
    selected_metrics = next(
        row for row in summary_rows if int(row["horizon"]) == selected_horizon
    )
    summary_rows.append(
        {
            "stage": "network1_selection",
            "selected_horizon": selected_horizon,
            "selected_coverage": selected_metrics["coverage"],
            "selected_median_area": selected_metrics["median_area"],
            "selected_mean_area": selected_metrics["mean_area"],
            "selection_decisions": str(decision["decisions"]),
            "network2_unlocked": float(selected_metrics["coverage"]) == 1.0,
            "runtime_seconds": time.perf_counter() - start,
        }
    )
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "training_history.csv", training_history)
    if float(selected_metrics["coverage"]) < 1.0:
        print(
            {
                "stage": "network1",
                "status": "failed_coverage_gate",
                "selected_horizon": selected_horizon,
                "coverage": selected_metrics["coverage"],
                "output_dir": str(OUTPUT_DIR),
            },
            flush=True,
        )
        return
    _cache_network1_regions(
        stage_states[selected_horizon],
        selected_horizon,
        train_positions,
        validation_positions,
        test_positions,
    )
    print(
        {
            "stage": "network1",
            "status": "passed_coverage_gate",
            "selected_horizon": selected_horizon,
            "coverage": selected_metrics["coverage"],
            "median_area": selected_metrics["median_area"],
            "output_dir": str(OUTPUT_DIR),
            "next": "network2_would_run_only_after_coverage_gate",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
