"""Region-based learned global search for continuous trajectory dynamics."""

from __future__ import annotations

import copy
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

import experiments.sho.run_stage1b_learned_black_box_search as previous
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as direct32
from experiments.sho.run_stage1b_continuous_c_search_comparison import _losses
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    os.environ.get(
        "REGION_SEARCH_OUTPUT_DIR",
        "outputs/sho/stage1B/region_learned_search",
    )
)
TRAIN_N = 4_096
VALIDATION_N = 512
TEST_N = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
CURRICULUM_HORIZONS = (5, 10, 15)
MAX_UPDATES_PER_STAGE = int(os.environ.get("REGION_SEARCH_MAX_UPDATES", "1000"))
EVAL_INTERVAL = int(os.environ.get("REGION_SEARCH_EVAL_INTERVAL", "50"))
PLATEAU_PATIENCE = int(os.environ.get("REGION_SEARCH_PATIENCE", "300"))
DISCOUNT = 0.97
NOVELTY_WEIGHT = 0.02
ENTROPY_WEIGHT = 0.01
LOG_EPSILON = 1e-30
LOSS_THRESHOLDS = (1e-1, 1e-2, 1e-3, 1e-4)
C_ERROR_THRESHOLDS = (1e-1, 1e-2, 1e-3)
MODEL_SEED = SEED + 1_960_000
DOMAIN_WIDTH = previous.UPPER - previous.LOWER


class RegionPolicy(nn.Module):
    """Permutation-equivariant region scorer with pooled global context."""

    def __init__(self) -> None:
        super().__init__()
        self.region_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, region_features: torch.Tensor) -> torch.Tensor:
        embedded = self.region_encoder(region_features)
        mean_context = embedded.mean(dim=1, keepdim=True)
        max_context = embedded.max(dim=1, keepdim=True).values
        context = torch.cat((mean_context, max_context), dim=2).expand(
            -1, embedded.shape[1], -1
        )
        return self.score_head(torch.cat((embedded, context), dim=2)).squeeze(-1)


@dataclass
class Rollout:
    query_c: torch.Tensor
    query_loss: torch.Tensor
    best_c: torch.Tensor
    best_loss: torch.Tensor
    candidate_lower: torch.Tensor
    candidate_upper: torch.Tensor
    selected_width: torch.Tensor
    selected_evaluation_count: torch.Tensor
    selected_underexplored: torch.Tensor
    selected_large: torch.Tensor
    entropy: torch.Tensor
    log_probabilities: list[torch.Tensor]
    rewards: list[torch.Tensor]


def _region_statistics(
    region_lower: torch.Tensor,
    region_upper: torch.Tensor,
    observed_c: torch.Tensor,
    observed_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = 0.5 * (region_lower + region_upper)
    widths = region_upper - region_lower
    if observed_c.shape[1] == 0:
        features = torch.stack(
            (
                centers / previous.UPPER,
                widths / DOMAIN_WIDTH,
                torch.zeros_like(centers),
                torch.zeros_like(centers),
                0.5 * widths / DOMAIN_WIDTH,
                torch.ones_like(centers),
            ),
            dim=2,
        )
        return features, torch.zeros_like(centers, dtype=torch.long)

    inside = (
        (observed_c[:, None, :] >= region_lower[:, :, None] - 1e-14)
        & (observed_c[:, None, :] <= region_upper[:, :, None] + 1e-14)
    )
    counts = inside.sum(dim=2)
    masked_loss = torch.where(
        inside,
        observed_loss[:, None, :],
        torch.full_like(observed_loss[:, None, :], float("inf")),
    )
    best_inside = masked_loss.min(dim=2).values
    global_best = observed_loss.min(dim=1).values[:, None]
    best_inside = torch.where(torch.isfinite(best_inside), best_inside, global_best)
    distances = torch.abs(observed_c[:, None, :] - centers[:, :, None])
    nearest = distances.min(dim=2).values
    empty = (counts == 0).to(DTYPE)
    features = torch.stack(
        (
            centers / previous.UPPER,
            widths / DOMAIN_WIDTH,
            torch.log(best_inside + LOG_EPSILON) / 10.0,
            torch.log1p(counts.to(DTYPE)) / math.log(16.0),
            nearest / DOMAIN_WIDTH,
            empty,
        ),
        dim=2,
    )
    return features, counts


def _split_regions(
    region_lower: torch.Tensor,
    region_upper: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, region_count = region_lower.shape
    next_lower = torch.empty(
        batch_size, region_count + 1, dtype=DTYPE, device=region_lower.device
    )
    next_upper = torch.empty_like(next_lower)
    for trajectory in range(batch_size):
        action = int(actions[trajectory].item())
        center = 0.5 * (
            region_lower[trajectory, action] + region_upper[trajectory, action]
        )
        next_lower[trajectory, :action] = region_lower[trajectory, :action]
        next_upper[trajectory, :action] = region_upper[trajectory, :action]
        next_lower[trajectory, action] = region_lower[trajectory, action]
        next_upper[trajectory, action] = center
        next_lower[trajectory, action + 1] = center
        next_upper[trajectory, action + 1] = region_upper[trajectory, action]
        next_lower[trajectory, action + 2 :] = region_lower[
            trajectory, action + 1 :
        ]
        next_upper[trajectory, action + 2 :] = region_upper[
            trajectory, action + 1 :
        ]
    return next_lower, next_upper


def _candidate_hull(
    region_lower: torch.Tensor,
    region_upper: torch.Tensor,
    best_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    touching = (
        (best_c[:, None] >= region_lower - 1e-14)
        & (best_c[:, None] <= region_upper + 1e-14)
    )
    lower = torch.where(
        touching,
        region_lower,
        torch.full_like(region_lower, float("inf")),
    ).min(dim=1).values
    upper = torch.where(
        touching,
        region_upper,
        torch.full_like(region_upper, -float("inf")),
    ).max(dim=1).values
    if not bool(torch.all(torch.isfinite(lower) & torch.isfinite(upper))):
        raise RuntimeError("Best sampled point is not covered by active regions")
    return lower, upper


def _rollout(
    model: RegionPolicy,
    positions: torch.Tensor,
    horizon: int,
    greedy: bool,
) -> Rollout:
    batch_size = positions.shape[0]
    region_lower = torch.full((batch_size, 1), previous.LOWER, dtype=DTYPE)
    region_upper = torch.full((batch_size, 1), previous.UPPER, dtype=DTYPE)
    observed_c = torch.empty(batch_size, 0, dtype=DTYPE)
    observed_loss = torch.empty(batch_size, 0, dtype=DTYPE)
    query_c_history: list[torch.Tensor] = []
    query_loss_history: list[torch.Tensor] = []
    best_c_history: list[torch.Tensor] = []
    best_loss_history: list[torch.Tensor] = []
    candidate_lower_history: list[torch.Tensor] = []
    candidate_upper_history: list[torch.Tensor] = []
    selected_width_history: list[torch.Tensor] = []
    selected_count_history: list[torch.Tensor] = []
    underexplored_history: list[torch.Tensor] = []
    large_history: list[torch.Tensor] = []
    entropy_history: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for _ in range(horizon):
        features, counts = _region_statistics(
            region_lower, region_upper, observed_c, observed_loss
        )
        logits = model(features)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=1) if greedy else distribution.sample()
        selected_lower = region_lower.gather(1, actions[:, None]).squeeze(1)
        selected_upper = region_upper.gather(1, actions[:, None]).squeeze(1)
        selected_width = selected_upper - selected_lower
        selected_count = counts.gather(1, actions[:, None]).squeeze(1)
        minimum_count = counts.min(dim=1).values
        median_width = region_upper.sub(region_lower).median(dim=1).values
        underexplored = selected_count == minimum_count
        selected_large = selected_width >= median_width - 1e-14
        query_c = 0.5 * (selected_lower + selected_upper)
        with torch.no_grad():
            query_loss = _losses(query_c, positions)
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
            novelty = 0.5 * selected_width / DOMAIN_WIDTH + 0.5 / (
                selected_count.to(DTYPE) + 1.0
            )
            reward = improvement + NOVELTY_WEIGHT * novelty

        observed_c = torch.cat((observed_c, query_c.detach()[:, None]), dim=1)
        observed_loss = torch.cat(
            (observed_loss, query_loss.detach()[:, None]), dim=1
        )
        region_lower, region_upper = _split_regions(
            region_lower, region_upper, actions
        )
        best_indices = torch.argmin(observed_loss, dim=1)
        best_c = observed_c.gather(1, best_indices[:, None]).squeeze(1)
        best_loss = observed_loss.gather(1, best_indices[:, None]).squeeze(1)
        candidate_lower, candidate_upper = _candidate_hull(
            region_lower, region_upper, best_c
        )

        query_c_history.append(query_c.detach())
        query_loss_history.append(query_loss.detach())
        best_c_history.append(best_c)
        best_loss_history.append(best_loss)
        candidate_lower_history.append(candidate_lower)
        candidate_upper_history.append(candidate_upper)
        selected_width_history.append(selected_width)
        selected_count_history.append(selected_count)
        underexplored_history.append(underexplored)
        large_history.append(selected_large)
        entropy_history.append(distribution.entropy())
        log_probabilities.append(distribution.log_prob(actions))
        rewards.append(reward)

    return Rollout(
        query_c=torch.stack(query_c_history, dim=1),
        query_loss=torch.stack(query_loss_history, dim=1),
        best_c=torch.stack(best_c_history, dim=1),
        best_loss=torch.stack(best_loss_history, dim=1),
        candidate_lower=torch.stack(candidate_lower_history, dim=1),
        candidate_upper=torch.stack(candidate_upper_history, dim=1),
        selected_width=torch.stack(selected_width_history, dim=1),
        selected_evaluation_count=torch.stack(selected_count_history, dim=1),
        selected_underexplored=torch.stack(underexplored_history, dim=1),
        selected_large=torch.stack(large_history, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        log_probabilities=log_probabilities,
        rewards=rewards,
    )


def _policy_loss(rollout: Rollout) -> torch.Tensor:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rollout.rewards[0])
    for reward in reversed(rollout.rewards):
        running = reward + DISCOUNT * running
        returns.append(running)
    returns.reverse()
    terms = []
    entropy_terms = []
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
def _evaluate_rollout(
    model: RegionPolicy, positions: torch.Tensor, horizon: int
) -> Rollout:
    model.eval()
    parts: list[Rollout] = []
    for start in range(0, positions.shape[0], BATCH_SIZE):
        parts.append(
            _rollout(model, positions[start : start + BATCH_SIZE], horizon, True)
        )
    tensor_fields = [
        "query_c",
        "query_loss",
        "best_c",
        "best_loss",
        "candidate_lower",
        "candidate_upper",
        "selected_width",
        "selected_evaluation_count",
        "selected_underexplored",
        "selected_large",
        "entropy",
    ]
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in tensor_fields
    }
    return Rollout(
        **combined,
        log_probabilities=[],
        rewards=[],
    )


def _validation_score(rollout: Rollout) -> float:
    return float(torch.log(rollout.best_loss[:, -1] + LOG_EPSILON).mean().item())


def _train_curriculum(
    train_positions: torch.Tensor, validation_positions: torch.Tensor
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    torch.manual_seed(MODEL_SEED)
    model = RegionPolicy().to(dtype=DTYPE)
    batch_generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    stage_states: dict[int, dict[str, torch.Tensor]] = {}
    stage_history: list[dict[str, Any]] = []
    total_update = 0

    for horizon in CURRICULUM_HORIZONS:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        initial_validation = _evaluate_rollout(
            model, validation_positions, horizon
        )
        best_score = _validation_score(initial_validation)
        best_state = copy.deepcopy(model.state_dict())
        best_update = 0
        stale_updates = 0
        stage_start = time.perf_counter()
        print(
            f"stage horizon={horizon} update=0000 "
            f"validation mean log best loss={best_score:.6e}",
            flush=True,
        )
        stop_reason = "maximum_updates"
        updates_run = 0
        for update in range(1, MAX_UPDATES_PER_STAGE + 1):
            model.train()
            indices = torch.randint(
                0,
                train_positions.shape[0],
                (BATCH_SIZE,),
                generator=batch_generator,
            )
            rollout = _rollout(model, train_positions[indices], horizon, False)
            loss = _policy_loss(rollout)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_update += 1
            updates_run = update

            if update % EVAL_INTERVAL == 0 or update == MAX_UPDATES_PER_STAGE:
                validation = _evaluate_rollout(
                    model, validation_positions, horizon
                )
                score = _validation_score(validation)
                if score < best_score - 1e-8:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    best_update = update
                    stale_updates = 0
                else:
                    stale_updates += EVAL_INTERVAL
                print(
                    f"stage horizon={horizon} update={update:04d} "
                    f"policy loss={float(loss.detach().item()):.6e} "
                    f"validation mean log best loss={score:.6e} "
                    f"best_update={best_update} stale={stale_updates}",
                    flush=True,
                )
                if stale_updates >= PLATEAU_PATIENCE:
                    stop_reason = "validation_plateau"
                    break

        model.load_state_dict(best_state)
        stage_states[horizon] = copy.deepcopy(best_state)
        stage_history.append(
            {
                "horizon": horizon,
                "updates_run": updates_run,
                "best_update": best_update,
                "best_validation_mean_log_best_loss": best_score,
                "stop_reason": stop_reason,
                "runtime_seconds": time.perf_counter() - stage_start,
                "cumulative_training_updates": total_update,
            }
        )
    return stage_states, stage_history


def _threshold_first_handoffs(
    rollout: Rollout, c_star: torch.Tensor, through_query: int
) -> dict[float, dict[str, torch.Tensor]]:
    output: dict[float, dict[str, torch.Tensor]] = {}
    for threshold in LOSS_THRESHOLDS:
        reached_history = rollout.best_loss[:, :through_query] < threshold
        reached = reached_history.any(dim=1)
        first_index = torch.argmax(reached_history.to(torch.long), dim=1)
        gather_index = first_index[:, None]
        lower = rollout.candidate_lower[:, :through_query].gather(
            1, gather_index
        ).squeeze(1)
        upper = rollout.candidate_upper[:, :through_query].gather(
            1, gather_index
        ).squeeze(1)
        output[threshold] = {
            "reached": reached,
            "first_index": first_index,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "covered": (c_star >= lower) & (c_star <= upper),
        }
    return output


def _per_query_rows(
    rollout: Rollout, c_star: torch.Tensor
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_number in range(1, rollout.best_loss.shape[1] + 1):
        handoffs = _threshold_first_handoffs(rollout, c_star, query_number)
        index = query_number - 1
        for trajectory in range(c_star.shape[0]):
            reference = float(c_star[trajectory].item())
            best_c = float(rollout.best_c[trajectory, index].item())
            error = abs(best_c - reference)
            lower = float(rollout.candidate_lower[trajectory, index].item())
            upper = float(rollout.candidate_upper[trajectory, index].item())
            row: dict[str, Any] = {
                "trajectory_index": trajectory,
                "adaptive_query_number": query_number,
                "reference_c_star_posthoc": reference,
                "queried_c": float(rollout.query_c[trajectory, index].item()),
                "queried_prefix_loss": float(
                    rollout.query_loss[trajectory, index].item()
                ),
                "best_observed_c": best_c,
                "absolute_best_c_error": error,
                "best_observed_prefix_loss": float(
                    rollout.best_loss[trajectory, index].item()
                ),
                "candidate_region_lower": lower,
                "candidate_region_upper": upper,
                "candidate_region_width": upper - lower,
                "candidate_region_contains_c_star": lower
                <= reference
                <= upper,
                "selected_region_width": float(
                    rollout.selected_width[trajectory, index].item()
                ),
                "selected_region_evaluation_count": int(
                    rollout.selected_evaluation_count[trajectory, index].item()
                ),
                "selected_underexplored_region": bool(
                    rollout.selected_underexplored[trajectory, index].item()
                ),
                "selected_large_region": bool(
                    rollout.selected_large[trajectory, index].item()
                ),
                "policy_entropy": float(rollout.entropy[trajectory, index].item()),
            }
            for threshold in C_ERROR_THRESHOLDS:
                exponent = int(round(-math.log10(threshold)))
                row[f"best_c_within_1e_{exponent}"] = error <= threshold
            for threshold, handoff in handoffs.items():
                exponent = int(round(-math.log10(threshold)))
                reached = bool(handoff["reached"][trajectory].item())
                first_query = int(handoff["first_index"][trajectory].item()) + 1
                row[f"loss_below_1e_{exponent}_reached"] = reached
                row[f"loss_below_1e_{exponent}_first_query"] = (
                    first_query if reached else ""
                )
                row[f"loss_below_1e_{exponent}_first_handoff_width"] = (
                    float(handoff["width"][trajectory].item()) if reached else ""
                )
                row[f"loss_below_1e_{exponent}_first_handoff_covered"] = (
                    bool(handoff["covered"][trajectory].item()) if reached else ""
                )
            rows.append(row)
    return rows


def _metrics(
    rollout: Rollout,
    c_star: torch.Tensor,
    query_number: int,
) -> dict[str, Any]:
    index = query_number - 1
    errors = torch.abs(rollout.best_c[:, index] - c_star)
    best_loss = rollout.best_loss[:, index]
    widths = rollout.candidate_upper[:, index] - rollout.candidate_lower[:, index]
    covered = (c_star >= rollout.candidate_lower[:, index]) & (
        c_star <= rollout.candidate_upper[:, index]
    )
    handoffs = _threshold_first_handoffs(rollout, c_star, query_number)
    metrics: dict[str, Any] = {
        "adaptive_query_number": query_number,
        "test_trajectories": c_star.shape[0],
        "median_best_c_error": float(torch.median(errors).item()),
        "worst_best_c_error": float(errors.max().item()),
        "median_best_observed_prefix_loss": float(torch.median(best_loss).item()),
        "worst_best_observed_prefix_loss": float(best_loss.max().item()),
        "median_candidate_region_width": float(torch.median(widths).item()),
        "mean_candidate_region_width": float(widths.mean().item()),
        "worst_candidate_region_width": float(widths.max().item()),
        "candidate_region_coverage": float(covered.to(DTYPE).mean().item()),
        "selected_underexplored_fraction": float(
            rollout.selected_underexplored[:, index].to(DTYPE).mean().item()
        ),
        "cumulative_underexplored_fraction": float(
            rollout.selected_underexplored[:, :query_number]
            .to(DTYPE)
            .mean()
            .item()
        ),
        "selected_large_region_fraction": float(
            rollout.selected_large[:, index].to(DTYPE).mean().item()
        ),
        "median_policy_entropy": float(
            torch.median(rollout.entropy[:, index]).item()
        ),
    }
    for threshold in C_ERROR_THRESHOLDS:
        exponent = int(round(-math.log10(threshold)))
        metrics[f"fraction_best_c_within_1e_{exponent}"] = float(
            (errors <= threshold).to(DTYPE).mean().item()
        )
    for threshold, handoff in handoffs.items():
        exponent = int(round(-math.log10(threshold)))
        reached = handoff["reached"]
        metrics[f"fraction_reaching_loss_below_1e_{exponent}"] = float(
            reached.to(DTYPE).mean().item()
        )
        if bool(reached.any()):
            metrics[f"median_first_handoff_width_loss_1e_{exponent}"] = float(
                torch.median(handoff["width"][reached]).item()
            )
            metrics[f"first_handoff_coverage_loss_1e_{exponent}"] = float(
                handoff["covered"][reached].to(DTYPE).mean().item()
            )
        else:
            metrics[f"median_first_handoff_width_loss_1e_{exponent}"] = ""
            metrics[f"first_handoff_coverage_loss_1e_{exponent}"] = ""
    return metrics


def _summary_rows(
    final_rollout: Rollout,
    stage_rollouts: dict[int, Rollout],
    c_star: torch.Tensor,
    stage_history: list[dict[str, Any]],
    total_runtime_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history_by_horizon = {int(row["horizon"]): row for row in stage_history}
    for query_number in range(1, CURRICULUM_HORIZONS[-1] + 1):
        row = {
            "row_type": "query_progress",
            "model_training_horizon": CURRICULUM_HORIZONS[-1],
            **_metrics(final_rollout, c_star, query_number),
            "training_updates_run": history_by_horizon[15]["updates_run"],
            "best_training_update": history_by_horizon[15]["best_update"],
            "stage_stop_reason": history_by_horizon[15]["stop_reason"],
            "total_runtime_seconds": total_runtime_seconds,
        }
        rows.append(row)
    for horizon in CURRICULUM_HORIZONS:
        history = history_by_horizon[horizon]
        rows.append(
            {
                "row_type": "curriculum_stage",
                "model_training_horizon": horizon,
                **_metrics(stage_rollouts[horizon], c_star, horizon),
                "training_updates_run": history["updates_run"],
                "best_training_update": history["best_update"],
                "best_validation_mean_log_best_loss": history[
                    "best_validation_mean_log_best_loss"
                ],
                "stage_stop_reason": history["stop_reason"],
                "stage_runtime_seconds": history["runtime_seconds"],
                "cumulative_training_updates": history[
                    "cumulative_training_updates"
                ],
                "total_runtime_seconds": total_runtime_seconds,
            }
        )
    return rows


def _plot(summary_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in summary_rows if row["row_type"] == "query_progress"]
    queries = [int(row["adaptive_query_number"]) for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(
        queries,
        [float(row["median_best_c_error"]) for row in rows],
        marker="o",
        label="median c error",
    )
    axes[0].plot(
        queries,
        [float(row["worst_best_c_error"]) for row in rows],
        marker="s",
        label="worst c error",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("best observed |c-c*|")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        queries,
        [float(row["median_candidate_region_width"]) for row in rows],
        marker="o",
        label="median handoff width",
    )
    second = axes[1].twinx()
    second.plot(
        queries,
        [float(row["candidate_region_coverage"]) for row in rows],
        color="black",
        marker="s",
        label="handoff coverage",
    )
    axes[1].set_ylabel("candidate region width")
    second.set_ylabel("coverage")
    second.set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = second.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b)
    for threshold in LOSS_THRESHOLDS:
        exponent = int(round(-math.log10(threshold)))
        axes[2].plot(
            queries,
            [float(row[f"fraction_reaching_loss_below_1e_{exponent}"]) for row in rows],
            marker="o",
            label=f"best loss < {threshold:g}",
        )
    axes[2].plot(
        queries,
        [float(row["selected_underexplored_fraction"]) for row in rows],
        color="black",
        linestyle="--",
        marker=".",
        label="selected underexplored",
    )
    axes[2].set_xlabel("region-selection queries")
    axes[2].set_ylabel("test fraction")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xticks(queries)
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "search_progress_vs_queries.png", dpi=180)
    plt.close(figure)


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = direct32._make_exact_labeled_split(
        TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions, _, _ = direct32._make_exact_labeled_split(
        VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels, test_hidden_c = direct32._make_exact_labeled_split(
        TEST_N, TEST_DATA_SEED + 1_810_300
    )
    print("training on-policy region-selection curriculum", flush=True)
    stage_states, stage_history = _train_curriculum(
        train_positions, validation_positions
    )

    # Freeze every stage policy and its test search history before loading c*.
    stage_rollouts: dict[int, Rollout] = {}
    for horizon in CURRICULUM_HORIZONS:
        model = RegionPolicy().to(dtype=DTYPE)
        model.load_state_dict(stage_states[horizon])
        stage_rollouts[horizon] = _evaluate_rollout(model, test_positions, horizon)
    final_rollout = stage_rollouts[CURRICULUM_HORIZONS[-1]]

    direct32._verify_test_labels(test_labels)
    test_c_star = previous._stored_test_c_star()
    if float(torch.max(torch.abs(test_hidden_c - test_c_star)).item()) > 1e-12:
        raise RuntimeError("Stored test c* does not match deterministic test data")
    result_rows = _per_query_rows(final_rollout, test_c_star)
    total_runtime_seconds = time.perf_counter() - start_time
    summary_rows = _summary_rows(
        final_rollout,
        stage_rollouts,
        test_c_star,
        stage_history,
        total_runtime_seconds,
    )
    _write_csv(OUTPUT_DIR / "region_search_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_query_results.csv", result_rows)
    _plot(summary_rows)
    stage_summary = {
        horizon: {
            "updates": next(
                row["updates_run"]
                for row in stage_history
                if row["horizon"] == horizon
            ),
            "median_c_error": _metrics(
                stage_rollouts[horizon], test_c_star, horizon
            )["median_best_c_error"],
            "candidate_coverage": _metrics(
                stage_rollouts[horizon], test_c_star, horizon
            )["candidate_region_coverage"],
        }
        for horizon in CURRICULUM_HORIZONS
    }
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "curriculum": stage_summary,
            "total_runtime_seconds": total_runtime_seconds,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
