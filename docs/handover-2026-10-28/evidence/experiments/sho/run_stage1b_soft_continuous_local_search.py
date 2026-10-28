"""Soft learned local search inside Network 1's saved safe intervals."""

from __future__ import annotations

import copy
import csv
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

import experiments.sho.run_stage1b_learned_black_box_search as black_box
import experiments.sho.run_stage1b_learned_local_refinement as prior_local
import experiments.sho.run_stage1b_region_learned_search as network1
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
        "SOFT_LOCAL_SEARCH_OUTPUT_DIR",
        "outputs/sho/stage1B/soft_continuous_local_search",
    )
)
SAVED_NETWORK1_TEST_INTERVALS = Path(
    "outputs/sho/stage1B/learned_local_refinement/per_query_results.csv"
)
TRAIN_N = 4_096
VALIDATION_N = 512
TEST_N = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MIXTURE_COMPONENTS = 3
FIXED_U = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE)
INITIAL_EVALUATIONS = len(FIXED_U)
EMERGENCY_EVALUATION_CAP = int(
    os.environ.get("SOFT_LOCAL_SEARCH_EVALUATION_CAP", "200")
)
MAX_LEARNED_QUERIES = EMERGENCY_EVALUATION_CAP - INITIAL_EVALUATIONS
TRAINING_HORIZONS = tuple(
    int(value)
    for value in os.environ.get(
        "SOFT_LOCAL_SEARCH_HORIZONS", "10,30,60,100"
    ).split(",")
)
DEFAULT_STAGE_UPDATES = (400, 400, 300, 200)
STAGE_UPDATES = tuple(
    int(value)
    for value in os.environ.get(
        "SOFT_LOCAL_SEARCH_STAGE_UPDATES",
        ",".join(str(value) for value in DEFAULT_STAGE_UPDATES),
    ).split(",")
)
if len(TRAINING_HORIZONS) != len(STAGE_UPDATES):
    raise ValueError("One update budget is required for each training horizon")
EVAL_INTERVAL = int(os.environ.get("SOFT_LOCAL_SEARCH_EVAL_INTERVAL", "50"))
PLATEAU_PATIENCE = int(os.environ.get("SOFT_LOCAL_SEARCH_PATIENCE", "150"))
LOG_EPSILON = 1e-30
SUCCESS_LOSS = 1e-20
LOSS_THRESHOLDS = (1e-4, 1e-8, 1e-12, 1e-16, 1e-20)
NOVELTY_WEIGHT = 2e-3
ENTROPY_WEIGHT = 1e-4
EVALUATION_COST = 1e-4
GUMBEL_TEMPERATURE = 0.75
MODEL_SEED = SEED + 2_010_000
EVALUATION_SEED = SEED + 2_020_000


class SoftGaussianMixtureController(nn.Module):
    """DeepSets controller producing a soft mixture on normalized [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 2 + 4, 64),
            nn.SiLU(),
            nn.Linear(64, MIXTURE_COMPONENTS * 3),
        )
        with torch.no_grad():
            final = self.head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Unexpected controller head")
            final.bias.zero_()
            mean_biases = torch.tensor(
                [
                    math.log(0.25 / 0.75),
                    0.0,
                    math.log(0.75 / 0.25),
                ],
                dtype=DTYPE,
            )
            final.bias[MIXTURE_COMPONENTS : 2 * MIXTURE_COMPONENTS] = (
                mean_biases
            )
            final.bias[2 * MIXTURE_COMPONENTS :] = -2.0

    def encode_pairs(
        self, sampled_u: torch.Tensor, sampled_loss: torch.Tensor
    ) -> torch.Tensor:
        pairs = torch.stack(
            (sampled_u, torch.log(sampled_loss + LOG_EPSILON) / 10.0),
            dim=1,
        )
        return self.element(pairs)

    def distribution_parameters(
        self,
        embedding_sum: torch.Tensor,
        embedding_max: torch.Tensor,
        valid_count: torch.Tensor,
        best_u: torch.Tensor,
        best_loss: torch.Tensor,
        sampled_u_sum: torch.Tensor,
        sampled_u_square_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_embedding = embedding_sum / valid_count[:, None]
        mean_u = sampled_u_sum / valid_count
        variance_u = torch.clamp(
            sampled_u_square_sum / valid_count - mean_u.square(), min=0.0
        )
        global_features = torch.stack(
            (
                best_u,
                torch.log(best_loss + LOG_EPSILON) / 10.0,
                torch.sqrt(variance_u + 1e-20),
                valid_count / float(EMERGENCY_EVALUATION_CAP),
            ),
            dim=1,
        )
        raw = self.head(
            torch.cat((mean_embedding, embedding_max, global_features), dim=1)
        )
        logits, raw_means, raw_scales = torch.split(
            raw, MIXTURE_COMPONENTS, dim=1
        )
        means = torch.sigmoid(raw_means)
        # This smooth parameterization spans useful exploratory scales down to
        # approximately the float64 local-polishing regime.
        log_scales = -2.0 - 26.0 * torch.sigmoid(raw_scales)
        scales = torch.exp(log_scales)
        return logits, means, scales


@dataclass
class SearchRollout:
    best_c: torch.Tensor
    best_loss: torch.Tensor
    best_loss_history: torch.Tensor
    query_c: torch.Tensor
    query_mask: torch.Tensor
    selected_component: torch.Tensor
    categorical_entropy: torch.Tensor
    entropy_upper_bound: torch.Tensor
    proposal_spread: torch.Tensor
    effective_modes: torch.Tensor
    active_weight_modes: torch.Tensor
    first_threshold_evaluation: dict[float, torch.Tensor]
    success: torch.Tensor
    evaluations_consumed: torch.Tensor
    training_objective: torch.Tensor | None


def _make_positions_only_split(count: int, seed: int) -> torch.Tensor:
    positions, _, _hidden_c = direct32._make_exact_labeled_split(count, seed)
    del _hidden_c
    return positions


def _load_saved_test_intervals() -> tuple[torch.Tensor, torch.Tensor]:
    if not SAVED_NETWORK1_TEST_INTERVALS.exists():
        raise FileNotFoundError(
            f"Missing saved Network 1 intervals: {SAVED_NETWORK1_TEST_INTERVALS}"
        )
    by_index: dict[int, tuple[float, float]] = {}
    with SAVED_NETWORK1_TEST_INTERVALS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["trajectory_index"])
            value = (
                float(row["network1_safe_lower"]),
                float(row["network1_safe_upper"]),
            )
            if index in by_index and by_index[index] != value:
                raise RuntimeError("Saved Network 1 interval changed across rows")
            by_index[index] = value
    if sorted(by_index) != list(range(TEST_N)):
        raise RuntimeError("Saved Network 1 test intervals are incomplete")
    lower = torch.tensor([by_index[i][0] for i in range(TEST_N)], dtype=DTYPE)
    upper = torch.tensor([by_index[i][1] for i in range(TEST_N)], dtype=DTYPE)
    return lower, upper


def _network1_intervals(
    state: dict[str, torch.Tensor], positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return prior_local._network1_intervals(state, positions)


@torch.no_grad()
def _fixed_losses(
    positions: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
) -> torch.Tensor:
    return prior_local._fixed_local_losses(
        positions, interval_lower, interval_upper
    )


def _sample_mixture(
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    generator: torch.Generator,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if training:
        uniform = torch.rand(
            logits.shape, dtype=DTYPE, generator=generator
        ).clamp_(1e-12, 1.0 - 1e-12)
        gumbel = -torch.log(-torch.log(uniform))
        soft = torch.softmax((logits + gumbel) / GUMBEL_TEMPERATURE, dim=1)
        hard_indices = torch.argmax(soft, dim=1)
        hard = torch.nn.functional.one_hot(
            hard_indices, MIXTURE_COMPONENTS
        ).to(DTYPE)
        assignment = hard + soft - soft.detach()
    else:
        weights = torch.softmax(logits, dim=1)
        uniform = torch.rand(
            (logits.shape[0], 1), dtype=DTYPE, generator=generator
        )
        hard_indices = torch.sum(
            uniform > torch.cumsum(weights, dim=1), dim=1
        ).clamp(max=MIXTURE_COMPONENTS - 1)
        assignment = torch.nn.functional.one_hot(
            hard_indices, MIXTURE_COMPONENTS
        ).to(DTYPE)
    noise = torch.randn(means.shape, dtype=DTYPE, generator=generator)
    component_samples = means + scales * noise
    query_u = torch.sum(assignment * component_samples, dim=1).clamp(0.0, 1.0)
    return query_u, hard_indices


def _rollout(
    model: SoftGaussianMixtureController,
    positions: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
    fixed_losses: torch.Tensor,
    learned_queries: int,
    generator: torch.Generator,
    training: bool,
) -> SearchRollout:
    batch_size = positions.shape[0]
    interval_width = interval_upper - interval_lower
    fixed_u = FIXED_U[None, :].expand(batch_size, -1)
    best_indices = torch.argmin(fixed_losses, dim=1)
    best_loss = fixed_losses.gather(1, best_indices[:, None]).squeeze(1)
    best_u = fixed_u.gather(1, best_indices[:, None]).squeeze(1)
    best_c = interval_lower + best_u * interval_width
    active = best_loss >= SUCCESS_LOSS

    flat_u = fixed_u.reshape(-1)
    flat_loss = fixed_losses.reshape(-1)
    embeddings = model.encode_pairs(flat_u, flat_loss).reshape(
        batch_size, INITIAL_EVALUATIONS, -1
    )
    embedding_sum = embeddings.sum(dim=1)
    embedding_max = embeddings.max(dim=1).values
    valid_count = torch.full(
        (batch_size,), float(INITIAL_EVALUATIONS), dtype=DTYPE
    )
    sampled_u_sum = fixed_u.sum(dim=1)
    sampled_u_square_sum = fixed_u.square().sum(dim=1)
    sampled_u = fixed_u

    query_c_history: list[torch.Tensor] = []
    query_mask_history: list[torch.Tensor] = []
    component_history: list[torch.Tensor] = []
    categorical_entropy_history: list[torch.Tensor] = []
    entropy_upper_history: list[torch.Tensor] = []
    spread_history: list[torch.Tensor] = []
    effective_modes_history: list[torch.Tensor] = []
    active_modes_history: list[torch.Tensor] = []
    best_loss_history = [best_loss.detach()]
    objective_terms: list[torch.Tensor] = []
    first_threshold_evaluation = {
        threshold: torch.where(
            best_loss.detach() < threshold,
            torch.full((batch_size,), INITIAL_EVALUATIONS, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long),
        )
        for threshold in LOSS_THRESHOLDS
    }
    evaluations_consumed = torch.full(
        (batch_size,), EMERGENCY_EVALUATION_CAP, dtype=torch.long
    )
    evaluations_consumed = torch.where(
        ~active,
        torch.full_like(evaluations_consumed, INITIAL_EVALUATIONS),
        evaluations_consumed,
    )

    for query_index in range(learned_queries):
        logits, means, scales = model.distribution_parameters(
            embedding_sum,
            embedding_max,
            valid_count,
            best_u,
            best_loss,
            sampled_u_sum,
            sampled_u_square_sum,
        )
        weights = torch.softmax(logits, dim=1)
        categorical_entropy = -torch.sum(
            weights * torch.log(weights + LOG_EPSILON), dim=1
        )
        normal_entropy = 0.5 * torch.log(
            2.0 * math.pi * math.e * scales.square()
        )
        entropy_upper = categorical_entropy + torch.sum(
            weights * normal_entropy, dim=1
        )
        mixture_mean = torch.sum(weights * means, dim=1)
        spread_u = torch.sqrt(
            torch.sum(
                weights
                * (scales.square() + (means - mixture_mean[:, None]).square()),
                dim=1,
            )
            + 1e-30
        )
        effective_modes = torch.exp(categorical_entropy)
        active_modes = torch.sum(weights >= 0.1, dim=1).to(DTYPE)
        query_u, selected_component = _sample_mixture(
            logits, means, scales, generator, training
        )
        query_c = interval_lower + query_u * interval_width
        query_loss = _losses(query_c, positions)
        query_mask = active.detach()
        prior_best_loss = best_loss.detach()
        log_improvement = torch.clamp(
            torch.log(prior_best_loss + LOG_EPSILON)
            - torch.log(query_loss + LOG_EPSILON),
            min=0.0,
            max=40.0,
        )
        nearest_distance = torch.min(
            torch.abs(query_u[:, None] - sampled_u), dim=1
        ).values
        if bool(torch.any(query_mask)):
            utility = (
                log_improvement
                + NOVELTY_WEIGHT * nearest_distance
                + ENTROPY_WEIGHT * entropy_upper
                - EVALUATION_COST
            )
            objective_terms.append(-utility[query_mask].mean())

        improved = query_mask & (query_loss.detach() < best_loss.detach())
        best_loss = torch.where(improved, query_loss.detach(), best_loss)
        best_u = torch.where(improved, query_u.detach(), best_u)
        best_c = interval_lower + best_u * interval_width
        total_evaluations = INITIAL_EVALUATIONS + query_index + 1
        for threshold, first_hit in first_threshold_evaluation.items():
            newly_reached = (
                query_mask
                & (first_hit == 0)
                & (best_loss.detach() < threshold)
            )
            first_threshold_evaluation[threshold] = torch.where(
                newly_reached,
                torch.full_like(first_hit, total_evaluations),
                first_hit,
            )
        newly_successful = query_mask & (best_loss.detach() < SUCCESS_LOSS)
        evaluations_consumed = torch.where(
            newly_successful,
            torch.full_like(evaluations_consumed, total_evaluations),
            evaluations_consumed,
        )
        active = query_mask & ~newly_successful

        query_u_detached = query_u.detach()
        query_loss_detached = query_loss.detach()
        new_embedding = model.encode_pairs(query_u_detached, query_loss_detached)
        mask_value = query_mask.to(DTYPE)
        embedding_sum = embedding_sum + new_embedding * mask_value[:, None]
        embedding_max = torch.where(
            query_mask[:, None],
            torch.maximum(embedding_max, new_embedding),
            embedding_max,
        )
        valid_count = valid_count + mask_value
        sampled_u_sum = sampled_u_sum + query_u_detached * mask_value
        sampled_u_square_sum = (
            sampled_u_square_sum + query_u_detached.square() * mask_value
        )
        sampled_u = torch.cat((sampled_u, query_u_detached[:, None]), dim=1)

        query_c_history.append(query_c.detach())
        query_mask_history.append(query_mask)
        component_history.append(selected_component.detach())
        categorical_entropy_history.append(categorical_entropy.detach())
        entropy_upper_history.append(entropy_upper.detach())
        spread_history.append((spread_u * interval_width).detach())
        effective_modes_history.append(effective_modes.detach())
        active_modes_history.append(active_modes.detach())
        best_loss_history.append(best_loss.detach())

    training_objective = (
        torch.stack(objective_terms).mean() if objective_terms else None
    )
    return SearchRollout(
        best_c=best_c.detach(),
        best_loss=best_loss.detach(),
        best_loss_history=torch.stack(best_loss_history, dim=1),
        query_c=torch.stack(query_c_history, dim=1),
        query_mask=torch.stack(query_mask_history, dim=1),
        selected_component=torch.stack(component_history, dim=1),
        categorical_entropy=torch.stack(categorical_entropy_history, dim=1),
        entropy_upper_bound=torch.stack(entropy_upper_history, dim=1),
        proposal_spread=torch.stack(spread_history, dim=1),
        effective_modes=torch.stack(effective_modes_history, dim=1),
        active_weight_modes=torch.stack(active_modes_history, dim=1),
        first_threshold_evaluation=first_threshold_evaluation,
        success=best_loss.detach() < SUCCESS_LOSS,
        evaluations_consumed=evaluations_consumed,
        training_objective=training_objective,
    )


@torch.no_grad()
def _evaluate(
    model: SoftGaussianMixtureController,
    positions: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
    fixed_losses: torch.Tensor,
    learned_queries: int,
    seed: int,
) -> SearchRollout:
    model.eval()
    parts: list[SearchRollout] = []
    for batch_number, start in enumerate(range(0, positions.shape[0], BATCH_SIZE)):
        stop = min(start + BATCH_SIZE, positions.shape[0])
        generator = torch.Generator().manual_seed(seed + batch_number)
        parts.append(
            _rollout(
                model,
                positions[start:stop],
                interval_lower[start:stop],
                interval_upper[start:stop],
                fixed_losses[start:stop],
                learned_queries,
                generator,
                training=False,
            )
        )
    tensor_fields = [
        "best_c",
        "best_loss",
        "best_loss_history",
        "query_c",
        "query_mask",
        "selected_component",
        "categorical_entropy",
        "entropy_upper_bound",
        "proposal_spread",
        "effective_modes",
        "active_weight_modes",
        "success",
        "evaluations_consumed",
    ]
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in tensor_fields
    }
    combined_thresholds = {
        threshold: torch.cat(
            [part.first_threshold_evaluation[threshold] for part in parts]
        )
        for threshold in LOSS_THRESHOLDS
    }
    return SearchRollout(
        **combined,
        first_threshold_evaluation=combined_thresholds,
        training_objective=None,
    )


def _validation_score(rollout: SearchRollout) -> float:
    gate_fraction = float(rollout.success.to(DTYPE).mean().item())
    mean_log_best = float(
        torch.log(rollout.best_loss + LOG_EPSILON).mean().item()
    )
    return -100.0 * gate_fraction + mean_log_best


def _train_controller(
    train_positions: torch.Tensor,
    train_lower: torch.Tensor,
    train_upper: torch.Tensor,
    train_fixed_losses: torch.Tensor,
    validation_positions: torch.Tensor,
    validation_lower: torch.Tensor,
    validation_upper: torch.Tensor,
    validation_fixed_losses: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    torch.manual_seed(MODEL_SEED)
    model = SoftGaussianMixtureController().to(dtype=DTYPE)
    batch_generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    query_generator = torch.Generator().manual_seed(MODEL_SEED + 2)
    history: list[dict[str, Any]] = []
    cumulative_updates = 0
    final_state = copy.deepcopy(model.state_dict())

    for horizon, update_budget in zip(TRAINING_HORIZONS, STAGE_UPDATES):
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        validation = _evaluate(
            model,
            validation_positions,
            validation_lower,
            validation_upper,
            validation_fixed_losses,
            horizon,
            EVALUATION_SEED + horizon,
        )
        best_score = _validation_score(validation)
        best_state = copy.deepcopy(model.state_dict())
        best_update = 0
        stale_updates = 0
        stop_reason = "maximum_updates"
        stage_start = time.perf_counter()
        updates_run = 0
        print(
            f"soft local horizon={horizon} update=0000 "
            f"validation gate={float(validation.success.to(DTYPE).mean().item()):.6f} "
            f"mean log best loss="
            f"{float(torch.log(validation.best_loss + LOG_EPSILON).mean().item()):.6e}",
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
            rollout = _rollout(
                model,
                train_positions[indices],
                train_lower[indices],
                train_upper[indices],
                train_fixed_losses[indices],
                horizon,
                query_generator,
                training=True,
            )
            if rollout.training_objective is None:
                raise RuntimeError("Training rollout produced no objective")
            optimizer.zero_grad()
            rollout.training_objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            cumulative_updates += 1
            updates_run = update

            if update % EVAL_INTERVAL == 0 or update == update_budget:
                validation = _evaluate(
                    model,
                    validation_positions,
                    validation_lower,
                    validation_upper,
                    validation_fixed_losses,
                    horizon,
                    EVALUATION_SEED + horizon,
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
                    f"soft local horizon={horizon} update={update:04d} "
                    f"objective={float(rollout.training_objective.detach().item()):.6e} "
                    f"validation gate="
                    f"{float(validation.success.to(DTYPE).mean().item()):.6f} "
                    f"mean log best loss="
                    f"{float(torch.log(validation.best_loss + LOG_EPSILON).mean().item()):.6e} "
                    f"best_update={best_update} stale={stale_updates}",
                    flush=True,
                )
                if stale_updates >= PLATEAU_PATIENCE:
                    stop_reason = "validation_plateau"
                    break

        model.load_state_dict(best_state)
        final_state = copy.deepcopy(best_state)
        history.append(
            {
                "horizon": horizon,
                "updates_run": updates_run,
                "best_update": best_update,
                "best_validation_score": best_score,
                "best_validation_gate_fraction": float(
                    _evaluate(
                        model,
                        validation_positions,
                        validation_lower,
                        validation_upper,
                        validation_fixed_losses,
                        horizon,
                        EVALUATION_SEED + horizon,
                    ).success.to(DTYPE).mean().item()
                ),
                "stop_reason": stop_reason,
                "runtime_seconds": time.perf_counter() - stage_start,
                "cumulative_updates": cumulative_updates,
            }
        )
    return final_state, history


def _previous_stall_mask() -> torch.Tensor:
    path = prior_local.OUTPUT_DIR / "per_query_results.csv"
    by_index: dict[int, dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["trajectory_index"])
            query = int(row["adaptive_query_number"])
            if query in (1, 15):
                by_index.setdefault(index, {})[query] = float(
                    row["best_observed_prefix_loss"]
                )
    stalled = []
    for index in range(TEST_N):
        values = by_index.get(index, {})
        if set(values) != {1, 15}:
            raise RuntimeError("Previous Network 2 results are incomplete")
        stalled.append(abs(values[1] - values[15]) <= 1e-15)
    return torch.tensor(stalled, dtype=torch.bool)


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> list[float]:
    return [float(value) for value in values[mask].tolist()]


def _group_summary(
    group_name: str,
    group_mask: torch.Tensor,
    rollout: SearchRollout,
    c_star: torch.Tensor,
    strict_one_side: torch.Tensor,
    dominant_one_side: torch.Tensor,
    distinct_modes: torch.Tensor,
) -> dict[str, Any]:
    count = int(group_mask.sum().item())
    success = rollout.success & group_mask
    failure = (~rollout.success) & group_mask
    errors = torch.abs(rollout.best_c - c_star)
    successful_evaluations = _masked_values(rollout.evaluations_consumed, success)
    termination_evaluations = _masked_values(
        rollout.evaluations_consumed, group_mask
    )
    row: dict[str, Any] = {
        "row_type": "group_summary",
        "group": group_name,
        "trajectories": count,
        "fraction_reaching_1e_20": float(
            rollout.success[group_mask].to(DTYPE).mean().item()
        ),
        "successful_trajectories": int(success.sum().item()),
        "emergency_cap_failures": int(failure.sum().item()),
        "median_success_evaluations": statistics.median(successful_evaluations)
        if successful_evaluations
        else "",
        "mean_success_evaluations": statistics.fmean(successful_evaluations)
        if successful_evaluations
        else "",
        "worst_success_evaluations": max(successful_evaluations)
        if successful_evaluations
        else "",
        "median_termination_evaluations": statistics.median(
            termination_evaluations
        ),
        "mean_termination_evaluations": statistics.fmean(
            termination_evaluations
        ),
        "worst_termination_evaluations": max(termination_evaluations),
        "median_final_prefix_loss": float(
            torch.median(rollout.best_loss[group_mask]).item()
        ),
        "worst_final_prefix_loss": float(
            rollout.best_loss[group_mask].max().item()
        ),
        "median_best_c_error_posthoc": float(
            torch.median(errors[group_mask]).item()
        ),
        "worst_best_c_error_posthoc": float(errors[group_mask].max().item()),
        "strict_one_side_fraction_among_failures": float(
            strict_one_side[failure].to(DTYPE).mean().item()
        )
        if bool(torch.any(failure))
        else "",
        "dominant_one_side_fraction_among_failures": float(
            dominant_one_side[failure].to(DTYPE).mean().item()
        )
        if bool(torch.any(failure))
        else "",
        "mean_distinct_selected_modes": float(
            distinct_modes[group_mask].to(DTYPE).mean().item()
        ),
        "fraction_using_multiple_selected_modes": float(
            (distinct_modes[group_mask] >= 2).to(DTYPE).mean().item()
        ),
    }
    valid = rollout.query_mask & group_mask[:, None]
    for name, tensor in (
        ("categorical_entropy", rollout.categorical_entropy),
        ("entropy_upper_bound", rollout.entropy_upper_bound),
        ("proposal_spread", rollout.proposal_spread),
        ("effective_modes", rollout.effective_modes),
        ("active_weight_modes", rollout.active_weight_modes),
    ):
        row[f"mean_{name}"] = float(tensor[valid].mean().item())
    for threshold in LOSS_THRESHOLDS:
        exponent = int(round(-math.log10(threshold)))
        hits = rollout.first_threshold_evaluation[threshold]
        reached = (hits > 0) & group_mask
        reached_values = _masked_values(hits, reached)
        row[f"fraction_reaching_1e_{exponent}"] = float(
            reached[group_mask].to(DTYPE).mean().item()
        )
        row[f"median_evaluations_to_1e_{exponent}"] = (
            statistics.median(reached_values) if reached_values else ""
        )
        row[f"mean_evaluations_to_1e_{exponent}"] = (
            statistics.fmean(reached_values) if reached_values else ""
        )
        row[f"worst_evaluations_to_1e_{exponent}"] = (
            max(reached_values) if reached_values else ""
        )
    return row


def _per_trajectory_rows(
    rollout: SearchRollout,
    c_star: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
    previous_stalled: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[dict[str, Any]] = []
    strict_one_side = torch.zeros(TEST_N, dtype=torch.bool)
    dominant_one_side = torch.zeros(TEST_N, dtype=torch.bool)
    distinct_modes = torch.zeros(TEST_N, dtype=torch.long)
    for trajectory in range(TEST_N):
        valid = rollout.query_mask[trajectory]
        queries = rollout.query_c[trajectory, valid]
        components = rollout.selected_component[trajectory, valid]
        reference = c_star[trajectory]
        left = int((queries < reference).sum().item())
        right = int((queries > reference).sum().item())
        total = max(left + right, 1)
        strict_one_side[trajectory] = left == 0 or right == 0
        dominant_one_side[trajectory] = max(left, right) / total >= 0.9
        distinct_modes[trajectory] = torch.unique(components).numel()
        row: dict[str, Any] = {
            "trajectory_index": trajectory,
            "previous_network2_stalled": bool(previous_stalled[trajectory]),
            "network1_safe_lower": float(interval_lower[trajectory].item()),
            "network1_safe_upper": float(interval_upper[trajectory].item()),
            "network1_safe_width": float(
                (interval_upper[trajectory] - interval_lower[trajectory]).item()
            ),
            "reference_c_star_posthoc": float(reference.item()),
            "success_1e_20": bool(rollout.success[trajectory]),
            "evaluations_consumed": int(
                rollout.evaluations_consumed[trajectory].item()
            ),
            "hit_emergency_cap": not bool(rollout.success[trajectory]),
            "best_observed_c": float(rollout.best_c[trajectory].item()),
            "best_observed_prefix_loss": float(
                rollout.best_loss[trajectory].item()
            ),
            "absolute_best_c_error_posthoc": float(
                torch.abs(rollout.best_c[trajectory] - reference).item()
            ),
            "learned_queries_evaluated": int(valid.sum().item()),
            "learned_queries_left_of_optimum_posthoc": left,
            "learned_queries_right_of_optimum_posthoc": right,
            "strictly_one_sided_failure_pattern": bool(
                strict_one_side[trajectory]
            ),
            "dominant_one_sided_failure_pattern": bool(
                dominant_one_side[trajectory]
            ),
            "distinct_selected_mixture_modes": int(
                distinct_modes[trajectory].item()
            ),
            "mean_categorical_entropy": float(
                rollout.categorical_entropy[trajectory, valid].mean().item()
            ),
            "mean_entropy_upper_bound": float(
                rollout.entropy_upper_bound[trajectory, valid].mean().item()
            ),
            "mean_proposal_spread": float(
                rollout.proposal_spread[trajectory, valid].mean().item()
            ),
            "final_proposal_spread": float(
                rollout.proposal_spread[trajectory, valid][-1].item()
            ),
            "mean_effective_modes": float(
                rollout.effective_modes[trajectory, valid].mean().item()
            ),
            "mean_active_weight_modes": float(
                rollout.active_weight_modes[trajectory, valid].mean().item()
            ),
        }
        for threshold in LOSS_THRESHOLDS:
            exponent = int(round(-math.log10(threshold)))
            first = int(
                rollout.first_threshold_evaluation[threshold][trajectory].item()
            )
            row[f"reached_1e_{exponent}"] = first > 0
            row[f"evaluations_to_1e_{exponent}"] = first if first > 0 else ""
        rows.append(row)
    return rows, strict_one_side, dominant_one_side, distinct_modes


def _summary_rows(
    rollout: SearchRollout,
    c_star: torch.Tensor,
    previous_stalled: torch.Tensor,
    strict_one_side: torch.Tensor,
    dominant_one_side: torch.Tensor,
    distinct_modes: torch.Tensor,
    training_history: list[dict[str, Any]],
    total_runtime_seconds: float,
) -> list[dict[str, Any]]:
    masks = {
        "all_test": torch.ones(TEST_N, dtype=torch.bool),
        "previous_network2_stalled": previous_stalled,
        "previous_network2_not_stalled": ~previous_stalled,
    }
    rows = [
        _group_summary(
            name,
            mask,
            rollout,
            c_star,
            strict_one_side,
            dominant_one_side,
            distinct_modes,
        )
        for name, mask in masks.items()
    ]
    shared = {
        "mixture_components": MIXTURE_COMPONENTS,
        "initial_fixed_evaluations": INITIAL_EVALUATIONS,
        "emergency_evaluation_cap": EMERGENCY_EVALUATION_CAP,
        "training_horizons": ";".join(str(value) for value in TRAINING_HORIZONS),
        "training_updates_by_stage": ";".join(
            str(row["updates_run"]) for row in training_history
        ),
        "best_updates_by_stage": ";".join(
            str(row["best_update"]) for row in training_history
        ),
        "stage_stop_reasons": ";".join(
            str(row["stop_reason"]) for row in training_history
        ),
        "total_runtime_seconds": total_runtime_seconds,
    }
    for row in rows:
        row.update(shared)
    checkpoints = [
        value
        for value in (3, 5, 10, 20, 50, 100, 150, 200)
        if value <= EMERGENCY_EVALUATION_CAP
    ]
    for evaluation in checkpoints:
        history_index = evaluation - INITIAL_EVALUATIONS
        loss = rollout.best_loss_history[:, history_index]
        rows.append(
            {
                "row_type": "evaluation_progress",
                "group": "all_test",
                "evaluations": evaluation,
                "median_best_prefix_loss": float(torch.median(loss).item()),
                "mean_best_prefix_loss": float(loss.mean().item()),
                "worst_best_prefix_loss": float(loss.max().item()),
                "fraction_below_1e_20": float(
                    (loss < SUCCESS_LOSS).to(DTYPE).mean().item()
                ),
                **shared,
            }
        )
    return rows


def _plot(rollout: SearchRollout) -> None:
    evaluations = list(range(INITIAL_EVALUATIONS, EMERGENCY_EVALUATION_CAP + 1))
    median_loss = torch.median(rollout.best_loss_history, dim=0).values
    quantile_90 = torch.quantile(rollout.best_loss_history, 0.9, dim=0)
    worst_loss = rollout.best_loss_history.max(dim=0).values
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(evaluations, median_loss, label="median")
    axes[0].plot(evaluations, quantile_90, label="90th percentile")
    axes[0].plot(evaluations, worst_loss, label="worst")
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
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions = _make_positions_only_split(
        TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions = _make_positions_only_split(
        VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels, _test_hidden_c = direct32._make_exact_labeled_split(
        TEST_N, TEST_DATA_SEED + 1_810_300
    )
    del _test_hidden_c

    saved_test_lower, saved_test_upper = _load_saved_test_intervals()
    print("replaying Network 1 for train/validation safe intervals", flush=True)
    network1_states, _network1_history = network1._train_curriculum(
        train_positions, validation_positions
    )
    network1_state = network1_states[network1.CURRICULUM_HORIZONS[-1]]
    train_lower, train_upper = _network1_intervals(
        network1_state, train_positions
    )
    validation_lower, validation_upper = _network1_intervals(
        network1_state, validation_positions
    )
    replayed_test_lower, replayed_test_upper = _network1_intervals(
        network1_state, test_positions
    )
    skip_replay_verification = (
        os.environ.get("SOFT_LOCAL_SEARCH_SKIP_INTERVAL_VERIFY", "0") == "1"
    )
    if (
        not skip_replay_verification
        and (
            not torch.equal(saved_test_lower, replayed_test_lower)
            or not torch.equal(saved_test_upper, replayed_test_upper)
        )
    ):
        maximum_difference = max(
            float(torch.max(torch.abs(saved_test_lower - replayed_test_lower)).item()),
            float(torch.max(torch.abs(saved_test_upper - replayed_test_upper)).item()),
        )
        raise RuntimeError(
            f"Replayed Network 1 intervals differ from saved values: {maximum_difference}"
        )
    test_lower, test_upper = saved_test_lower, saved_test_upper

    print("precomputing fixed local prefix-loss probes", flush=True)
    train_fixed_losses = _fixed_losses(train_positions, train_lower, train_upper)
    validation_fixed_losses = _fixed_losses(
        validation_positions, validation_lower, validation_upper
    )
    test_fixed_losses = _fixed_losses(test_positions, test_lower, test_upper)
    print("training soft continuous local-search policy", flush=True)
    final_state, training_history = _train_controller(
        train_positions,
        train_lower,
        train_upper,
        train_fixed_losses,
        validation_positions,
        validation_lower,
        validation_upper,
        validation_fixed_losses,
    )

    model = SoftGaussianMixtureController().to(dtype=DTYPE)
    model.load_state_dict(final_state)
    print("evaluating frozen policy through the emergency cap", flush=True)
    test_rollout = _evaluate(
        model,
        test_positions,
        test_lower,
        test_upper,
        test_fixed_losses,
        MAX_LEARNED_QUERIES,
        EVALUATION_SEED + 10_000,
    )

    # Reference values and previous failure identities remain unopened until
    # every search query and stopping decision has been frozen.
    direct32._verify_test_labels(test_labels)
    test_c_star = black_box._stored_test_c_star()
    previous_stalled = _previous_stall_mask()
    trajectory_rows, strict_one_side, dominant_one_side, distinct_modes = (
        _per_trajectory_rows(
            test_rollout,
            test_c_star,
            test_lower,
            test_upper,
            previous_stalled,
        )
    )
    total_runtime_seconds = time.perf_counter() - start_time
    summary_rows = _summary_rows(
        test_rollout,
        test_c_star,
        previous_stalled,
        strict_one_side,
        dominant_one_side,
        distinct_modes,
        training_history,
        total_runtime_seconds,
    )
    _write_csv(OUTPUT_DIR / "local_search_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_trajectory_search.csv", trajectory_rows)
    _plot(test_rollout)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "success_fraction": float(
                test_rollout.success.to(DTYPE).mean().item()
            ),
            "median_best_loss": float(torch.median(test_rollout.best_loss).item()),
            "worst_best_loss": float(test_rollout.best_loss.max().item()),
            "previous_stalled_count": int(previous_stalled.sum().item()),
            "runtime_seconds": total_runtime_seconds,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
