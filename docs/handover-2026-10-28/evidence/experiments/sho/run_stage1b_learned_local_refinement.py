"""Learned local prefix-loss refinement inside Network 1 safe intervals."""

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

import experiments.sho.run_stage1b_learned_black_box_search as black_box
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
        "LOCAL_REFINEMENT_OUTPUT_DIR",
        "outputs/sho/stage1B/learned_local_refinement",
    )
)
TRAIN_N = 4_096
VALIDATION_N = 512
TEST_N = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
CURRICULUM_HORIZONS = (3, 6, 10, 15)
MAX_UPDATES_PER_STAGE = int(
    os.environ.get("LOCAL_REFINEMENT_MAX_UPDATES", "1000")
)
EVAL_INTERVAL = int(os.environ.get("LOCAL_REFINEMENT_EVAL_INTERVAL", "50"))
PLATEAU_PATIENCE = int(os.environ.get("LOCAL_REFINEMENT_PATIENCE", "300"))
LOG_INPUT_EPSILON = 1e-30
LOG_OBJECTIVE_EPSILON = 1e-24
C_ERROR_THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-6)
FIXED_LOCAL_U = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE)
MODEL_SEED = SEED + 1_980_000


class LocalDeepSetsController(nn.Module):
    """Permutation-invariant local-query controller in normalized coordinates."""

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
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sampled_u: torch.Tensor,
        sampled_loss: torch.Tensor,
        candidate_lower_u: torch.Tensor,
        candidate_upper_u: torch.Tensor,
    ) -> torch.Tensor:
        pairs = torch.stack(
            (
                sampled_u,
                torch.log(sampled_loss + LOG_INPUT_EPSILON) / 10.0,
            ),
            dim=2,
        )
        embedded = self.element(pairs)
        context = torch.cat(
            (embedded.mean(dim=1), embedded.max(dim=1).values), dim=1
        )
        candidate_width = candidate_upper_u - candidate_lower_u
        count_feature = torch.full_like(
            candidate_lower_u, sampled_u.shape[1] / 18.0
        )
        global_features = torch.stack(
            (
                candidate_lower_u,
                candidate_upper_u,
                candidate_width,
                count_feature,
            ),
            dim=1,
        )
        raw = self.head(torch.cat((context, global_features), dim=1)).squeeze(1)
        # The candidate bracket informs the policy but is not an irreversible
        # constraint: every query remains possible inside Network 1's safe box.
        return torch.sigmoid(raw)


@dataclass
class LocalRollout:
    query_u: torch.Tensor
    query_c: torch.Tensor
    query_loss: torch.Tensor
    best_c: torch.Tensor
    best_loss: torch.Tensor
    candidate_lower: torch.Tensor
    candidate_upper: torch.Tensor
    improved_best: torch.Tensor
    training_objective: torch.Tensor | None


def _network1_intervals(
    state: dict[str, torch.Tensor], positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    model = network1.RegionPolicy().to(dtype=DTYPE)
    model.load_state_dict(state)
    rollout = network1._evaluate_rollout(
        model, positions, network1.CURRICULUM_HORIZONS[-1]
    )
    return rollout.candidate_lower[:, -1], rollout.candidate_upper[:, -1]


@torch.no_grad()
def _fixed_local_losses(
    positions: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, positions.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, positions.shape[0])
        lower = interval_lower[start:stop]
        width = interval_upper[start:stop] - lower
        batch_losses = []
        for value in FIXED_LOCAL_U:
            c_value = lower + float(value.item()) * width
            batch_losses.append(_losses(c_value, positions[start:stop]))
        parts.append(torch.stack(batch_losses, dim=1))
    return torch.cat(parts, dim=0)


def _candidate_bracket_u(
    sampled_u: torch.Tensor, sampled_loss: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = sampled_u.shape[0]
    lower = torch.empty(batch_size, dtype=DTYPE)
    upper = torch.empty(batch_size, dtype=DTYPE)
    detached_u = sampled_u.detach()
    detached_loss = sampled_loss.detach()
    for trajectory in range(batch_size):
        values = detached_u[trajectory]
        losses = detached_loss[trajectory]
        support: list[float] = []
        for index in torch.argsort(losses).tolist():
            value = float(values[index].item())
            if all(abs(value - existing) > 1e-14 for existing in support):
                support.append(value)
            if len(support) == 2:
                break
        support_lower = min(support)
        support_upper = max(support)
        lower_values = values[values < support_lower - 1e-14]
        upper_values = values[values > support_upper + 1e-14]
        lower[trajectory] = (
            torch.max(lower_values) if lower_values.numel() else 0.0
        )
        upper[trajectory] = (
            torch.min(upper_values) if upper_values.numel() else 1.0
        )
    return lower, upper


def _unroll(
    model: LocalDeepSetsController,
    positions: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
    fixed_losses: torch.Tensor,
    horizon: int,
) -> LocalRollout:
    batch_size = positions.shape[0]
    interval_width = interval_upper - interval_lower
    sampled_u = FIXED_LOCAL_U[None, :].expand(batch_size, -1)
    sampled_loss = fixed_losses
    candidate_lower_u, candidate_upper_u = _candidate_bracket_u(
        sampled_u, sampled_loss
    )
    query_u_history: list[torch.Tensor] = []
    query_c_history: list[torch.Tensor] = []
    query_loss_history: list[torch.Tensor] = []
    best_c_history: list[torch.Tensor] = []
    best_loss_history: list[torch.Tensor] = []
    candidate_lower_history: list[torch.Tensor] = []
    candidate_upper_history: list[torch.Tensor] = []
    improved_history: list[torch.Tensor] = []
    objective_terms: list[torch.Tensor] = []

    for _ in range(horizon):
        query_u = model(
            sampled_u,
            sampled_loss,
            candidate_lower_u,
            candidate_upper_u,
        )
        query_c = interval_lower + query_u * interval_width
        query_loss = _losses(query_c, positions)
        prior_best = sampled_loss.min(dim=1).values
        improved = query_loss.detach() < prior_best
        objective_terms.append(torch.log(query_loss + LOG_OBJECTIVE_EPSILON))
        sampled_u = torch.cat((sampled_u, query_u.detach()[:, None]), dim=1)
        sampled_loss = torch.cat(
            (sampled_loss, query_loss.detach()[:, None]), dim=1
        )
        candidate_lower_u, candidate_upper_u = _candidate_bracket_u(
            sampled_u, sampled_loss
        )
        best_indices = torch.argmin(sampled_loss, dim=1)
        best_u = sampled_u.gather(1, best_indices[:, None]).squeeze(1)
        best_loss = sampled_loss.gather(1, best_indices[:, None]).squeeze(1)

        query_u_history.append(query_u.detach())
        query_c_history.append(query_c.detach())
        query_loss_history.append(query_loss.detach())
        best_c_history.append(interval_lower + best_u * interval_width)
        best_loss_history.append(best_loss)
        candidate_lower_history.append(
            interval_lower + candidate_lower_u * interval_width
        )
        candidate_upper_history.append(
            interval_lower + candidate_upper_u * interval_width
        )
        improved_history.append(improved)

    objective = torch.stack(objective_terms, dim=1).mean()
    return LocalRollout(
        query_u=torch.stack(query_u_history, dim=1),
        query_c=torch.stack(query_c_history, dim=1),
        query_loss=torch.stack(query_loss_history, dim=1),
        best_c=torch.stack(best_c_history, dim=1),
        best_loss=torch.stack(best_loss_history, dim=1),
        candidate_lower=torch.stack(candidate_lower_history, dim=1),
        candidate_upper=torch.stack(candidate_upper_history, dim=1),
        improved_best=torch.stack(improved_history, dim=1),
        training_objective=objective,
    )


@torch.no_grad()
def _evaluate(
    model: LocalDeepSetsController,
    positions: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
    fixed_losses: torch.Tensor,
    horizon: int,
) -> LocalRollout:
    model.eval()
    parts: list[LocalRollout] = []
    for start in range(0, positions.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, positions.shape[0])
        parts.append(
            _unroll(
                model,
                positions[start:stop],
                interval_lower[start:stop],
                interval_upper[start:stop],
                fixed_losses[start:stop],
                horizon,
            )
        )
    fields = [
        "query_u",
        "query_c",
        "query_loss",
        "best_c",
        "best_loss",
        "candidate_lower",
        "candidate_upper",
        "improved_best",
    ]
    combined = {
        field: torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in fields
    }
    return LocalRollout(**combined, training_objective=None)


def _validation_score(rollout: LocalRollout) -> float:
    return float(
        torch.log(rollout.best_loss[:, -1] + LOG_INPUT_EPSILON).mean().item()
    )


def _train_curriculum(
    train_positions: torch.Tensor,
    train_lower: torch.Tensor,
    train_upper: torch.Tensor,
    train_fixed_losses: torch.Tensor,
    validation_positions: torch.Tensor,
    validation_lower: torch.Tensor,
    validation_upper: torch.Tensor,
    validation_fixed_losses: torch.Tensor,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    torch.manual_seed(MODEL_SEED)
    model = LocalDeepSetsController().to(dtype=DTYPE)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    stage_states: dict[int, dict[str, torch.Tensor]] = {}
    stage_history: list[dict[str, Any]] = []
    cumulative_updates = 0

    for horizon in CURRICULUM_HORIZONS:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        validation = _evaluate(
            model,
            validation_positions,
            validation_lower,
            validation_upper,
            validation_fixed_losses,
            horizon,
        )
        best_score = _validation_score(validation)
        best_state = copy.deepcopy(model.state_dict())
        best_update = 0
        stale_updates = 0
        updates_run = 0
        stop_reason = "maximum_updates"
        stage_start = time.perf_counter()
        print(
            f"local horizon={horizon} update=0000 "
            f"validation mean log best loss={best_score:.6e}",
            flush=True,
        )
        for update in range(1, MAX_UPDATES_PER_STAGE + 1):
            model.train()
            indices = torch.randint(
                0,
                train_positions.shape[0],
                (BATCH_SIZE,),
                generator=generator,
            )
            rollout = _unroll(
                model,
                train_positions[indices],
                train_lower[indices],
                train_upper[indices],
                train_fixed_losses[indices],
                horizon,
            )
            if rollout.training_objective is None:
                raise RuntimeError("Training rollout has no objective")
            optimizer.zero_grad()
            rollout.training_objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            updates_run = update
            cumulative_updates += 1

            if update % EVAL_INTERVAL == 0 or update == MAX_UPDATES_PER_STAGE:
                validation = _evaluate(
                    model,
                    validation_positions,
                    validation_lower,
                    validation_upper,
                    validation_fixed_losses,
                    horizon,
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
                    f"local horizon={horizon} update={update:04d} "
                    f"train mean log query loss="
                    f"{float(rollout.training_objective.detach().item()):.6e} "
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
                "cumulative_updates": cumulative_updates,
            }
        )
    return stage_states, stage_history


def _metrics(
    rollout: LocalRollout,
    c_star: torch.Tensor,
    query_number: int,
) -> dict[str, Any]:
    index = query_number - 1
    errors = torch.abs(rollout.best_c[:, index] - c_star)
    best_loss = rollout.best_loss[:, index]
    lower = rollout.candidate_lower[:, index]
    upper = rollout.candidate_upper[:, index]
    widths = upper - lower
    covered = (c_star >= lower) & (c_star <= upper)
    metrics: dict[str, Any] = {
        "adaptive_query_number": query_number,
        "test_trajectories": c_star.shape[0],
        "median_best_c_error": float(torch.median(errors).item()),
        "worst_best_c_error": float(errors.max().item()),
        "median_best_observed_prefix_loss": float(torch.median(best_loss).item()),
        "worst_best_observed_prefix_loss": float(best_loss.max().item()),
        "candidate_region_coverage": float(covered.to(DTYPE).mean().item()),
        "median_candidate_region_width": float(torch.median(widths).item()),
        "mean_candidate_region_width": float(widths.mean().item()),
        "worst_candidate_region_width": float(widths.max().item()),
        "new_query_improved_best_loss_fraction": float(
            rollout.improved_best[:, index].to(DTYPE).mean().item()
        ),
    }
    for threshold in C_ERROR_THRESHOLDS:
        exponent = int(round(-math.log10(threshold)))
        metrics[f"fraction_best_c_within_1e_{exponent}"] = float(
            (errors <= threshold).to(DTYPE).mean().item()
        )
    return metrics


def _per_query_rows(
    rollout: LocalRollout,
    c_star: torch.Tensor,
    network1_lower: torch.Tensor,
    network1_upper: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_number in range(1, rollout.best_loss.shape[1] + 1):
        index = query_number - 1
        for trajectory in range(c_star.shape[0]):
            reference = float(c_star[trajectory].item())
            best_c = float(rollout.best_c[trajectory, index].item())
            lower = float(rollout.candidate_lower[trajectory, index].item())
            upper = float(rollout.candidate_upper[trajectory, index].item())
            error = abs(best_c - reference)
            row: dict[str, Any] = {
                "trajectory_index": trajectory,
                "adaptive_query_number": query_number,
                "reference_c_star_posthoc": reference,
                "network1_safe_lower": float(network1_lower[trajectory].item()),
                "network1_safe_upper": float(network1_upper[trajectory].item()),
                "queried_normalized_position": float(
                    rollout.query_u[trajectory, index].item()
                ),
                "queried_c": float(rollout.query_c[trajectory, index].item()),
                "queried_prefix_loss": float(
                    rollout.query_loss[trajectory, index].item()
                ),
                "new_query_improved_best_loss": bool(
                    rollout.improved_best[trajectory, index].item()
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
            }
            for threshold in C_ERROR_THRESHOLDS:
                exponent = int(round(-math.log10(threshold)))
                row[f"best_c_within_1e_{exponent}"] = error <= threshold
            rows.append(row)
    return rows


def _summary_rows(
    final_rollout: LocalRollout,
    stage_rollouts: dict[int, LocalRollout],
    c_star: torch.Tensor,
    network1_lower: torch.Tensor,
    network1_upper: torch.Tensor,
    stage_history: list[dict[str, Any]],
    network1_history: list[dict[str, Any]],
    total_runtime_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history_by_horizon = {int(row["horizon"]): row for row in stage_history}
    initial_width = network1_upper - network1_lower
    initial_coverage = (c_star >= network1_lower) & (c_star <= network1_upper)
    shared = {
        "network1_input_coverage": float(initial_coverage.to(DTYPE).mean().item()),
        "network1_input_median_width": float(torch.median(initial_width).item()),
        "network1_input_worst_width": float(initial_width.max().item()),
        "network1_training_updates": sum(
            int(row["updates_run"]) for row in network1_history
        ),
        "total_runtime_seconds": total_runtime_seconds,
    }
    for query_number in range(1, CURRICULUM_HORIZONS[-1] + 1):
        history = history_by_horizon[15]
        rows.append(
            {
                "row_type": "query_progress",
                "model_training_horizon": 15,
                **_metrics(final_rollout, c_star, query_number),
                "training_updates_run": history["updates_run"],
                "best_training_update": history["best_update"],
                "stage_stop_reason": history["stop_reason"],
                **shared,
            }
        )
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
                "cumulative_training_updates": history["cumulative_updates"],
                **shared,
            }
        )
    return rows


def _plot(summary_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in summary_rows if row["row_type"] == "query_progress"]
    queries = [int(row["adaptive_query_number"]) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
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
    for threshold in C_ERROR_THRESHOLDS:
        exponent = int(round(-math.log10(threshold)))
        axes[1].plot(
            queries,
            [float(row[f"fraction_best_c_within_1e_{exponent}"]) for row in rows],
            marker="o",
            label=f"within {threshold:g}",
        )
    axes[1].plot(
        queries,
        [float(row["candidate_region_coverage"]) for row in rows],
        color="black",
        linestyle="--",
        marker="s",
        label="local-region coverage",
    )
    axes[1].set_xlabel("learned local-refinement queries")
    axes[1].set_ylabel("test fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xticks(queries)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "local_error_vs_queries.png", dpi=180)
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

    print("replaying Network 1 to generate safe intervals", flush=True)
    network1_states, network1_history = network1._train_curriculum(
        train_positions, validation_positions
    )
    network1_state = network1_states[network1.CURRICULUM_HORIZONS[-1]]
    train_lower, train_upper = _network1_intervals(
        network1_state, train_positions
    )
    validation_lower, validation_upper = _network1_intervals(
        network1_state, validation_positions
    )
    test_lower, test_upper = _network1_intervals(network1_state, test_positions)
    print("precomputing three local probe losses", flush=True)
    train_fixed_losses = _fixed_local_losses(
        train_positions, train_lower, train_upper
    )
    validation_fixed_losses = _fixed_local_losses(
        validation_positions, validation_lower, validation_upper
    )
    test_fixed_losses = _fixed_local_losses(test_positions, test_lower, test_upper)
    print("training recurrent local-refinement curriculum", flush=True)
    local_states, local_history = _train_curriculum(
        train_positions,
        train_lower,
        train_upper,
        train_fixed_losses,
        validation_positions,
        validation_lower,
        validation_upper,
        validation_fixed_losses,
    )

    # Freeze every local policy and test history before loading c*.
    stage_rollouts: dict[int, LocalRollout] = {}
    for horizon in CURRICULUM_HORIZONS:
        model = LocalDeepSetsController().to(dtype=DTYPE)
        model.load_state_dict(local_states[horizon])
        stage_rollouts[horizon] = _evaluate(
            model,
            test_positions,
            test_lower,
            test_upper,
            test_fixed_losses,
            horizon,
        )
    final_rollout = stage_rollouts[CURRICULUM_HORIZONS[-1]]

    direct32._verify_test_labels(test_labels)
    test_c_star = black_box._stored_test_c_star()
    if float(torch.max(torch.abs(test_hidden_c - test_c_star)).item()) > 1e-12:
        raise RuntimeError("Stored test c* does not match deterministic test data")
    result_rows = _per_query_rows(
        final_rollout, test_c_star, test_lower, test_upper
    )
    total_runtime_seconds = time.perf_counter() - start_time
    summary_rows = _summary_rows(
        final_rollout,
        stage_rollouts,
        test_c_star,
        test_lower,
        test_upper,
        local_history,
        network1_history,
        total_runtime_seconds,
    )
    _write_csv(OUTPUT_DIR / "local_refinement_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_query_results.csv", result_rows)
    _plot(summary_rows)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "network1_input_coverage": summary_rows[0][
                "network1_input_coverage"
            ],
            "curriculum": {
                horizon: {
                    "updates": next(
                        row["updates_run"]
                        for row in local_history
                        if row["horizon"] == horizon
                    ),
                    "median_c_error": _metrics(
                        stage_rollouts[horizon], test_c_star, horizon
                    )["median_best_c_error"],
                    "coverage": _metrics(
                        stage_rollouts[horizon], test_c_star, horizon
                    )["candidate_region_coverage"],
                }
                for horizon in CURRICULUM_HORIZONS
            },
            "total_runtime_seconds": total_runtime_seconds,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
