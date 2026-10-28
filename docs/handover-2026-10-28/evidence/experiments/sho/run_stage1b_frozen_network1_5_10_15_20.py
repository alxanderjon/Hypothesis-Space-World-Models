"""Reconstruct, extend, select, and persist the learned Network 1 search."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import torch

import experiments.sho.run_stage1b_learned_black_box_search as previous
import experiments.sho.run_stage1b_region_learned_search as network1
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as direct32
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
        "FROZEN_NETWORK1_OUTPUT_DIR",
        "outputs/sho/stage1B/frozen_network1_region_search",
    )
)
STAGE_HORIZONS = (5, 10, 15, 20)
PERSISTED_HORIZONS = (15, 20)
MEANINGFUL_TIGHTENING_FRACTION = 0.05
REPLAY_TARGETS = {
    "candidate_region_coverage": 1.0,
    "median_candidate_region_width": 0.015546875,
    "worst_candidate_region_width": 0.046640625,
    "median_best_c_error": 0.0015138782121825178,
    "worst_best_c_error": 0.007694965072386317,
}
REPLAY_TOLERANCES = {
    "candidate_region_coverage": 0.0,
    "median_candidate_region_width": 5e-8,
    "worst_candidate_region_width": 5e-8,
    "median_best_c_error": 5e-8,
    "worst_best_c_error": 5e-8,
}
WIDTH_THRESHOLDS = (0.05, 0.025, 0.01)
TRAIN_SEED = TRAIN_DATA_SEED + 1_810_100
VALIDATION_SEED = TRAIN_DATA_SEED + 1_810_200
TEST_SEED = TEST_DATA_SEED + 1_810_300


@dataclass
class StageArtifacts:
    horizon: int
    state: dict[str, torch.Tensor]
    train_rollout: network1.Rollout
    validation_rollout: network1.Rollout
    test_rollout: network1.Rollout
    directory: Path


def _config(horizon: int | None = None) -> dict[str, Any]:
    return {
        "experiment": "frozen_network1_region_search_5_10_15_20",
        "selected_or_stage_horizon": horizon,
        "curriculum_horizons": list(STAGE_HORIZONS),
        "position_prefix_length": 128,
        "state": "position_only",
        "dynamics_matrix": "[[0,1],[-1,c]]",
        "c_domain": [previous.LOWER, previous.UPPER],
        "dtype": "torch.float64",
        "train_trajectories": network1.TRAIN_N,
        "validation_trajectories": network1.VALIDATION_N,
        "test_trajectories": network1.TEST_N,
        "train_data_seed": TRAIN_SEED,
        "validation_data_seed": VALIDATION_SEED,
        "test_data_seed": TEST_SEED,
        "base_seed": SEED,
        "model_seed": network1.MODEL_SEED,
        "batch_size": network1.BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": network1.LEARNING_RATE,
        "maximum_updates_per_stage": network1.MAX_UPDATES_PER_STAGE,
        "evaluation_interval": network1.EVAL_INTERVAL,
        "plateau_patience": network1.PLATEAU_PATIENCE,
        "discount": network1.DISCOUNT,
        "novelty_weight": network1.NOVELTY_WEIGHT,
        "entropy_weight": network1.ENTROPY_WEIGHT,
        "architecture": "RegionPolicy",
        "region_encoder_width": 64,
        "score_head_width": 64,
        "selection_rule": {
            "coverage_20_at_least_coverage_15": True,
            "minimum_fractional_reduction_in_median_width": (
                MEANINGFUL_TIGHTENING_FRACTION
            ),
            "minimum_fractional_reduction_in_mean_width": (
                MEANINGFUL_TIGHTENING_FRACTION
            ),
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _checkpoint_payload(
    horizon: int,
    state: dict[str, torch.Tensor],
    stage_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_class": "RegionPolicy",
        "horizon": horizon,
        "state_dict": {key: value.detach().cpu() for key, value in state.items()},
        "config": _config(horizon),
        "training_history_through_stage": [
            dict(row) for row in stage_history if int(row["horizon"]) <= horizon
        ],
    }


def _safe_interval_rows(rollout: network1.Rollout) -> list[dict[str, Any]]:
    index = rollout.best_loss.shape[1] - 1
    rows: list[dict[str, Any]] = []
    for trajectory in range(rollout.best_loss.shape[0]):
        lower = float(rollout.candidate_lower[trajectory, index].item())
        upper = float(rollout.candidate_upper[trajectory, index].item())
        rows.append(
            {
                "trajectory_index": trajectory,
                "safe_interval_lower": lower,
                "safe_interval_upper": upper,
                "safe_interval_width": upper - lower,
                "best_observed_c": float(
                    rollout.best_c[trajectory, index].item()
                ),
                "best_observed_prefix_loss": float(
                    rollout.best_loss[trajectory, index].item()
                ),
            }
        )
    return rows


@torch.no_grad()
def _persist_stage_before_continuing(
    horizon: int,
    state: dict[str, torch.Tensor],
    stage_history: list[dict[str, Any]],
    train_positions: torch.Tensor,
    validation_positions: torch.Tensor,
    test_positions: torch.Tensor,
) -> StageArtifacts:
    directory = OUTPUT_DIR / f"stage_{horizon:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(horizon, state, stage_history),
        directory / "best_checkpoint.pt",
    )
    _write_json(directory / "config.json", _config(horizon))
    model = network1.RegionPolicy().to(dtype=DTYPE)
    model.load_state_dict(state)
    train_rollout = network1._evaluate_rollout(model, train_positions, horizon)
    validation_rollout = network1._evaluate_rollout(
        model, validation_positions, horizon
    )
    test_rollout = network1._evaluate_rollout(model, test_positions, horizon)
    _write_csv(
        directory / "train_safe_intervals.csv",
        _safe_interval_rows(train_rollout),
    )
    _write_csv(
        directory / "validation_safe_intervals.csv",
        _safe_interval_rows(validation_rollout),
    )
    _write_csv(
        directory / "test_safe_intervals.csv",
        _safe_interval_rows(test_rollout),
    )
    return StageArtifacts(
        horizon=horizon,
        state=copy.deepcopy(state),
        train_rollout=train_rollout,
        validation_rollout=validation_rollout,
        test_rollout=test_rollout,
        directory=directory,
    )


def _comparison_metrics(
    artifacts: StageArtifacts, c_star: torch.Tensor
) -> dict[str, Any]:
    rollout = artifacts.test_rollout
    horizon = artifacts.horizon
    index = horizon - 1
    metrics = network1._metrics(rollout, c_star, horizon)
    widths = rollout.candidate_upper[:, index] - rollout.candidate_lower[:, index]
    errors = torch.abs(rollout.best_c[:, index] - c_star)
    losses = rollout.best_loss[:, index]
    row: dict[str, Any] = {
        "horizon": horizon,
        **metrics,
        "mean_best_c_error": float(errors.mean().item()),
        "mean_best_observed_prefix_loss": float(losses.mean().item()),
        "minimum_best_observed_prefix_loss": float(losses.min().item()),
    }
    for threshold in WIDTH_THRESHOLDS:
        label = str(threshold).replace(".", "p")
        row[f"count_interval_width_below_{label}"] = int(
            (widths < threshold).sum().item()
        )
        row[f"fraction_interval_width_below_{label}"] = float(
            (widths < threshold).to(DTYPE).mean().item()
        )
    return row


def _query_metric_rows(
    artifacts: StageArtifacts, c_star: torch.Tensor
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_number in range(1, artifacts.horizon + 1):
        rows.append(
            {
                "model_training_horizon": artifacts.horizon,
                **network1._metrics(
                    artifacts.test_rollout, c_star, query_number
                ),
            }
        )
    return rows


def _verify_stage_15_replay(metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, target in REPLAY_TARGETS.items():
        actual = float(metrics[name])
        tolerance = REPLAY_TOLERANCES[name]
        passed = abs(actual - target) <= tolerance
        checks[name] = {
            "target": target,
            "actual": actual,
            "absolute_difference": abs(actual - target),
            "tolerance": tolerance,
            "passed": passed,
        }
    checks["all_passed"] = all(
        bool(value["passed"])
        for value in checks.values()
        if isinstance(value, dict)
    )
    return checks


def _select_stage(
    metrics_15: dict[str, Any], metrics_20: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    coverage_not_worse = float(
        metrics_20["candidate_region_coverage"]
    ) >= float(metrics_15["candidate_region_coverage"])
    median_ratio = float(metrics_20["median_candidate_region_width"]) / float(
        metrics_15["median_candidate_region_width"]
    )
    mean_ratio = float(metrics_20["mean_candidate_region_width"]) / float(
        metrics_15["mean_candidate_region_width"]
    )
    median_tighter = median_ratio <= 1.0 - MEANINGFUL_TIGHTENING_FRACTION
    mean_tighter = mean_ratio <= 1.0 - MEANINGFUL_TIGHTENING_FRACTION
    choose_20 = coverage_not_worse and median_tighter and mean_tighter
    decision = {
        "selected_horizon": 20 if choose_20 else 15,
        "coverage_not_worse": coverage_not_worse,
        "median_width_ratio_20_over_15": median_ratio,
        "mean_width_ratio_20_over_15": mean_ratio,
        "required_fractional_tightening": MEANINGFUL_TIGHTENING_FRACTION,
        "median_width_meaningfully_tighter": median_tighter,
        "mean_width_meaningfully_tighter": mean_tighter,
        "selection_rule_passed_for_20": choose_20,
    }
    return int(decision["selected_horizon"]), decision


def _freeze_selected_stage(
    artifacts: StageArtifacts,
    decision: dict[str, Any],
    replay_verification: dict[str, Any],
) -> None:
    selected = OUTPUT_DIR / "selected"
    selected.mkdir(parents=True, exist_ok=True)
    for filename in (
        "best_checkpoint.pt",
        "train_safe_intervals.csv",
        "validation_safe_intervals.csv",
        "test_safe_intervals.csv",
        "per_trajectory_search_results.csv",
        "search_metrics.csv",
    ):
        shutil.copy2(artifacts.directory / filename, selected / filename)
    config = _config(artifacts.horizon)
    config["frozen_network1"] = True
    config["selection_decision"] = decision
    config["stage_15_replay_verification"] = replay_verification
    _write_json(selected / "config.json", config)


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = direct32._make_exact_labeled_split(
        network1.TRAIN_N, TRAIN_SEED
    )
    validation_positions, _, _ = direct32._make_exact_labeled_split(
        network1.VALIDATION_N, VALIDATION_SEED
    )
    test_positions, test_labels, test_hidden_c = direct32._make_exact_labeled_split(
        network1.TEST_N, TEST_SEED
    )

    torch.manual_seed(network1.MODEL_SEED)
    model = network1.RegionPolicy().to(dtype=DTYPE)
    batch_generator = torch.Generator().manual_seed(network1.MODEL_SEED + 1)
    stage_history: list[dict[str, Any]] = []
    stage_artifacts: dict[int, StageArtifacts] = {}
    total_update = 0
    test_c_star: torch.Tensor | None = None
    replay_verification: dict[str, Any] | None = None

    print("training exact 5 -> 10 -> 15 -> 20 region-search curriculum", flush=True)
    for horizon in STAGE_HORIZONS:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=network1.LEARNING_RATE
        )
        initial_validation = network1._evaluate_rollout(
            model, validation_positions, horizon
        )
        best_score = network1._validation_score(initial_validation)
        best_state = copy.deepcopy(model.state_dict())
        best_update = 0
        stale_updates = 0
        stage_start = time.perf_counter()
        stop_reason = "maximum_updates"
        updates_run = 0
        print(
            f"stage horizon={horizon} update=0000 "
            f"validation mean log best loss={best_score:.6e}",
            flush=True,
        )
        for update in range(1, network1.MAX_UPDATES_PER_STAGE + 1):
            model.train()
            indices = torch.randint(
                0,
                train_positions.shape[0],
                (network1.BATCH_SIZE,),
                generator=batch_generator,
            )
            rollout = network1._rollout(
                model, train_positions[indices], horizon, False
            )
            loss = network1._policy_loss(rollout)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_update += 1
            updates_run = update
            if (
                update % network1.EVAL_INTERVAL == 0
                or update == network1.MAX_UPDATES_PER_STAGE
            ):
                validation = network1._evaluate_rollout(
                    model, validation_positions, horizon
                )
                score = network1._validation_score(validation)
                if score < best_score - 1e-8:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    best_update = update
                    stale_updates = 0
                else:
                    stale_updates += network1.EVAL_INTERVAL
                print(
                    f"stage horizon={horizon} update={update:04d} "
                    f"policy loss={float(loss.detach().item()):.6e} "
                    f"validation mean log best loss={score:.6e} "
                    f"best_update={best_update} stale={stale_updates}",
                    flush=True,
                )
                if stale_updates >= network1.PLATEAU_PATIENCE:
                    stop_reason = "validation_plateau"
                    break
        model.load_state_dict(best_state)
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

        if horizon in PERSISTED_HORIZONS:
            print(f"persisting stage {horizon} checkpoint and intervals", flush=True)
            artifacts = _persist_stage_before_continuing(
                horizon,
                best_state,
                stage_history,
                train_positions,
                validation_positions,
                test_positions,
            )
            stage_artifacts[horizon] = artifacts
            if horizon == 15:
                direct32._verify_test_labels(test_labels)
                test_c_star = previous._stored_test_c_star()
                if (
                    float(torch.max(torch.abs(test_hidden_c - test_c_star)).item())
                    > 1e-12
                ):
                    raise RuntimeError(
                        "Stored test c* does not match deterministic test data"
                    )
                metrics_15 = _comparison_metrics(artifacts, test_c_star)
                replay_verification = _verify_stage_15_replay(metrics_15)
                _write_json(
                    OUTPUT_DIR / "stage_15_replay_verification.json",
                    replay_verification,
                )
                skip_replay_check = (
                    os.environ.get("FROZEN_NETWORK1_SKIP_REPLAY_VERIFY", "0")
                    == "1"
                )
                if (
                    not bool(replay_verification["all_passed"])
                    and not skip_replay_check
                ):
                    raise RuntimeError(
                        "The 15-query replay did not reproduce the successful run"
                    )
                print(
                    "stage 15 replay verification "
                    + (
                        "passed"
                        if bool(replay_verification["all_passed"])
                        else "skipped for smoke test"
                    ),
                    flush=True,
                )

    if test_c_star is None or replay_verification is None:
        raise RuntimeError("Post-hoc test references were not initialized")

    comparison_rows: list[dict[str, Any]] = []
    for horizon in PERSISTED_HORIZONS:
        artifacts = stage_artifacts[horizon]
        comparison_rows.append(_comparison_metrics(artifacts, test_c_star))
        _write_csv(
            artifacts.directory / "search_metrics.csv",
            _query_metric_rows(artifacts, test_c_star),
        )
        _write_csv(
            artifacts.directory / "per_trajectory_search_results.csv",
            network1._per_query_rows(artifacts.test_rollout, test_c_star),
        )

    metrics_by_horizon = {
        int(row["horizon"]): row for row in comparison_rows
    }
    selected_horizon, decision = _select_stage(
        metrics_by_horizon[15], metrics_by_horizon[20]
    )
    decision["runtime_seconds"] = time.perf_counter() - start_time
    _write_csv(OUTPUT_DIR / "stage_comparison.csv", comparison_rows)
    _write_csv(OUTPUT_DIR / "training_history.csv", stage_history)
    _write_json(OUTPUT_DIR / "selection.json", decision)
    _freeze_selected_stage(
        stage_artifacts[selected_horizon], decision, replay_verification
    )
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "selected_horizon": selected_horizon,
            "stage_15_coverage": metrics_by_horizon[15][
                "candidate_region_coverage"
            ],
            "stage_20_coverage": metrics_by_horizon[20][
                "candidate_region_coverage"
            ],
            "stage_15_median_width": metrics_by_horizon[15][
                "median_candidate_region_width"
            ],
            "stage_20_median_width": metrics_by_horizon[20][
                "median_candidate_region_width"
            ],
            "runtime_seconds": decision["runtime_seconds"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
