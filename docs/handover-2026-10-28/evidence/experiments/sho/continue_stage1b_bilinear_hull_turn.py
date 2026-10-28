#!/usr/bin/env python3
"""Faithfully continue the epoch-259 bilinear hull-turn run to plateau/500."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_balanced_hull_turn_smokescreen as balanced
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.validate_stage1b_bilinear_hull_turn as validation


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/bilinear_hull_turn_continuation")
SUMMARY_PATH = OUTPUT_DIR / "bilinear_continuation.csv"
FULL_VALIDATION_PATH = OUTPUT_DIR / "bilinear_continuation_full_validation.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "bilinear_continuation_checkpoint.pt"

SOURCE_EPOCH = 259
MAX_EPOCHS = 500
PATIENCE = smoke.PATIENCE
PLATEAU_RELATIVE_IMPROVEMENT = smoke.PLATEAU_RELATIVE_IMPROVEMENT


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_weight() -> tuple[np.ndarray, dict[str, str]]:
    weight, summary, _ = validation._load_frozen_source()
    if int(summary["best_epoch"]) != SOURCE_EPOCH:
        raise RuntimeError("source no longer identifies epoch 259")
    return weight, summary


def _train_and_continue(
    training: dict[str, np.ndarray], validation_set: dict[str, np.ndarray]
) -> tuple[smoke.BilinearModel, list[dict[str, Any]], dict[str, Any]]:
    recorded_weight, source_summary = _source_weight()
    torch.manual_seed(smoke.MODEL_SEED)
    model = smoke.BilinearModel().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=smoke.LEARNING_RATE)
    train_inputs = torch.as_tensor(training["inputs"], dtype=torch.float64)
    train_target = torch.as_tensor(training["target"], dtype=torch.float64)
    generator = np.random.default_rng(smoke.MODEL_SEED + 97)

    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_generator_state: dict[str, Any] | None = None
    best_validation_mse = math.inf
    plateau_validation_mse = math.inf
    best_epoch = SOURCE_EPOCH
    stale = 0
    stop_epoch = MAX_EPOCHS
    stopping_reason = "maximum_epochs"
    replay_maximum_weight_error = math.inf
    replay_metrics_match = False
    rows: list[dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(train_target))
        total_mse = 0.0
        model.train()
        for start in range(0, len(order), smoke.BATCH_SIZE):
            indices = torch.as_tensor(
                order[start : start + smoke.BATCH_SIZE], dtype=torch.long
            )
            predicted = model(train_inputs[indices])
            loss = nn.functional.mse_loss(predicted, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_mse += float(loss.item()) * len(indices)

        if epoch < SOURCE_EPOCH:
            continue

        validation_prediction = smoke._predict(model, validation_set["inputs"])
        validation_mse = float(
            np.mean((validation_prediction - validation_set["target"]) ** 2)
        )
        metrics = {
            **smoke._continuous_metrics(
                validation_set["target"], validation_prediction
            ),
            **smoke._decision_metrics(validation_set, validation_prediction),
        }
        if epoch == SOURCE_EPOCH:
            replay_weight = model.weight.detach().cpu().numpy()
            replay_maximum_weight_error = float(
                np.max(np.abs(replay_weight - recorded_weight))
            )
            replay_metrics_match = bool(
                replay_maximum_weight_error == 0.0
                and metrics["rmse"] == float(source_summary["rmse"])
                and metrics["maximum_absolute_error"]
                == float(source_summary["maximum_absolute_error"])
            )
            if not replay_metrics_match:
                raise RuntimeError(
                    "deterministic replay did not reproduce epoch-259 checkpoint exactly"
                )
            best_validation_mse = validation_mse
            plateau_validation_mse = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_generator_state = copy.deepcopy(generator.bit_generator.state)
        else:
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_generator_state = copy.deepcopy(generator.bit_generator.state)
            if validation_mse < plateau_validation_mse * (
                1.0 - PLATEAU_RELATIVE_IMPROVEMENT
            ):
                plateau_validation_mse = validation_mse
                stale = 0
            else:
                stale += 1

        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "phase": "source_checkpoint" if epoch == SOURCE_EPOCH else "continuation",
                "training_mse": total_mse / len(order),
                "validation_mse": validation_mse,
                "plateau_stale_epochs": stale,
                **metrics,
            }
        )
        if epoch == SOURCE_EPOCH or epoch % 10 == 0:
            print(
                f"epoch={epoch} rmse={metrics['rmse']:.6e} "
                f"max={metrics['maximum_absolute_error']:.6e} "
                f"decision={metrics['decision_accuracy']:.6f} stale={stale}",
                flush=True,
            )
        if epoch > SOURCE_EPOCH and stale >= PATIENCE:
            stop_epoch = epoch
            stopping_reason = "validation_plateau"
            break

    if best_state is None or best_optimizer_state is None or best_generator_state is None:
        raise RuntimeError("continuation never established a best checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    best_prediction = smoke._predict(model, validation_set["inputs"])
    best_metrics = {
        **smoke._continuous_metrics(validation_set["target"], best_prediction),
        **smoke._decision_metrics(validation_set, best_prediction),
    }
    weight = model.weight.detach().cpu().numpy()
    reference = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    metadata: dict[str, Any] = {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "maximum_epochs": MAX_EPOCHS,
        "stopping_reason": stopping_reason,
        "plateau_patience": PATIENCE,
        "plateau_relative_improvement": PLATEAU_RELATIVE_IMPROVEMENT,
        "source_epoch": SOURCE_EPOCH,
        "source_replay_exact": replay_metrics_match,
        "source_replay_maximum_weight_error": replay_maximum_weight_error,
        "continuation_epochs": stop_epoch - SOURCE_EPOCH,
        "continuation_optimizer_updates": (stop_epoch - SOURCE_EPOCH)
        * math.ceil(smoke.TRAIN_EVENTS / smoke.BATCH_SIZE),
        **best_metrics,
        "W_00": float(weight[0, 0]),
        "W_01": float(weight[0, 1]),
        "W_10": float(weight[1, 0]),
        "W_11": float(weight[1, 1]),
        "posthoc_reference_frobenius_error": float(
            np.linalg.norm(weight - reference)
        ),
        "posthoc_reference_maximum_coefficient_error": float(
            np.max(np.abs(weight - reference))
        ),
        "source_reference_frobenius_error": float(
            np.linalg.norm(recorded_weight - reference)
        ),
        "source_reference_maximum_coefficient_error": float(
            np.max(np.abs(recorded_weight - reference))
        ),
        "classification_target_used": False,
        "hull_reconstruction_run": False,
        "trajectory_search_run": False,
        "lock_status": "CANDIDATE_NOT_LOCKED",
        "best_optimizer_state": best_optimizer_state,
        "best_generator_state": best_generator_state,
    }
    return model, rows, metadata


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED
    )
    validation_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    train_states = turn._build_states(payload, train_indices, workers, "train replay")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "validation replay"
    )
    train_pool = turn._flatten_triples(train_states)
    validation_pool = turn._flatten_triples(validation_states)
    train_selected, _ = balanced._balanced_sample(
        train_pool, smoke.TRAIN_EVENTS, smoke.DATA_SEED + 11
    )
    validation_selected, _ = balanced._balanced_sample(
        validation_pool, smoke.VALIDATION_EVENTS, smoke.DATA_SEED + 12
    )
    training = smoke._dataset(train_selected, train_states)
    validation_set = smoke._dataset(validation_selected, validation_states)

    model, history, metadata = _train_and_continue(training, validation_set)
    checkpoint = {
        "format_version": 1,
        "component": "bilinear_hull_turn_continuation",
        "architecture": "y_hat = u1^T W u2",
        "source_epoch": SOURCE_EPOCH,
        "best_epoch": metadata["best_epoch"],
        "stop_epoch": metadata["stop_epoch"],
        "state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": metadata.pop("best_optimizer_state"),
        "numpy_generator_state": metadata.pop("best_generator_state"),
        "config": {
            "learning_rate": smoke.LEARNING_RATE,
            "batch_size": smoke.BATCH_SIZE,
            "model_seed": smoke.MODEL_SEED,
            "data_seed": smoke.DATA_SEED,
            "plateau_patience": PATIENCE,
            "plateau_relative_improvement": PLATEAU_RELATIVE_IMPROVEMENT,
        },
        "lock_status": "CANDIDATE_NOT_LOCKED",
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    checkpoint_hash = _sha256(CHECKPOINT_PATH)
    restored = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint_readback_exact = bool(
        torch.equal(restored["state_dict"]["weight"], model.weight.detach().cpu())
    )
    summary_row = {
        "record_type": "final_summary",
        **metadata,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_readback_exact": checkpoint_readback_exact,
        "training_resumed_from_epoch_259": True,
        "success_stopping_disabled_for_continuation": True,
    }
    _write_csv(SUMMARY_PATH, history + [summary_row])

    full_dataset = validation._event_dataset(validation_pool, validation_states)
    best_weight = model.weight.detach().cpu().numpy()
    full_prediction = validation._predict(best_weight, full_dataset["inputs"])
    full_rows, full_passed, full_summary = validation._full_validation_rows(
        full_dataset, full_prediction
    )
    full_summary["checkpoint_epoch"] = metadata["best_epoch"]
    full_summary["checkpoint_sha256"] = checkpoint_hash
    full_summary["full_validation_passed"] = full_passed
    _write_csv(FULL_VALIDATION_PATH, full_rows)
    print(
        f"stop_epoch={metadata['stop_epoch']} best_epoch={metadata['best_epoch']} "
        f"reason={metadata['stopping_reason']} rmse={metadata['rmse']:.6e}",
        flush=True,
    )
    print(
        f"matrix_max_error={metadata['posthoc_reference_maximum_coefficient_error']:.6e} "
        f"full_pool_mistakes={full_summary['mistake_count']} "
        f"full_pool_passed={full_passed}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
