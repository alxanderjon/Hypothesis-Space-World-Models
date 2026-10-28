#!/usr/bin/env python3
"""Learn only largest-versus-ordinary routing among selected eDIRECT cells."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_largest_cell_score as largest
import experiments.sho.run_stage1b_learned_global_search_imitation as imitation
import experiments.sho.run_stage1b_ordinary_cell_score_full as ordinary_full
import experiments.sho.run_stage1b_ordinary_cell_score_network as score_base
import experiments.sho.run_stage1b_specialized_cell_filter as specialized
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/largest_cell_router")
CHECKPOINT_PATH = OUTPUT_DIR / "best_router_checkpoint.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
ERRORS_PATH = OUTPUT_DIR / "routing_errors.csv"
SCREEN_PATH = OUTPUT_DIR / "screen_per_trajectory.csv"

ORDINARY_DATASET = ordinary_full.DATASET_PATH
LARGEST_DATASET = largest.DATASET_PATH
ORDINARY_CHECKPOINT = ordinary_full.CHECKPOINT_PATH
LARGEST_CHECKPOINT = largest.CHECKPOINT_PATH
SELECTOR_CHECKPOINT = specialized.FROZEN_SELECTOR_PATH
SELECTOR_SOURCE = specialized.SOURCE_SELECTOR
REFERENCE_TRACE = largest.SEARCH_TRACE_PATH

TRAIN_STATES = 2_000
VALIDATION_STATES = 500
MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 256
LEARNING_RATE = 2.0e-3
HIDDEN_DIM = 8
MODEL_SEED = SEED + 17_200_000
DATA_SEED = SEED + 17_300_000
STATIC_GATE = 1.0
SCREEN_COUNT = 16
GATE = imitation.GATE
NORMAL_CAP = imitation.NORMAL_CAP
FULL_BOUNDS = specialized.FULL_BOUNDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(
    path: Path, rows: list[dict[str, Any]], default_fields: tuple[str, ...] = ()
) -> None:
    fields: list[str] = list(default_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class LargestRouter(nn.Module):
    """Shared scalar scorer; softmax is applied across selected cells."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, sizes: torch.Tensor) -> torch.Tensor:
        return self.network(sizes).squeeze(-1)


def _state_records(split: str) -> list[dict[str, Any]]:
    ordinary_payload = torch.load(
        ORDINARY_DATASET, map_location="cpu", weights_only=False
    )[split]
    largest_payload = torch.load(
        LARGEST_DATASET, map_location="cpu", weights_only=False
    )[split]
    ordinary_groups: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for index in range(len(ordinary_payload["counts"])):
        key = (
            int(ordinary_payload["trajectory_id"][index]),
            int(ordinary_payload["iteration"][index]),
        )
        ordinary_groups.setdefault(key, []).append(
            (
                int(ordinary_payload["site_index"][index]),
                float(ordinary_payload["cell_size"][index]),
            )
        )
    records: list[dict[str, Any]] = []
    for index in range(len(largest_payload["counts"])):
        trajectory_id = int(largest_payload["trajectory_id"][index])
        iteration = int(largest_payload["iteration"][index])
        key = (trajectory_id, iteration)
        cells = list(ordinary_groups.get(key, []))
        largest_site = int(largest_payload["site_index"][index])
        largest_size = float(largest_payload["cell_size"][index])
        cells.append((largest_site, largest_size))
        cells.sort(key=lambda item: (item[1], item[0]))
        site_ids = np.asarray([item[0] for item in cells], dtype=np.int32)
        sizes = np.asarray([item[1] for item in cells], dtype=np.float64)
        target = int(np.flatnonzero(site_ids == largest_site)[0])
        deterministic = int(np.argmax(sizes))
        if target != deterministic:
            raise RuntimeError(
                f"saved largest-cell label disagrees with argmax for {key}: "
                f"target={target} argmax={deterministic}"
            )
        order = np.random.default_rng(
            DATA_SEED + 104729 * trajectory_id + 1009 * iteration
        ).permutation(len(sizes))
        sizes = sizes[order]
        site_ids = site_ids[order]
        target = int(np.flatnonzero(site_ids == largest_site)[0])
        sorted_sizes = np.sort(sizes)
        second = float(sorted_sizes[-2]) if len(sorted_sizes) > 1 else -math.inf
        records.append(
            {
                "trajectory_id": trajectory_id,
                "iteration": iteration,
                "site_ids": site_ids,
                "sizes": sizes,
                "target": target,
                "largest_site": largest_site,
                "selected_count": len(sizes),
                "margin": largest_size - second,
                "exact_tie": int(np.sum(sizes == largest_size)) > 1,
            }
        )
    return records


def _sample_records(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(records), count, replace=False)
    return [records[int(index)] for index in indices]


def _pad(records: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(int(record["selected_count"]) for record in records)
    sizes = np.zeros((len(records), maximum), dtype=np.float64)
    mask = np.zeros((len(records), maximum), dtype=np.bool_)
    targets = np.empty(len(records), dtype=np.int64)
    for index, record in enumerate(records):
        count = int(record["selected_count"])
        sizes[index, :count] = record["sizes"]
        mask[index, :count] = True
        targets[index] = int(record["target"])
    return {
        "sizes": torch.as_tensor(sizes, dtype=torch.float64),
        "mask": torch.as_tensor(mask),
        "targets": torch.as_tensor(targets, dtype=torch.long),
        "records": records,
    }


def _logits(model: LargestRouter, sizes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(sizes.unsqueeze(-1))
    return logits.masked_fill(~mask, -torch.inf)


def _evaluate(model: LargestRouter, padded: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(padded["targets"]), BATCH_SIZE):
            stop = start + BATCH_SIZE
            logits = _logits(
                model, padded["sizes"][start:stop], padded["mask"][start:stop]
            )
            targets = padded["targets"][start:stop]
            loss_sum += float(
                nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            )
            count += len(targets)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    prediction = np.concatenate(predictions)
    target = padded["targets"].cpu().numpy()
    return {
        "loss": loss_sum / count,
        "accuracy": float(np.mean(prediction == target)),
        "predictions": prediction,
    }


def _train(
    train: dict[str, Any], validation: dict[str, Any]
) -> tuple[LargestRouter, list[dict[str, Any]], int, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = LargestRouter().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = np.random.default_rng(MODEL_SEED)
    best_state = copy.deepcopy(model.state_dict())
    best = _evaluate(model, validation)
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = generator.permutation(len(train["targets"]))
        train_loss = 0.0
        train_correct = 0
        for start in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            logits = _logits(model, train["sizes"][indices], train["mask"][indices])
            targets = train["targets"][indices]
            loss = nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(indices)
            train_correct += int(torch.sum(torch.argmax(logits, dim=1) == targets).item())
        validation_metrics = _evaluate(model, validation)
        history.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "train_loss": train_loss / len(train["targets"]),
                "train_accuracy": train_correct / len(train["targets"]),
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        accuracy_improved = validation_metrics["accuracy"] > best["accuracy"] + 1.0e-15
        loss_tie_break = (
            validation_metrics["accuracy"] >= best["accuracy"] - 1.0e-15
            and validation_metrics["loss"] < best["loss"] - 1.0e-10
        )
        if accuracy_improved or loss_tie_break:
            best = validation_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0 if accuracy_improved else stale + 1
        else:
            stale += 1
        print(
            f"epoch={epoch} train_acc={train_correct / len(train['targets']):.6f} "
            f"val_acc={validation_metrics['accuracy']:.6f} "
            f"val_loss={validation_metrics['loss']:.3e}",
            flush=True,
        )
        if epoch >= 10 and stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, _evaluate(model, validation)


def _diagnostics(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    predictions: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = np.asarray([record["target"] for record in validation_records])
    hits = predictions == targets
    rows: list[dict[str, Any]] = []
    for selected_count in sorted(
        set(int(record["selected_count"]) for record in validation_records)
    ):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(validation_records)
                if int(record["selected_count"]) == selected_count
            ],
            dtype=np.int64,
        )
        rows.append(
            {
                "record_type": "accuracy_by_selected_count",
                "bin": selected_count,
                "count": len(indices),
                "accuracy": float(np.mean(hits[indices])),
            }
        )
    finite_train = np.asarray(
        [record["margin"] for record in train_records if np.isfinite(record["margin"])],
        dtype=np.float64,
    )
    edges = np.quantile(finite_train, (0.25, 0.5, 0.75))
    validation_margins = np.asarray(
        [record["margin"] for record in validation_records], dtype=np.float64
    )
    bins = np.searchsorted(edges, validation_margins, side="right")
    for bin_index in range(4):
        indices = np.flatnonzero(bins == bin_index)
        rows.append(
            {
                "record_type": "accuracy_by_margin_quartile",
                "bin": f"Q{bin_index + 1}",
                "count": len(indices),
                "lower": -math.inf if bin_index == 0 else edges[bin_index - 1],
                "upper": math.inf if bin_index == 3 else edges[bin_index],
                "accuracy": float(np.mean(hits[indices])) if len(indices) else math.nan,
            }
        )
    errors: list[dict[str, Any]] = []
    for index in np.flatnonzero(~hits):
        record = validation_records[int(index)]
        prediction = int(predictions[index])
        errors.append(
            {
                "validation_index": int(index),
                "trajectory_id": record["trajectory_id"],
                "iteration": record["iteration"],
                "selected_count": record["selected_count"],
                "margin": record["margin"],
                "target_position": record["target"],
                "predicted_position": prediction,
                "target_site_id": int(record["site_ids"][record["target"]]),
                "predicted_site_id": int(record["site_ids"][prediction]),
                "target_size": float(record["sizes"][record["target"]]),
                "predicted_size": float(record["sizes"][prediction]),
            }
        )
    return rows, errors


def _save_checkpoint(
    model: LargestRouter,
    best_epoch: int,
    metrics: dict[str, Any],
    hashes: dict[str, str],
    screen_preserved: bool | None,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "LargestRouter(1,8,1; SiLU; shared scorer)",
            "best_epoch": best_epoch,
            "validation_loss": metrics["loss"],
            "validation_accuracy": metrics["accuracy"],
            "inputs": ["selected_cell_size"],
            "screen_reliability_preserved": screen_preserved,
            "frozen_hashes": hashes,
        },
        CHECKPOINT_PATH,
    )


def _joint_cell_proposals(
    router: LargestRouter,
    ordinary_model: score_base.ScoreNetwork,
    largest_model: largest.LargestCellScorer,
    points: np.ndarray,
    losses: np.ndarray,
    generator_id: int,
    iteration: int,
) -> tuple[np.ndarray, dict[str, int]]:
    data = edirect._cell_data(points, losses, generator_id, iteration)
    surface = edirect.GaussianLocalResponse(points, losses)
    selected_sizes = data.sizes[data.selected]
    with torch.no_grad():
        router_scores = router(
            torch.as_tensor(selected_sizes[:, None], dtype=torch.float64)
        ).cpu().numpy()
    learned_largest_position = int(np.argmax(router_scores))
    deterministic_largest_position = int(np.argmax(selected_sizes))
    size_weight = edirect._normalize(selected_sizes)
    loo_weight = edirect._normalize(surface.loo_errors(data.selected))
    u_values = 0.5 * size_weight + 0.5 * loo_weight
    proposals: list[np.ndarray] = []
    for selected_position, site_index_value in enumerate(data.selected):
        site_index = int(site_index_value)
        mask = data.owners == site_index
        candidates = data.points[mask]
        distances = data.distances[mask]
        if len(candidates) == 0:
            continue
        normalized_distance = edirect._normalize(distances)
        if selected_position == learned_largest_position:
            scale = max(float(data.sizes[site_index]), 1.0e-12)
            inputs = np.column_stack(
                ((candidates - points[site_index][None, :]) / scale, normalized_distance)
            )
            with torch.no_grad():
                scores = largest_model(
                    torch.as_tensor(inputs, dtype=torch.float64)
                ).cpu().numpy()
            winner = int(np.argmax(scores))
        else:
            prediction = surface.predict(candidates)
            normalized_prediction = edirect._normalize(prediction)
            u_value = float(u_values[selected_position])
            inputs = np.column_stack(
                (
                    normalized_distance,
                    normalized_prediction,
                    np.full(len(candidates), u_value),
                )
            )
            with torch.no_grad():
                scores = ordinary_model(
                    torch.as_tensor(inputs, dtype=torch.float64)
                ).cpu().numpy()
            winner = int(np.argmax(scores))
        proposal = candidates[winner]
        if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
            proposals.append(proposal)
    if not proposals:
        nearest = edirect.cKDTree(points).query(data.points, k=1)[0]
        for index in np.argsort(nearest)[::-1]:
            if nearest[index] > 1.0e-12:
                proposals.append(data.points[index])
                break
    return np.asarray(proposals, dtype=np.float64), {
        "router_calls": 1,
        "router_disagreements": int(
            learned_largest_position != deterministic_largest_position
        ),
    }


def _run_normal(
    trajectory_id: int,
    position: np.ndarray,
    router: LargestRouter,
    ordinary_model: score_base.ScoreNetwork,
    largest_model: largest.LargestCellScorer,
    selector_model: imitation.CandidateRanker,
) -> tuple[imitation.ModeRun, dict[str, int]]:
    split = "validation"
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = np.empty((0, 2), dtype=np.float64)
    losses = np.empty(0, dtype=np.float64)
    evaluations = iteration = 0
    router_calls = router_disagreements = 0
    events: list[dict[str, Any]] = []
    pending = edirect._initial_design().copy()
    while len(pending) and evaluations < NORMAL_CAP and (
        not len(losses) or float(np.min(losses)) > GATE
    ):
        if len(points):
            global_features = imitation._global_features(
                points,
                losses,
                len(pending),
                evaluations,
                NORMAL_CAP,
                iteration,
                FULL_BOUNDS,
                "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model,
                points,
                losses,
                pending,
                "initial_design",
                global_features,
            )
        else:
            choice = 0
        proposal = pending[int(choice)]
        loss = float(
            edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0]
        )
        points = np.concatenate((points, proposal[None, :]))
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, int(choice), axis=0)
        evaluations += 1
    while evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
        pending, diagnostics = _joint_cell_proposals(
            router,
            ordinary_model,
            largest_model,
            points,
            losses,
            generator_id,
            iteration,
        )
        router_calls += diagnostics["router_calls"]
        router_disagreements += diagnostics["router_disagreements"]
        if diagnostics["router_disagreements"]:
            events.append(
                {
                    "iteration": iteration,
                    "evaluation": evaluations,
                    **diagnostics,
                }
            )
        while len(pending) and evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
            global_features = imitation._global_features(
                points,
                losses,
                len(pending),
                evaluations,
                NORMAL_CAP,
                iteration,
                FULL_BOUNDS,
                "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, pending, "cell", global_features
            )
            proposal = pending[int(choice)]
            loss = float(
                edirect._prefix_losses_numpy(
                    (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
                )[0]
            )
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, int(choice), axis=0)
            evaluations += 1
        if evaluations >= NORMAL_CAP or float(np.min(losses)) <= GATE:
            break
        greedy = edirect._greedy_proposal(points, losses, generator_id, iteration)
        if greedy is not None:
            loss = float(
                edirect._prefix_losses_numpy(
                    (LOWER + DOMAIN_WIDTH * greedy)[None, :], position
                )[0]
            )
            points = np.concatenate((points, greedy[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            evaluations += 1
        iteration += 1
    return (
        imitation.ModeRun(
            mode="normal_learned_largest_router",
            bounds=FULL_BOUNDS.copy(),
            points=points,
            losses=losses,
            new_evaluations=evaluations,
            success=float(np.min(losses)) <= GATE,
            decisions=events,
        ),
        {
            "router_calls": router_calls,
            "router_disagreements": router_disagreements,
        },
    )


_WORKER_ROUTER: LargestRouter | None = None
_WORKER_ORDINARY: score_base.ScoreNetwork | None = None
_WORKER_LARGEST: largest.LargestCellScorer | None = None
_WORKER_SELECTOR: imitation.CandidateRanker | None = None


def _load_models() -> tuple[
    LargestRouter,
    score_base.ScoreNetwork,
    largest.LargestCellScorer,
    imitation.CandidateRanker,
]:
    global _WORKER_ROUTER, _WORKER_ORDINARY, _WORKER_LARGEST, _WORKER_SELECTOR
    if _WORKER_ROUTER is None:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _WORKER_ROUTER = LargestRouter().to(torch.float64)
        _WORKER_ROUTER.load_state_dict(checkpoint["state_dict"])
        _WORKER_ROUTER.eval()
    if _WORKER_ORDINARY is None:
        checkpoint = torch.load(
            ORDINARY_CHECKPOINT, map_location="cpu", weights_only=False
        )
        _WORKER_ORDINARY = score_base.ScoreNetwork().to(torch.float64)
        _WORKER_ORDINARY.load_state_dict(checkpoint["state_dict"])
        _WORKER_ORDINARY.eval()
    if _WORKER_LARGEST is None:
        checkpoint = torch.load(
            LARGEST_CHECKPOINT, map_location="cpu", weights_only=False
        )
        _WORKER_LARGEST = largest.LargestCellScorer().to(torch.float64)
        _WORKER_LARGEST.load_state_dict(checkpoint["state_dict"])
        _WORKER_LARGEST.eval()
    if _WORKER_SELECTOR is None:
        checkpoint = torch.load(
            SELECTOR_CHECKPOINT, map_location="cpu", weights_only=False
        )
        _WORKER_SELECTOR = imitation.CandidateRanker().to(torch.float32)
        _WORKER_SELECTOR.load_state_dict(checkpoint["state_dict"])
        _WORKER_SELECTOR.eval()
    return _WORKER_ROUTER, _WORKER_ORDINARY, _WORKER_LARGEST, _WORKER_SELECTOR


def _screen_worker(arguments: tuple[int, np.ndarray]) -> dict[str, Any]:
    trajectory_id, position = arguments
    router, ordinary_model, largest_model, selector_model = _load_models()
    normal, diagnostics = _run_normal(
        trajectory_id,
        position,
        router,
        ordinary_model,
        largest_model,
        selector_model,
    )
    generator_id = imitation.SPLIT_ID_OFFSET["validation"] + trajectory_id
    modes = [normal]
    total = normal.new_evaluations
    boundary_run = nested_run = None
    if not normal.success:
        boundary_run, _ = imitation._run_boundary_rescue(
            split="validation",
            trajectory_id=trajectory_id,
            generator_id=generator_id,
            position=position,
            normal=normal,
            collect_trace=False,
        )
        modes.append(boundary_run)
        total += boundary_run.new_evaluations
    if boundary_run is not None and not boundary_run.success:
        nested_run = imitation._run_nested_rescue(
            split="validation",
            trajectory_id=trajectory_id,
            position=position,
            boundary_run=boundary_run,
            collect_trace=False,
        )
        modes.append(nested_run)
        total += nested_run.new_evaluations
    final = next((mode for mode in modes[::-1] if mode.success), modes[-1])
    return {
        "trajectory_id": trajectory_id,
        "normal_success": normal.success,
        "final_success": any(mode.success for mode in modes),
        "normal_evaluations": normal.new_evaluations,
        "total_evaluations": total,
        "final_loss": final.best_loss,
        "rescue_invoked": boundary_run is not None,
        "nested_invoked": nested_run is not None,
        **diagnostics,
    }


def _run_screen(positions: np.ndarray, workers: int) -> list[dict[str, Any]]:
    tasks = [(index, positions[index]) for index in range(SCREEN_COUNT)]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.append(_screen_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_screen_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 4 == 0 or len(results) == len(tasks):
                    print(f"assembled screen {len(results)}/{len(tasks)}", flush=True)
    return sorted(results, key=lambda row: int(row["trajectory_id"]))


def _reference_fields(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "normal_success": bool(run["normal_success"]),
        "final_success": bool(run["success"]),
        "normal_evaluations": int(run["normal_evaluations"]),
        "total_evaluations": int(run["total_evaluations"]),
    }


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_paths = {
        "ordinary_cell_winner": ORDINARY_CHECKPOINT,
        "largest_cell_winner": LARGEST_CHECKPOINT,
        "final_selector": SELECTOR_CHECKPOINT,
    }
    hashes_before = {name: _sha256(path) for name, path in frozen_paths.items()}
    if hashes_before["final_selector"] != _sha256(SELECTOR_SOURCE):
        raise RuntimeError("frozen final selector differs from source checkpoint")

    train_records_all = _state_records("train")
    validation_records_all = _state_records("validation")
    train_records = _sample_records(train_records_all, TRAIN_STATES, DATA_SEED)
    validation_records = _sample_records(
        validation_records_all, VALIDATION_STATES, DATA_SEED + 1
    )
    del train_records_all, validation_records_all
    gc.collect()
    deterministic_accuracy = float(
        np.mean(
            [
                int(np.argmax(record["sizes"])) == int(record["target"])
                for record in validation_records
            ]
        )
    )
    exact_ties_train = sum(bool(record["exact_tie"]) for record in train_records)
    exact_ties_validation = sum(
        bool(record["exact_tie"]) for record in validation_records
    )
    train_padded = _pad(train_records)
    validation_padded = _pad(validation_records)
    model, rows, best_epoch, metrics = _train(train_padded, validation_padded)
    predictions = metrics.pop("predictions")
    diagnostic_rows, errors = _diagnostics(
        train_records, validation_records, predictions
    )
    rows.append(
        {
            "record_type": "static_summary",
            "best_epoch": best_epoch,
            "train_states": TRAIN_STATES,
            "validation_states": VALIDATION_STATES,
            "validation_loss": metrics["loss"],
            "validation_top1_accuracy": metrics["accuracy"],
            "exact_ties_train": exact_ties_train,
            "exact_ties_validation": exact_ties_validation,
            "deterministic_argmax_accuracy": deterministic_accuracy,
            "static_gate": STATIC_GATE,
            "passed": metrics["accuracy"] >= STATIC_GATE
            and deterministic_accuracy == 1.0,
        }
    )
    rows.extend(diagnostic_rows)
    _write_csv(ERRORS_PATH, errors, default_fields=("validation_index",))
    static_passed = bool(
        metrics["accuracy"] >= STATIC_GATE and deterministic_accuracy == 1.0
    )
    _save_checkpoint(model, best_epoch, metrics, hashes_before, None)
    if not static_passed:
        hashes_after = {name: _sha256(path) for name, path in frozen_paths.items()}
        rows.append(
            {
                "record_type": "integrity",
                **{
                    f"{name}_unchanged": hashes_before[name] == hashes_after[name]
                    for name in frozen_paths
                },
                "assembled_screen_run": False,
            }
        )
        _write_csv(SUMMARY_PATH, rows)
        _write_csv(SCREEN_PATH, [], default_fields=("trajectory_id",))
        print("static routing was not perfect; assembled screen skipped", flush=True)
        return

    validation_positions, _, _ = _make_exact_labeled_split(
        imitation.VALIDATION_COUNT, imitation.VALIDATION_SEED
    )
    positions = validation_positions.to(dtype=DTYPE).cpu().numpy()
    learned = _run_screen(positions, workers)
    reference_payload = torch.load(
        REFERENCE_TRACE, map_location="cpu", weights_only=False
    )
    reference = reference_payload["screen"]
    reference_by_id = {int(row["trajectory_id"]): row for row in reference}
    comparisons: list[dict[str, Any]] = []
    for run_result in learned:
        teacher = _reference_fields(reference_by_id[int(run_result["trajectory_id"])])
        comparisons.append(
            {
                **run_result,
                "reference_normal_success": teacher["normal_success"],
                "reference_final_success": teacher["final_success"],
                "reference_normal_evaluations": teacher["normal_evaluations"],
                "reference_total_evaluations": teacher["total_evaluations"],
                "normal_outcome_agreement": bool(run_result["normal_success"])
                == teacher["normal_success"],
                "final_outcome_agreement": bool(run_result["final_success"])
                == teacher["final_success"],
                "evaluation_difference": int(run_result["total_evaluations"])
                - teacher["total_evaluations"],
            }
        )
    screen_preserved = bool(
        all(row["normal_outcome_agreement"] for row in comparisons)
        and all(row["final_outcome_agreement"] for row in comparisons)
        and sum(bool(row["final_success"]) for row in comparisons)
        >= sum(bool(row["reference_final_success"]) for row in comparisons)
    )
    rows.append(
        {
            "record_type": "assembled_screen",
            "trajectories": len(comparisons),
            "learned_normal_successes": sum(
                bool(row["normal_success"]) for row in comparisons
            ),
            "reference_normal_successes": sum(
                bool(row["reference_normal_success"]) for row in comparisons
            ),
            "learned_final_successes": sum(
                bool(row["final_success"]) for row in comparisons
            ),
            "reference_final_successes": sum(
                bool(row["reference_final_success"]) for row in comparisons
            ),
            "normal_outcome_agreement": float(
                np.mean([row["normal_outcome_agreement"] for row in comparisons])
            ),
            "final_outcome_agreement": float(
                np.mean([row["final_outcome_agreement"] for row in comparisons])
            ),
            "router_calls": sum(int(row["router_calls"]) for row in comparisons),
            "router_disagreements": sum(
                int(row["router_disagreements"]) for row in comparisons
            ),
            "learned_median_total_evaluations": float(
                np.median([row["total_evaluations"] for row in comparisons])
            ),
            "reference_median_total_evaluations": float(
                np.median([row["reference_total_evaluations"] for row in comparisons])
            ),
            "screen_reliability_preserved": screen_preserved,
            "full_validation_or_test_run": False,
        }
    )
    hashes_after = {name: _sha256(path) for name, path in frozen_paths.items()}
    rows.append(
        {
            "record_type": "integrity",
            **{
                f"{name}_before": hashes_before[name]
                for name in frozen_paths
            },
            **{
                f"{name}_after": hashes_after[name]
                for name in frozen_paths
            },
            **{
                f"{name}_unchanged": hashes_before[name] == hashes_after[name]
                for name in frozen_paths
            },
            "assembled_screen_run": True,
        }
    )
    _save_checkpoint(model, best_epoch, metrics, hashes_before, screen_preserved)
    _write_csv(SUMMARY_PATH, rows)
    _write_csv(SCREEN_PATH, comparisons, default_fields=("trajectory_id",))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
