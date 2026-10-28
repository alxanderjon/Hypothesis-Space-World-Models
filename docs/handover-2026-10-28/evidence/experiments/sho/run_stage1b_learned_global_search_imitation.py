#!/usr/bin/env python3
"""Imitate deterministic eDIRECT candidate priority for 2-D K(a,b) search."""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_adaptive_boundary_rescue as boundary
import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_nested1d_rescue as nested
from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    UPPER,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_global_search_imitation")
TEACHER_TRACE_PATH = OUTPUT_DIR / "teacher_traces.pt"
CHECKPOINT_PATH = OUTPUT_DIR / "best_selector_checkpoint.pt"
SEARCH_TRACE_PATH = OUTPUT_DIR / "validation_test_search_traces.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"

TRAIN_COUNT = 256
VALIDATION_COUNT = 64
TEST_COUNT = 128
TRAIN_SEED = FRESH_TEST_SEED + 1_100_000
VALIDATION_SEED = FRESH_TEST_SEED + 1_200_000
TEST_SEED = FRESH_TEST_SEED + 1_300_000
SPLIT_ID_OFFSET = {"train": 0, "validation": 100_000, "test": 200_000}
NORMAL_CAP = edirect.MAX_EVALUATIONS
RESCUE_CAP = boundary.RESCUE_BUDGET
GATE = edirect.HANDOFF_GATE
TRACE_PERMUTATION_SEED = SEED + 12_100_000
MODEL_SEED = SEED + 12_200_000
FEATURE_DIM = 14
GLOBAL_DIM = 16
HIDDEN_DIM = 64
TRAIN_SAMPLE_CAP = 100_000
INTERNAL_VALIDATION_SAMPLE_CAP = 25_000
BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1.0e-3

MODE_INDEX = {"normal_edirect": 0, "boundary_rescue": 1, "nested_1d": 2}


@dataclass(slots=True)
class ModeRun:
    mode: str
    bounds: np.ndarray
    points: np.ndarray
    losses: np.ndarray
    new_evaluations: int
    success: bool
    decisions: list[dict[str, Any]] = field(default_factory=list)
    teacher_agreements: int = 0
    learned_decisions: int = 0

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.losses))

    @property
    def best_loss(self) -> float:
        return float(self.losses[self.best_index])

    @property
    def best_physical(self) -> np.ndarray:
        return self.bounds[:, 0] + self.points[self.best_index] * (
            self.bounds[:, 1] - self.bounds[:, 0]
        )


@dataclass(slots=True)
class PipelineRun:
    split: str
    trajectory_id: int
    generator_id: int
    selector: str
    success: bool
    final_loss: float
    final_theta: np.ndarray
    total_evaluations: int
    normal_success: bool
    normal_evaluations: int
    boundary_invoked: bool
    boundary_success: bool
    boundary_evaluations: int
    nested_invoked: bool
    nested_success: bool
    nested_evaluations: int
    teacher_agreements: int
    learned_decisions: int
    modes: list[ModeRun]


class CandidateRanker(nn.Module):
    """Permutation-equivariant scores over a variable candidate set."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * HIDDEN_DIM + GLOBAL_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(candidate_features)
        mask_float = mask[:, :, None].to(encoded.dtype)
        mean = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        negative = torch.finfo(encoded.dtype).min
        maximum = torch.where(mask[:, :, None], encoded, negative).max(dim=1).values
        context = torch.cat((mean, maximum, global_features), dim=1)
        expanded = context[:, None, :].expand(-1, encoded.shape[1], -1)
        scores = self.head(torch.cat((encoded, expanded), dim=2)).squeeze(2)
        return scores.masked_fill(~mask, -torch.inf)


def _physical(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return bounds[:, 0] + points * (bounds[:, 1] - bounds[:, 0])


def _evaluate(points: np.ndarray, position: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return edirect._prefix_losses_numpy(_physical(points, bounds), position)


def _global_features(
    points: np.ndarray,
    losses: np.ndarray,
    candidate_count: int,
    evaluations: int,
    cap: int,
    iteration: int,
    bounds: np.ndarray,
    mode: str,
) -> np.ndarray:
    log_losses = np.log10(np.maximum(losses, edirect.LOSS_FLOOR))
    best_index = int(np.argmin(losses))
    recent_gap = max(0.0, float(log_losses[-1] - log_losses[best_index]))
    one_hot = np.zeros(3, dtype=np.float64)
    one_hot[MODE_INDEX[mode]] = 1.0
    return np.asarray(
        (
            evaluations / float(max(1, cap)),
            math.log1p(len(points)) / math.log1p(max(2, cap)),
            float(log_losses[best_index]) / 20.0,
            float(np.median(log_losses)) / 20.0,
            float(np.std(log_losses)) / 20.0,
            float(points[best_index, 0]),
            float(points[best_index, 1]),
            min(candidate_count, 32) / 32.0,
            math.log1p(iteration) / math.log1p(max(2, cap)),
            float(bounds[0, 1] - bounds[0, 0]) / DOMAIN_WIDTH,
            float(bounds[1, 1] - bounds[1, 0]) / DOMAIN_WIDTH,
            min(recent_gap, 60.0) / 60.0,
            float(one_hot[0]),
            float(one_hot[1]),
            float(one_hot[2]),
            float(np.ptp(log_losses)) / 60.0,
        ),
        dtype=np.float32,
    )


def _candidate_features(
    points: np.ndarray,
    losses: np.ndarray,
    candidates: np.ndarray,
    candidate_type: str,
) -> np.ndarray:
    candidates = np.atleast_2d(candidates).astype(np.float64, copy=False)
    tree = edirect.cKDTree(points)
    distances, owners = tree.query(candidates, k=1)
    best_index = int(np.argmin(losses))
    best = points[best_index]
    log_losses = np.log10(np.maximum(losses, edirect.LOSS_FLOOR))
    if len(points) > 1:
        unique_owners, owner_inverse = np.unique(owners, return_inverse=True)
        owner_spacing = tree.query(points[unique_owners], k=2)[0][:, 1]
        candidate_spacing = owner_spacing[owner_inverse]
    else:
        candidate_spacing = np.ones(len(candidates), dtype=np.float64)
    surface = edirect.GaussianLocalResponse(points, losses)
    prediction = surface.predict(candidates)
    center = np.mean(candidates, axis=0)
    features = np.empty((len(candidates), FEATURE_DIM), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        radius = max(
            float(distances[index]) * 2.0,
            float(candidate_spacing[index]),
            1e-12,
        )
        local_density = len(tree.query_ball_point(candidate, radius)) / max(1, len(points))
        boundary_distance = float(np.min(np.concatenate((candidate, 1.0 - candidate))))
        features[index] = np.asarray(
            (
                candidate[0],
                candidate[1],
                distances[index],
                np.linalg.norm(candidate - best),
                prediction[index] / 20.0,
                log_losses[owners[index]] / 20.0,
                np.clip(prediction[index] - log_losses[best_index], 0.0, 60.0) / 60.0,
                boundary_distance,
                candidate_spacing[index],
                local_density,
                np.linalg.norm(candidate - center),
                float(candidate_type == "greedy"),
                float(candidate_type == "initial_design"),
                math.log1p(len(points)) / math.log1p(NORMAL_CAP),
            ),
            dtype=np.float32,
        )
    return features


def _permuted_teacher_record(
    split: str,
    trajectory_id: int,
    mode: str,
    decision_index: int,
    points: np.ndarray,
    losses: np.ndarray,
    candidates: np.ndarray,
    candidate_type: str,
    global_features: np.ndarray,
    resulting_loss: float,
) -> dict[str, Any]:
    features = _candidate_features(points, losses, candidates, candidate_type)
    seed = (
        TRACE_PERMUTATION_SEED
        + 1_000_003 * trajectory_id
        + 10_007 * decision_index
        + 101 * MODE_INDEX[mode]
    )
    permutation = np.random.default_rng(seed).permutation(len(candidates))
    selected_index = int(np.flatnonzero(permutation == 0)[0])
    return {
        "split": split,
        "trajectory_id": trajectory_id,
        "mode": mode,
        "history_count": len(points),
        "candidate_type": candidate_type,
        "candidates": candidates[permutation].astype(np.float64),
        "candidate_features": features[permutation],
        "global_features": global_features.astype(np.float32),
        "selected_index": selected_index,
        "resulting_prefix_loss": resulting_loss,
    }


def _model_choice(
    model: CandidateRanker,
    points: np.ndarray,
    losses: np.ndarray,
    candidates: np.ndarray,
    candidate_type: str,
    global_features: np.ndarray,
) -> tuple[int, np.ndarray]:
    features = _candidate_features(points, losses, candidates, candidate_type)
    with torch.no_grad():
        scores = model(
            torch.tensor(features, dtype=torch.float32)[None, :, :],
            torch.ones((1, len(candidates)), dtype=torch.bool),
            torch.tensor(global_features, dtype=torch.float32)[None, :],
        )[0]
    return int(torch.argmax(scores)), scores.cpu().numpy()


def _run_edirect_mode(
    *,
    split: str,
    trajectory_id: int,
    generator_id: int,
    position: np.ndarray,
    bounds: np.ndarray,
    cap: int,
    mode: str,
    selector: str,
    model: CandidateRanker | None,
    seed_points: np.ndarray | None = None,
    seed_losses: np.ndarray | None = None,
    collect_trace: bool = True,
) -> ModeRun:
    points = (
        np.empty((0, 2), dtype=np.float64)
        if seed_points is None
        else seed_points.astype(np.float64, copy=True)
    )
    losses = (
        np.empty(0, dtype=np.float64)
        if seed_losses is None
        else seed_losses.astype(np.float64, copy=True)
    )
    decisions: list[dict[str, Any]] = []
    new_evaluations = 0
    iteration = 0
    decision_index = 0
    agreements = 0
    learned_decisions = 0

    initial = []
    for proposal in edirect._initial_design():
        if len(points) and np.min(np.linalg.norm(points - proposal[None, :], axis=1)) <= 1e-12:
            continue
        initial.append(proposal)
    pending = np.asarray(initial, dtype=np.float64)
    while len(pending) and new_evaluations < cap and (not len(losses) or float(np.min(losses)) > GATE):
        global_features = _global_features(
            points if len(points) else np.asarray(((0.5, 0.5),)),
            losses if len(losses) else np.asarray((1.0,)),
            len(pending),
            new_evaluations,
            cap,
            iteration,
            bounds,
            mode,
        )
        teacher_choice = 0
        if selector == "learned" and len(points):
            if model is None:
                raise RuntimeError("Learned selector is missing")
            choice, _ = _model_choice(
                model, points, losses, pending, "initial_design", global_features
            )
            agreements += int(choice == teacher_choice)
            learned_decisions += 1
        else:
            choice = teacher_choice
        proposal = pending[choice]
        loss = float(_evaluate(proposal[None, :], position, bounds)[0])
        if collect_trace and selector == "teacher" and len(points):
            decisions.append(
                _permuted_teacher_record(
                    split,
                    trajectory_id,
                    mode,
                    decision_index,
                    points,
                    losses,
                    pending,
                    "initial_design",
                    global_features,
                    loss,
                )
            )
        elif collect_trace and selector == "learned" and len(points):
            decisions.append(
                {
                    "mode": mode,
                    "history_count": len(points),
                    "candidate_type": "initial_design",
                    "candidate_count": len(pending),
                    "teacher_choice": teacher_choice,
                    "learned_choice": choice,
                    "agreement": int(choice == teacher_choice),
                    "selected_candidate": proposal.copy(),
                    "resulting_prefix_loss": loss,
                }
            )
        points = np.concatenate((points, proposal[None, :]), axis=0)
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, choice, axis=0)
        new_evaluations += 1
        decision_index += 1

    while new_evaluations < cap and float(np.min(losses)) > GATE:
        proposals = edirect._cell_proposals(points, losses, generator_id, iteration)
        proposals = np.asarray(proposals, dtype=np.float64).reshape(-1, 2)
        pending = proposals.copy()
        while len(pending) and new_evaluations < cap and float(np.min(losses)) > GATE:
            global_features = _global_features(
                points,
                losses,
                len(pending),
                new_evaluations,
                cap,
                iteration,
                bounds,
                mode,
            )
            teacher_choice = 0
            if selector == "learned":
                if model is None:
                    raise RuntimeError("Learned selector is missing")
                choice, _ = _model_choice(
                    model, points, losses, pending, "cell", global_features
                )
                agreements += int(choice == teacher_choice)
                learned_decisions += 1
            else:
                choice = teacher_choice
            proposal = pending[choice]
            loss = float(_evaluate(proposal[None, :], position, bounds)[0])
            if collect_trace and selector == "teacher":
                decisions.append(
                    _permuted_teacher_record(
                        split,
                        trajectory_id,
                        mode,
                        decision_index,
                        points,
                        losses,
                        pending,
                        "cell",
                        global_features,
                        loss,
                    )
                )
            elif collect_trace:
                decisions.append(
                    {
                        "mode": mode,
                        "history_count": len(points),
                        "candidate_type": "cell",
                        "candidate_count": len(pending),
                        "teacher_choice": teacher_choice,
                        "learned_choice": choice,
                        "agreement": int(choice == teacher_choice),
                        "selected_candidate": proposal.copy(),
                        "resulting_prefix_loss": loss,
                    }
                )
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, choice, axis=0)
            new_evaluations += 1
            decision_index += 1
        if new_evaluations >= cap or float(np.min(losses)) <= GATE:
            break
        greedy = edirect._greedy_proposal(points, losses, generator_id, iteration)
        if greedy is not None:
            loss = float(_evaluate(greedy[None, :], position, bounds)[0])
            global_features = _global_features(
                points, losses, 1, new_evaluations, cap, iteration, bounds, mode
            )
            if collect_trace and selector == "teacher":
                decisions.append(
                    _permuted_teacher_record(
                        split,
                        trajectory_id,
                        mode,
                        decision_index,
                        points,
                        losses,
                        greedy[None, :],
                        "greedy",
                        global_features,
                        loss,
                    )
                )
            elif collect_trace:
                decisions.append(
                    {
                        "mode": mode,
                        "history_count": len(points),
                        "candidate_type": "greedy",
                        "candidate_count": 1,
                        "teacher_choice": 0,
                        "learned_choice": 0,
                        "agreement": 1,
                        "selected_candidate": greedy.copy(),
                        "resulting_prefix_loss": loss,
                    }
                )
            points = np.concatenate((points, greedy[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray((loss,))))
            new_evaluations += 1
            decision_index += 1
        iteration += 1
    return ModeRun(
        mode=mode,
        bounds=bounds.copy(),
        points=points,
        losses=losses,
        new_evaluations=new_evaluations,
        success=float(np.min(losses)) <= GATE,
        decisions=decisions,
        teacher_agreements=agreements,
        learned_decisions=learned_decisions,
    )


def _run_boundary_rescue(
    *,
    split: str,
    trajectory_id: int,
    generator_id: int,
    position: np.ndarray,
    normal: ModeRun,
    collect_trace: bool,
) -> tuple[ModeRun, dict[str, object]]:
    physical = _physical(normal.points, normal.bounds)
    best = physical[normal.best_index]
    face, axis, is_lower, distance = boundary._nearest_face(best)
    spacing = boundary._local_normal_spacing(physical, best, axis)
    bounds, delta = boundary._strip_bounds(best, axis, is_lower, spacing)
    inside = np.all(
        (physical >= bounds[:, 0] - 1e-14)
        & (physical <= bounds[:, 1] + 1e-14),
        axis=1,
    )
    seed_physical = physical[inside]
    seed_points = (seed_physical - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    seed_losses = normal.losses[inside]
    run = _run_edirect_mode(
        split=split,
        trajectory_id=trajectory_id,
        generator_id=generator_id,
        position=position,
        bounds=bounds,
        cap=RESCUE_CAP,
        mode="boundary_rescue",
        selector="teacher",
        model=None,
        seed_points=seed_points,
        seed_losses=seed_losses,
        collect_trace=collect_trace,
    )
    return run, {
        "face": face,
        "axis": axis,
        "distance": distance,
        "spacing": spacing,
        "delta": delta,
    }


class _GateReached(Exception):
    pass


def _run_nested_rescue(
    *,
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    boundary_run: ModeRun,
    collect_trace: bool,
) -> ModeRun:
    bounds = boundary_run.bounds
    best = boundary_run.best_physical
    evaluated_points: list[np.ndarray] = []
    evaluated_losses: list[float] = []
    decisions: list[dict[str, Any]] = []
    decision_index = 0

    def evaluate_candidates(candidates: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        nonlocal decision_index
        results: list[tuple[np.ndarray, float]] = []
        pending = np.asarray(candidates, dtype=np.float64)
        while len(pending):
            proposal = pending[0]
            loss = float(edirect._prefix_losses_numpy(proposal[None, :], position)[0])
            if collect_trace and evaluated_points:
                history_physical = np.asarray(evaluated_points)
                history_normalized = (history_physical - bounds[:, 0]) / (
                    bounds[:, 1] - bounds[:, 0]
                )
                candidate_normalized = (pending - bounds[:, 0]) / (
                    bounds[:, 1] - bounds[:, 0]
                )
                global_features = _global_features(
                    history_normalized,
                    np.asarray(evaluated_losses),
                    len(pending),
                    len(evaluated_points),
                    10_000,
                    decision_index,
                    bounds,
                    "nested_1d",
                )
                decisions.append(
                    _permuted_teacher_record(
                        split,
                        trajectory_id,
                        "nested_1d",
                        decision_index,
                        history_normalized,
                        np.asarray(evaluated_losses),
                        candidate_normalized,
                        "cell",
                        global_features,
                        loss,
                    )
                )
            evaluated_points.append(proposal.copy())
            evaluated_losses.append(loss)
            results.append((proposal.copy(), loss))
            pending = pending[1:]
            decision_index += 1
            if loss <= GATE:
                raise _GateReached
        return results

    def refine_a(parameter_b: float) -> tuple[float, float]:
        lower, upper = float(bounds[0, 0]), float(bounds[0, 1])
        best_a = float(best[0])
        best_loss = math.inf
        levels = [(nested.INNER_INITIAL_POINTS, True)] + [
            (nested.REFINEMENT_POINTS, False) for _ in range(nested.REFINEMENT_LEVELS)
        ]
        for count, first in levels:
            values = nested._ordered_unique(
                np.linspace(lower, upper, count), best_a if first else None
            )
            evaluated = evaluate_candidates(
                [np.asarray((value, parameter_b)) for value in values]
            )
            ordered = sorted((float(point[0]), loss) for point, loss in evaluated)
            index = min(range(len(ordered)), key=lambda item: ordered[item][1])
            if ordered[index][1] < best_loss:
                best_a, best_loss = ordered[index]
            lower = ordered[max(0, index - 1)][0]
            upper = ordered[min(len(ordered) - 1, index + 1)][0]
            if upper - lower <= 1e-15:
                break
        return best_a, best_loss

    cache: dict[float, tuple[float, float]] = {}
    try:
        lower, upper = float(bounds[1, 0]), float(bounds[1, 1])
        best_b = float(best[1])
        levels = [(nested.OUTER_INITIAL_POINTS, True)] + [
            (nested.REFINEMENT_POINTS, False) for _ in range(nested.REFINEMENT_LEVELS)
        ]
        for count, first in levels:
            values = nested._ordered_unique(
                np.linspace(lower, upper, count), best_b if first else None
            )
            level: list[tuple[float, float]] = []
            for value in values:
                if value not in cache:
                    cache[value] = refine_a(value)
                level.append((value, cache[value][1]))
            level.sort()
            index = min(range(len(level)), key=lambda item: level[item][1])
            best_b = level[index][0]
            lower = level[max(0, index - 1)][0]
            upper = level[min(len(level) - 1, index + 1)][0]
            if upper - lower <= 1e-15:
                break
    except _GateReached:
        pass
    points_physical = np.asarray(evaluated_points)
    losses = np.asarray(evaluated_losses)
    points = (points_physical - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return ModeRun(
        mode="nested_1d",
        bounds=bounds.copy(),
        points=points,
        losses=losses,
        new_evaluations=len(losses),
        success=float(np.min(losses)) <= GATE,
        decisions=decisions,
    )


def _run_pipeline(
    *,
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    selector: str,
    model: CandidateRanker | None,
    collect_trace: bool,
) -> PipelineRun:
    generator_id = SPLIT_ID_OFFSET[split] + trajectory_id
    full_bounds = np.asarray(((LOWER, UPPER), (LOWER, UPPER)), dtype=np.float64)
    normal = _run_edirect_mode(
        split=split,
        trajectory_id=trajectory_id,
        generator_id=generator_id,
        position=position,
        bounds=full_bounds,
        cap=NORMAL_CAP,
        mode="normal_edirect",
        selector=selector,
        model=model,
        collect_trace=collect_trace,
    )
    modes = [normal]
    total = normal.new_evaluations
    boundary_run: ModeRun | None = None
    nested_run: ModeRun | None = None
    if not normal.success:
        boundary_run, _ = _run_boundary_rescue(
            split=split,
            trajectory_id=trajectory_id,
            generator_id=generator_id,
            position=position,
            normal=normal,
            collect_trace=collect_trace,
        )
        modes.append(boundary_run)
        total += boundary_run.new_evaluations
    if boundary_run is not None and not boundary_run.success:
        nested_run = _run_nested_rescue(
            split=split,
            trajectory_id=trajectory_id,
            position=position,
            boundary_run=boundary_run,
            collect_trace=collect_trace,
        )
        modes.append(nested_run)
        total += nested_run.new_evaluations
    final = next((mode for mode in modes[::-1] if mode.success), modes[-1])
    agreements = sum(mode.teacher_agreements for mode in modes)
    learned_decisions = sum(mode.learned_decisions for mode in modes)
    return PipelineRun(
        split=split,
        trajectory_id=trajectory_id,
        generator_id=generator_id,
        selector=selector,
        success=any(mode.success for mode in modes),
        final_loss=final.best_loss,
        final_theta=final.best_physical,
        total_evaluations=total,
        normal_success=normal.success,
        normal_evaluations=normal.new_evaluations,
        boundary_invoked=boundary_run is not None,
        boundary_success=bool(boundary_run and boundary_run.success),
        boundary_evaluations=boundary_run.new_evaluations if boundary_run else 0,
        nested_invoked=nested_run is not None,
        nested_success=bool(nested_run and nested_run.success),
        nested_evaluations=nested_run.new_evaluations if nested_run else 0,
        teacher_agreements=agreements,
        learned_decisions=learned_decisions,
        modes=modes,
    )


def _teacher_worker(arguments: tuple[str, int, np.ndarray, bool]) -> PipelineRun:
    split, trajectory_id, position, collect_trace = arguments
    return _run_pipeline(
        split=split,
        trajectory_id=trajectory_id,
        position=position,
        selector="teacher",
        model=None,
        collect_trace=collect_trace,
    )


def _run_teachers(
    split: str,
    positions: np.ndarray,
    workers: int,
    collect_trace: bool,
) -> list[PipelineRun]:
    tasks = [
        (split, trajectory_id, positions[trajectory_id], collect_trace)
        for trajectory_id in range(len(positions))
    ]
    if workers <= 1:
        results = []
        for task in tasks:
            results.append(_teacher_worker(task))
            print(f"teacher {split} {len(results)}/{len(tasks)}", flush=True)
        return results
    results: list[PipelineRun] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_teacher_worker, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 8 == 0 or len(results) == len(tasks):
                print(f"teacher {split} {len(results)}/{len(tasks)}", flush=True)
    results.sort(key=lambda item: item.trajectory_id)
    return results


def _trace_payload(runs: list[PipelineRun]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for run in runs:
        mode_payload: list[dict[str, Any]] = []
        for mode in run.modes:
            decisions.extend(mode.decisions)
            mode_payload.append(
                {
                    "mode": mode.mode,
                    "bounds": mode.bounds,
                    "evaluated_hypotheses_normalized": mode.points,
                    "evaluated_prefix_losses": mode.losses,
                    "new_evaluations": mode.new_evaluations,
                    "success": mode.success,
                }
            )
        trajectories.append(
            {
                "split": run.split,
                "trajectory_id": run.trajectory_id,
                "generator_id": run.generator_id,
                "selector": run.selector,
                "success": run.success,
                "total_evaluations": run.total_evaluations,
                "final_loss": run.final_loss,
                "final_theta": run.final_theta,
                "modes": mode_payload,
            }
        )
    return {"trajectories": trajectories, "decisions": decisions}


def _decision_indices(
    decisions: list[dict[str, Any]], internal_validation: bool
) -> list[int]:
    indices = [
        index
        for index, decision in enumerate(decisions)
        if len(decision["candidate_features"]) > 1
        and ((decision["trajectory_id"] % 5 == 0) == internal_validation)
    ]
    cap = INTERNAL_VALIDATION_SAMPLE_CAP if internal_validation else TRAIN_SAMPLE_CAP
    if len(indices) > cap:
        generator = np.random.default_rng(MODEL_SEED + int(internal_validation))
        indices = sorted(generator.choice(indices, cap, replace=False).tolist())
    return indices


def _batch(
    decisions: list[dict[str, Any]], indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(decisions[index]["candidate_features"]) for index in indices)
    features = torch.zeros((len(indices), maximum, FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((len(indices), maximum), dtype=torch.bool)
    global_features = torch.empty((len(indices), GLOBAL_DIM), dtype=torch.float32)
    targets = torch.empty(len(indices), dtype=torch.long)
    for row, index in enumerate(indices):
        decision = decisions[index]
        count = len(decision["candidate_features"])
        features[row, :count] = torch.tensor(decision["candidate_features"])
        mask[row, :count] = True
        global_features[row] = torch.tensor(decision["global_features"])
        targets[row] = int(decision["selected_index"])
    return features, mask, global_features, targets


def _evaluate_ranking(
    model: CandidateRanker,
    decisions: list[dict[str, Any]],
    indices: list[int],
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    mode_correct: dict[str, list[int]] = {}
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            selected = indices[start : start + BATCH_SIZE]
            features, mask, global_features, targets = _batch(decisions, selected)
            scores = model(features, mask, global_features)
            loss = nn.functional.cross_entropy(scores, targets, reduction="sum")
            predictions = torch.argmax(scores, dim=1)
            total_loss += float(loss)
            total_correct += int(torch.sum(predictions == targets))
            for local_index, decision_index in enumerate(selected):
                mode = decisions[decision_index]["mode"]
                counts = mode_correct.setdefault(mode, [0, 0])
                counts[0] += int(predictions[local_index] == targets[local_index])
                counts[1] += 1
    mode_accuracy = {
        mode: correct / count for mode, (correct, count) in mode_correct.items()
    }
    return total_loss / max(1, len(indices)), total_correct / max(1, len(indices)), mode_accuracy


def _train_selector(
    decisions: list[dict[str, Any]],
) -> tuple[CandidateRanker, list[dict[str, object]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = CandidateRanker().to(dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_indices = _decision_indices(decisions, False)
    validation_indices = _decision_indices(decisions, True)
    if not train_indices or not validation_indices:
        raise RuntimeError("Insufficient nontrivial teacher decisions")
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator).tolist()
        epoch_loss = 0.0
        epoch_correct = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch_positions = permutation[start : start + BATCH_SIZE]
            selected = [train_indices[position] for position in batch_positions]
            features, mask, global_features, targets = _batch(decisions, selected)
            optimizer.zero_grad(set_to_none=True)
            scores = model(features, mask, global_features)
            loss = nn.functional.cross_entropy(scores, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * len(selected)
            epoch_correct += int(torch.sum(torch.argmax(scores, dim=1) == targets))
        validation_loss, validation_accuracy, mode_accuracy = _evaluate_ranking(
            model, decisions, validation_indices
        )
        row = {
            "row_type": "training_epoch",
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_indices),
            "train_accuracy": epoch_correct / len(train_indices),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_mode_accuracy": ";".join(
                f"{mode}:{value:.8f}" for mode, value in sorted(mode_accuracy.items())
            ),
        }
        history.append(row)
        improved = validation_accuracy > best_accuracy + 1e-12 or (
            abs(validation_accuracy - best_accuracy) <= 1e-12
            and validation_loss < best_loss
        )
        if improved:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": best_state,
                    "epoch": best_epoch,
                    "validation_accuracy": best_accuracy,
                    "validation_loss": best_loss,
                    "feature_dim": FEATURE_DIM,
                    "global_dim": GLOBAL_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "train_seed": TRAIN_SEED,
                    "model_seed": MODEL_SEED,
                    "candidate_generation": "unchanged deterministic eDIRECT proposal sets",
                },
                CHECKPOINT_PATH,
            )
        else:
            stale += 1
        print(
            f"selector epoch={epoch} train_acc={row['train_accuracy']:.4f} "
            f"val_acc={validation_accuracy:.4f}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    _, _, best_mode_accuracy = _evaluate_ranking(
        model, decisions, validation_indices
    )
    validation_mode_counts = {
        mode: sum(decisions[index]["mode"] == mode for index in validation_indices)
        for mode in MODE_INDEX
    }
    return model, history, {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        **{
            f"heldout_{mode}_accuracy": best_mode_accuracy.get(mode, "")
            for mode in MODE_INDEX
        },
        **{
            f"heldout_{mode}_examples": validation_mode_counts[mode]
            for mode in MODE_INDEX
        },
    }


_FROZEN_WORKER_MODEL: CandidateRanker | None = None


def _learned_worker(arguments: tuple[str, int, np.ndarray]) -> PipelineRun:
    global _FROZEN_WORKER_MODEL
    split, trajectory_id, position = arguments
    if _FROZEN_WORKER_MODEL is None:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _FROZEN_WORKER_MODEL = CandidateRanker().to(dtype=torch.float32)
        _FROZEN_WORKER_MODEL.load_state_dict(checkpoint["state_dict"])
        _FROZEN_WORKER_MODEL.eval()
    return _run_pipeline(
        split=split,
        trajectory_id=trajectory_id,
        position=position,
        selector="learned",
        model=_FROZEN_WORKER_MODEL,
        collect_trace=True,
    )


def _run_learned(
    split: str,
    positions: np.ndarray,
    model: CandidateRanker,
    workers: int = 1,
) -> list[PipelineRun]:
    tasks = [
        (split, trajectory_id, positions[trajectory_id])
        for trajectory_id in range(len(positions))
    ]
    if workers <= 1:
        results = []
        for task in tasks:
            results.append(
                _run_pipeline(
                    split=task[0],
                    trajectory_id=task[1],
                    position=task[2],
                    selector="learned",
                    model=model,
                    collect_trace=True,
                )
            )
            if len(results) % 8 == 0 or len(results) == len(positions):
                print(f"learned {split} {len(results)}/{len(positions)}", flush=True)
        return results
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_learned_worker, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 8 == 0 or len(results) == len(positions):
                print(f"learned {split} {len(results)}/{len(positions)}", flush=True)
    results.sort(key=lambda item: item.trajectory_id)
    return results


def _pipeline_rows(
    runs: list[PipelineRun], row_type: str = "search_summary"
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in sorted({run.split for run in runs}):
        split_runs = [run for run in runs if run.split == split]
        for selector in sorted({run.selector for run in split_runs}):
            selected = [run for run in split_runs if run.selector == selector]
            successes = [run for run in selected if run.success]
            decisions = sum(run.learned_decisions for run in selected)
            rows.append(
                {
                    "row_type": row_type,
                    "split": split,
                    "selector": selector,
                    "trajectories": len(selected),
                    "normal_success_count": sum(run.normal_success for run in selected),
                    "normal_success_fraction": sum(run.normal_success for run in selected) / len(selected),
                    "boundary_rescue_invocations": sum(run.boundary_invoked for run in selected),
                    "boundary_rescue_successes": sum(run.boundary_success for run in selected),
                    "nested_1d_invocations": sum(run.nested_invoked for run in selected),
                    "nested_1d_successes": sum(run.nested_success for run in selected),
                    "final_success_count": len(successes),
                    "final_success_fraction": len(successes) / len(selected),
                    "median_total_evaluations_success": (
                        float(np.median([run.total_evaluations for run in successes]))
                        if successes
                        else ""
                    ),
                    "mean_total_evaluations_success": (
                        float(np.mean([run.total_evaluations for run in successes]))
                        if successes
                        else ""
                    ),
                    "worst_total_evaluations_success": max(
                        [run.total_evaluations for run in successes], default=""
                    ),
                    "teacher_decision_agreement": (
                        sum(run.teacher_agreements for run in selected) / decisions
                        if decisions
                        else ""
                    ),
                    "learned_decision_count": decisions,
                    "gate": GATE,
                    "normal_cap": NORMAL_CAP,
                    "rescue_cap": RESCUE_CAP,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume-teacher",
        action="store_true",
        help="load the already-persisted training teacher trace",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        action="store_true",
        help="skip fitting and load the persisted best selector checkpoint",
    )
    args = parser.parse_args()
    counts = (
        (8, 4, 4) if args.smoke else (TRAIN_COUNT, VALIDATION_COUNT, TEST_COUNT)
    )
    train_positions, _, _ = _make_exact_labeled_split(counts[0], TRAIN_SEED)
    validation_positions, _, _ = _make_exact_labeled_split(counts[1], VALIDATION_SEED)
    test_positions, _, _ = _make_exact_labeled_split(counts[2], TEST_SEED)
    train_numpy = train_positions.to(dtype=DTYPE).cpu().numpy()
    validation_numpy = validation_positions.to(dtype=DTYPE).cpu().numpy()
    test_numpy = test_positions.to(dtype=DTYPE).cpu().numpy()

    started = time.perf_counter()
    teacher_train: list[PipelineRun] | None = None
    if args.resume_teacher or args.evaluate_checkpoint:
        if not TEACHER_TRACE_PATH.exists():
            raise FileNotFoundError(f"missing teacher trace: {TEACHER_TRACE_PATH}")
        saved_teacher = torch.load(
            TEACHER_TRACE_PATH, map_location="cpu", weights_only=False
        )
        train_payload = {
            "trajectories": saved_teacher["trajectories"],
            "decisions": saved_teacher["decisions"],
        }
        print(
            f"loaded teacher trace with {len(train_payload['decisions'])} decisions",
            flush=True,
        )
    else:
        teacher_train = _run_teachers("train", train_numpy, args.workers, True)
        train_payload = _trace_payload(teacher_train)
    if args.smoke:
        nontrivial = sum(
            len(decision["candidate_features"]) > 1
            for decision in train_payload["decisions"]
        )
        print(f"smoke teacher decisions={len(train_payload['decisions'])} nontrivial={nontrivial}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        smoke_model, _, smoke_best = _train_selector(train_payload["decisions"])
        smoke_learned = _run_learned(
            "validation", validation_numpy[:1], smoke_model
        )[0]
        print(
            f"smoke selector val_acc={smoke_best['best_validation_accuracy']:.4f} "
            f"learned_success={smoke_learned.success} "
            f"evaluations={smoke_learned.total_evaluations}",
            flush=True,
        )
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (args.resume_teacher or args.evaluate_checkpoint):
        torch.save(
            {
                "config": {
                    "train_count": TRAIN_COUNT,
                    "train_seed": TRAIN_SEED,
                    "normal_cap": NORMAL_CAP,
                    "rescue_cap": RESCUE_CAP,
                    "gate": GATE,
                    "operator": "K(a,b)=[[0,1],[a,b]]",
                },
                **train_payload,
            },
            TEACHER_TRACE_PATH,
        )
    if args.evaluate_checkpoint:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        model = CandidateRanker().to(dtype=torch.float32)
        model.load_state_dict(checkpoint["state_dict"])
        validation_indices = _decision_indices(train_payload["decisions"], True)
        validation_loss, validation_accuracy, mode_accuracy = _evaluate_ranking(
            model, train_payload["decisions"], validation_indices
        )
        mode_counts = {
            mode: sum(
                train_payload["decisions"][index]["mode"] == mode
                for index in validation_indices
            )
            for mode in MODE_INDEX
        }
        training_history = []
        training_best = {
            "best_epoch": checkpoint["epoch"],
            "best_validation_accuracy": validation_accuracy,
            "best_validation_loss": validation_loss,
            "train_examples": len(_decision_indices(train_payload["decisions"], False)),
            "validation_examples": len(validation_indices),
            **{
                f"heldout_{mode}_accuracy": mode_accuracy.get(mode, "")
                for mode in MODE_INDEX
            },
            **{
                f"heldout_{mode}_examples": mode_counts[mode]
                for mode in MODE_INDEX
            },
        }
        print(
            f"loaded checkpoint epoch={checkpoint['epoch']} "
            f"heldout_accuracy={validation_accuracy:.4f}",
            flush=True,
        )
    else:
        model, training_history, training_best = _train_selector(
            train_payload["decisions"]
        )
    teacher_validation = _run_teachers(
        "validation", validation_numpy, args.workers, False
    )
    learned_validation = _run_learned(
        "validation", validation_numpy, model, args.workers
    )
    teacher_test = _run_teachers("test", test_numpy, args.workers, False)
    learned_test = _run_learned("test", test_numpy, model, args.workers)
    torch.save(
        {
            "config": {
                "validation_count": VALIDATION_COUNT,
                "test_count": TEST_COUNT,
                "validation_seed": VALIDATION_SEED,
                "test_seed": TEST_SEED,
                "normal_cap": NORMAL_CAP,
                "rescue_cap": RESCUE_CAP,
                "gate": GATE,
            },
            "validation_teacher": _trace_payload(teacher_validation),
            "validation_learned": _trace_payload(learned_validation),
            "test_teacher": _trace_payload(teacher_test),
            "test_learned": _trace_payload(learned_test),
        },
        SEARCH_TRACE_PATH,
    )
    summary_rows: list[dict[str, object]] = []
    summary_rows.extend(training_history)
    summary_rows.append(
        {
            "row_type": "training_best",
            **training_best,
            "teacher_trace_decisions": len(train_payload["decisions"]),
            "teacher_train_final_success_fraction": sum(
                bool(run["success"]) for run in train_payload["trajectories"]
            )
            / len(train_payload["trajectories"]),
            "runtime_seconds_before_frozen_evaluation": time.perf_counter() - started,
        }
    )
    summary_rows.extend(
        _pipeline_rows(
            teacher_validation
            + learned_validation
            + teacher_test
            + learned_test
        )
    )
    _write_csv(SUMMARY_PATH, summary_rows)
    print(f"wrote learned global-search imitation artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
