#!/usr/bin/env python3
"""Nested deterministic 1-D rescue for unresolved adaptive-strip cases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.sho.run_stage1b_edirect_vs_direct import (
    HANDOFF_GATE,
    _prefix_losses_numpy,
    _read_csv,
    _write_csv,
)
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    _posthoc_reference,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/nested1d_and_network2_2d")
SOURCE = Path(
    "outputs/sho/stage1B/adaptive_boundary_rescue/adaptive_boundary_rescue_cases.csv"
)
TRAJECTORY_IDS = (36, 230)
INNER_INITIAL_POINTS = 33
OUTER_INITIAL_POINTS = 17
REFINEMENT_POINTS = 9
REFINEMENT_LEVELS = 5

SEARCH_RULE = (
    "nested coordinate-decomposed global search: for every b, approximate "
    "F(b)=min_a L(a,b) with the saved a seed plus 33 full-domain evenly "
    "spaced a values and 5 neighbor-retaining refinement levels of 9 values; "
    "search b inside the unchanged saved rescue strip using its saved b seed, "
    "17 full-strip values, and the same 5x9 refinement; cache duplicate (a,b) "
    "evaluations and stop immediately when any actual loss<=1e-4"
)


class GateReached(Exception):
    pass


@dataclass(slots=True)
class SearchState:
    position: np.ndarray
    count: int = 0
    best_loss: float = math.inf
    best_a: float = math.nan
    best_b: float = math.nan

    def evaluate(self, parameter_a: float, parameter_b: float) -> float:
        loss = float(
            _prefix_losses_numpy(
                np.asarray(((parameter_a, parameter_b),)), self.position
            )[0]
        )
        self.count += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_a = parameter_a
            self.best_b = parameter_b
        if loss <= HANDOFF_GATE:
            raise GateReached
        return loss


def _ordered_unique(values: np.ndarray, seed: float | None = None) -> list[float]:
    result: list[float] = []
    if seed is not None:
        result.append(float(seed))
    for value in values:
        value_float = float(value)
        if not any(abs(value_float - prior) <= 1.0e-15 for prior in result):
            result.append(value_float)
    return result


def _refine_1d(
    objective,
    lower: float,
    upper: float,
    initial_points: int,
    seed: float | None,
) -> tuple[float, float]:
    interval_lower = lower
    interval_upper = upper
    best_x = float(seed) if seed is not None else 0.5 * (lower + upper)
    best_value = math.inf
    levels = [(initial_points, True)] + [
        (REFINEMENT_POINTS, False) for _ in range(REFINEMENT_LEVELS)
    ]
    for point_count, is_initial in levels:
        grid = np.linspace(interval_lower, interval_upper, point_count)
        candidates = _ordered_unique(grid, seed if is_initial else None)
        level_values: list[tuple[float, float]] = []
        for candidate in candidates:
            value = float(objective(candidate))
            level_values.append((candidate, value))
            if value < best_value:
                best_x, best_value = candidate, value
        ordered = sorted(level_values)
        level_best = min(range(len(ordered)), key=lambda index: ordered[index][1])
        left_index = max(0, level_best - 1)
        right_index = min(len(ordered) - 1, level_best + 1)
        interval_lower = ordered[left_index][0]
        interval_upper = ordered[right_index][0]
        if interval_upper - interval_lower <= 1.0e-15:
            break
    return best_x, best_value


def _nested_search(position: np.ndarray, saved: dict[str, str]) -> SearchState:
    state = SearchState(position=position)
    a_lower = float(saved["strip_a_lower"])
    a_upper = float(saved["strip_a_upper"])
    b_lower = float(saved["strip_b_lower"])
    b_upper = float(saved["strip_b_upper"])
    seed_a = float(saved["final_best_a"])
    seed_b = float(saved["final_best_b"])
    outer_cache: dict[float, tuple[float, float]] = {}

    def outer_objective(parameter_b: float) -> float:
        key = float(parameter_b)
        cached = outer_cache.get(key)
        if cached is not None:
            return cached[1]

        def inner_objective(parameter_a: float) -> float:
            return state.evaluate(parameter_a, parameter_b)

        inner_best_a, inner_best_loss = _refine_1d(
            inner_objective,
            a_lower,
            a_upper,
            INNER_INITIAL_POINTS,
            seed_a,
        )
        outer_cache[key] = (inner_best_a, inner_best_loss)
        return inner_best_loss

    try:
        _refine_1d(
            outer_objective,
            b_lower,
            b_upper,
            OUTER_INITIAL_POINTS,
            seed_b,
        )
    except GateReached:
        pass
    return state


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    saved_rows = {
        int(row["trajectory_id"]): row
        for row in _read_csv(SOURCE)
        if int(row["trajectory_id"]) in TRAJECTORY_IDS
    }
    if set(saved_rows) != set(TRAJECTORY_IDS):
        raise RuntimeError("Missing unresolved adaptive-strip cases")
    positions, _, hidden_c = _make_exact_labeled_split(256, FRESH_TEST_SEED)
    positions = positions.to(dtype=DTYPE).cpu().numpy()
    references = _posthoc_reference(hidden_c).cpu().numpy()

    rows: list[dict[str, object]] = []
    for trajectory_id in TRAJECTORY_IDS:
        saved = saved_rows[trajectory_id]
        state = _nested_search(positions[trajectory_id], saved)
        reference = references[trajectory_id]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "crossed_gate": int(state.best_loss <= HANDOFF_GATE),
                "total_actual_prefix_loss_evaluations": state.count,
                "final_loss": state.best_loss,
                "final_a": state.best_a,
                "final_b": state.best_b,
                "posthoc_parameter_error": math.hypot(
                    state.best_a - reference[0], state.best_b - reference[1]
                ),
                "starting_saved_rescue_loss": float(saved["final_loss"]),
                "strip_a_lower": float(saved["strip_a_lower"]),
                "strip_a_upper": float(saved["strip_a_upper"]),
                "strip_b_lower": float(saved["strip_b_lower"]),
                "strip_b_upper": float(saved["strip_b_upper"]),
                "gate": HANDOFF_GATE,
                "search_rule": SEARCH_RULE,
            }
        )
        print(
            f"nested1d {trajectory_id}: success={state.best_loss <= HANDOFF_GATE} "
            f"evaluations={state.count} loss={state.best_loss:.6e}",
            flush=True,
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "nested1d_rescue.csv", rows)


if __name__ == "__main__":
    main()
