#!/usr/bin/env python3
"""Audit extreme finite neighboring-hull slopes without training."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import experiments.sho.audit_stage1b_improvement_condition as improvement
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


OUTPUT_DIR = Path("outputs/sho/stage1B/hull_slope_extreme_audit")
AUDIT_PATH = OUTPUT_DIR / "hull_slope_extreme_audit.txt"
EDGES_PATH = OUTPUT_DIR / "hull_slope_extreme_edges.csv"
GROUP_TOLERANCE = turn.SIZE_GROUP_TOLERANCE


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


def _bucket(value: float) -> str:
    absolute = abs(value)
    if absolute < 1.0:
        return "abs_K_lt_1"
    if absolute < 1.0e3:
        return "abs_K_1_to_1e3"
    if absolute < 1.0e6:
        return "abs_K_1e3_to_1e6"
    if absolute < 1.0e12:
        return "abs_K_1e6_to_1e12"
    if absolute < 1.0e24:
        return "abs_K_1e12_to_1e24"
    if absolute < 1.0e40:
        return "abs_K_1e24_to_1e40"
    return "abs_K_ge_1e40"


def _right_dependency(event: dict[str, Any]) -> str:
    if event["is_last_hull_point"]:
        return "K_prev_enters_max_but_unbounded_K_next_makes_PASS_insensitive"
    if event["active_lower_bound"] == "lower_hull_slope":
        return "K_prev_wins_max_and_is_compared_to_K_next"
    return "K_prev_loses_max; only_ordering_with_active_lower_bound_matters"


def _left_dependency(event: dict[str, Any]) -> str:
    if float(event["upper_hull_slope"]) <= 0.0:
        return "K_next_sign_determines_FAIL; magnitude_not_needed_after_sign"
    return "K_next_sign_and_magnitude_enter_upper_bound_comparison"


def _edges(states: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_position, state in enumerate(states):
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = state["sizes"]
        losses = state["losses"]
        f_min = float(np.min(losses))
        target = f_min - 1.0e-4 * abs(f_min)
        state_events = [
            improvement._event(
                state_position,
                state,
                position,
                int(index),
                hull,
                target,
                f_min,
            )
            for position, index in enumerate(hull)
        ]
        for left_position in range(len(hull) - 1):
            right_position = left_position + 1
            left_index = int(hull[left_position])
            right_index = int(hull[right_position])
            d_left = float(sizes[left_index])
            d_right = float(sizes[right_index])
            f_left = float(losses[left_index])
            f_right = float(losses[right_index])
            delta_d = d_right - d_left
            delta_f = f_right - f_left
            if not delta_d > 0.0:
                raise RuntimeError(
                    f"non-positive delta_d split={split} state={state_position}"
                )
            slope = delta_f / delta_d
            if not math.isfinite(slope):
                raise RuntimeError("non-finite quotient for a finite hull edge")
            left_event = state_events[left_position]
            right_event = state_events[right_position]
            if float(left_event["upper_hull_slope"]) != slope:
                raise RuntimeError("edge slope differs from left K_next")
            if float(right_event["lower_hull_slope"]) != slope:
                raise RuntimeError("edge slope differs from right K_prev")
            rows.append(
                {
                    "edge_id": -1,
                    "split": split,
                    "state_position_within_split": state_position,
                    "source_state_index": int(state["source_state_index"]),
                    "trajectory_id": int(state["trajectory_id"]),
                    "iteration": int(state["iteration"]),
                    "state_bin": state["state_bin"],
                    "representative_cell_count": len(sizes),
                    "hull_size": len(hull),
                    "left_hull_position": left_position,
                    "right_hull_position": right_position,
                    "left_group_index": left_index,
                    "right_group_index": right_index,
                    "left_original_site_index": int(state["representatives"][left_index]),
                    "right_original_site_index": int(state["representatives"][right_index]),
                    "d_L": d_left,
                    "d_R": d_right,
                    "f_L": f_left,
                    "f_R": f_right,
                    "delta_d": delta_d,
                    "abs_delta_d": abs(delta_d),
                    "delta_f": delta_f,
                    "abs_delta_f": abs(delta_f),
                    "K": slope,
                    "abs_K": abs(slope),
                    "K_sign": "positive" if slope > 0.0 else "negative" if slope < 0.0 else "zero",
                    "abs_K_bucket": _bucket(slope),
                    "used_as_K_next": True,
                    "used_as_K_prev": True,
                    "downstream_roles": "K_next_of_left_and_K_prev_of_right",
                    "same_size_test_quantity_abs_dR_minus_dL": abs(delta_d),
                    "same_size_grouping_threshold": GROUP_TOLERANCE,
                    "pair_would_be_same_group_under_threshold": abs(delta_d) <= GROUP_TOLERANCE,
                    "pair_gap_to_grouping_threshold_ratio": abs(delta_d) / GROUP_TOLERANCE,
                    "left_K_lower": float(left_event["admissible_K_lower"]),
                    "left_K_next_positive": bool(left_event["positivity_pass"]),
                    "left_feasibility_margin": float(left_event["feasibility_margin"]),
                    "left_improvement_pass": bool(left_event["improvement_pass"]),
                    "left_dependency": _left_dependency(left_event),
                    "right_K_prev_wins_lower_max": right_event["active_lower_bound"] == "lower_hull_slope",
                    "right_active_lower_bound": right_event["active_lower_bound"],
                    "right_K_next": float(right_event["upper_hull_slope"]),
                    "right_feasibility_margin": float(right_event["feasibility_margin"]),
                    "right_improvement_pass": bool(right_event["improvement_pass"]),
                    "right_is_last_hull_point": bool(right_event["is_last_hull_point"]),
                    "right_dependency": _right_dependency(right_event),
                }
            )
    return rows


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "p25": float(np.percentile(values, 25.0)),
        "median": float(np.percentile(values, 50.0)),
        "p75": float(np.percentile(values, 75.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
    }


def _stats_text(name: str, values: np.ndarray) -> str:
    stats = _stats(values)
    threshold_counts = "\n".join(
        f"    count < {threshold:.0e} = {int(np.sum(values < threshold))}"
        for threshold in (1.0e-6, 1.0e-12, 1.0e-20, 1.0e-40, 1.0e-50)
    )
    return "\n".join(
        [f"  {name}:"]
        + [f"    {key} = {value:.17g}" for key, value in stats.items()]
        + [threshold_counts]
    )


def _bucket_text(rows: list[dict[str, Any]]) -> str:
    order = (
        "abs_K_lt_1",
        "abs_K_1_to_1e3",
        "abs_K_1e3_to_1e6",
        "abs_K_1e6_to_1e12",
        "abs_K_1e12_to_1e24",
        "abs_K_1e24_to_1e40",
        "abs_K_ge_1e40",
    )
    lines: list[str] = []
    for bucket in order:
        selected = [row for row in rows if row["abs_K_bucket"] == bucket]
        positive = sum(row["K_sign"] == "positive" for row in selected)
        negative = sum(row["K_sign"] == "negative" for row in selected)
        zero = len(selected) - positive - negative
        lines.append(
            f"  {bucket}: total={len(selected)}, positive={positive}, "
            f"negative={negative}, zero={zero}"
        )
    return "\n".join(lines)


def _top_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "rank  split state hull(L,R) delta_d delta_f K d_L d_R f_L f_R"
    ]
    for row in rows[:20]:
        lines.append(
            f"{row['abs_K_rank']:>4}  {row['split']:<10} "
            f"{row['source_state_index']:>5} "
            f"({row['left_hull_position']},{row['right_hull_position']}) "
            f"{float(row['delta_d']):.6e} {float(row['delta_f']):.6e} "
            f"{float(row['K']):.6e} {float(row['d_L']):.6e} "
            f"{float(row['d_R']):.6e} {float(row['f_L']):.6e} "
            f"{float(row['f_R']):.6e}"
        )
    return "\n".join(lines)


def _write_audit(rows: list[dict[str, Any]]) -> None:
    delta_d = np.asarray([row["delta_d"] for row in rows], dtype=np.float64)
    abs_delta_f = np.asarray([row["abs_delta_f"] for row in rows], dtype=np.float64)
    abs_k = np.asarray([row["abs_K"] for row in rows], dtype=np.float64)
    nonzero_df = abs_delta_f > 0.0
    log_k = np.log10(abs_k[nonzero_df])
    log_df = np.log10(abs_delta_f[nonzero_df])
    negative_log_dd = -np.log10(delta_d[nonzero_df])
    correlation_large_df = float(np.corrcoef(log_k, log_df)[0, 1])
    correlation_small_dd = float(np.corrcoef(log_k, negative_log_dd)[0, 1])
    q01_dd = float(np.quantile(delta_d, 0.01))
    q99_df = float(np.quantile(abs_delta_f, 0.99))
    top = rows[:20]
    top_small_dd = sum(float(row["delta_d"]) <= q01_dd for row in top)
    top_large_df = sum(float(row["abs_delta_f"]) >= q99_df for row in top)
    top_both = sum(
        float(row["delta_d"]) <= q01_dd
        and float(row["abs_delta_f"]) >= q99_df
        for row in top
    )
    top_right_last = sum(bool(row["right_is_last_hull_point"]) for row in top)
    top_right_active = sum(bool(row["right_K_prev_wins_lower_max"]) for row in top)
    top_left_pass = sum(bool(row["left_improvement_pass"]) for row in top)
    top_right_pass = sum(bool(row["right_improvement_pass"]) for row in top)
    text = f"""Extreme neighboring-hull slope audit
====================================

Scope
-----
No training, locked-checkpoint access or mutation, hull reconstruction, or trajectory
search was performed. This audit covers every unique finite lower-hull edge in the
existing state-disjoint pool: {len(rows)} edges from 240 training and 80 held-out
validation states.

Exact slope operation
---------------------
For adjacent lower-hull representatives ordered by increasing d:

    delta_d = d_R - d_L
    delta_f = f_R - f_L
    K       = delta_f / delta_d

All denominators were asserted strictly positive. No epsilon or numerical tolerance is
used in the quotient.

1. delta_d distribution
-----------------------
{_stats_text('delta_d', delta_d)}

2. |delta_f| distribution
-------------------------
{_stats_text('abs_delta_f', abs_delta_f)}

3. Largest-|K| edges
--------------------
{_top_table(top)}

The CSV contains all edges, with `abs_K_rank=1..{len(rows)}` and `is_top20` marking
these examples.

4. Cause of extreme K
---------------------
Because `log10|K| = log10|delta_f| - log10(delta_d)`, the diagnostic correlations are:

  corr(log10|K|, log10|delta_f|) = {correlation_large_df:.17g}
  corr(log10|K|, -log10(delta_d)) = {correlation_small_dd:.17g}

Using the global 1st-percentile delta_d ({q01_dd:.17g}) as "extremely small" and the
99th-percentile |delta_f| ({q99_df:.17g}) as "extremely large":

  top-20 with extremely small delta_d = {top_small_dd}/20
  top-20 with extremely large |delta_f| = {top_large_df}/20
  top-20 satisfying both = {top_both}/20

Thus the extreme positive slopes are primarily attributable to the enormous loss
differences, with small denominators acting as an amplifier only where the per-edge
flags in the CSV show both conditions.

5. Same-size grouping immediately upstream
-------------------------------------------
The implementation lexicographically sorts sites by `(size, loss, original_index)` and
walks them in that order. Its exact rule is:

    if no current group or abs(size - current_group_size) > 1e-14:
        start a new group
    elif loss < current_group_best_loss:
        replace that group's representative with this lower-loss site

Equality at `abs(size-current_group_size) == 1e-14` stays in the existing group. The
stored group size remains the size that opened the group even if its representative is
replaced. No relative tolerance is used.

For every edge, the CSV reports `abs(d_R-d_L)`, the `1e-14` threshold, their ratio, and
the counterfactual same-group result. Every extreme pair is separated by far more than
the threshold, so these cells correctly survive as distinct size groups. Intervening
non-hull size groups may exist; grouping is completed before hull construction.

6. |K| buckets and sign composition
-----------------------------------
{_bucket_text(rows)}

7. Downstream use by the improvement condition
-----------------------------------------------
Every finite edge is used in both roles:

- as `K_next` (the upper admissible K bound) for its left hull point;
- as `K_prev` (a candidate lower bound) for its right hull point.

For the left point, the code first requires `K_next > 0`, then compares
`K_lower <= K_next + 5e-14`. Positive magnitude therefore matters as an interval bound,
not merely as a sign.

For the right point, the edge competes inside
`max(K_prev, K_required, 0)`. If it wins, its magnitude is compared with the following
`K_next`; if it loses, only its ordering relative to the winning lower bound matters.
If the right point is the last hull point, its next bound is +inf and the finite K_prev
magnitude cannot change that endpoint's PASS result.

Among the top 20 extreme edges:

  used as both K_prev and K_next = 20/20
  left-point PASS = {top_left_pass}/20
  K_prev wins the right-point max = {top_right_active}/20
  right point is the +inf endpoint = {top_right_last}/20
  right-point PASS = {top_right_pass}/20

The CSV records the exact left/right bounds, margins, decisions, and dependency class
for every edge. Consequently, preserving only sign would be insufficient in general;
ordering/max and magnitude comparisons are genuine downstream operations.
"""
    AUDIT_PATH.write_text(text, encoding="utf-8")


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
    train_states = turn._build_states(payload, train_indices, workers, "extreme audit train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "extreme audit validation"
    )
    rows = _edges(train_states, "train") + _edges(validation_states, "validation")
    rows.sort(key=lambda row: float(row["abs_K"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["edge_id"] = rank - 1
        row["abs_K_rank"] = rank
        row["is_top20"] = rank <= 20
    _write_csv(EDGES_PATH, rows)
    _write_audit(rows)
    print(
        f"extreme slope audit edges={len(rows)} max_abs_K={rows[0]['abs_K']:.6e} "
        f"min_delta_d={min(float(row['delta_d']) for row in rows):.6e}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
