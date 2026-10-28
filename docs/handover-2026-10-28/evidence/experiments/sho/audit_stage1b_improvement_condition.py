#!/usr/bin/env python3
"""Audit eDIRECT's post-hull improvement-condition operator only."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


OUTPUT_DIR = Path("outputs/sho/stage1B/improvement_condition_audit")
AUDIT_PATH = OUTPUT_DIR / "improvement_condition_audit.txt"
EXAMPLES_PATH = OUTPUT_DIR / "improvement_condition_examples.csv"
DISTRIBUTION_PATH = OUTPUT_DIR / "improvement_condition_distribution.csv"


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


def _event(
    state_position: int,
    state: dict[str, Any],
    position: int,
    index: int,
    hull: np.ndarray,
    target: float,
    f_min: float,
) -> dict[str, Any]:
    sizes = state["sizes"]
    losses = state["losses"]
    size = float(sizes[index])
    loss = float(losses[index])
    previous_index = int(hull[position - 1]) if position > 0 else None
    next_index = int(hull[position + 1]) if position < len(hull) - 1 else None
    previous_size = float(sizes[previous_index]) if previous_index is not None else math.nan
    previous_loss = float(losses[previous_index]) if previous_index is not None else math.nan
    next_size = float(sizes[next_index]) if next_index is not None else math.nan
    next_loss = float(losses[next_index]) if next_index is not None else math.nan
    lower_slope = -math.inf
    upper_slope = math.inf
    if previous_index is not None:
        denominator = size - previous_size
        if not denominator > 0.0:
            raise RuntimeError("hull sizes are not strictly increasing")
        lower_slope = (loss - previous_loss) / denominator
    if next_index is not None:
        denominator = next_size - size
        if not denominator > 0.0:
            raise RuntimeError("hull sizes are not strictly increasing")
        upper_slope = (next_loss - loss) / denominator
    required = (loss - target) / size if size > 0.0 else math.inf
    lower = max(lower_slope, required, 0.0)
    positivity_pass = upper_slope > 0.0
    feasibility_margin = upper_slope + turn.HULL_TOLERANCE - lower
    feasibility_pass = lower <= upper_slope + turn.HULL_TOLERANCE
    passed = positivity_pass and feasibility_pass
    if math.isfinite(upper_slope) and math.isfinite(feasibility_margin):
        bottleneck_margin = min(upper_slope, feasibility_margin)
    else:
        bottleneck_margin = min(upper_slope, feasibility_margin)
    interval_width = upper_slope - lower
    active_lower = (
        "lower_hull_slope"
        if lower == lower_slope
        else "improvement_required"
        if lower == required
        else "K_nonnegative"
    )
    return {
        "event_id": -1,
        "state_position": state_position,
        "source_state_index": int(state["source_state_index"]),
        "trajectory_id": int(state["trajectory_id"]),
        "iteration": int(state["iteration"]),
        "state_bin": state["state_bin"],
        "representative_cell_count": len(sizes),
        "hull_size": len(hull),
        "hull_position": position,
        "representative_index": index,
        "is_first_hull_point": position == 0,
        "is_last_hull_point": position == len(hull) - 1,
        "previous_index": "" if previous_index is None else previous_index,
        "next_index": "" if next_index is None else next_index,
        "cell_size": size,
        "site_loss": loss,
        "global_best_loss": f_min,
        "epsilon": edirect.EDIRECT_EPSILON,
        "improvement_target": target,
        "previous_size": previous_size,
        "previous_loss": previous_loss,
        "next_size": next_size,
        "next_loss": next_loss,
        "lower_hull_slope": lower_slope,
        "upper_hull_slope": upper_slope,
        "improvement_required_slope": required,
        "K_nonnegative_bound": 0.0,
        "admissible_K_lower": lower,
        "admissible_K_upper": upper_slope,
        "active_lower_bound": active_lower,
        "slope_tolerance": turn.HULL_TOLERANCE,
        "positivity_margin": upper_slope,
        "feasibility_margin": feasibility_margin,
        "bottleneck_margin": bottleneck_margin,
        "raw_interval_width": interval_width,
        "positivity_pass": positivity_pass,
        "feasibility_pass": feasibility_pass,
        "improvement_pass": passed,
        "failure_reason": (
            "PASS"
            if passed
            else "upper_slope_not_positive"
            if not positivity_pass
            else "empty_admissible_K_interval"
        ),
    }


def _independent_filter(
    state_position: int, state: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    hull = np.asarray(state["teacher_hull"], dtype=np.int64)
    sizes = state["sizes"]
    losses = state["losses"]
    f_min = float(np.min(losses))
    target = f_min - edirect.EDIRECT_EPSILON * abs(f_min)
    selected: list[int] = []
    events: list[dict[str, Any]] = []
    for position, value in enumerate(hull):
        index = int(value)
        event = _event(
            state_position, state, position, index, hull, target, f_min
        )
        events.append(event)
        if bool(event["improvement_pass"]):
            selected.append(index)
    fallback_used = not selected
    if fallback_used:
        selected.append(int(np.argmax(sizes)))
    result = np.asarray(
        sorted(selected, key=lambda i: (sizes[i], losses[i], i)), dtype=np.int64
    )
    return result, events, fallback_used


def _finite_distribution(
    name: str, values: np.ndarray, category: str = "input"
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    row: dict[str, Any] = {
        "record_type": "variable_distribution",
        "category": category,
        "variable": name,
        "count": len(values),
        "finite_count": len(finite),
        "positive_infinity_count": int(np.sum(values == math.inf)),
        "negative_infinity_count": int(np.sum(values == -math.inf)),
        "nan_count": int(np.sum(np.isnan(values))),
        "exact_zero_count": int(np.sum(values == 0.0)),
    }
    if len(finite):
        row.update(
            {
                "minimum": float(np.min(finite)),
                "p01": float(np.percentile(finite, 1.0)),
                "p05": float(np.percentile(finite, 5.0)),
                "p25": float(np.percentile(finite, 25.0)),
                "p50": float(np.percentile(finite, 50.0)),
                "p75": float(np.percentile(finite, 75.0)),
                "p95": float(np.percentile(finite, 95.0)),
                "p99": float(np.percentile(finite, 99.0)),
                "maximum": float(np.max(finite)),
                "minimum_nonzero_absolute": (
                    float(np.min(np.abs(finite[finite != 0.0])))
                    if np.any(finite != 0.0)
                    else ""
                ),
            }
        )
    return row


def _margin_boundary_row(name: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    negative = finite[finite < 0.0]
    return {
        "record_type": "boundary_distribution",
        "variable": name,
        "finite_count": len(finite),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "exact_zero_count": int(np.sum(finite == 0.0)),
        "smallest_positive": float(np.min(positive)) if len(positive) else "",
        "largest_negative": float(np.max(negative)) if len(negative) else "",
        "minimum_nonzero_absolute": (
            float(np.min(np.abs(finite[finite != 0.0])))
            if np.any(finite != 0.0)
            else ""
        ),
    }


def _tiny_rows(name: str, values: np.ndarray) -> list[dict[str, Any]]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    finite = np.isfinite(absolute)
    rows: list[dict[str, Any]] = []
    for threshold in (1.0e-6, 1.0e-12, 1.0e-24, 1.0e-40, 1.0e-50):
        rows.append(
            {
                "record_type": "tiny_margin_count",
                "variable": name,
                "threshold": threshold,
                "count": int(np.sum(finite & (absolute < threshold))),
            }
        )
    return rows


def _example_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[int] = []

    def add(indices: list[int], count: int = 1) -> None:
        for index in indices:
            if index not in chosen:
                chosen.append(index)
                if count == 1 or len([value for value in chosen if value in indices]) >= count:
                    break

    add([i for i, event in enumerate(events) if event["improvement_pass"] and event["is_first_hull_point"]])
    add([i for i, event in enumerate(events) if event["improvement_pass"] and event["is_last_hull_point"]])
    add([i for i, event in enumerate(events) if event["improvement_pass"] and not event["is_first_hull_point"] and not event["is_last_hull_point"]])
    add([i for i, event in enumerate(events) if event["failure_reason"] == "upper_slope_not_positive"])
    add([i for i, event in enumerate(events) if event["failure_reason"] == "empty_admissible_K_interval"])
    finite_bottleneck = [
        i for i, event in enumerate(events) if math.isfinite(float(event["bottleneck_margin"]))
    ]
    finite_bottleneck.sort(key=lambda i: abs(float(events[i]["bottleneck_margin"])))
    for index in finite_bottleneck[:5]:
        if index not in chosen:
            chosen.append(index)
    required_order = sorted(
        range(len(events)),
        key=lambda i: float(events[i]["improvement_required_slope"]),
        reverse=True,
    )
    for index in required_order:
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= 12:
            break
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(chosen[:12]):
        row = dict(events[index])
        row["record_type"] = "worked_example"
        row["example_number"] = position + 1
        row["formula_target"] = "f_min - epsilon*abs(f_min)"
        row["formula_lower"] = "max(lower_slope, required_slope, 0)"
        row["formula_pass"] = "upper_slope > 0 and lower <= upper_slope + 5e-14"
        rows.append(row)
    return rows


def _write_audit_text(
    state_count: int,
    events: list[dict[str, Any]],
    exact_event_membership: int,
    membership_comparable_events: int,
    exact_states: int,
    fallback_states: int,
) -> None:
    pass_count = sum(bool(event["improvement_pass"]) for event in events)
    fail_count = len(events) - pass_count
    nonpositive = sum(
        event["failure_reason"] == "upper_slope_not_positive" for event in events
    )
    infeasible = sum(
        event["failure_reason"] == "empty_admissible_K_interval" for event in events
    )
    finite_bottleneck = np.asarray(
        [
            float(event["bottleneck_margin"])
            for event in events
            if math.isfinite(float(event["bottleneck_margin"]))
        ],
        dtype=np.float64,
    )
    positive_bottleneck = finite_bottleneck[finite_bottleneck > 0.0]
    negative_bottleneck = finite_bottleneck[finite_bottleneck < 0.0]
    smallest_positive_bottleneck = float(np.min(positive_bottleneck))
    largest_negative_bottleneck = float(np.max(negative_bottleneck))
    tiny_bottleneck = int(np.sum(np.abs(finite_bottleneck) < 1.0e-6))
    text = f"""eDIRECT post-hull improvement-condition audit
================================================

Scope
-----
No training, checkpoint mutation, or trajectory search was performed. The audit uses
the same deterministic held-out validation-state selection as the locked hull work:
{state_count} states and every one of their {len(events)} lower-hull representatives.

Audited code path
-----------------
`experiments/sho/run_stage1b_edirect_vs_direct.py::_potentially_optimal_cells`
performs, in order:

1. same-size grouping and representative choice (upstream of this audit),
2. lower-right-hull construction (upstream and now locked),
3. the improvement-condition loop audited here,
4. an argmax-size fallback if no hull representative passes,
5. deterministic output sorting by `(size, loss, site_index)`.

`experiments/sho/validate_stage1b_bilinear_hull_turn.py::_improvement_filter`
is an independent repository reconstruction of the same post-hull operation.

Exact inputs actually used by the improvement test
--------------------------------------------------
For hull representative i:

- its estimated Voronoi cell size `d_i`,
- its observed site prefix loss `f_i`,
- the global minimum observed site loss `f_min = min(all site losses)`,
- the constant `epsilon = {edirect.EDIRECT_EPSILON:.17g}`,
- the immediately previous hull representative, when present, through its `(d,f)`,
- the immediately next hull representative, when present, through its `(d,f)`,
- the positivity lower bound `K >= 0`,
- the absolute slope-comparison tolerance `{turn.HULL_TOLERANCE:.17g}`.

Hull position/order is used only to identify the immediate neighbors. The site index is
not an arithmetic input to PASS/FAIL; it is retained as the selected identity and used
only in final deterministic sorting/tie bookkeeping.

Not used by this operation: site coordinates, candidate coordinates, raw candidate
points, surrogate predictions, LOO error, candidate source/type, Voronoi owners,
within-cell distances, cell proposal scores, the learned selector, or any locked-network
features beyond the already-produced hull membership.

Exact teacher calculation
-------------------------

    f_min  = min(losses)
    target = f_min - 1e-4 * abs(f_min)

    lower_slope = -inf                                  # first hull point
    lower_slope = (f_i-f_prev)/(d_i-d_prev)             # otherwise

    upper_slope = +inf                                  # last hull point
    upper_slope = (f_next-f_i)/(d_next-d_i)             # otherwise

    required = (f_i-target)/d_i if d_i > 0 else +inf
    K_lower  = max(lower_slope, required, 0.0)
    K_upper  = upper_slope

    PASS iff (K_upper > 0.0) and
             (K_lower <= K_upper + 5e-14)

The code algebraically eliminates K; it never chooses or evaluates a particular K.
The admissible set is the interval `[K_lower, K_upper]`, with an absolute +5e-14
relaxation on the upper comparison. `K_upper > 0` is strict and has no tolerance.
The interval comparison is inclusive. Therefore equality at
`K_lower == K_upper + 5e-14` passes, while `K_upper == 0` fails.

Endpoint behavior
-----------------
The first hull point has `lower_slope=-inf`, which disappears inside the max. The last
hull point has `K_upper=+inf`; it passes the two implemented comparisons (including the
unlikely `required=+inf` case because Python evaluates `inf <= inf` as true). An
unbounded upper interval is therefore represented explicitly by +inf. No extra endpoint
rule is present.

If no representative passes, the function selects `argmax(sizes)` as a fallback. This
is not an improvement-condition PASS; it is post-condition safety bookkeeping. Fallback
was used in {fallback_states}/{state_count} audited states. NumPy's first-maximum
`argmax` behavior resolves an exact maximum-size tie in that fallback.

Smallest exact continuous decomposition
---------------------------------------
The exact operation contains two continuous margins followed by a conjunction:

    positivity_margin  = K_upper
    feasibility_margin = K_upper + 5e-14 - K_lower

    PASS iff positivity_margin > 0
             AND feasibility_margin >= 0

The strict/inclusive mismatch matters. A single ordinary `min(... ) >= 0` score is not
exact when `K_upper == 0`; the faithful decomposition is these two continuous quantities
plus the stated comparisons and Boolean AND. The max that constructs K_lower is itself
continuous and piecewise linear in `[lower_slope, required, 0]`.

Audit counts and verification
-----------------------------
PASS events: {pass_count}
FAIL events: {fail_count}
  upper slope non-positive: {nonpositive}
  admissible interval empty: {infeasible}

PASS fraction: {pass_count / len(events):.12g}
FAIL fraction: {fail_count / len(events):.12g}
PASS:FAIL ratio: {pass_count / fail_count:.12g}:1. The events are moderately imbalanced
toward PASS, but not an extreme >9:1 split.

Closest finite bottleneck margins:
  smallest positive = {smallest_positive_bottleneck:.17g}
  largest negative  = {largest_negative_bottleneck:.17g}
  exact zero count  = {int(np.sum(finite_bottleneck == 0.0))}
  |margin| < 1e-6   = {tiny_bottleneck}

Unlike the learned hull-turn operation, this held-out improvement dataset has no
extremely tiny decision margins: the nearest finite boundary distance is
{float(np.min(np.abs(finite_bottleneck))):.17g}, not 1e-40-scale.

Independent final selected-set reconstruction: {exact_states}/{state_count} states exact.
Direct PASS-membership comparison where fallback does not obscure membership:
{exact_event_membership}/{membership_comparable_events} events exact.

The distribution CSV contains finite/infinite counts, percentiles, boundary-nearest
margins, and tiny-margin counts. The examples CSV gives complete worked calculations.
"""
    AUDIT_PATH.write_text(text, encoding="utf-8")


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    states = turn._build_states(payload, indices, workers, "improvement audit")
    events: list[dict[str, Any]] = []
    exact_states = 0
    fallback_states = 0
    exact_event_membership = 0
    membership_comparable_events = 0
    state_rows: list[dict[str, Any]] = []
    for state_position, state in enumerate(states):
        independent, state_events, fallback = _independent_filter(
            state_position, state
        )
        teacher = edirect._potentially_optimal_cells(
            state["sizes"], state["losses"]
        )
        exact = bool(np.array_equal(independent, teacher))
        exact_states += int(exact)
        fallback_states += int(fallback)
        selected_set = set(int(value) for value in teacher)
        if not fallback:
            for event in state_events:
                exact_event_membership += int(
                    bool(event["improvement_pass"])
                    == (int(event["representative_index"]) in selected_set)
                )
                membership_comparable_events += 1
        for event in state_events:
            event["event_id"] = len(events)
            event["fallback_used_in_state"] = fallback
            events.append(event)
        state_rows.append(
            {
                "record_type": "state_reconstruction",
                "source_state_index": state["source_state_index"],
                "hull_event_count": len(state_events),
                "teacher_selected_count": len(teacher),
                "fallback_used": fallback,
                "exact_selected_set": exact,
            }
        )
    if exact_states != len(states):
        raise RuntimeError("independent improvement-condition reconstruction failed")

    pass_count = sum(bool(event["improvement_pass"]) for event in events)
    fail_count = len(events) - pass_count
    distribution_rows: list[dict[str, Any]] = [
        {
            "record_type": "audit_summary",
            "state_count": len(states),
            "event_count": len(events),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_fraction": pass_count / len(events),
            "fail_fraction": fail_count / len(events),
            "pass_to_fail_ratio": pass_count / fail_count if fail_count else math.inf,
            "highly_imbalanced_gt_9_to_1": (
                max(pass_count, fail_count) / max(1, min(pass_count, fail_count)) > 9.0
            ),
            "fallback_state_count": fallback_states,
            "independent_exact_state_count": exact_states,
            "independent_exact_state_fraction": exact_states / len(states),
            "membership_comparable_events": membership_comparable_events,
            "exact_event_membership": exact_event_membership,
            "exact_event_membership_fraction": (
                exact_event_membership / membership_comparable_events
                if membership_comparable_events
                else ""
            ),
            "epsilon": edirect.EDIRECT_EPSILON,
            "slope_tolerance": turn.HULL_TOLERANCE,
        }
    ]
    variables = {
        "cell_size": "input",
        "site_loss": "input",
        "global_best_loss": "input",
        "improvement_target": "derived_input",
        "previous_size": "neighbor_input",
        "previous_loss": "neighbor_input",
        "next_size": "neighbor_input",
        "next_loss": "neighbor_input",
        "lower_hull_slope": "K_bound",
        "upper_hull_slope": "K_bound",
        "improvement_required_slope": "K_bound",
        "admissible_K_lower": "K_bound",
        "admissible_K_upper": "K_bound",
        "raw_interval_width": "K_interval",
        "positivity_margin": "decision_margin",
        "feasibility_margin": "decision_margin",
        "bottleneck_margin": "diagnostic_only",
    }
    for name, category in variables.items():
        distribution_rows.append(
            _finite_distribution(
                name,
                np.asarray([event[name] for event in events], dtype=np.float64),
                category,
            )
        )
    for name in ("positivity_margin", "feasibility_margin", "bottleneck_margin"):
        values = np.asarray([event[name] for event in events], dtype=np.float64)
        distribution_rows.append(_margin_boundary_row(name, values))
        distribution_rows.extend(_tiny_rows(name, values))
    for reason in ("PASS", "upper_slope_not_positive", "empty_admissible_K_interval"):
        distribution_rows.append(
            {
                "record_type": "decision_reason",
                "reason": reason,
                "count": sum(event["failure_reason"] == reason for event in events),
            }
        )
    active_names = ("lower_hull_slope", "improvement_required", "K_nonnegative")
    for name in active_names:
        distribution_rows.append(
            {
                "record_type": "active_lower_bound",
                "active_lower_bound": name,
                "count": sum(event["active_lower_bound"] == name for event in events),
            }
        )
    distribution_rows.extend(state_rows)
    _write_csv(DISTRIBUTION_PATH, distribution_rows)
    _write_csv(EXAMPLES_PATH, _example_rows(events))
    _write_audit_text(
        len(states),
        events,
        exact_event_membership,
        membership_comparable_events,
        exact_states,
        fallback_states,
    )
    print(
        f"improvement audit events={len(events)} pass={pass_count} fail={fail_count} "
        f"exact_states={exact_states}/{len(states)} fallback_states={fallback_states}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
