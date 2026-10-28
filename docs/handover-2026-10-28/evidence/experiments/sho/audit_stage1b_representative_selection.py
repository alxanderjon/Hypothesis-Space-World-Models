#!/usr/bin/env python3
"""Audit representative selection inside already-formed same-size groups."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import experiments.sho.audit_stage1b_same_size_grouping as grouping_audit
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


OUTPUT_DIR = Path("outputs/sho/stage1B/representative_selection_audit")
AUDIT_PATH = OUTPUT_DIR / "representative_selection_audit.txt"
DISTRIBUTION_PATH = OUTPUT_DIR / "representative_selection_distribution.csv"


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


def _percentile_rows(quantity: str, values: np.ndarray) -> list[dict[str, Any]]:
    if len(values) == 0:
        return []
    rows: list[dict[str, Any]] = []
    for name, percentile in (
        ("min", 0.0),
        ("p01", 1.0),
        ("p05", 5.0),
        ("p10", 10.0),
        ("p25", 25.0),
        ("median", 50.0),
        ("p75", 75.0),
        ("p90", 90.0),
        ("p95", 95.0),
        ("p99", 99.0),
        ("max", 100.0),
    ):
        rows.append(
            {
                "record_type": "numeric_distribution",
                "quantity": quantity,
                "statistic": name,
                "value": float(np.percentile(values, percentile)),
                "sample_count": len(values),
            }
        )
    return rows


def _event_rows(states: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact_final_groups = 0
    total_groups = 0
    exact_states = 0
    groups_multiple_replacements = 0
    groups_with_comparisons = 0
    for state_position, state in enumerate(states):
        trace = grouping_audit._teacher_trace(state["sizes"], state["losses"])
        losses = state["losses"]
        independently_selected: list[int] = []
        state_exact = True
        for group_index, group in enumerate(trace["groups"]):
            total_groups += 1
            members = [int(value) for value in group["members"]]
            best_index = members[0]
            replacement_count = 0
            if len(members) > 1:
                groups_with_comparisons += 1
            for comparison_position, challenger_index in enumerate(members[1:], start=1):
                f_i = float(losses[challenger_index])
                f_best = float(losses[best_index])
                delta = f_best - f_i
                teacher_replace = bool(f_i < f_best)
                reconstructed_replace = bool(delta > 0.0)
                scale = max(abs(f_i), abs(f_best))
                norm_defined = scale > 0.0
                delta_norm = delta / scale if norm_defined else math.nan
                log_defined = f_i > 0.0 and f_best > 0.0
                log_ratio = (
                    math.log(f_best) - math.log(f_i) if log_defined else math.nan
                )
                norm_sign_matches = bool(
                    not norm_defined
                    or np.signbit(delta_norm) == np.signbit(delta)
                    and (delta_norm == 0.0) == (delta == 0.0)
                )
                log_sign_matches = bool(
                    not log_defined
                    or np.signbit(log_ratio) == np.signbit(delta)
                    and (log_ratio == 0.0) == (delta == 0.0)
                )
                row = {
                    "record_type": "comparison_event",
                    "split": state["split"],
                    "state_position": state_position,
                    "source_state_index": state["source_state_index"],
                    "group_index": group_index,
                    "comparison_position": comparison_position,
                    "current_best_index": best_index,
                    "challenger_index": challenger_index,
                    "f_i": f_i,
                    "f_best": f_best,
                    "delta_f_best_minus_f_i": delta,
                    "teacher_REPLACE": teacher_replace,
                    "reconstructed_REPLACE": reconstructed_replace,
                    "replacement_exact": teacher_replace == reconstructed_replace,
                    "scale_S": scale,
                    "delta_norm_defined": norm_defined,
                    "delta_norm": delta_norm,
                    "delta_norm_sign_and_zero_exact": norm_sign_matches,
                    "log_ratio_defined": log_defined,
                    "log_ratio": log_ratio,
                    "log_ratio_sign_and_zero_exact": log_sign_matches,
                }
                rows.append(row)
                if teacher_replace:
                    best_index = challenger_index
                    replacement_count += 1
            if replacement_count >= 2:
                groups_multiple_replacements += 1
            teacher_representative = int(group["representative_index"])
            exact_group = best_index == teacher_representative
            exact_final_groups += int(exact_group)
            state_exact = state_exact and exact_group
            independently_selected.append(best_index)
        reference = trace["representatives"]
        state_exact = state_exact and np.array_equal(
            np.asarray(independently_selected, dtype=np.int64), reference
        )
        exact_states += int(state_exact)
    return rows, {
        "state_count": len(states),
        "total_group_count": total_groups,
        "groups_with_comparisons": groups_with_comparisons,
        "groups_with_multiple_replacement_events": groups_multiple_replacements,
        "exact_final_representative_groups": exact_final_groups,
        "exact_final_representative_states": exact_states,
    }


def _analyze(
    events: list[dict[str, Any]], reconstruction: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    f_i = np.asarray([row["f_i"] for row in events], dtype=np.float64)
    f_best = np.asarray([row["f_best"] for row in events], dtype=np.float64)
    delta = np.asarray(
        [row["delta_f_best_minus_f_i"] for row in events], dtype=np.float64
    )
    replace = np.asarray([row["teacher_REPLACE"] for row in events], dtype=bool)
    norm_defined = np.asarray(
        [row["delta_norm_defined"] for row in events], dtype=bool
    )
    delta_norm = np.asarray(
        [row["delta_norm"] for row in events], dtype=np.float64
    )
    log_defined = np.asarray(
        [row["log_ratio_defined"] for row in events], dtype=bool
    )
    log_ratio = np.asarray([row["log_ratio"] for row in events], dtype=np.float64)
    norm_exact = np.asarray(
        [row["delta_norm_sign_and_zero_exact"] for row in events], dtype=bool
    )
    log_exact = np.asarray(
        [row["log_ratio_sign_and_zero_exact"] for row in events], dtype=bool
    )
    positive_delta = delta[delta > 0.0]
    negative_delta = delta[delta < 0.0]
    zero_delta_count = int(np.sum(delta == 0.0))
    norm_zero_collapse = int(np.sum(norm_defined & (delta != 0.0) & (delta_norm == 0.0)))
    log_zero_collapse = int(np.sum(log_defined & (delta != 0.0) & (log_ratio == 0.0)))
    summary = {
        "record_type": "summary",
        "comparison_event_count": len(events),
        "REPLACE_count": int(np.sum(replace)),
        "KEEP_count": int(np.sum(~replace)),
        "f_i_exact_zero_count": int(np.sum(f_i == 0.0)),
        "f_best_exact_zero_count": int(np.sum(f_best == 0.0)),
        "closest_positive_delta_to_zero": (
            float(np.min(positive_delta)) if len(positive_delta) else ""
        ),
        "closest_negative_delta_to_zero": (
            float(np.max(negative_delta)) if len(negative_delta) else ""
        ),
        "exact_zero_delta_count": zero_delta_count,
        "delta_norm_defined_count": int(np.sum(norm_defined)),
        "delta_norm_undefined_count": int(np.sum(~norm_defined)),
        "delta_norm_sign_or_zero_information_loss_count": int(np.sum(~norm_exact)),
        "delta_norm_nonzero_to_zero_collapse_count": norm_zero_collapse,
        "log_ratio_defined_count": int(np.sum(log_defined)),
        "log_ratio_undefined_count": int(np.sum(~log_defined)),
        "log_ratio_sign_or_zero_information_loss_count": int(np.sum(~log_exact)),
        "log_ratio_nonzero_to_zero_collapse_count": log_zero_collapse,
        "replacement_decision_reconstruction_mistakes": int(
            sum(not bool(row["replacement_exact"]) for row in events)
        ),
        **reconstruction,
    }

    distribution_rows: list[dict[str, Any]] = [summary]
    distribution_rows.extend(_percentile_rows("f_i", f_i))
    distribution_rows.extend(_percentile_rows("f_best", f_best))
    distribution_rows.extend(_percentile_rows("delta", delta))
    distribution_rows.extend(_percentile_rows("delta_REPLACE", delta[replace]))
    distribution_rows.extend(_percentile_rows("delta_KEEP", delta[~replace]))
    distribution_rows.extend(
        _percentile_rows("delta_norm", delta_norm[norm_defined])
    )
    distribution_rows.extend(
        _percentile_rows("delta_norm_REPLACE", delta_norm[norm_defined & replace])
    )
    distribution_rows.extend(
        _percentile_rows("delta_norm_KEEP", delta_norm[norm_defined & ~replace])
    )
    distribution_rows.extend(
        _percentile_rows("log_ratio", log_ratio[log_defined])
    )
    distribution_rows.extend(
        _percentile_rows("log_ratio_REPLACE", log_ratio[log_defined & replace])
    )
    distribution_rows.extend(
        _percentile_rows("log_ratio_KEEP", log_ratio[log_defined & ~replace])
    )
    distribution_rows.extend(events)

    raw_range = float(np.max(delta) - np.min(delta))
    norm_range = float(
        np.max(delta_norm[norm_defined]) - np.min(delta_norm[norm_defined])
    )
    log_range = float(np.max(log_ratio[log_defined]) - np.min(log_ratio[log_defined]))
    if (
        np.all(norm_defined)
        and np.all(norm_exact)
        and norm_zero_collapse == 0
    ):
        safest = "B. normalized delta"
        reason = (
            "It is defined on every real event, preserves sign and exact-zero status on "
            "every event, and bounds the observed comparison scale without applying a "
            "transcendental transform."
        )
    elif np.all(log_defined) and np.all(log_exact) and log_zero_collapse == 0:
        safest = "C. log-ratio"
        reason = (
            "Normalized delta is not lossless on this pool, while log-ratio is defined "
            "and sign-preserving on every event."
        )
    else:
        safest = "A. raw delta"
        reason = (
            "Both transformed candidates lose sign/zero information or are undefined "
            "on real events, so only the exact teacher subtraction is universally safe."
        )

    audit_text = f"""Representative selection within same-size groups
================================================

Scope
-----
This audit uses the existing saved eDIRECT teacher traces: {turn.TRAIN_SOURCE_STATES}
training and {turn.VALIDATION_SOURCE_STATES} held-out states. It performs no training,
trajectory search, or checkpoint writes.

Exact teacher operation
-----------------------
After anchored same-size groups have been formed, the first cell in the sorted group is
the initial representative. Each later group member is processed sequentially:

    delta = f_best - f_i
    REPLACE iff f_i < f_best

Equivalently, REPLACE iff `delta > 0`. Equality (`delta == 0`) keeps the current
representative. There is no tolerance. When replacement occurs, only the current best
loss and representative site index change; the same-size group's anchor size remains
unchanged.

Scale and decision distribution
-------------------------------
Comparison events: {len(events)}. REPLACE: {summary['REPLACE_count']}; KEEP:
{summary['KEEP_count']}. `f_i` exact zeros: {summary['f_i_exact_zero_count']}; `f_best`
exact zeros: {summary['f_best_exact_zero_count']}.

Closest positive delta to zero: {summary['closest_positive_delta_to_zero']}.
Closest negative delta to zero: {summary['closest_negative_delta_to_zero']}.
Exact-zero delta events: {zero_delta_count}.

Groups with comparisons: {reconstruction['groups_with_comparisons']}; groups containing
multiple replacement events: {reconstruction['groups_with_multiple_replacement_events']}.
The CSV contains min/max and percentiles for raw losses, all deltas, and separate
REPLACE/KEEP margins.

Normalization audit
-------------------
For `S=max(|f_i|,|f_best|)`, normalized delta was defined on
{summary['delta_norm_defined_count']}/{len(events)} events. Undefined events:
{summary['delta_norm_undefined_count']}; sign/zero information losses:
{summary['delta_norm_sign_or_zero_information_loss_count']}; nonzero-to-zero collapses:
{norm_zero_collapse}. Its observed range width is {norm_range:.17g}.

`log(f_best)-log(f_i)` was defined on {summary['log_ratio_defined_count']}/{len(events)}
events. Undefined events: {summary['log_ratio_undefined_count']}; sign/zero information
losses: {summary['log_ratio_sign_or_zero_information_loss_count']}; nonzero-to-zero
collapses: {log_zero_collapse}. Its observed range width is {log_range:.17g}.

The raw delta range width is {raw_range:.17g}.

Safest continuous target
------------------------
{safest} is numerically safest on the observed teacher population. {reason}

Independent verification
------------------------
The independent `delta > 0` replay made
{summary['replacement_decision_reconstruction_mistakes']} replacement-decision errors.
It recovered the exact final representative for
{reconstruction['exact_final_representative_groups']}/{reconstruction['total_group_count']}
groups and the exact ordered representative list for
{reconstruction['exact_final_representative_states']}/{reconstruction['state_count']}
states.
"""
    return distribution_rows, audit_text


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED
    )
    held_out_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    indexed = [(index, "training") for index in train_indices]
    indexed.extend((index, "held_out") for index in held_out_indices)
    states = grouping_audit._build_states(payload, indexed, workers)
    events, reconstruction = _event_rows(states)
    distribution_rows, audit_text = _analyze(events, reconstruction)
    _write_csv(DISTRIBUTION_PATH, distribution_rows)
    AUDIT_PATH.write_text(audit_text, encoding="utf-8")
    summary = distribution_rows[0]
    print(
        "representative selection audit complete "
        f"events={summary['comparison_event_count']} "
        f"groups={summary['exact_final_representative_groups']}/"
        f"{summary['total_group_count']} states="
        f"{summary['exact_final_representative_states']}/"
        f"{summary['state_count']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    run(args.workers)


if __name__ == "__main__":
    main()
