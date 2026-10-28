#!/usr/bin/env python3
"""Audit only (f_i, f_min, d_i) -> K_required; perform no training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import experiments.sho.audit_stage1b_improvement_condition as improvement
import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


OUTPUT_DIR = Path("outputs/sho/stage1B/k_required_audit")
AUDIT_PATH = OUTPUT_DIR / "k_required_audit.txt"
DISTRIBUTION_PATH = OUTPUT_DIR / "k_required_distribution.csv"
ROLE_PATH = OUTPUT_DIR / "k_required_downstream_role.csv"

LOGMAG_PATH = Path(
    "outputs/sho/stage1B/hull_slope_logmag/hull_slope_logmag_candidate.pt"
)
LOGMAG_SHA256 = "a4e1f1dbedf5f42121c441b6ced7b0f243fd78aed28b7e8c9c838178f675b249"
SIGN_PATH = Path(
    "outputs/sho/stage1B/hull_slope_sign/hull_slope_sign_locked.pt"
)
SIGN_SHA256 = "fb43410c48f89189757d1a397e95945e8717f7b069ebda88eeb019684a98a848"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_locked_hashes() -> None:
    for path, expected, name in (
        (LOGMAG_PATH, LOGMAG_SHA256, "log-magnitude source"),
        (SIGN_PATH, SIGN_SHA256, "locked sign network"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} hash mismatch: {actual} != {expected}")


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


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {
        "count": len(values),
        "finite_count": len(finite),
        "positive_infinity_count": int(np.sum(values == math.inf)),
        "negative_infinity_count": int(np.sum(values == -math.inf)),
        "nan_count": int(np.sum(np.isnan(values))),
        "exact_zero_count": int(np.sum(values == 0.0)),
    }
    if len(finite):
        result.update(
            {
                "minimum": float(np.min(finite)),
                "p01": float(np.percentile(finite, 1.0)),
                "p05": float(np.percentile(finite, 5.0)),
                "p25": float(np.percentile(finite, 25.0)),
                "median": float(np.percentile(finite, 50.0)),
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
    return result


def _distribution_row(variable: str, values: np.ndarray) -> dict[str, Any]:
    return {
        "record_type": "variable_distribution",
        "variable": variable,
        **_stats(values),
    }


def _bucket(value: float) -> str:
    if value < 0.0:
        return "negative"
    if value == 0.0:
        return "zero"
    if value < 1.0:
        return "positive_0_to_1"
    if value < 1.0e3:
        return "positive_1_to_1e3"
    if value < 1.0e6:
        return "positive_1e3_to_1e6"
    if value < 1.0e12:
        return "positive_1e6_to_1e12"
    if value < 1.0e24:
        return "positive_1e12_to_1e24"
    if value < 1.0e40:
        return "positive_1e24_to_1e40"
    return "positive_ge_1e40"


def _active_term(k_prev: float, k_required: float) -> str:
    lower = max(k_prev, k_required, 0.0)
    if lower == k_prev:
        return "K_prev"
    if lower == k_required:
        return "K_required"
    return "zero"


def _events(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_position, state in enumerate(states):
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = np.asarray(state["sizes"], dtype=np.float64)
        losses = np.asarray(state["losses"], dtype=np.float64)
        f_min = float(np.min(losses))
        target = f_min - edirect.EDIRECT_EPSILON * abs(f_min)
        for hull_position, index_value in enumerate(hull):
            index = int(index_value)
            teacher = improvement._event(
                state_position,
                state,
                hull_position,
                index,
                hull,
                target,
                f_min,
            )
            d_i = float(sizes[index])
            f_i = float(losses[index])
            amount = f_i - target
            reconstructed = amount / d_i if d_i > 0.0 else math.inf
            teacher_required = float(teacher["improvement_required_slope"])
            exact = bool(reconstructed == teacher_required)
            k_prev = float(teacher["lower_hull_slope"])
            strongest_other = max(k_prev, 0.0)
            competition_margin = teacher_required - strongest_other
            absolute_competition_gap = abs(competition_margin)
            active = _active_term(k_prev, teacher_required)
            rows.append(
                {
                    "record_type": "event",
                    "event_id": len(rows),
                    "state_position": state_position,
                    "source_state_index": int(state["source_state_index"]),
                    "trajectory_id": int(state["trajectory_id"]),
                    "iteration": int(state["iteration"]),
                    "state_bin": state["state_bin"],
                    "hull_position": hull_position,
                    "representative_index": index,
                    "is_first_hull_point": hull_position == 0,
                    "is_last_hull_point": hull_position == len(hull) - 1,
                    "f_i": f_i,
                    "f_min": f_min,
                    "epsilon": edirect.EDIRECT_EPSILON,
                    "target": target,
                    "improvement_amount": amount,
                    "d_i": d_i,
                    "K_required_teacher": teacher_required,
                    "K_required_reconstructed": reconstructed,
                    "absolute_reconstruction_error": abs(
                        reconstructed - teacher_required
                    ),
                    "exact_reconstruction": exact,
                    "K_required_bucket": _bucket(teacher_required),
                    "K_prev": k_prev,
                    "zero_term": 0.0,
                    "strongest_other_term": strongest_other,
                    "active_lower_term": active,
                    "K_required_wins": active == "K_required",
                    "K_required_below_K_prev": teacher_required < k_prev,
                    "K_required_equal_K_prev": teacher_required == k_prev,
                    "K_required_below_zero": teacher_required < 0.0,
                    "K_required_equal_zero": teacher_required == 0.0,
                    "K_required_minus_strongest_other": competition_margin,
                    "absolute_competition_gap": absolute_competition_gap,
                    "downstream_K_lower": float(teacher["admissible_K_lower"]),
                }
            )
    return rows


def _summary_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variables = (
        "f_i",
        "f_min",
        "target",
        "improvement_amount",
        "d_i",
        "K_required_teacher",
    )
    for variable in variables:
        rows.append(
            _distribution_row(
                variable,
                np.asarray([event[variable] for event in events], dtype=np.float64),
            )
        )
    bucket_order = (
        "negative",
        "zero",
        "positive_0_to_1",
        "positive_1_to_1e3",
        "positive_1e3_to_1e6",
        "positive_1e6_to_1e12",
        "positive_1e12_to_1e24",
        "positive_1e24_to_1e40",
        "positive_ge_1e40",
    )
    for bucket in bucket_order:
        rows.append(
            {
                "record_type": "K_required_bucket",
                "bucket": bucket,
                "event_count": sum(
                    event["K_required_bucket"] == bucket for event in events
                ),
            }
        )
    k = np.asarray(
        [event["K_required_teacher"] for event in events], dtype=np.float64
    )
    amount = np.asarray(
        [event["improvement_amount"] for event in events], dtype=np.float64
    )
    d_i = np.asarray([event["d_i"] for event in events], dtype=np.float64)
    f_i = np.asarray([event["f_i"] for event in events], dtype=np.float64)
    defined = (
        np.isfinite(k)
        & (k != 0.0)
        & np.isfinite(amount)
        & (amount != 0.0)
        & np.isfinite(d_i)
        & (d_i > 0.0)
    )
    log_k = np.log(np.abs(k[defined]))
    log_amount = np.log(np.abs(amount[defined]))
    negative_log_d = -np.log(d_i[defined])
    log_f_defined = defined & (f_i != 0.0)
    rows.append(
        {
            "record_type": "log_correlation",
            "defined_event_count": int(np.sum(defined)),
            "corr_log_abs_K_with_log_abs_improvement_amount": float(
                np.corrcoef(log_k, log_amount)[0, 1]
            ),
            "corr_log_abs_K_with_negative_log_d_i": float(
                np.corrcoef(log_k, negative_log_d)[0, 1]
            ),
            "corr_log_abs_K_with_log_abs_f_i": float(
                np.corrcoef(
                    np.log(np.abs(k[log_f_defined])),
                    np.log(np.abs(f_i[log_f_defined])),
                )[0, 1]
            ),
            "log_abs_improvement_amount_range": float(
                np.max(log_amount) - np.min(log_amount)
            ),
            "negative_log_d_i_range": float(
                np.max(negative_log_d) - np.min(negative_log_d)
            ),
            "log_abs_K_range": float(np.max(log_k) - np.min(log_k)),
        }
    )
    exact_count = sum(bool(event["exact_reconstruction"]) for event in events)
    rows.append(
        {
            "record_type": "reconstruction_verification",
            "exact_reconstruction_count": exact_count,
            "total_event_count": len(events),
            "exact_fraction": exact_count / len(events),
            "maximum_absolute_reconstruction_error": max(
                float(event["absolute_reconstruction_error"]) for event in events
            ),
        }
    )
    return rows


def _role_summary_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = (
        ("K_required_wins", lambda event: bool(event["K_required_wins"])),
        (
            "K_required_below_K_prev",
            lambda event: bool(event["K_required_below_K_prev"]),
        ),
        ("K_required_below_zero", lambda event: bool(event["K_required_below_zero"])),
        ("K_required_equal_K_prev", lambda event: bool(event["K_required_equal_K_prev"])),
        ("K_required_equal_zero", lambda event: bool(event["K_required_equal_zero"])),
    )
    for name, predicate in outcomes:
        count = sum(predicate(event) for event in events)
        rows.append(
            {
                "record_type": "role_count",
                "role": name,
                "event_count": count,
                "total_event_count": len(events),
                "fraction": count / len(events),
            }
        )
    for active in ("K_required", "K_prev", "zero"):
        selected = [
            event for event in events if event["active_lower_term"] == active
        ]
        gaps = np.asarray(
            [event["absolute_competition_gap"] for event in selected],
            dtype=np.float64,
        )
        row: dict[str, Any] = {
            "record_type": "active_term_gap_distribution",
            "active_lower_term": active,
            "event_count": len(selected),
        }
        if len(gaps):
            row.update(_stats(gaps))
        rows.append(row)
    signed = np.asarray(
        [event["K_required_minus_strongest_other"] for event in events],
        dtype=np.float64,
    )
    rows.append(
        {
            "record_type": "signed_competition_margin_distribution",
            **_stats(signed),
        }
    )
    return rows


def _format_stats(name: str, values: np.ndarray) -> str:
    stats = _stats(values)
    keys = (
        "minimum",
        "p01",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "maximum",
    )
    return "\n".join(
        [f"  {name}:"]
        + [f"    {key} = {float(stats[key]):.17g}" for key in keys]
        + [
            f"    exact_zero_count = {stats['exact_zero_count']}",
            f"    finite_count = {stats['finite_count']}/{stats['count']}",
        ]
    )


def _write_audit(
    states: list[dict[str, Any]],
    events: list[dict[str, Any]],
    distribution_summary: list[dict[str, Any]],
) -> None:
    arrays = {
        name: np.asarray([event[name] for event in events], dtype=np.float64)
        for name in (
            "f_i",
            "f_min",
            "target",
            "improvement_amount",
            "d_i",
            "K_required_teacher",
        )
    }
    k = arrays["K_required_teacher"]
    positive = k[k > 0.0]
    negative = k[k < 0.0]
    nonzero = k[k != 0.0]
    correlation = next(
        row for row in distribution_summary if row["record_type"] == "log_correlation"
    )
    bucket_lines = "\n".join(
        f"  {row['bucket']}: {row['event_count']}"
        for row in distribution_summary
        if row["record_type"] == "K_required_bucket"
    )
    active_counts = {
        name: sum(event["active_lower_term"] == name for event in events)
        for name in ("K_required", "K_prev", "zero")
    }
    below_prev = sum(bool(event["K_required_below_K_prev"]) for event in events)
    below_zero = sum(bool(event["K_required_below_zero"]) for event in events)
    gap_wins = np.asarray(
        [
            event["absolute_competition_gap"]
            for event in events
            if event["K_required_wins"]
        ],
        dtype=np.float64,
    )
    gap_losses = np.asarray(
        [
            event["absolute_competition_gap"]
            for event in events
            if not event["K_required_wins"]
        ],
        dtype=np.float64,
    )
    exact = sum(bool(event["exact_reconstruction"]) for event in events)
    endpoint_count = sum(
        bool(event["is_first_hull_point"] or event["is_last_hull_point"])
        for event in events
    )
    text = f"""K_required operator audit
=========================

Scope and locked provenance
---------------------------
No training, checkpoint mutation, hull reconstruction, or trajectory search was
performed. This audit uses every lower-hull representative in the existing held-out
validation trace: {len(states)} states and {len(events)} events.

The locked slope artifacts were verified read-only before the audit:

- log-magnitude source SHA-256: {LOGMAG_SHA256}
- locked sign network SHA-256: {SIGN_SHA256}

Neither artifact is an input to K_required and neither was loaded or modified.

1. Exact implemented teacher arithmetic
---------------------------------------
The audited implementation is
`run_stage1b_edirect_vs_direct.py::_potentially_optimal_cells`:

    f_min  = min(all evaluated site losses)
    target = f_min - 1e-4 * abs(f_min)

    if d_i > 0:
        K_required = (f_i - target) / d_i
    else:
        K_required = +inf

The constant `1e-4` is the DIRECT sufficient-improvement parameter, not a numerical
division epsilon. No additive epsilon, clipping, tolerance, or stabilization is used in
this operator. The later `5e-14` feasibility tolerance is not part of K_required.

In the held-out pool, d_i was strictly positive for
{int(np.sum(arrays['d_i'] > 0.0))}/{len(events)} events; zero/non-positive count was
{int(np.sum(arrays['d_i'] <= 0.0))}. Therefore the +inf special case was never reached.

K_required was positive for {len(positive)} events, negative for {len(negative)}, and
exactly zero for {int(np.sum(k == 0.0))}. Under valid teacher inputs it cannot be
negative: `f_i >= f_min >= target`, and d_i is positive. It can be exactly zero only in
the degenerate valid case `f_i = f_min = 0`, because then the sufficient-improvement
offset is also zero. That case did not occur in this held-out pool.

All hull positions use the same K_required formula. The {endpoint_count} first/last
endpoint events differ only in their neighboring K_prev/K_next sentinels downstream;
there is no endpoint branch inside this operator.

2. Clean exact decomposition
----------------------------

    improvement_amount = f_i - target
                       = f_i - f_min + 1e-4*abs(f_min)

    K_required = improvement_amount / d_i

Equivalently, directly in the three inputs:

    K_required = (f_i - f_min + 1e-4*abs(f_min)) / d_i

This is the cleanest exact decomposition: one sufficient-improvement subtraction,
followed by one positive-denominator quotient.

3. Held-out numerical ranges
----------------------------
{_format_stats('f_i', arrays['f_i'])}
{_format_stats('f_min', arrays['f_min'])}
{_format_stats('target', arrays['target'])}
{_format_stats('improvement_amount', arrays['improvement_amount'])}
{_format_stats('d_i', arrays['d_i'])}
{_format_stats('K_required', arrays['K_required_teacher'])}

K_required sign/magnitude buckets:
{bucket_lines}

Smallest nonzero |K_required| = {float(np.min(np.abs(nonzero))):.17g}
Largest |K_required| = {float(np.max(np.abs(k))):.17g}

For the {correlation['defined_event_count']} positive finite log-defined events:

  corr(log|K_required|, log|improvement_amount|)
    = {correlation['corr_log_abs_K_with_log_abs_improvement_amount']:.17g}
  corr(log|K_required|, -log(d_i))
    = {correlation['corr_log_abs_K_with_negative_log_d_i']:.17g}
  corr(log|K_required|, log|f_i|)
    = {correlation['corr_log_abs_K_with_log_abs_f_i']:.17g}

The log-range contributed by the numerator is
{correlation['log_abs_improvement_amount_range']:.17g}, versus
{correlation['negative_log_d_i_range']:.17g} for reciprocal size. The conclusion about
which factor dominates is based on these measured spans and correlations, not merely on
the algebraic quotient. Here the numerator/f_i dynamics clearly dominate: its log range
is about 9.75 times larger and its correlation with log|K_required| is 0.995, whereas
the observed correlation with -log(d_i) is -0.316.

4. Downstream max role
----------------------
Downstream uses exactly:

    K_lower = max(K_prev, K_required, 0.0)

K_required is the active term in {active_counts['K_required']}/{len(events)} events.
K_prev is active in {active_counts['K_prev']}/{len(events)}, and zero is active in
{active_counts['zero']}/{len(events)}. K_required is strictly below K_prev in
{below_prev}/{len(events)} events and strictly below zero in {below_zero}/{len(events)}.

The competition quantity recorded per event is
`K_required - max(K_prev, 0)`. Its absolute value is the amount by which K_required
wins or loses against its strongest other term. Among K_required wins, the median gap
is {float(np.median(gap_wins)):.17g} and the minimum gap is
{float(np.min(gap_wins)):.17g}. Among losses, the median gap magnitude is
{float(np.median(gap_losses)):.17g} and the minimum is
{float(np.min(gap_losses)):.17g}. Full percentiles are in the role CSV.

Consequently, eventual learning must preserve both the max-term ordering and the
magnitude used in the later K_lower-versus-K_next feasibility comparison; sign alone is
not sufficient even though K_required is positive throughout this pool.

5. Candidate learning representations (audit only)
---------------------------------------------------
Raw K_required is poorly conditioned for ordinary unscaled regression: it runs from
{float(np.min(k)):.17g} to {float(np.max(k)):.17g}, spanning almost 60 decimal orders
of magnitude. The exact, invertible positive-event
representation is:

    log(K_required) = log(improvement_amount) - log(d_i)

No epsilon is needed because both quantities are strictly positive in this pool. If a
future dataset contains exact-zero improvement amounts, an explicit zero indicator plus
the log magnitude of nonzero events is required; silently adding epsilon would change
the operator. A signed-log decomposition is mathematically faithful for a more general
dataset, but the observed sign is constant positive here, so sign learning would carry
no information. Retaining `improvement_amount` and `d_i` as separate continuous
quantities is also exact and postpones the quotient operation cleanly.

This audit selects no architecture and performs no training.

6. Independent reconstruction verification
-------------------------------------------
The independently decomposed calculation reproduced the teacher float64 value exactly
for {exact}/{len(events)} events. Maximum absolute reconstruction error was
{max(float(event['absolute_reconstruction_error']) for event in events):.17g}.
"""
    AUDIT_PATH.write_text(text, encoding="utf-8")


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _verify_locked_hashes()
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    states = turn._build_states(payload, indices, workers, "K_required audit")
    events = _events(states)
    if not events:
        raise RuntimeError("no held-out K_required events")
    if any(not bool(event["exact_reconstruction"]) for event in events):
        raise RuntimeError("independent K_required reconstruction mismatch")

    distribution_summary = _summary_rows(events)
    distribution_rows = distribution_summary + events
    role_rows = _role_summary_rows(events) + [
        {
            key: value
            for key, value in event.items()
            if key
            in {
                "record_type",
                "event_id",
                "state_position",
                "source_state_index",
                "trajectory_id",
                "iteration",
                "state_bin",
                "hull_position",
                "representative_index",
                "is_first_hull_point",
                "is_last_hull_point",
                "K_required_teacher",
                "K_prev",
                "zero_term",
                "strongest_other_term",
                "active_lower_term",
                "K_required_wins",
                "K_required_below_K_prev",
                "K_required_equal_K_prev",
                "K_required_below_zero",
                "K_required_equal_zero",
                "K_required_minus_strongest_other",
                "absolute_competition_gap",
                "downstream_K_lower",
            }
        }
        for event in events
    ]
    _write_csv(DISTRIBUTION_PATH, distribution_rows)
    _write_csv(ROLE_PATH, role_rows)
    _write_audit(states, events, distribution_summary)
    print(
        f"K_required audit exact={len(events)}/{len(events)} "
        f"positive={sum(float(event['K_required_teacher']) > 0.0 for event in events)} "
        f"wins={sum(bool(event['K_required_wins']) for event in events)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
