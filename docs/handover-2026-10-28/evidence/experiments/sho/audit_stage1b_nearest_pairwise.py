#!/usr/bin/env python3
"""Audit the decisive pairwise comparisons behind nearest-site selection.

SciPy's cKDTree does not expose its internal tree-traversal comparisons.  The
auditable comparison is therefore the exact nearest-versus-runner-up pair
returned by cKDTree.  That pair is sufficient to reconstruct each unique
argmin; exact distance ties retain the production cKDTree result.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.run_stage1b_distance_metric as distance_metric
import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/nearest_pairwise_audit")
AUDIT_PATH = OUTPUT_DIR / "nearest_pairwise_audit.txt"
PROBE_PATH = OUTPUT_DIR / "nearest_pairwise_probe.csv"
SPACING_PATH = OUTPUT_DIR / "nearest_pairwise_spacing.csv"

THRESHOLDS = (1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15, 1.0e-16, 1.0e-17)
LOCKED_DISTANCE_PATH = Path(
    "outputs/sho/stage1B/distance_metric_assembled_validation/"
    "geometric_distance_metric_locked.pt"
)
LOCKED_DISTANCE_SHA256 = (
    "29d1e5947a45f6fb79e2d8e5baf7ca4b75406e3e4637d69fe58558e2ebef0c1c"
)

FIELDS = [
    "record_type",
    "split",
    "source_state_index",
    "trajectory_id",
    "generator_id",
    "iteration",
    "query_index",
    "current_site_index",
    "candidate_site_index",
    "d_current",
    "d_candidate",
    "raw_margin_m",
    "scale_S",
    "normalized_margin",
    "teacher_decision",
    "teacher_winner_site_index",
    "reconstructed_winner_site_index",
    "exact_tie",
    "ulp_separation",
    "one_ulp_apart",
    "reconstruction_exact",
    "quantity",
    "statistic",
    "value",
    "sample_count",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ulp_separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return representable-float separation for finite nonnegative float64."""

    if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise RuntimeError("ULP audit received a non-finite distance")
    if np.any(first < 0.0) or np.any(second < 0.0):
        raise RuntimeError("ULP audit received a negative distance")
    first_bits = np.ascontiguousarray(first, dtype=np.float64).view(np.uint64)
    second_bits = np.ascontiguousarray(second, dtype=np.float64).view(np.uint64)
    return np.where(
        first_bits >= second_bits,
        first_bits - second_bits,
        second_bits - first_bits,
    ).astype(np.uint64)


def _ordered_pair(
    first_site: np.ndarray,
    second_site: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Order each pair by original site index to define current/candidate."""

    first_is_current = first_site < second_site
    current_site = np.where(first_is_current, first_site, second_site).astype(np.int64)
    candidate_site = np.where(first_is_current, second_site, first_site).astype(
        np.int64
    )
    current_distance = np.where(
        first_is_current, first_distance, second_distance
    ).astype(np.float64)
    candidate_distance = np.where(
        first_is_current, second_distance, first_distance
    ).astype(np.float64)
    return current_site, candidate_site, current_distance, candidate_distance


def _write_rows(
    writer: csv.DictWriter,
    *,
    split: str,
    source_state_index: int,
    record: dict[str, Any],
    query_indices: np.ndarray,
    current_sites: np.ndarray,
    candidate_sites: np.ndarray,
    current_distances: np.ndarray,
    candidate_distances: np.ndarray,
    teacher_winners: np.ndarray,
) -> dict[str, Any]:
    margins = current_distances - candidate_distances
    ties = margins == 0.0
    scales = np.maximum(current_distances, candidate_distances)
    non_ties = ~ties
    if np.any(scales[non_ties] <= 0.0):
        raise RuntimeError("non-tied distance comparison has nonpositive scale")
    normalized = np.full(len(margins), np.nan, dtype=np.float64)
    normalized[non_ties] = margins[non_ties] / scales[non_ties]
    if not np.array_equal(margins[non_ties] > 0.0, normalized[non_ties] > 0.0):
        raise RuntimeError("normalization changed a positive comparison sign")
    if not np.array_equal(margins[non_ties] < 0.0, normalized[non_ties] < 0.0):
        raise RuntimeError("normalization changed a negative comparison sign")

    reconstructed = np.where(
        margins > 0.0, candidate_sites, current_sites
    ).astype(np.int64)
    # An exact tie is intentionally outside the learned comparison and retains
    # the production cKDTree bookkeeping result.
    reconstructed[ties] = teacher_winners[ties]
    exact = reconstructed == teacher_winners
    ulps = _ulp_separation(current_distances, candidate_distances)

    for position in range(len(query_indices)):
        tie = bool(ties[position])
        if tie:
            decision = "TIE_CKDTREE"
        elif margins[position] > 0.0:
            decision = "REPLACE"
        else:
            decision = "KEEP"
        writer.writerow(
            {
                "record_type": "exact_tie" if tie else "comparison",
                "split": split,
                "source_state_index": source_state_index,
                "trajectory_id": int(record["trajectory_id"]),
                "generator_id": int(record["generator_id"]),
                "iteration": int(record["iteration"]),
                "query_index": int(query_indices[position]),
                "current_site_index": int(current_sites[position]),
                "candidate_site_index": int(candidate_sites[position]),
                "d_current": float(current_distances[position]),
                "d_candidate": float(candidate_distances[position]),
                "raw_margin_m": float(margins[position]),
                "scale_S": float(scales[position]),
                "normalized_margin": (
                    "" if tie else float(normalized[position])
                ),
                "teacher_decision": decision,
                "teacher_winner_site_index": int(teacher_winners[position]),
                "reconstructed_winner_site_index": int(reconstructed[position]),
                "exact_tie": tie,
                "ulp_separation": int(ulps[position]),
                "one_ulp_apart": bool(ulps[position] == 1),
                "reconstruction_exact": bool(exact[position]),
            }
        )
    return {
        "margins": margins[non_ties],
        "normalized": normalized[non_ties],
        "tie_count": int(np.sum(ties)),
        "replace_count": int(np.sum(margins > 0.0)),
        "keep_count": int(np.sum(margins < 0.0)),
        "one_ulp_count": int(np.sum((ulps == 1) & non_ties)),
        "reconstruction_mismatch_count": int(np.sum(~exact)),
        "event_count": len(margins),
    }


def _percentile_rows(kind: str, quantity: str, values: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, percentile in (
        ("min", 0.0),
        ("p01", 1.0),
        ("p05", 5.0),
        ("p25", 25.0),
        ("median", 50.0),
        ("p75", 75.0),
        ("p95", 95.0),
        ("p99", 99.0),
        ("max", 100.0),
    ):
        rows.append(
            {
                "record_type": "summary",
                "split": kind,
                "quantity": quantity,
                "statistic": name,
                "value": float(np.percentile(values, percentile)),
                "sample_count": len(values),
            }
        )
    return rows


def _summary_rows(kind: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    margins = np.concatenate(summary["margin_parts"])
    normalized = np.concatenate(summary["normalized_parts"])
    rows = _percentile_rows(kind, "raw_margin_m", margins)
    rows.extend(_percentile_rows(kind, "normalized_margin", normalized))
    scalar_values: dict[str, int | float] = {
        "total_pair_count": summary["event_count"],
        "non_tied_comparison_count": len(margins),
        "replace_count": summary["replace_count"],
        "keep_count": summary["keep_count"],
        "exact_tie_count": summary["tie_count"],
        "one_ulp_apart_count": summary["one_ulp_count"],
        "reconstruction_mismatch_count": summary["reconstruction_mismatch_count"],
        "closest_positive_margin_to_zero": float(np.min(margins[margins > 0.0])),
        "closest_negative_margin_to_zero": float(np.max(margins[margins < 0.0])),
        "normalized_sign_change_count": int(
            np.sum(np.signbit(margins) != np.signbit(normalized))
        ),
    }
    for threshold in THRESHOLDS:
        scalar_values[f"count_abs_margin_below_{threshold:.0e}"] = int(
            np.sum(np.abs(margins) < threshold)
        )
    for quantity, value in scalar_values.items():
        rows.append(
            {
                "record_type": "summary",
                "split": kind,
                "quantity": quantity,
                "statistic": "count" if "count" in quantity else "value",
                "value": value,
                "sample_count": len(margins),
            }
        )
    summary["margins"] = margins
    summary["normalized"] = normalized
    return rows


def _new_summary() -> dict[str, Any]:
    return {
        "margin_parts": [],
        "normalized_parts": [],
        "event_count": 0,
        "replace_count": 0,
        "keep_count": 0,
        "tie_count": 0,
        "one_ulp_count": 0,
        "reconstruction_mismatch_count": 0,
    }


def _accumulate(total: dict[str, Any], state: dict[str, Any]) -> None:
    total["margin_parts"].append(state["margins"])
    total["normalized_parts"].append(state["normalized"])
    for key in (
        "event_count",
        "replace_count",
        "keep_count",
        "tie_count",
        "one_ulp_count",
        "reconstruction_mismatch_count",
    ):
        total[key] += state[key]


def _format_distribution(values: np.ndarray) -> str:
    return (
        f"min={np.min(values):.17g}, median={np.median(values):.17g}, "
        f"max={np.max(values):.17g}"
    )


def _threshold_lines(margins: np.ndarray) -> Iterable[str]:
    for threshold in THRESHOLDS:
        yield f"  |m| < {threshold:.0e}: {int(np.sum(np.abs(margins) < threshold)):,}"


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LOCKED_DISTANCE_PATH.exists():
        raise FileNotFoundError(LOCKED_DISTANCE_PATH)
    locked_hash_before = _sha256(LOCKED_DISTANCE_PATH)
    if locked_hash_before != LOCKED_DISTANCE_SHA256:
        raise RuntimeError(
            f"locked distance hash mismatch: {locked_hash_before}"
        )

    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    held_out_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )

    probe_total = _new_summary()
    spacing_total = _new_summary()
    with PROBE_PATH.open("w", newline="", encoding="utf-8") as probe_handle, \
        SPACING_PATH.open("w", newline="", encoding="utf-8") as spacing_handle:
        probe_writer = csv.DictWriter(probe_handle, fieldnames=FIELDS)
        spacing_writer = csv.DictWriter(spacing_handle, fieldnames=FIELDS)
        probe_writer.writeheader()
        spacing_writer.writeheader()

        for position, source_state_index in enumerate(held_out_indices, start=1):
            source_state_index = int(source_state_index)
            sites, probes, record = distance_metric._state_geometry(
                payload, trajectories, source_state_index
            )
            trajectory = trajectories[int(record["trajectory_id"])]
            count = int(record["history_count"])
            losses = np.asarray(
                trajectory["evaluated_prefix_losses"][:count], dtype=np.float64
            )
            production = edirect._cell_data(
                sites,
                losses,
                int(record["generator_id"]),
                int(record["iteration"]),
            )
            if not np.array_equal(probes, production.points):
                raise RuntimeError("regenerated probe pool differs from production")
            tree = cKDTree(sites)

            probe_distances, probe_sites = tree.query(probes, k=2)
            probe_current, probe_candidate, d_current, d_candidate = _ordered_pair(
                probe_sites[:, 0],
                probe_sites[:, 1],
                probe_distances[:, 0],
                probe_distances[:, 1],
            )
            probe_state = _write_rows(
                probe_writer,
                split="held_out",
                source_state_index=source_state_index,
                record=record,
                query_indices=np.arange(len(probes), dtype=np.int64),
                current_sites=probe_current,
                candidate_sites=probe_candidate,
                current_distances=d_current,
                candidate_distances=d_candidate,
                teacher_winners=np.asarray(production.owners, dtype=np.int64),
            )
            _accumulate(probe_total, probe_state)

            spacing_teacher_distance, spacing_teacher_site = tree.query(sites, k=2)
            spacing_distances, spacing_sites = tree.query(sites, k=3)
            spacing_current, spacing_candidate, s_current, s_candidate = _ordered_pair(
                spacing_sites[:, 1],
                spacing_sites[:, 2],
                spacing_distances[:, 1],
                spacing_distances[:, 2],
            )
            spacing_state = _write_rows(
                spacing_writer,
                split="held_out",
                source_state_index=source_state_index,
                record=record,
                query_indices=np.arange(len(sites), dtype=np.int64),
                current_sites=spacing_current,
                candidate_sites=spacing_candidate,
                current_distances=s_current,
                candidate_distances=s_candidate,
                teacher_winners=np.asarray(spacing_teacher_site[:, 1], dtype=np.int64),
            )
            _accumulate(spacing_total, spacing_state)

            if position % 10 == 0 or position == len(held_out_indices):
                print(f"nearest comparison audit states {position}/{len(held_out_indices)}", flush=True)

        probe_writer.writerows(_summary_rows("probe_assignment", probe_total))
        spacing_writer.writerows(_summary_rows("site_spacing", spacing_total))

    if probe_total["reconstruction_mismatch_count"] != 0:
        raise RuntimeError("probe-owner reconstruction did not exactly match teacher")
    if spacing_total["reconstruction_mismatch_count"] != 0:
        raise RuntimeError("spacing-owner reconstruction did not exactly match teacher")
    locked_hash_after = _sha256(LOCKED_DISTANCE_PATH)
    if locked_hash_after != locked_hash_before:
        raise RuntimeError("locked distance checkpoint changed during audit")

    probe_m = probe_total["margins"]
    probe_n = probe_total["normalized"]
    spacing_m = spacing_total["margins"]
    spacing_n = spacing_total["normalized"]
    lines = [
        "PAIRWISE NEAREST-DISTANCE COMPARISON AUDIT",
        "==========================================",
        "",
        "Scope",
        "-----",
        f"Held-out states: {len(held_out_indices)}",
        "No training, trajectory search, or checkpoint mutation was performed.",
        f"Locked distance checkpoint SHA-256 (unchanged): {locked_hash_after}",
        "",
        "Instrumentation note",
        "--------------------",
        "SciPy cKDTree does not expose internal node-by-node traversal comparisons.",
        "This audit records the exact nearest-versus-runner-up comparison returned",
        "by cKDTree for every query. That decisive pair is sufficient to reconstruct",
        "every unique argmin. Pair members are ordered by original site index: the",
        "lower index is current and the higher index is candidate. Exact equal-distance",
        "ties bypass the comparator and retain production cKDTree k=1 bookkeeping.",
        "",
        "Teacher comparison",
        "------------------",
        "m = d_current - d_candidate",
        "m > 0: REPLACE current with candidate",
        "m < 0: KEEP current",
        "m = 0: exact tie; retain deterministic cKDTree owner",
        "",
        "Probe -> evaluated-site assignment",
        "----------------------------------",
        f"Total decisive pairs: {probe_total['event_count']:,}",
        f"Non-tied comparisons: {len(probe_m):,}",
        f"REPLACE: {probe_total['replace_count']:,}",
        f"KEEP: {probe_total['keep_count']:,}",
        f"Exact ties: {probe_total['tie_count']:,}",
        f"Raw m distribution: {_format_distribution(probe_m)}",
        f"Closest positive m to zero: {np.min(probe_m[probe_m > 0.0]):.17g}",
        f"Closest negative m to zero: {np.max(probe_m[probe_m < 0.0]):.17g}",
        *_threshold_lines(probe_m),
        f"Non-tied comparisons exactly 1 ULP apart: {probe_total['one_ulp_count']:,}",
        f"Normalized m range: [{np.min(probe_n):.17g}, {np.max(probe_n):.17g}]",
        f"Normalized sign changes: {int(np.sum(np.signbit(probe_m) != np.signbit(probe_n))):,}",
        f"Reconstructed owner mismatches: {probe_total['reconstruction_mismatch_count']:,}",
        "",
        "Site -> nearest-other-site spacing",
        "----------------------------------",
        f"Total decisive pairs: {spacing_total['event_count']:,}",
        f"Non-tied comparisons: {len(spacing_m):,}",
        f"REPLACE: {spacing_total['replace_count']:,}",
        f"KEEP: {spacing_total['keep_count']:,}",
        f"Exact ties: {spacing_total['tie_count']:,}",
        f"Raw m distribution: {_format_distribution(spacing_m)}",
        f"Closest positive m to zero: {np.min(spacing_m[spacing_m > 0.0]):.17g}",
        f"Closest negative m to zero: {np.max(spacing_m[spacing_m < 0.0]):.17g}",
        *_threshold_lines(spacing_m),
        f"Non-tied comparisons exactly 1 ULP apart: {spacing_total['one_ulp_count']:,}",
        f"Normalized m range: [{np.min(spacing_n):.17g}, {np.max(spacing_n):.17g}]",
        f"Normalized sign changes: {int(np.sum(np.signbit(spacing_m) != np.signbit(spacing_n))):,}",
        f"Reconstructed nearest-neighbor mismatches: {spacing_total['reconstruction_mismatch_count']:,}",
        "",
        "Numerical representation",
        "------------------------",
        "For every non-tied event S=max(d_current,d_candidate) was strictly positive.",
        "m_norm=m/S preserved the sign of every non-tied comparison exactly and bounds",
        "the scale, but it does not remove the genuine near-boundary cases: comparisons",
        "one representable float apart remain intrinsically sensitive. Raw m is therefore",
        "well-defined but not uniformly well-conditioned as a learned threshold target;",
        "m_norm is better scaled while retaining the same ultra-close ordering burden.",
        "",
        "Independent reconstruction",
        "--------------------------",
        f"Probe owners: {probe_total['event_count'] - probe_total['reconstruction_mismatch_count']:,}/{probe_total['event_count']:,} exact",
        f"Site nearest neighbors: {spacing_total['event_count'] - spacing_total['reconstruction_mismatch_count']:,}/{spacing_total['event_count']:,} exact",
    ]
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {AUDIT_PATH}")
    print(f"wrote {PROBE_PATH}")
    print(f"wrote {SPACING_PATH}")


if __name__ == "__main__":
    run()
