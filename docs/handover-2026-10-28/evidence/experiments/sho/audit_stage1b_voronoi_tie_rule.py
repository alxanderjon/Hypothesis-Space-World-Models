#!/usr/bin/env python3
"""Counterfactual audit of explicit lowest-site-index Voronoi tie ownership."""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.audit_stage1b_same_size_grouping as grouping_audit
import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/voronoi_tie_rule_audit")
AUDIT_PATH = OUTPUT_DIR / "voronoi_tie_rule_audit.txt"
CHANGES_PATH = OUTPUT_DIR / "voronoi_tie_changes.csv"
DOWNSTREAM_PATH = OUTPUT_DIR / "voronoi_tie_downstream_comparison.csv"
EXACT_TIE_CANDIDATE_SCREEN = 1.0e-12


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


def _fmt_indices(values: np.ndarray | list[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def _fmt_coordinates(indices: np.ndarray, points: np.ndarray) -> str:
    return ";".join(
        f"{int(index)}:({float(points[int(index), 0]):.17g},"
        f"{float(points[int(index), 1]):.17g})"
        for index in indices
    )


def _group_lists(trace: dict[str, Any]) -> list[tuple[int, ...]]:
    return [
        tuple(int(value) for value in group["members"])
        for group in trace["groups"]
    ]


def _hull_site_ids(trace: dict[str, Any]) -> np.ndarray:
    hull, _ = turn._teacher_hull_and_triples(
        trace["grouped_sizes"], trace["grouped_losses"]
    )
    return trace["representatives"][hull]


def _symmetric_difference_count(first: np.ndarray, second: np.ndarray) -> int:
    return len(set(int(value) for value in first) ^ set(int(value) for value in second))


def _ordered_position_change_count(first: np.ndarray, second: np.ndarray) -> int:
    common = min(len(first), len(second))
    return int(np.sum(first[:common] != second[:common])) + abs(len(first) - len(second))


def _analyze_state(
    source_state_index: int,
    record: dict[str, Any],
    trajectory: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    history_count = int(record["history_count"])
    points = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:history_count],
        dtype=np.float64,
    )
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:history_count], dtype=np.float64
    )
    production = edirect._cell_data(
        points,
        losses,
        int(record["generator_id"]),
        int(record["iteration"]),
    )
    tree = cKDTree(points)
    two_distances, two_owners = tree.query(production.points, k=2)
    ckd_exact_tie = two_distances[:, 0] == two_distances[:, 1]
    nearest_gap = two_distances[:, 1] - two_distances[:, 0]
    # This conservative screen only avoids evaluating every site for millions
    # of clearly separated probes. It never labels a tie: the proposed rule
    # below still requires exact equality of directly computed squared distance.
    audit_probe_indices = np.flatnonzero(
        nearest_gap <= EXACT_TIE_CANDIDATE_SCREEN
    )

    proposed_owners = production.owners.copy()
    tie_rows: list[dict[str, Any]] = []
    explicit_tie_count = 0
    changed_owner_count = 0
    arithmetic_disagreement_count = 0
    for probe_index_value in audit_probe_indices:
        probe_index = int(probe_index_value)
        probe = production.points[probe_index]
        delta = points - probe[None, :]
        squared_distances = np.square(delta[:, 0]) + np.square(delta[:, 1])
        minimum_squared_distance = float(np.min(squared_distances))
        tied_sites = np.flatnonzero(squared_distances == minimum_squared_distance)
        direct_exact_tie = len(tied_sites) >= 2
        if not direct_exact_tie:
            proposed_owner = int(np.argmin(squared_distances))
        else:
            explicit_tie_count += 1
            proposed_owner = int(np.min(tied_sites))
        current_exact_tie = bool(ckd_exact_tie[probe_index])
        arithmetic_disagreement_count += int(current_exact_tie != direct_exact_tie)
        old_owner = int(production.owners[probe_index])
        if old_owner not in tied_sites:
            raise RuntimeError(
                "production owner is absent from direct exact-minimum set in "
                f"state {source_state_index}, probe {probe_index}"
            )
        proposed_owners[probe_index] = proposed_owner
        changed = proposed_owner != old_owner
        changed_owner_count += int(changed)
        if not (current_exact_tie or direct_exact_tie or changed):
            continue
        tie_rows.append(
            {
                "record_type": "exact_tie_probe",
                "split": split,
                "source_state_index": source_state_index,
                "trajectory_id": int(record["trajectory_id"]),
                "iteration": int(record["iteration"]),
                "probe_index": probe_index,
                "probe_x_normalized": float(probe[0]),
                "probe_y_normalized": float(probe[1]),
                "ckdtree_tied_distance": float(two_distances[probe_index, 0]),
                "direct_minimum_squared_distance": minimum_squared_distance,
                "direct_minimum_distance": math.sqrt(minimum_squared_distance),
                "tied_site_count": len(tied_sites),
                "tied_site_indices": _fmt_indices(tied_sites),
                "tied_site_coordinates_normalized": _fmt_coordinates(
                    tied_sites, points
                ),
                "old_ckdtree_owner": old_owner,
                "new_lowest_index_owner": proposed_owner,
                "owner_changed": changed,
                "ckdtree_query_reports_exact_tie": current_exact_tie,
                "direct_arithmetic_confirms_exact_tie": direct_exact_tie,
            }
        )

    current_radius = np.zeros(history_count, dtype=np.float64)
    proposed_radius = np.zeros(history_count, dtype=np.float64)
    np.maximum.at(current_radius, production.owners, production.distances)
    # The counterfactual changes ownership only. Exact-tied probes retain the
    # same nearest distance returned by the production query.
    np.maximum.at(proposed_radius, proposed_owners, production.distances)
    nearest_spacing = tree.query(points, k=2)[0][:, 1]
    spacing_floor = 0.5 * nearest_spacing
    current_sizes = np.maximum(current_radius, spacing_floor)
    proposed_sizes = np.maximum(proposed_radius, spacing_floor)
    if not np.array_equal(current_sizes, production.sizes):
        raise RuntimeError("current cell-size reconstruction differs from teacher")

    radius_changed = current_radius != proposed_radius
    size_changed = current_sizes != proposed_sizes
    current_trace = grouping_audit._teacher_trace(current_sizes, losses)
    proposed_trace = grouping_audit._teacher_trace(proposed_sizes, losses)
    current_groups = _group_lists(current_trace)
    proposed_groups = _group_lists(proposed_trace)
    groups_exact = current_groups == proposed_groups
    group_membership_changes = int(
        np.sum(current_trace["membership"] != proposed_trace["membership"])
    )
    current_representatives = current_trace["representatives"]
    proposed_representatives = proposed_trace["representatives"]
    representatives_exact = bool(
        np.array_equal(current_representatives, proposed_representatives)
    )
    current_hull_sites = _hull_site_ids(current_trace)
    proposed_hull_sites = _hull_site_ids(proposed_trace)
    hull_exact = bool(np.array_equal(current_hull_sites, proposed_hull_sites))
    current_po = edirect._potentially_optimal_cells(current_sizes, losses)
    proposed_po = edirect._potentially_optimal_cells(proposed_sizes, losses)
    po_exact = bool(np.array_equal(current_po, proposed_po))

    current_farthest_count = np.zeros(history_count, dtype=np.int64)
    current_is_farthest = production.distances == current_radius[production.owners]
    np.add.at(current_farthest_count, production.owners, current_is_farthest.astype(np.int64))
    proposed_is_farthest = production.distances == proposed_radius[proposed_owners]
    proposed_farthest_count = np.zeros(history_count, dtype=np.int64)
    np.add.at(
        proposed_farthest_count,
        proposed_owners,
        proposed_is_farthest.astype(np.int64),
    )
    tied_row_by_probe = {int(row["probe_index"]): row for row in tie_rows}
    changed_farthest_either_count = 0
    changed_unique_farthest_either_count = 0
    for probe_index, row in tied_row_by_probe.items():
        old_owner = int(production.owners[probe_index])
        new_owner = int(proposed_owners[probe_index])
        distance = float(production.distances[probe_index])
        old_farthest = bool(distance == current_radius[old_owner])
        old_unique_farthest = bool(
            old_farthest and current_farthest_count[old_owner] == 1
        )
        new_farthest = bool(distance == proposed_radius[new_owner])
        new_unique_farthest = bool(
            new_farthest and proposed_farthest_count[new_owner] == 1
        )
        either_farthest = old_farthest or new_farthest
        either_unique = old_unique_farthest or new_unique_farthest
        if bool(row["owner_changed"]):
            changed_farthest_either_count += int(either_farthest)
            changed_unique_farthest_either_count += int(either_unique)
        row.update(
            {
                "distance_used_for_radius": distance,
                "old_owner_current_radius": float(current_radius[old_owner]),
                "new_owner_current_radius": float(current_radius[new_owner]),
                "old_owner_proposed_radius": float(proposed_radius[old_owner]),
                "new_owner_proposed_radius": float(proposed_radius[new_owner]),
                "probe_was_old_owner_farthest": old_farthest,
                "probe_was_old_owner_unique_farthest": old_unique_farthest,
                "probe_is_new_owner_farthest": new_farthest,
                "probe_is_new_owner_unique_farthest": new_unique_farthest,
                "probe_farthest_for_either_affected_site": either_farthest,
                "probe_unique_farthest_for_either_affected_site": either_unique,
                "old_owner_radius_changed": bool(radius_changed[old_owner]),
                "new_owner_radius_changed": bool(radius_changed[new_owner]),
                "old_owner_cell_size_changed": bool(size_changed[old_owner]),
                "new_owner_cell_size_changed": bool(size_changed[new_owner]),
            }
        )

    group_anchor_max_difference = math.nan
    if len(current_trace["grouped_sizes"]) == len(proposed_trace["grouped_sizes"]):
        group_anchor_max_difference = float(
            np.max(
                np.abs(
                    current_trace["grouped_sizes"]
                    - proposed_trace["grouped_sizes"]
                )
            )
        )

    state_row = {
        "record_type": "state_comparison",
        "split": split,
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": int(record["iteration"]),
        "site_count": history_count,
        "raw_probe_count": len(production.points),
        "exact_tie_candidate_screen_count": len(audit_probe_indices),
        "ckdtree_exact_tie_probe_count": int(np.sum(ckd_exact_tie)),
        "direct_exact_tie_probe_count": explicit_tie_count,
        "tie_arithmetic_disagreement_count": arithmetic_disagreement_count,
        "owner_change_count": changed_owner_count,
        "changed_tied_probe_farthest_either_count": changed_farthest_either_count,
        "changed_tied_probe_unique_farthest_either_count": (
            changed_unique_farthest_either_count
        ),
        "site_radius_change_count": int(np.sum(radius_changed)),
        "maximum_assigned_radius_absolute_difference": float(
            np.max(np.abs(current_radius - proposed_radius))
        ),
        "cell_size_change_count": int(np.sum(size_changed)),
        "maximum_cell_size_absolute_difference": float(
            np.max(np.abs(current_sizes - proposed_sizes))
        ),
        "current_group_count": len(current_groups),
        "proposed_group_count": len(proposed_groups),
        "ordered_group_index_assignment_change_count": group_membership_changes,
        "group_composition_changed_cell_count": sum(
            frozenset(current_groups[int(current_trace["membership"][index])])
            != frozenset(proposed_groups[int(proposed_trace["membership"][index])])
            for index in range(history_count)
        ),
        "same_size_group_set_symmetric_difference_count": len(
            set(frozenset(group) for group in current_groups)
            ^ set(frozenset(group) for group in proposed_groups)
        ),
        "same_size_groups_exact": groups_exact,
        "maximum_group_anchor_size_absolute_difference": group_anchor_max_difference,
        "representative_ordered_position_change_count": (
            _ordered_position_change_count(
                current_representatives, proposed_representatives
            )
        ),
        "representative_symmetric_difference_count": (
            _symmetric_difference_count(
                current_representatives, proposed_representatives
            )
        ),
        "representatives_exact": representatives_exact,
        "lower_hull_ordered_position_change_count": (
            _ordered_position_change_count(current_hull_sites, proposed_hull_sites)
        ),
        "lower_hull_symmetric_difference_count": (
            _symmetric_difference_count(current_hull_sites, proposed_hull_sites)
        ),
        "lower_hull_exact": hull_exact,
        "current_lower_hull_site_indices": _fmt_indices(current_hull_sites),
        "proposed_lower_hull_site_indices": _fmt_indices(proposed_hull_sites),
        "current_improvement_filtered_po_count": len(current_po),
        "proposed_improvement_filtered_po_count": len(proposed_po),
        "po_ordered_position_change_count": _ordered_position_change_count(
            current_po, proposed_po
        ),
        "po_symmetric_difference_count": _symmetric_difference_count(
            current_po, proposed_po
        ),
        "improvement_filtered_po_set_exact": po_exact,
        "current_improvement_filtered_po_indices": _fmt_indices(current_po),
        "proposed_improvement_filtered_po_indices": _fmt_indices(proposed_po),
    }
    return {"state_row": state_row, "tie_rows": tie_rows}


def _analyze_trajectory_states(
    arguments: tuple[list[tuple[int, dict[str, Any], str]], dict[str, Any]]
) -> list[dict[str, Any]]:
    records, trajectory = arguments
    return [
        _analyze_state(source_index, record, trajectory, split)
        for source_index, record, split in records
    ]


def _build_results(
    payload: dict[str, Any], indexed_records: list[tuple[int, str]], workers: int
) -> list[dict[str, Any]]:
    trajectories = ordinary_data._trajectory_map(payload)
    grouped: dict[int, list[tuple[int, dict[str, Any], str]]] = {}
    for source_index, split in indexed_records:
        record = payload["states"][source_index]
        grouped.setdefault(int(record["trajectory_id"]), []).append(
            (source_index, record, split)
        )
    tasks = [
        (records, trajectories[trajectory_id])
        for trajectory_id, records in sorted(grouped.items())
    ]
    built: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            built.extend(_analyze_trajectory_states(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_analyze_trajectory_states, task) for task in tasks
            ]
            for future in as_completed(futures):
                batch = future.result()
                built.extend(batch)
                if len(built) % 40 < len(batch) or len(built) == len(indexed_records):
                    print(
                        f"Voronoi tie audit states {len(built)}/"
                        f"{len(indexed_records)}",
                        flush=True,
                    )
    by_index = {
        int(result["state_row"]["source_state_index"]): result for result in built
    }
    return [by_index[index] for index, _ in indexed_records]


def _stage_row(
    stage: str,
    changed_item_count: int,
    changed_state_count: int,
    maximum_absolute_difference: float | str = "",
    symmetric_difference_count: int | str = "",
) -> dict[str, Any]:
    return {
        "record_type": "stage_summary",
        "stage": stage,
        "changed_item_count": changed_item_count,
        "changed_state_count": changed_state_count,
        "maximum_absolute_difference": maximum_absolute_difference,
        "symmetric_difference_count": symmetric_difference_count,
    }


def _aggregate(results: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    state_rows = [result["state_row"] for result in results]
    tie_rows = [row for result in results for row in result["tie_rows"]]
    changed_tie_rows = [row for row in tie_rows if bool(row["owner_changed"])]

    def total(key: str) -> int:
        return sum(int(row[key]) for row in state_rows)

    def changed_states(key: str) -> int:
        return sum(int(row[key]) > 0 for row in state_rows)

    def nonexact_states(key: str) -> int:
        return sum(not bool(row[key]) for row in state_rows)

    def max_value(key: str) -> float:
        return max(float(row[key]) for row in state_rows)

    tie_summary = {
        "record_type": "summary",
        "state_count": len(state_rows),
        "evaluated_site_count": total("site_count"),
        "raw_probe_count": total("raw_probe_count"),
        "exact_tie_candidate_screen_distance": EXACT_TIE_CANDIDATE_SCREEN,
        "exact_tie_candidate_screen_probe_count": total(
            "exact_tie_candidate_screen_count"
        ),
        "ckdtree_exact_tied_probe_count": total("ckdtree_exact_tie_probe_count"),
        "direct_arithmetic_exact_tied_probe_count": total(
            "direct_exact_tie_probe_count"
        ),
        "tie_arithmetic_disagreement_count": total(
            "tie_arithmetic_disagreement_count"
        ),
        "owner_changed_probe_count": len(changed_tie_rows),
        "owner_changes_on_direct_exact_ties": sum(
            bool(row["direct_arithmetic_confirms_exact_tie"])
            for row in changed_tie_rows
        ),
        "owner_changes_on_ckdtree_only_ties": sum(
            bool(row["ckdtree_query_reports_exact_tie"])
            and not bool(row["direct_arithmetic_confirms_exact_tie"])
            for row in changed_tie_rows
        ),
        "owner_unchanged_audited_tie_probe_count": (
            len(tie_rows) - len(changed_tie_rows)
        ),
        "changed_tied_probe_farthest_either_count": sum(
            bool(row["probe_farthest_for_either_affected_site"])
            for row in changed_tie_rows
        ),
        "changed_tied_probe_unique_farthest_either_count": sum(
            bool(row["probe_unique_farthest_for_either_affected_site"])
            for row in changed_tie_rows
        ),
        "changed_tied_probe_directly_changed_affected_radius_count": sum(
            bool(row["old_owner_radius_changed"])
            or bool(row["new_owner_radius_changed"])
            for row in changed_tie_rows
        ),
        "changed_tied_probe_directly_changed_affected_cell_size_count": sum(
            bool(row["old_owner_cell_size_changed"])
            or bool(row["new_owner_cell_size_changed"])
            for row in changed_tie_rows
        ),
    }
    tie_change_rows: list[dict[str, Any]] = [tie_summary, *changed_tie_rows]

    radius_changes = total("site_radius_change_count")
    size_changes = total("cell_size_change_count")
    ordered_group_index_changes = total(
        "ordered_group_index_assignment_change_count"
    )
    group_composition_changed_cells = total(
        "group_composition_changed_cell_count"
    )
    group_set_symmetric_difference = total(
        "same_size_group_set_symmetric_difference_count"
    )
    representative_position_changes = total(
        "representative_ordered_position_change_count"
    )
    hull_position_changes = total("lower_hull_ordered_position_change_count")
    po_position_changes = total("po_ordered_position_change_count")
    downstream_summary = {
        "record_type": "summary",
        **{key: value for key, value in tie_summary.items() if key != "record_type"},
        "site_radius_change_count": radius_changes,
        "maximum_assigned_radius_absolute_difference": max_value(
            "maximum_assigned_radius_absolute_difference"
        ),
        "cell_size_change_count": size_changes,
        "maximum_cell_size_absolute_difference": max_value(
            "maximum_cell_size_absolute_difference"
        ),
        "ordered_group_index_assignment_change_count": ordered_group_index_changes,
        "group_composition_changed_cell_count": group_composition_changed_cells,
        "same_size_group_set_symmetric_difference_count": (
            group_set_symmetric_difference
        ),
        "same_size_group_changed_state_count": nonexact_states(
            "same_size_groups_exact"
        ),
        "representative_position_change_count": representative_position_changes,
        "representative_symmetric_difference_count": total(
            "representative_symmetric_difference_count"
        ),
        "representative_changed_state_count": nonexact_states(
            "representatives_exact"
        ),
        "lower_hull_position_change_count": hull_position_changes,
        "lower_hull_symmetric_difference_count": total(
            "lower_hull_symmetric_difference_count"
        ),
        "lower_hull_changed_state_count": nonexact_states("lower_hull_exact"),
        "improvement_filtered_po_position_change_count": po_position_changes,
        "improvement_filtered_po_symmetric_difference_count": total(
            "po_symmetric_difference_count"
        ),
        "improvement_filtered_po_changed_state_count": nonexact_states(
            "improvement_filtered_po_set_exact"
        ),
        "improvement_filtered_po_changed_source_states": ";".join(
            str(int(row["source_state_index"]))
            for row in state_rows
            if not bool(row["improvement_filtered_po_set_exact"])
        ),
        "improvement_filtered_po_changed_trajectory_ids": ";".join(
            str(int(row["trajectory_id"]))
            for row in state_rows
            if not bool(row["improvement_filtered_po_set_exact"])
        ),
    }
    stage_rows = [
        _stage_row(
            "probe_ownership",
            len(changed_tie_rows),
            changed_states("owner_change_count"),
        ),
        _stage_row(
            "max_assigned_probe_radius",
            radius_changes,
            changed_states("site_radius_change_count"),
            downstream_summary["maximum_assigned_radius_absolute_difference"],
        ),
        _stage_row(
            "final_cell_size",
            size_changes,
            changed_states("cell_size_change_count"),
            downstream_summary["maximum_cell_size_absolute_difference"],
        ),
        _stage_row(
            "same_size_groups",
            group_composition_changed_cells,
            downstream_summary["same_size_group_changed_state_count"],
            max(
                float(row["maximum_group_anchor_size_absolute_difference"])
                for row in state_rows
                if math.isfinite(
                    float(row["maximum_group_anchor_size_absolute_difference"])
                )
            ),
        ),
        _stage_row(
            "representatives",
            representative_position_changes,
            downstream_summary["representative_changed_state_count"],
            "",
            downstream_summary["representative_symmetric_difference_count"],
        ),
        _stage_row(
            "lower_hull",
            hull_position_changes,
            downstream_summary["lower_hull_changed_state_count"],
            "",
            downstream_summary["lower_hull_symmetric_difference_count"],
        ),
        _stage_row(
            "improvement_filtered_po_set",
            po_position_changes,
            downstream_summary["improvement_filtered_po_changed_state_count"],
            "",
            downstream_summary[
                "improvement_filtered_po_symmetric_difference_count"
            ],
        ),
    ]
    return tie_change_rows, [downstream_summary, *stage_rows, *state_rows], downstream_summary


def _audit_text(summary: dict[str, Any]) -> str:
    downstream_unchanged = bool(
        int(summary["cell_size_change_count"]) == 0
        and int(summary["group_composition_changed_cell_count"]) == 0
        and int(summary["representative_position_change_count"]) == 0
        and int(summary["lower_hull_position_change_count"]) == 0
        and int(summary["improvement_filtered_po_position_change_count"]) == 0
    )
    conclusion = (
        "Yes. The explicit exact-tie -> lowest-site-index rule changes ownership "
        "but leaves every final cell size and every audited downstream result unchanged."
        if downstream_unchanged
        else "No. The counterfactual tie rule changes at least one meaningful downstream result."
    )
    return f"""Voronoi exact-tie ownership audit
=====================================

Scope and rule
--------------
The same 320 existing teacher states were replayed. No training, checkpoint
write, or trajectory search was performed. Current ownership is the site index
returned by `cKDTree.query(raw_probes, k=1)`. The counterfactual owner is the
smallest original evaluated-site index among sites whose directly computed
two-term squared Euclidean distance equals the exact minimum. No distance
tolerance is used. Distances themselves are unchanged; only exact-tie ownership
is replaced.

To avoid an all-sites calculation for 6.86 million clearly separated probes,
direct equality was evaluated for the conservative candidate set whose KD-tree
nearest/second-nearest gap is <= {EXACT_TIE_CANDIDATE_SCREEN:.1e}. This screen
is not the tie rule: a candidate is tied only when directly computed squared
distances are exactly equal. The prior audit found no gaps between 1e-15 and
1e-10, so this covers the complete observed near-tie population with two
orders of magnitude to spare.

Tie results
-----------
- States: {summary['state_count']}
- Evaluated sites: {summary['evaluated_site_count']}
- Raw probes: {summary['raw_probe_count']}
- cKDTree exact-distance tied probes: {summary['ckdtree_exact_tied_probe_count']}
- Direct-arithmetic exact tied probes: {summary['direct_arithmetic_exact_tied_probe_count']}
- KD-tree/direct exact-tie arithmetic disagreements: {summary['tie_arithmetic_disagreement_count']}
- Tied probes whose owner changes: {summary['owner_changed_probe_count']}
- Owner changes on direct-arithmetic exact ties: {summary['owner_changes_on_direct_exact_ties']}
- Owner changes on KD-tree-only equality cases: {summary['owner_changes_on_ckdtree_only_ties']}
- Audited tied probes already owned by their proposed owner: {summary['owner_unchanged_audited_tie_probe_count']}

Farthest-probe relevance
------------------------
- Changed tied probes that are a farthest assigned probe for the old or new
  affected site: {summary['changed_tied_probe_farthest_either_count']}
- Changed tied probes that are the unique farthest probe for at least one
  affected site: {summary['changed_tied_probe_unique_farthest_either_count']}
- Changed tied probes associated with an actual affected-site radius change:
  {summary['changed_tied_probe_directly_changed_affected_radius_count']}
- Changed tied probes associated with an actual affected-site final d_i change:
  {summary['changed_tied_probe_directly_changed_affected_cell_size_count']}

Downstream propagation
----------------------
- Site radii changed: {summary['site_radius_change_count']}; maximum absolute
  radius difference: {summary['maximum_assigned_radius_absolute_difference']:.17g}
- Final d_i values changed: {summary['cell_size_change_count']}; maximum absolute
  d_i difference: {summary['maximum_cell_size_absolute_difference']:.17g}
- Cells whose actual same-size group composition changes:
  {summary['group_composition_changed_cell_count']}; group-set symmetric
  difference: {summary['same_size_group_set_symmetric_difference_count']};
  ordered group-index shifts: {summary['ordered_group_index_assignment_change_count']}
  across {summary['same_size_group_changed_state_count']} states
- Representative positional changes: {summary['representative_position_change_count']};
  representative symmetric difference: {summary['representative_symmetric_difference_count']}
- Lower-hull positional changes: {summary['lower_hull_position_change_count']};
  lower-hull symmetric difference: {summary['lower_hull_symmetric_difference_count']}
- Improvement-filtered PO positional changes:
  {summary['improvement_filtered_po_position_change_count']}; PO symmetric
  difference: {summary['improvement_filtered_po_symmetric_difference_count']};
  changed source state(s): {summary['improvement_filtered_po_changed_source_states']}
  (trajectory {summary['improvement_filtered_po_changed_trajectory_ids']})

Conclusion
----------
{conclusion}

The production teacher implementation was not changed.
"""


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    source_train = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED
    )
    source_validation = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    indexed = [(index, "training") for index in source_train]
    indexed.extend((index, "held_out") for index in source_validation)
    results = _build_results(payload, indexed, workers)
    tie_rows, downstream_rows, summary = _aggregate(results)
    _write_csv(CHANGES_PATH, tie_rows)
    _write_csv(DOWNSTREAM_PATH, downstream_rows)
    AUDIT_PATH.write_text(_audit_text(summary), encoding="utf-8")
    print(
        "Voronoi tie-rule audit complete: "
        f"ties={summary['ckdtree_exact_tied_probe_count']} "
        f"owner_changes={summary['owner_changed_probe_count']} "
        f"radius_changes={summary['site_radius_change_count']} "
        f"size_changes={summary['cell_size_change_count']} "
        f"po_changes={summary['improvement_filtered_po_position_change_count']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    run(args.workers)


if __name__ == "__main__":
    main()
