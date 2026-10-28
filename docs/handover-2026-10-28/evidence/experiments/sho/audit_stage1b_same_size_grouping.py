#!/usr/bin/env python3
"""Audit eDIRECT same-size grouping and representative selection."""

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

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/same_size_grouping_audit")
AUDIT_PATH = OUTPUT_DIR / "same_size_grouping_audit.txt"
DISTRIBUTION_PATH = OUTPUT_DIR / "same_size_grouping_distribution.csv"
EXAMPLES_PATH = OUTPUT_DIR / "representative_selection_examples.csv"

SIZE_TOLERANCE = 1.0e-14
DIAGNOSTIC_NEAR_LOSS_TOLERANCE = 1.0e-14
BOUNDARY_EXAMPLE_COUNT = 10
REPRESENTATIVE_EXAMPLE_COUNT = 20


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


def _raw_state(
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
    pool = edirect._candidate_pool(
        history_count, int(record["generator_id"]), int(record["iteration"])
    )
    tree = cKDTree(points)
    distances, owners = tree.query(pool, k=1)
    sizes = np.zeros(history_count, dtype=np.float64)
    np.maximum.at(sizes, owners, distances)
    if history_count > 1:
        nearest = tree.query(points, k=2)[0][:, 1]
        sizes = np.maximum(sizes, 0.5 * nearest)
    if not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(losses)):
        raise RuntimeError(f"nonfinite size/loss in state {source_state_index}")
    return {
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": int(record["iteration"]),
        "history_count": history_count,
        "split": split,
        "sizes": sizes,
        "losses": losses,
    }


def _build_trajectory_states(
    arguments: tuple[list[tuple[int, dict[str, Any], str]], dict[str, Any]]
) -> list[dict[str, Any]]:
    records, trajectory = arguments
    return [
        _raw_state(source_index, record, trajectory, split)
        for source_index, record, split in records
    ]


def _build_states(
    payload: dict[str, Any],
    indexed_records: list[tuple[int, str]],
    workers: int,
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
            built.extend(_build_trajectory_states(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_trajectory_states, task) for task in tasks]
            for future in as_completed(futures):
                batch = future.result()
                built.extend(batch)
                if len(built) % 40 < len(batch) or len(built) == len(indexed_records):
                    print(
                        f"same-size audit states {len(built)}/{len(indexed_records)}",
                        flush=True,
                    )
    by_index = {int(state["source_state_index"]): state for state in built}
    return [by_index[index] for index, _ in indexed_records]


def _teacher_trace(sizes: np.ndarray, losses: np.ndarray) -> dict[str, Any]:
    """Literal trace of the production grouping loop."""
    order = np.lexsort((np.arange(len(sizes)), losses, sizes))
    groups: list[dict[str, Any]] = []
    membership = np.full(len(sizes), -1, dtype=np.int64)
    decisions: list[dict[str, Any]] = []
    for order_position, index_value in enumerate(order):
        index = int(index_value)
        size = float(sizes[index])
        loss = float(losses[index])
        if not groups:
            groups.append(
                {
                    "anchor_size": size,
                    "representative_loss": loss,
                    "representative_index": index,
                    "members": [index],
                    "member_order_positions": [order_position],
                }
            )
            membership[index] = 0
            decisions.append(
                {
                    "index": index,
                    "order_position": order_position,
                    "prior_group": -1,
                    "prior_anchor": math.nan,
                    "comparison_gap": math.nan,
                    "same_group": False,
                    "started_new_group": True,
                    "replaced_representative": False,
                }
            )
            continue
        group = groups[-1]
        gap = abs(size - float(group["anchor_size"]))
        if gap > SIZE_TOLERANCE:
            groups.append(
                {
                    "anchor_size": size,
                    "representative_loss": loss,
                    "representative_index": index,
                    "members": [index],
                    "member_order_positions": [order_position],
                }
            )
            membership[index] = len(groups) - 1
            decisions.append(
                {
                    "index": index,
                    "order_position": order_position,
                    "prior_group": len(groups) - 2,
                    "prior_anchor": float(group["anchor_size"]),
                    "comparison_gap": gap,
                    "same_group": False,
                    "started_new_group": True,
                    "replaced_representative": False,
                }
            )
        else:
            group_index = len(groups) - 1
            replaced = loss < float(group["representative_loss"])
            group["members"].append(index)
            group["member_order_positions"].append(order_position)
            if replaced:
                group["representative_loss"] = loss
                group["representative_index"] = index
            membership[index] = group_index
            decisions.append(
                {
                    "index": index,
                    "order_position": order_position,
                    "prior_group": group_index,
                    "prior_anchor": float(group["anchor_size"]),
                    "comparison_gap": gap,
                    "same_group": True,
                    "started_new_group": False,
                    "replaced_representative": replaced,
                }
            )
    return {
        "order": order.astype(np.int64),
        "groups": groups,
        "membership": membership,
        "decisions": decisions,
        "grouped_sizes": np.asarray(
            [group["anchor_size"] for group in groups], dtype=np.float64
        ),
        "grouped_losses": np.asarray(
            [group["representative_loss"] for group in groups], dtype=np.float64
        ),
        "representatives": np.asarray(
            [group["representative_index"] for group in groups], dtype=np.int64
        ),
    }


def _independent_reconstruction(
    sizes: np.ndarray, losses: np.ndarray
) -> dict[str, np.ndarray]:
    """Independent group-slice reconstruction from anchor boundaries."""
    order = np.asarray(
        sorted(
            range(len(sizes)),
            key=lambda index: (float(sizes[index]), float(losses[index]), index),
        ),
        dtype=np.int64,
    )
    group_slices: list[list[int]] = []
    current: list[int] = []
    anchor = math.nan
    for index_value in order:
        index = int(index_value)
        size = float(sizes[index])
        if not current or abs(size - anchor) > SIZE_TOLERANCE:
            if current:
                group_slices.append(current)
            current = [index]
            anchor = size
        else:
            current.append(index)
    if current:
        group_slices.append(current)

    membership = np.full(len(sizes), -1, dtype=np.int64)
    order_position = np.empty(len(order), dtype=np.int64)
    order_position[order] = np.arange(len(order), dtype=np.int64)
    representatives: list[int] = []
    grouped_sizes: list[float] = []
    grouped_losses: list[float] = []
    for group_index, members in enumerate(group_slices):
        for index in members:
            membership[index] = group_index
        representative = min(
            members,
            key=lambda index: (
                float(losses[index]),
                int(order_position[index]),
            ),
        )
        representatives.append(representative)
        grouped_sizes.append(float(sizes[members[0]]))
        grouped_losses.append(float(losses[representative]))
    return {
        "order": order,
        "membership": membership,
        "representatives": np.asarray(representatives, dtype=np.int64),
        "grouped_sizes": np.asarray(grouped_sizes, dtype=np.float64),
        "grouped_losses": np.asarray(grouped_losses, dtype=np.float64),
    }


def _percentile_rows(
    quantity: str, values: np.ndarray, split: str = "all"
) -> list[dict[str, Any]]:
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
                "split": split,
                "quantity": quantity,
                "statistic": name,
                "value": float(np.percentile(values, percentile)),
                "sample_count": len(values),
            }
        )
    return rows


def _fmt(values: list[int] | np.ndarray) -> str:
    return ";".join(str(int(value)) for value in values)


def _analyze(states: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    traces: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    assigned_gaps: list[float] = []
    comparison_gaps_inside: list[float] = []
    comparison_gaps_outside: list[float] = []
    comparison_records_inside: list[dict[str, Any]] = []
    comparison_records_outside: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    loss_gaps: list[float] = []
    relative_loss_gaps: list[float] = []
    groups_exact_loss_tie = 0
    groups_near_abs_loss_tie = 0
    groups_near_scaled_loss_tie = 0
    replacement_count = 0
    all_groups: list[dict[str, Any]] = []

    for state_position, state in enumerate(states):
        sizes = state["sizes"]
        losses = state["losses"]
        trace = _teacher_trace(sizes, losses)
        independent = _independent_reconstruction(sizes, losses)
        reference_sizes, reference_losses, reference_representatives = (
            turn._same_size_representatives(sizes, losses)
        )
        membership_exact = np.array_equal(
            trace["membership"], independent["membership"]
        )
        representatives_exact = np.array_equal(
            trace["representatives"], independent["representatives"]
        )
        ordered_exact = bool(
            np.array_equal(trace["order"], independent["order"])
            and np.array_equal(trace["grouped_sizes"], independent["grouped_sizes"])
            and np.array_equal(trace["grouped_losses"], independent["grouped_losses"])
            and np.array_equal(trace["representatives"], reference_representatives)
            and np.array_equal(trace["grouped_sizes"], reference_sizes)
            and np.array_equal(trace["grouped_losses"], reference_losses)
        )
        state_rows.append(
            {
                "record_type": "state_reconstruction",
                "split": state["split"],
                "state_position": state_position,
                "source_state_index": state["source_state_index"],
                "cell_count": len(sizes),
                "group_count": len(trace["groups"]),
                "group_membership_exact": membership_exact,
                "representatives_exact": representatives_exact,
                "final_ordered_representative_list_exact": ordered_exact,
            }
        )
        if not membership_exact or not representatives_exact or not ordered_exact:
            raise RuntimeError(
                f"same-size reconstruction mismatch in state {state['source_state_index']}"
            )
        for decision in trace["decisions"]:
            if math.isfinite(float(decision["comparison_gap"])):
                record = {
                    "split": state["split"],
                    "state_position": state_position,
                    "source_state_index": state["source_state_index"],
                    "cell_index": decision["index"],
                    "comparison_gap": float(decision["comparison_gap"]),
                    "distance_from_boundary": abs(
                        float(decision["comparison_gap"]) - SIZE_TOLERANCE
                    ),
                    "same_group": bool(decision["same_group"]),
                    "started_new_group": bool(decision["started_new_group"]),
                    "group_anchor_size_before_decision": decision["prior_anchor"],
                    "cell_size": float(sizes[int(decision["index"])]),
                }
                if decision["same_group"]:
                    comparison_gaps_inside.append(record["comparison_gap"])
                    comparison_records_inside.append(record)
                else:
                    comparison_gaps_outside.append(record["comparison_gap"])
                    comparison_records_outside.append(record)
            replacement_count += int(decision["replaced_representative"])

        for group_index, group in enumerate(trace["groups"]):
            members = [int(value) for value in group["members"]]
            anchor = float(group["anchor_size"])
            representative = int(group["representative_index"])
            member_losses = np.asarray([losses[index] for index in members])
            member_sizes = np.asarray([sizes[index] for index in members])
            member_gaps = np.abs(member_sizes - anchor)
            assigned_gaps.extend(float(value) for value in member_gaps)
            group_sizes.append(len(members))
            sorted_losses = np.sort(member_losses)
            winner_loss = float(losses[representative])
            if len(members) > 1:
                runner_loss = float(sorted_losses[1])
                loss_gap = runner_loss - winner_loss
                scale = max(1.0, abs(winner_loss), abs(runner_loss))
                loss_gaps.append(loss_gap)
                relative_loss_gaps.append(loss_gap / scale)
                exact_tie = bool(loss_gap == 0.0)
                near_abs = bool(loss_gap <= DIAGNOSTIC_NEAR_LOSS_TOLERANCE)
                near_scaled = bool(
                    loss_gap <= DIAGNOSTIC_NEAR_LOSS_TOLERANCE * scale
                )
                groups_exact_loss_tie += int(exact_tie)
                groups_near_abs_loss_tie += int(near_abs)
                groups_near_scaled_loss_tie += int(near_scaled)
            else:
                runner_loss = math.nan
                loss_gap = math.nan
                exact_tie = False
                near_abs = False
                near_scaled = False
            group_record = {
                "record_type": "group_example",
                "split": state["split"],
                "state_position": state_position,
                "source_state_index": state["source_state_index"],
                "group_index": group_index,
                "group_size": len(members),
                "group_anchor_size": anchor,
                "representative_index": representative,
                "representative_actual_size": float(sizes[representative]),
                "representative_loss": winner_loss,
                "representative_differs_from_group_opener": representative != members[0],
                "runner_up_loss": runner_loss,
                "winner_runner_up_loss_gap": loss_gap,
                "exact_winner_loss_tie": exact_tie,
                "diagnostic_near_abs_loss_tie_1e-14": near_abs,
                "diagnostic_near_scaled_loss_tie_1e-14": near_scaled,
                "member_indices_in_sorted_order": _fmt(members),
                "member_sizes_in_sorted_order": ";".join(
                    f"{float(sizes[index]):.17g}" for index in members
                ),
                "member_losses_in_sorted_order": ";".join(
                    f"{float(losses[index]):.17g}" for index in members
                ),
            }
            all_groups.append(group_record)
        traces.append(trace)

    state_count = len(states)
    total_cells = sum(len(state["sizes"]) for state in states)
    total_groups = len(group_sizes)
    singleton_groups = int(sum(size == 1 for size in group_sizes))
    multi_groups = total_groups - singleton_groups
    exact_states = sum(
        bool(row["group_membership_exact"])
        and bool(row["representatives_exact"])
        and bool(row["final_ordered_representative_list_exact"])
        for row in state_rows
    )

    distribution_rows: list[dict[str, Any]] = [
        {
            "record_type": "overall_summary",
            "state_count": state_count,
            "training_state_count": sum(state["split"] == "training" for state in states),
            "held_out_state_count": sum(state["split"] == "held_out" for state in states),
            "total_cells_entering_stage": total_cells,
            "total_same_size_groups": total_groups,
            "singleton_group_count": singleton_groups,
            "multi_cell_group_count": multi_groups,
            "largest_group_size": max(group_sizes),
            "representative_replacement_count": replacement_count,
            "exact_boundary_case_count": int(
                sum(gap == SIZE_TOLERANCE for gap in comparison_gaps_inside)
                + sum(gap == SIZE_TOLERANCE for gap in comparison_gaps_outside)
            ),
            "multi_groups_with_exact_winner_loss_tie": groups_exact_loss_tie,
            "multi_groups_with_abs_loss_gap_le_1e-14": groups_near_abs_loss_tie,
            "multi_groups_with_scaled_loss_gap_le_1e-14": groups_near_scaled_loss_tie,
            "state_level_exact_reconstruction": exact_states,
            "state_level_total": state_count,
        }
    ]
    distribution_rows.extend(state_rows)
    distribution_rows.extend(
        _percentile_rows("group_size", np.asarray(group_sizes, dtype=np.float64))
    )
    distribution_rows.extend(
        _percentile_rows(
            "abs_cell_size_minus_assigned_group_anchor",
            np.asarray(assigned_gaps, dtype=np.float64),
        )
    )
    distribution_rows.extend(
        _percentile_rows(
            "same_group_comparison_gap",
            np.asarray(comparison_gaps_inside, dtype=np.float64),
        )
    )
    distribution_rows.extend(
        _percentile_rows(
            "new_group_comparison_gap",
            np.asarray(comparison_gaps_outside, dtype=np.float64),
        )
    )
    distribution_rows.extend(
        _percentile_rows(
            "winner_runner_up_loss_gap",
            np.asarray(loss_gaps, dtype=np.float64),
        )
    )
    distribution_rows.extend(
        _percentile_rows(
            "winner_runner_up_scaled_loss_gap",
            np.asarray(relative_loss_gaps, dtype=np.float64),
        )
    )
    for label, lower, upper in (
        ("1", 1, 1),
        ("2", 2, 2),
        ("3_to_4", 3, 4),
        ("5_to_8", 5, 8),
        ("9_to_16", 9, 16),
        ("17_to_32", 17, 32),
        ("33_plus", 33, math.inf),
    ):
        distribution_rows.append(
            {
                "record_type": "group_size_bucket",
                "group_size_bucket": label,
                "group_count": int(sum(lower <= size <= upper for size in group_sizes)),
            }
        )
    closest_inside = sorted(
        comparison_records_inside, key=lambda row: row["distance_from_boundary"]
    )[:BOUNDARY_EXAMPLE_COUNT]
    closest_outside = sorted(
        comparison_records_outside, key=lambda row: row["distance_from_boundary"]
    )[:BOUNDARY_EXAMPLE_COUNT]
    for side, records in (("same_side", closest_inside), ("new_group_side", closest_outside)):
        for rank, record in enumerate(records, start=1):
            distribution_rows.append(
                {
                    "record_type": "closest_boundary_example",
                    "boundary_side": side,
                    "rank": rank,
                    "tolerance": SIZE_TOLERANCE,
                    **record,
                }
            )

    selected_examples: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()

    def add_examples(category: str, records: list[dict[str, Any]]) -> None:
        for record in records[:REPRESENTATIVE_EXAMPLE_COUNT]:
            key = (int(record["source_state_index"]), int(record["group_index"]))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected_examples.append({"selection_category": category, **record})

    multi = [row for row in all_groups if int(row["group_size"]) > 1]
    add_examples(
        "largest_groups",
        sorted(multi, key=lambda row: (-int(row["group_size"]), row["source_state_index"])),
    )
    add_examples(
        "smallest_winner_runner_up_gap",
        sorted(multi, key=lambda row: float(row["winner_runner_up_loss_gap"])),
    )
    add_examples(
        "representative_replaced_group_opener",
        [row for row in multi if row["representative_differs_from_group_opener"]],
    )
    add_examples(
        "exact_or_near_loss_ties",
        [row for row in multi if row["diagnostic_near_scaled_loss_tie_1e-14"]],
    )

    closest_inside_gap = (
        closest_inside[0]["comparison_gap"] if closest_inside else math.nan
    )
    closest_outside_gap = (
        closest_outside[0]["comparison_gap"] if closest_outside else math.nan
    )
    audit_text = f"""Same-size grouping and representative-selection audit
=====================================================

Scope
-----
This is a code/trace audit only. It uses the existing saved eDIRECT teacher trace and
reconstructs the same {turn.TRAIN_SOURCE_STATES} training plus
{turn.VALIDATION_SOURCE_STATES} held-out search states used by the hull experiments.
It performs no trajectory search, no training, and no checkpoint writes.

Exact implementation
--------------------
Production function:
`experiments/sho/run_stage1b_edirect_vs_direct.py::_potentially_optimal_cells`.

The exact sort is:

    order = np.lexsort((original_site_index, loss, size))

Because NumPy lexsort uses the last key as primary, this orders cells by increasing
`(size, loss, original_site_index)`.

The exact grouping loop is:

    for index in order:
        d = sizes[index]
        f = losses[index]
        if no current group or abs(d - current_group_anchor_size) > 1e-14:
            start group (anchor_size=d, representative_loss=f, representative=index)
        elif f < current_group_representative_loss:
            replace representative_loss and representative index

The tolerance is the absolute float64 constant `1e-14`; there is no relative scaling.
Equality (`abs(d-anchor)==1e-14`) remains in the current group. The comparison is
anchored to the size that opened the group, never the preceding cell and never the
current representative's actual size. Replacing a representative does not change the
anchor size. A new group starts only when the next sorted cell differs from that anchor
by strictly more than `1e-14`.

The underlying pairwise relation `abs(d_i-d_j)<=1e-14` is not transitive. The implemented
grouping is nevertheless deterministic because every membership decision is anchored
to the first cell of the current group. Thus chaining cannot extend a group: A and B may
be within tolerance and B and C within tolerance, but C starts a new group when C is
more than `1e-14` from A.

Representative selection and ties
----------------------------------
The lowest observed site loss within the anchored group wins. Replacement uses strict
`f < best_f`; there is no loss tolerance. An exactly equal loss never replaces the
current representative. Consequently a loss tie keeps the member encountered first in
the `(size, loss, original_site_index)` order: smaller size first, then (only for equal
size and loss) smaller original site index. Near-equal but unequal losses are treated by
their ordinary float64 ordering.

Coordinates, surrogate predictions, LOO error, candidate/Voronoi geometry beyond the
already computed scalar size, sampling density, and candidate points are not used.

Before the learned hull stage, groups stay in increasing anchor-size order. The hull
receives `(group_anchor_size, representative_loss, representative_site_index)`. If the
representative was replaced, its own actual size can differ slightly from the anchor;
the anchor is still what enters the hull geometry.

Exact learnable-operation decomposition (architecture intentionally unspecified)
-------------------------------------------------------------------------
1. Continuous grouping quantity: `delta_size = abs(d_i - d_group_anchor)`.
2. Discrete group decision: SAME iff `delta_size <= 1e-14`; otherwise NEW GROUP.
3. Within a group, continuous comparison quantity for challenger `i`:
   `delta_loss = f_i - f_current_representative`.
4. Discrete replacement: replace iff `delta_loss < 0`; equality retains the current
   representative.
5. Deterministic scan bookkeeping preserves the anchor, membership, and group order.

Data distribution
-----------------
States: {state_count}; cells entering stage: {total_cells}; groups: {total_groups}.
Singleton groups: {singleton_groups}; multi-cell groups: {multi_groups}; largest group:
{max(group_sizes)} cells. Representative replacements: {replacement_count}.

Closest real SAME-side comparison gap to the boundary: {closest_inside_gap:.17g}.
Closest real NEW-GROUP-side comparison gap: {closest_outside_gap:.17g}.
Exact `1e-14` boundary cases: {distribution_rows[0]['exact_boundary_case_count']}.

Among multi-cell groups, exact winner/runner-up loss ties: {groups_exact_loss_tie};
diagnostic absolute gaps <=1e-14: {groups_near_abs_loss_tie}; diagnostic scale-relative
gaps <=1e-14*max(1,|winner|,|runner-up|): {groups_near_scaled_loss_tie}. These near-tie
thresholds are audit summaries only; the teacher uses no loss tolerance.

Independent reconstruction
--------------------------
An independent implementation sorted with Python tuple ordering, formed anchor-based
group slices, then selected each representative by minimum loss with first-in-sorted-
order tie handling. It exactly matched the literal production trace for membership and
representatives, and matched the existing teacher preprocessing for the final ordered
representative list, on {exact_states}/{state_count} states.
"""
    return distribution_rows, selected_examples, audit_text


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
    indexed_records = [(index, "training") for index in train_indices]
    indexed_records.extend((index, "held_out") for index in held_out_indices)
    states = _build_states(payload, indexed_records, workers)
    distribution_rows, example_rows, audit_text = _analyze(states)
    _write_csv(DISTRIBUTION_PATH, distribution_rows)
    _write_csv(EXAMPLES_PATH, example_rows)
    AUDIT_PATH.write_text(audit_text, encoding="utf-8")
    summary = distribution_rows[0]
    print(
        "same-size grouping audit complete "
        f"states={summary['state_level_exact_reconstruction']}/{summary['state_level_total']} "
        f"cells={summary['total_cells_entering_stage']} "
        f"groups={summary['total_same_size_groups']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    run(args.workers)


if __name__ == "__main__":
    main()
