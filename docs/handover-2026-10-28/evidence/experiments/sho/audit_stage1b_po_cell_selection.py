#!/usr/bin/env python3
"""Audit eDIRECT's potentially-optimal Voronoi-cell selection only."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/po_cell_selection_audit")
AUDIT_PATH = OUTPUT_DIR / "po_cell_selection_audit.md"
TRACE_PATH = OUTPUT_DIR / "po_cell_selection_trace.csv"
RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH
STATE_COUNT = 15
DETAIL_COUNT = 3
SIZE_GROUP_TOLERANCE = 1.0e-14
HULL_TOLERANCE = 5.0e-14


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


def _representative_indices(states: list[dict[str, Any]]) -> tuple[list[int], set[int]]:
    cell_indices = [
        index for index, state in enumerate(states) if state["candidate_type"] == "cell"
    ]
    ordered = sorted(
        cell_indices,
        key=lambda index: (
            int(states[index]["history_count"]),
            len(states[index]["teacher_curated_points"]),
            int(states[index]["trajectory_id"]),
            index,
        ),
    )
    small = [index for index in ordered if int(states[index]["history_count"]) <= 60]
    detail_positions = np.rint(
        np.linspace(0, len(small) - 1, DETAIL_COUNT)
    ).astype(int)
    detailed = {small[int(position)] for position in detail_positions}

    larger = [index for index in ordered if index not in detailed]
    positions = np.rint(
        np.linspace(0, len(larger) - 1, STATE_COUNT - DETAIL_COUNT)
    ).astype(int)
    chosen = list(detailed) + [larger[int(position)] for position in positions]
    chosen = sorted(
        set(chosen),
        key=lambda index: (
            int(states[index]["history_count"]),
            len(states[index]["teacher_curated_points"]),
            int(states[index]["trajectory_id"]),
            index,
        ),
    )
    if len(chosen) != STATE_COUNT:
        for index in ordered:
            if index not in chosen:
                chosen.append(index)
            if len(chosen) == STATE_COUNT:
                break
        chosen.sort(key=lambda index: int(states[index]["history_count"]))
    return chosen, detailed


def _reconstruct_sizes(
    points: np.ndarray, generator_id: int, iteration: int
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    pool = edirect._candidate_pool(len(points), generator_id, iteration)
    tree = cKDTree(points)
    distances, owners = tree.query(pool, k=1)
    assigned_radius = np.zeros(len(points), dtype=np.float64)
    np.maximum.at(assigned_radius, owners, distances)
    if len(points) > 1:
        nearest_neighbor = tree.query(points, k=2)[0][:, 1]
        spacing_floor = 0.5 * nearest_neighbor
    else:
        spacing_floor = np.zeros(1, dtype=np.float64)
    sizes = np.maximum(assigned_radius, spacing_floor)
    return pool, owners, distances, assigned_radius, spacing_floor, sizes


def _standalone_po_selection(
    sizes: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Independent transcription of the audited rule, with trace metadata."""

    sample_count = len(sizes)
    order = np.lexsort((np.arange(sample_count), losses, sizes))
    groups: list[dict[str, Any]] = []
    cell_group = np.full(sample_count, -1, dtype=np.int64)
    for index_value in order:
        index = int(index_value)
        size = float(sizes[index])
        loss = float(losses[index])
        if not groups or abs(size - float(groups[-1]["size"])) > SIZE_GROUP_TOLERANCE:
            groups.append({"size": size, "loss": loss, "sample_index": index})
        elif loss < float(groups[-1]["loss"]):
            groups[-1]["loss"] = loss
            groups[-1]["sample_index"] = index
        cell_group[index] = len(groups) - 1

    hull: list[int] = []
    popped_groups: set[int] = set()
    for group_index, group in enumerate(groups):
        size = float(group["size"])
        loss = float(group["loss"])
        while len(hull) >= 2:
            first = groups[hull[-2]]
            second = groups[hull[-1]]
            cross = (float(second["size"]) - float(first["size"])) * (
                loss - float(first["loss"])
            ) - (float(second["loss"]) - float(first["loss"])) * (
                size - float(first["size"])
            )
            scale = max(
                1.0,
                abs(float(first["loss"])),
                abs(float(second["loss"])),
                abs(loss),
            )
            if cross < -HULL_TOLERANCE * scale:
                popped_groups.add(hull.pop())
            else:
                break
        hull.append(group_index)

    minimum_loss = float(np.min(losses))
    target = minimum_loss - edirect.EDIRECT_EPSILON * abs(minimum_loss)
    selected_unsorted: list[int] = []
    for position, group_index in enumerate(hull):
        group = groups[group_index]
        size = float(group["size"])
        loss = float(group["loss"])
        lower_slope = -math.inf
        upper_slope = math.inf
        if position > 0:
            previous = groups[hull[position - 1]]
            lower_slope = (loss - float(previous["loss"])) / (
                size - float(previous["size"])
            )
        if position < len(hull) - 1:
            following = groups[hull[position + 1]]
            upper_slope = (float(following["loss"]) - loss) / (
                float(following["size"]) - size
            )
        required = (loss - target) / size if size > 0.0 else math.inf
        feasible_lower = max(lower_slope, required, 0.0)
        epsilon_pass = bool(
            upper_slope > 0.0
            and feasible_lower <= upper_slope + HULL_TOLERANCE
        )
        group.update(
            {
                "hull_position": position,
                "lower_slope": lower_slope,
                "upper_slope": upper_slope,
                "required_slope": required,
                "feasible_lower_slope": feasible_lower,
                "epsilon_pass": epsilon_pass,
            }
        )
        if epsilon_pass:
            selected_unsorted.append(int(group["sample_index"]))

    fallback_used = not selected_unsorted
    if fallback_used:
        selected_unsorted.append(int(np.argmax(sizes)))
    selected = np.asarray(
        sorted(
            selected_unsorted,
            key=lambda index: (sizes[index], losses[index], index),
        ),
        dtype=np.int64,
    )

    pareto_dominated = np.zeros(sample_count, dtype=np.bool_)
    for index in range(sample_count):
        better_or_equal = (sizes >= sizes[index]) & (losses <= losses[index])
        strictly_better = (sizes > sizes[index]) | (losses < losses[index])
        better_or_equal[index] = False
        pareto_dominated[index] = bool(np.any(better_or_equal & strictly_better))

    return selected, {
        "order": order,
        "groups": groups,
        "cell_group": cell_group,
        "hull": hull,
        "popped_groups": popped_groups,
        "target": target,
        "minimum_loss": minimum_loss,
        "fallback_used": fallback_used,
        "pareto_dominated": pareto_dominated,
    }


def _fmt_indices(indices: np.ndarray | list[int]) -> str:
    return ";".join(str(int(index)) for index in indices)


def _finite(value: float) -> str | float:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value


def _trace_state(
    audit_state: int,
    source_state_index: int,
    record: dict[str, Any],
    trajectory: dict[str, Any],
    detailed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history_count = int(record["history_count"])
    iteration = int(record["iteration"])
    generator_id = int(record["generator_id"])
    points = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:history_count],
        dtype=np.float64,
    )
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:history_count], dtype=np.float64
    )
    (
        pool,
        owners,
        distances,
        assigned_radius,
        spacing_floor,
        sizes,
    ) = _reconstruct_sizes(points, generator_id, iteration)
    independent, metadata = _standalone_po_selection(sizes, losses)
    teacher = edirect._potentially_optimal_cells(sizes, losses)
    selected_match = bool(np.array_equal(independent, teacher))
    independent_largest = int(independent[int(np.argmax(sizes[independent]))])
    teacher_largest = int(teacher[int(np.argmax(sizes[teacher]))])
    largest_match = independent_largest == teacher_largest
    selected_order = {int(index): position for position, index in enumerate(independent)}
    hull_position = {
        int(group_index): position
        for position, group_index in enumerate(metadata["hull"])
    }
    representative_by_group = {
        group_index: int(group["sample_index"])
        for group_index, group in enumerate(metadata["groups"])
    }
    rows: list[dict[str, Any]] = [
        {
            "audit_state": audit_state,
            "source_state_index": source_state_index,
            "row_type": "state_summary",
            "detailed_state": detailed,
            "trajectory_id": int(record["trajectory_id"]),
            "generator_id": generator_id,
            "iteration": iteration,
            "history_count": history_count,
            "cell_count_before_po": len(points),
            "raw_voronoi_candidate_count": len(pool),
            "equal_size_group_count": len(metadata["groups"]),
            "lower_right_hull_count": len(metadata["hull"]),
            "pareto_dominated_cell_count": int(np.sum(metadata["pareto_dominated"])),
            "same_size_discarded_count": sum(
                int(representative_by_group[int(metadata["cell_group"][index])] != index)
                for index in range(len(points))
            ),
            "minimum_observed_loss": metadata["minimum_loss"],
            "sufficient_decrease_target": metadata["target"],
            "epsilon": edirect.EDIRECT_EPSILON,
            "fallback_largest_cell_used": metadata["fallback_used"],
            "teacher_selected_indices": _fmt_indices(teacher),
            "standalone_selected_indices": _fmt_indices(independent),
            "selected_set_and_order_exact": selected_match,
            "teacher_largest_selected_index": teacher_largest,
            "standalone_largest_selected_index": independent_largest,
            "largest_designation_exact": largest_match,
        }
    ]
    cell_rows: list[dict[str, Any]] = []
    for index in range(len(points)):
        group_index = int(metadata["cell_group"][index])
        group = metadata["groups"][group_index]
        representative = representative_by_group[group_index] == index
        hull_member = group_index in hull_position and representative
        row = {
            "audit_state": audit_state,
            "source_state_index": source_state_index,
            "row_type": "cell",
            "detailed_state": detailed,
            "trajectory_id": int(record["trajectory_id"]),
            "iteration": iteration,
            "history_count": history_count,
            "cell_index": index,
            "site_x_normalized": float(points[index, 0]),
            "site_y_normalized": float(points[index, 1]),
            "observed_site_loss": float(losses[index]),
            "assigned_candidate_count": int(np.sum(owners == index)),
            "max_assigned_candidate_distance": float(assigned_radius[index]),
            "half_nearest_site_spacing": float(spacing_floor[index]),
            "estimated_cell_size": float(sizes[index]),
            "size_determined_by": (
                "assigned_radius"
                if assigned_radius[index] >= spacing_floor[index]
                else "half_nearest_site_spacing"
            ),
            "equal_size_group": group_index,
            "is_equal_size_representative": representative,
            "same_size_dominated": not representative,
            "pareto_dominated_size_loss": bool(metadata["pareto_dominated"][index]),
            "is_lower_right_hull_member": hull_member,
            "hull_position": hull_position.get(group_index, "") if representative else "",
            "lower_neighbor_slope": (
                _finite(float(group["lower_slope"])) if hull_member else ""
            ),
            "upper_neighbor_slope": (
                _finite(float(group["upper_slope"])) if hull_member else ""
            ),
            "required_slope_for_epsilon": (
                _finite(float(group["required_slope"])) if hull_member else ""
            ),
            "feasible_lower_slope": (
                _finite(float(group["feasible_lower_slope"])) if hull_member else ""
            ),
            "passes_epsilon_slope_test": (
                bool(group["epsilon_pass"]) if hull_member else False
            ),
            "is_potentially_optimal": index in selected_order,
            "selected_output_order": selected_order.get(index, ""),
            "is_largest_selected_cell": index == independent_largest,
            "loo_error_used_for_po_selection": False,
            "normalized_quantity_used_for_po_selection": False,
        }
        rows.append(row)
        cell_rows.append(row)
    return rows, {
        "audit_state": audit_state,
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": iteration,
        "history_count": history_count,
        "pool_count": len(pool),
        "groups": len(metadata["groups"]),
        "hull_count": len(metadata["hull"]),
        "target": metadata["target"],
        "teacher": teacher,
        "independent": independent,
        "largest": independent_largest,
        "selected_match": selected_match,
        "largest_match": largest_match,
        "fallback": metadata["fallback_used"],
        "cells": cell_rows,
        "detailed": detailed,
    }


def _manual_section(summary: dict[str, Any]) -> str:
    lines = [
        f"### Audit state {summary['audit_state']}: trajectory "
        f"{summary['trajectory_id']}, iteration {summary['iteration']}",
        "",
        f"History/cells: {summary['history_count']}; raw Voronoi probes: "
        f"{summary['pool_count']}; equal-size groups: {summary['groups']}; hull groups: "
        f"{summary['hull_count']}.",
        "",
        f"Sufficient-decrease target: `{summary['target']:.12g}`. Selected sites in "
        f"final sorted order: `{_fmt_indices(summary['independent'])}`. Largest selected "
        f"site: `{summary['largest']}`.",
        "",
        "| Site | d=size | f=loss | Equal-size rep | Pareto dominated | Hull | "
        "K lower | K required | K upper | PO | Largest |",
        "|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in summary["cells"]:
        lower = row["lower_neighbor_slope"]
        required = row["required_slope_for_epsilon"]
        upper = row["upper_neighbor_slope"]
        lines.append(
            f"| {row['cell_index']} | {row['estimated_cell_size']:.9g} | "
            f"{row['observed_site_loss']:.9g} | {row['is_equal_size_representative']} | "
            f"{row['pareto_dominated_size_loss']} | {row['is_lower_right_hull_member']} | "
            f"{lower if lower != '' else '—'} | {required if required != '' else '—'} | "
            f"{upper if upper != '' else '—'} | {row['is_potentially_optimal']} | "
            f"{row['is_largest_selected_cell']} |"
        )
    lines.extend(
        [
            "",
            "Manual reproduction: sort by `(d,f,index)`, collapse equal-`d` groups, "
            "apply the cross-product hull scan, then check whether each hull site's "
            "feasible `K` lower bound is no larger than its upper-neighbor slope. The "
            "table gives every numerical term needed for that final test.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown(summaries: list[dict[str, Any]]) -> str:
    selected_matches = sum(summary["selected_match"] for summary in summaries)
    largest_matches = sum(summary["largest_match"] for summary in summaries)
    detailed = "\n".join(
        _manual_section(summary) for summary in summaries if summary["detailed"]
    )
    return f"""# Potentially-optimal Voronoi-cell selection audit

## Result

The eDIRECT implementation uses a **DIRECT-style global lower-right-hull test in
`(estimated Voronoi-cell size, observed site loss)` space**. It is not an independent
per-cell threshold and it does not use the surrogate or LOO error at this stage. The
standalone reconstruction exactly reproduced the ordered selected-cell set in
**{selected_matches}/{len(summaries)}** audited states and exactly reproduced the
largest-selected-cell designation in **{largest_matches}/{len(summaries)}** states.

The audited source is `experiments/sho/run_stage1b_edirect_vs_direct.py`, specifically
`_candidate_pool`, `_cell_data`, `_potentially_optimal_cells`, and the first lines of
`_cell_proposals`. No search or model was run.

## Exact data path and available features

1. A deterministic scrambled-Sobol plus closed-edge probe pool is generated in normalized
   `[0,1]^2` coordinates.
2. A KD-tree assigns every probe to its nearest evaluated site. For each site, the code
   has its coordinates, real observed prefix loss, assigned probe count/distances, maximum
   assigned distance, and nearest evaluated-site spacing.
3. The cell size `d_i` is
   `max(max assigned probe distance, 0.5 * nearest evaluated-site distance)`. The second
   term prevents a spuriously zero/small radius when the finite probe pool misses a cell.
4. PO selection receives **only** the vectors `d_i` and observed losses `f_i`. Site
   coordinates are available but unused. No normalized values, surrogate predictions,
   LOO errors, candidate source labels, or learned features enter the rule.
5. LOO error and normalized selected-cell size are computed only after PO selection, for
   ordinary-cell versus largest-cell proposal scoring.

All quantities required for PO selection are cheap at inference: the history and observed
losses already exist; the same KD-tree query used for Voronoi assignment supplies the
distances; and grouping/hull arithmetic is linear after sorting.

## Exact potentially-optimal rule

Let `d_i` be estimated cell size and `f_i` its observed real prefix loss.

1. Sort sites lexicographically by `(d_i, f_i, original_site_index)`.
2. Collapse consecutive sizes within absolute tolerance `1e-14`. Keep only the
   lowest-loss site in each size group. Loss ties therefore keep the smallest original
   site index through the lexicographic order. All other same-size sites are dominated
   for PO purposes.
3. Scan the size groups in increasing `d`. Maintain a hull stack. For consecutive stack
   points `p,q` and new point `r`, compute
   `cross=(d_q-d_p)(f_r-f_p)-(f_q-f_p)(d_r-d_p)`. Pop `q` only when
   `cross < -5e-14 * max(1,|f_p|,|f_q|,|f_r|)`. Collinear points are retained.
4. Define `f_min=min_i f_i` and the sufficient-decrease target
   `T=f_min-1e-4*|f_min|`.
5. For hull site `i`, neighbor slopes bound the Lipschitz-like value `K` for which it can
   minimize `f-Kd`. The previous-hull slope is the lower bound and the next-hull slope is
   the upper bound (missing endpoints use `-inf`/`+inf`). Sufficient decrease additionally
   requires `K >= (f_i-T)/d_i`, and `K >= 0`.
6. Select the site iff its upper slope is positive and
   `max(previous_slope, (f_i-T)/d_i, 0) <= next_slope + 5e-14`.
7. If no site passes, use `argmax(d_i)` as a safety fallback; NumPy's first maximum wins.
8. Return selected site indices sorted by `(d_i, f_i, original_site_index)`.

This is the classical DIRECT `f_i-Kd_i` potentially-optimal/sufficient-decrease idea,
but applied to approximate Voronoi radii rather than DIRECT hyperrectangle diagonals. The
implementation also makes its numerical choices explicit: finite-probe cell radii, a
`1e-14` size-group tolerance, a scaled `5e-14` hull tolerance, retained collinear groups,
`epsilon=1e-4`, and the largest-cell fallback. Thus the optimization criterion is
DIRECT-style; the geometry and numerical realization are this eDIRECT implementation's
adaptation.

Selection is **global and simultaneous**. Whether cell `i` is selected depends on all
other cells through equal-size competition, the hull, `f_min`, and adjacent hull slopes.
There is no sequential accept/reject interaction among selected cells.

## Largest selected cell

After PO selection, `_cell_proposals` forms `selected_sizes = sizes[selected]` and uses
`largest_position = np.argmax(selected_sizes)`. Therefore the same approximate radius
`d_i` determines the designation. `np.argmax` returns the first maximum. Since equal-size
groups normally leave only one representative and `selected` is sorted by
`(size,loss,index)`, exact largest-size ties have already been resolved; the fallback also
uses NumPy's first-index behavior.

## Concise pseudocode

```text
for each evaluated site i:
    assigned = raw Voronoi probes nearest to i
    d[i] = max(max_distance(i, assigned), 0.5 * nearest_site_spacing(i))
    f[i] = observed real prefix loss at i

groups = sort by (d, f, index), then keep minimum-f site per equal-d group
hull = lower_right_hull(groups, retain_collinear=True)
target = min(f) - 1e-4 * abs(min(f))

selected = []
for i in hull:
    K_low  = max(slope(previous_hull, i), (f[i]-target)/d[i], 0)
    K_high = slope(i, next_hull)
    if K_high > 0 and K_low <= K_high + 5e-14:
        selected.append(i)

if selected is empty:
    selected = [argmax(d)]
selected = sort(selected, key=(d, f, index))
largest = selected[argmax(d[selected])]
```

## Natural neural analogue

The natural learned analogue is a **permutation-equivariant set-to-mask model over
cells**, preferably a small Set Transformer or attention-based DeepSets model. Each cell
token would contain `(site coordinates, observed loss, estimated size)`; global attention
or pooled context is necessary because hull membership and slope feasibility are relational
and depend on the whole set. It should output one PO logit per cell plus either derive the
largest designation deterministically from selected sizes or use a second masked argmax
head. A plain independent per-cell MLP is structurally mismatched to this operation.

Because the exact deterministic selector is already inexpensive and fully observable, a
learned replacement should be judged against exact set recovery, not merely per-cell
accuracy.

## Three manually reproducible states

{detailed}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    indices, detailed_indices = _representative_indices(payload["states"])
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for audit_state, source_state_index in enumerate(indices):
        record = payload["states"][source_state_index]
        trajectory = trajectories[int(record["trajectory_id"])]
        state_rows, summary = _trace_state(
            audit_state,
            source_state_index,
            record,
            trajectory,
            source_state_index in detailed_indices,
        )
        rows.extend(state_rows)
        summaries.append(summary)
        print(
            f"state {audit_state + 1}/{len(indices)} history={summary['history_count']} "
            f"selected={len(summary['independent'])} exact={summary['selected_match']} "
            f"largest={summary['largest_match']}",
            flush=True,
        )
    if not all(
        summary["selected_match"] and summary["largest_match"]
        for summary in summaries
    ):
        raise RuntimeError("standalone PO reconstruction did not exactly match teacher")
    _write_csv(TRACE_PATH, rows)
    AUDIT_PATH.write_text(_markdown(summaries), encoding="utf-8")
    print(f"wrote {AUDIT_PATH} and {TRACE_PATH}", flush=True)


if __name__ == "__main__":
    main()
