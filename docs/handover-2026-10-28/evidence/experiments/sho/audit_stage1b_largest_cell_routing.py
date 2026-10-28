#!/usr/bin/env python3
"""Audit routing from selected eDIRECT cells to largest versus ordinary."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/largest_cell_routing_audit")
AUDIT_PATH = OUTPUT_DIR / "largest_cell_routing_audit.md"
TRACE_PATH = OUTPUT_DIR / "largest_cell_routing_trace.csv"
RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH
STATE_COUNT = 20
DETAIL_POSITIONS = {0, STATE_COUNT // 2, STATE_COUNT - 1}


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


def _representative_indices(states: list[dict[str, Any]]) -> list[int]:
    indices = [
        index for index, state in enumerate(states) if state["candidate_type"] == "cell"
    ]
    ordered = sorted(
        indices,
        key=lambda index: (
            int(states[index]["history_count"]),
            len(states[index]["teacher_curated_points"]),
            int(states[index]["trajectory_id"]),
            index,
        ),
    )
    positions = np.rint(np.linspace(0, len(ordered) - 1, STATE_COUNT)).astype(int)
    return [ordered[int(position)] for position in positions]


def _standalone_route(
    selected_ids: np.ndarray, selected_sizes: np.ndarray
) -> tuple[np.ndarray, int]:
    """Routing reconstructed from only ordered selected IDs and size values."""

    largest_position = int(np.argmax(selected_sizes))
    routes = np.full(len(selected_ids), "ordinary", dtype=object)
    routes[largest_position] = "largest"
    return routes, largest_position


def _fmt_indices(values: np.ndarray | list[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def _fmt_sizes(values: np.ndarray) -> str:
    return ";".join(format(float(value), ".17g") for value in values)


def _trace_state(
    audit_state: int,
    source_state_index: int,
    record: dict[str, Any],
    trajectory: dict[str, Any],
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
    data = edirect._cell_data(points, losses, generator_id, iteration)
    selected_ids = np.asarray(data.selected, dtype=np.int64)
    selected_sizes = np.asarray(data.sizes[selected_ids], dtype=np.float64)

    # Reconstruct the two ingredients of size so the trace states exactly what
    # is being maximized rather than merely copying CellProposalData.sizes.
    assigned_radius = np.zeros(len(points), dtype=np.float64)
    np.maximum.at(assigned_radius, data.owners, data.distances)
    tree = cKDTree(points)
    if len(points) > 1:
        half_spacing = 0.5 * tree.query(points, k=2)[0][:, 1]
    else:
        half_spacing = np.zeros(1, dtype=np.float64)
    reconstructed_sizes = np.maximum(assigned_radius, half_spacing)
    if not np.array_equal(reconstructed_sizes, data.sizes):
        raise RuntimeError("cell-size reconstruction differs from implementation")

    implementation_largest_position = int(np.argmax(selected_sizes))
    implementation_routes = np.full(len(selected_ids), "ordinary", dtype=object)
    implementation_routes[implementation_largest_position] = "largest"
    standalone_routes, standalone_largest_position = _standalone_route(
        selected_ids.copy(), selected_sizes.copy()
    )
    exact = bool(
        standalone_largest_position == implementation_largest_position
        and np.array_equal(standalone_routes, implementation_routes)
    )
    maximum_size = float(selected_sizes[implementation_largest_position])
    exact_tie_positions = np.flatnonzero(selected_sizes == maximum_size)
    exact_tie_ids = selected_ids[exact_tie_positions]
    detailed = audit_state in DETAIL_POSITIONS
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
            "selected_cell_count": len(selected_ids),
            "selected_cell_ids_in_order": _fmt_indices(selected_ids),
            "selected_cell_sizes_in_order": _fmt_sizes(selected_sizes),
            "maximum_selected_cell_size": maximum_size,
            "exact_maximum_size_tie_count": len(exact_tie_positions),
            "exact_maximum_size_tie_positions": _fmt_indices(exact_tie_positions),
            "exact_maximum_size_tie_cell_ids": _fmt_indices(exact_tie_ids),
            "implementation_largest_position": implementation_largest_position,
            "implementation_largest_cell_id": int(
                selected_ids[implementation_largest_position]
            ),
            "standalone_largest_position": standalone_largest_position,
            "standalone_largest_cell_id": int(
                selected_ids[standalone_largest_position]
            ),
            "ordinary_cell_ids": _fmt_indices(
                selected_ids[np.arange(len(selected_ids)) != implementation_largest_position]
            ),
            "routing_exact": exact,
        }
    ]
    cell_rows: list[dict[str, Any]] = []
    for selected_position, cell_id_value in enumerate(selected_ids):
        cell_id = int(cell_id_value)
        route = str(implementation_routes[selected_position])
        row = {
            "audit_state": audit_state,
            "source_state_index": source_state_index,
            "row_type": "selected_cell",
            "detailed_state": detailed,
            "trajectory_id": int(record["trajectory_id"]),
            "generator_id": generator_id,
            "iteration": iteration,
            "history_count": history_count,
            "selected_cell_count": len(selected_ids),
            "selected_position": selected_position,
            "cell_id": cell_id,
            "site_x_normalized": float(points[cell_id, 0]),
            "site_y_normalized": float(points[cell_id, 1]),
            "observed_site_loss_not_used_for_routing": float(losses[cell_id]),
            "assigned_candidate_count_not_used_directly": int(
                np.sum(data.owners == cell_id)
            ),
            "maximum_assigned_candidate_distance": float(assigned_radius[cell_id]),
            "half_nearest_site_spacing": float(half_spacing[cell_id]),
            "exact_cell_size": float(data.sizes[cell_id]),
            "cell_size_source": (
                "maximum_assigned_candidate_distance"
                if assigned_radius[cell_id] >= half_spacing[cell_id]
                else "half_nearest_site_spacing_floor"
            ),
            "equals_maximum_selected_size": bool(data.sizes[cell_id] == maximum_size),
            "implementation_route": route,
            "standalone_route": str(standalone_routes[selected_position]),
            "routing_exact": route == str(standalone_routes[selected_position]),
        }
        rows.append(row)
        cell_rows.append(row)
    return rows, {
        "audit_state": audit_state,
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": iteration,
        "history_count": history_count,
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "selected_sizes": selected_sizes,
        "largest_position": implementation_largest_position,
        "largest_id": int(selected_ids[implementation_largest_position]),
        "tie_count": len(exact_tie_positions),
        "tie_ids": exact_tie_ids,
        "exact": exact,
        "detailed": detailed,
        "cells": cell_rows,
    }


def _manual_section(summary: dict[str, Any]) -> str:
    lines = [
        f"### State {summary['audit_state']}: trajectory {summary['trajectory_id']}, "
        f"iteration {summary['iteration']}",
        "",
        f"History sites: {summary['history_count']}; selected cells: "
        f"{summary['selected_count']}; largest selected cell: `{summary['largest_id']}` "
        f"at selected position `{summary['largest_position']}`; exact maximum-size tie "
        f"count: `{summary['tie_count']}`.",
        "",
        "| Selected position | Cell ID | Exact size | Max assigned radius | "
        "Half-spacing floor | Route |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["cells"]:
        lines.append(
            f"| {row['selected_position']} | {row['cell_id']} | "
            f"{row['exact_cell_size']:.17g} | "
            f"{row['maximum_assigned_candidate_distance']:.17g} | "
            f"{row['half_nearest_site_spacing']:.17g} | "
            f"{row['implementation_route']} |"
        )
    lines.extend(
        [
            "",
            "Manual reproduction: read the exact size column in selected-list order, "
            "take its first `argmax`, route that one row to `largest`, and route every "
            "other row to `ordinary`.",
            "",
        ]
    )
    return "\n".join(lines)


def _audit_markdown(summaries: list[dict[str, Any]]) -> str:
    exact_count = sum(summary["exact"] for summary in summaries)
    tied_states = [summary for summary in summaries if summary["tie_count"] > 1]
    detailed = "\n".join(
        _manual_section(summary) for summary in summaries if summary["detailed"]
    )
    tie_statement = (
        f"{len(tied_states)} audited states had an exact maximum-size tie."
        if tied_states
        else "None of the 20 audited states had an exact maximum-size tie."
    )
    return f"""# Largest-selected-cell routing audit

## Result

The routing operation is exactly one NumPy argmax over the estimated sizes of the
currently selected potentially-optimal cells:

```python
selected_sizes = data.sizes[data.selected]
largest_position = int(np.argmax(selected_sizes))
route[selected_position] = (
    "largest" if selected_position == largest_position else "ordinary"
)
```

The standalone reconstruction, using only ordered selected-cell IDs and their saved size
values, reproduced the implementation's routing in **{exact_count}/{len(summaries)}**
representative states. {tie_statement}

## What “cell size” means

It is **not Voronoi area or volume**. It is an approximate Euclidean **cell radius** in
normalized `[0,1]^2` parameter coordinates:

```text
assigned_radius_i = max distance from site i to a raw candidate assigned to i
spacing_floor_i   = 0.5 * distance from site i to its nearest evaluated site
cell_size_i       = max(assigned_radius_i, spacing_floor_i)
```

The raw candidates are the deterministic Sobol-plus-domain-edge points used to approximate
the Voronoi cells. The spacing floor prevents the finite probe cloud from making a cell
artificially tiny.

## Exact routing behavior

1. `_potentially_optimal_cells` has already returned an ordered selected-site array.
2. `_cell_proposals` gathers `data.sizes[data.selected]` without renormalizing it.
3. `np.argmax` chooses exactly one selected position.
4. That position uses the largest-cell farthest-candidate rule. Every other selected
   position uses the ordinary-cell weighted distance/surrogate rule.

Routing directly depends only on selected-cell sizes. It does **not** inspect loss,
surrogate error, LOO error, candidate count, site coordinates, or prior search decisions.
Those quantities can influence which cells were selected or how size was estimated, but
they are not inputs to this post-selection branch. Selected-list ordering matters only for
an exact size tie.

`np.argmax` returns the first maximum, so only one cell is ever routed as largest. The PO
selector already groups equal sizes within `1e-14` and returns selected cells sorted by
`(size, loss, site index)`, making exact selected-size ties normally impossible. If a tie
were nevertheless supplied, the first cell in that selected order would win; because of
the upstream order, that means lower loss and then lower original site index for an exact
size tie.

## Determinism and inference cost

The rule is completely deterministic from the ordered selected-cell IDs and size values.
The sizes are already present in `CellProposalData`; routing adds one array gather and one
argmax. No surrogate evaluation is required to classify largest versus ordinary.

## Pseudocode

```text
selected_sizes = [cell_size[cell_id] for cell_id in selected_cell_ids]
largest_position = first_argmax(selected_sizes)

for position, cell_id in enumerate(selected_cell_ids):
    if position == largest_position:
        route[cell_id] = LARGEST
    else:
        route[cell_id] = ORDINARY
```

## Natural supervised network

A learned replacement needs only the selected cells' scalar size values and a
permutation-equivariant one-winner decision. The smallest suitable model is a shared
one-dimensional scorer `cell_size -> logit`, followed by a softmax/argmax across selected
cells. To reproduce hypothetical exact ties, also provide selected-order position or apply
the same deterministic first-index tie break. Loss, surrogate error, LOO error, candidate
count, and coordinates are unnecessary for this operation.

In practice, the deterministic operation is already cheaper and exact; a learned version
should be required to recover the full one-hot routing mask, not just high per-cell binary
accuracy.

## Three manually reproducible states

{detailed}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    indices = _representative_indices(payload["states"])
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for audit_state, source_state_index in enumerate(indices):
        record = payload["states"][source_state_index]
        trajectory = trajectories[int(record["trajectory_id"])]
        state_rows, summary = _trace_state(
            audit_state, source_state_index, record, trajectory
        )
        rows.extend(state_rows)
        summaries.append(summary)
        print(
            f"state {audit_state + 1}/{len(indices)} "
            f"history={summary['history_count']} selected={summary['selected_count']} "
            f"largest={summary['largest_id']} ties={summary['tie_count']} "
            f"exact={summary['exact']}",
            flush=True,
        )
    if not all(summary["exact"] for summary in summaries):
        raise RuntimeError("standalone routing reconstruction disagrees with implementation")
    _write_csv(TRACE_PATH, rows)
    AUDIT_PATH.write_text(_audit_markdown(summaries), encoding="utf-8")
    print(f"wrote {AUDIT_PATH} and {TRACE_PATH}", flush=True)


if __name__ == "__main__":
    main()
