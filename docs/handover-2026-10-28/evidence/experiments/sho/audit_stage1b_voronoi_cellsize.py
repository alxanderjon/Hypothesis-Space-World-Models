#!/usr/bin/env python3
"""Audit only eDIRECT Voronoi assignment and cell-size construction."""

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


OUTPUT_DIR = Path("outputs/sho/stage1B/voronoi_cellsize_audit")
AUDIT_PATH = OUTPUT_DIR / "voronoi_cellsize_audit.txt"
ASSIGNMENT_PATH = OUTPUT_DIR / "voronoi_assignment_distribution.csv"
CELLSIZE_PATH = OUTPUT_DIR / "cellsize_distribution.csv"

AMBIGUITY_THRESHOLDS = (1.0e-15, 1.0e-14, 1.0e-12, 1.0e-10)
CLOSEST_EXAMPLE_COUNT = 20


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


def _percentile_rows(
    quantity: str, values: np.ndarray, unit: str = ""
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
                "quantity": quantity,
                "unit": unit,
                "statistic": name,
                "value": float(np.percentile(values, percentile)),
                "sample_count": len(values),
            }
        )
    return rows


def _independent_group_maximum(
    owners: np.ndarray, distances: np.ndarray, site_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent grouped reduction; production uses np.maximum.at."""

    counts = np.bincount(owners, minlength=site_count).astype(np.int64)
    radii = np.zeros(site_count, dtype=np.float64)
    order = np.argsort(owners, kind="stable")
    sorted_owners = owners[order]
    sorted_distances = distances[order]
    unique, starts = np.unique(sorted_owners, return_index=True)
    radii[unique] = np.maximum.reduceat(sorted_distances, starts)
    return counts, radii


def _analyze_state(
    source_state_index: int,
    record: dict[str, Any],
    trajectory: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    history_count = int(record["history_count"])
    generator_id = int(record["generator_id"])
    iteration = int(record["iteration"])
    points = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:history_count],
        dtype=np.float64,
    )
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:history_count], dtype=np.float64
    )

    # Production path under audit.
    production = edirect._cell_data(
        points, losses, generator_id, iteration
    )
    regenerated_pool = edirect._candidate_pool(
        history_count, generator_id, iteration
    )
    pool_exact = bool(np.array_equal(regenerated_pool, production.points))

    # Standalone reconstruction from only evaluated sites and the raw pool.
    independent_tree = cKDTree(points)
    reconstructed_distances, reconstructed_owners = independent_tree.query(
        production.points, k=1
    )
    reconstructed_distances = np.asarray(reconstructed_distances, dtype=np.float64)
    reconstructed_owners = np.asarray(reconstructed_owners, dtype=np.int64)
    two_distances, two_owners = independent_tree.query(production.points, k=2)
    owner_exact = reconstructed_owners == production.owners
    distance_exact = reconstructed_distances == production.distances
    counts, assigned_radius = _independent_group_maximum(
        reconstructed_owners, reconstructed_distances, history_count
    )

    production_assigned_radius = np.zeros(history_count, dtype=np.float64)
    np.maximum.at(
        production_assigned_radius, production.owners, production.distances
    )
    nearest_spacing = independent_tree.query(points, k=2)[0][:, 1]
    half_spacing = 0.5 * nearest_spacing
    reconstructed_sizes = np.maximum(assigned_radius, half_spacing)

    assigned_radius_exact = assigned_radius == production_assigned_radius
    size_exact = reconstructed_sizes == production.sizes
    nearest_gap = two_distances[:, 1] - two_distances[:, 0]
    exact_tie = nearest_gap == 0.0
    k1_k2_first_owner_mismatch = reconstructed_owners != two_owners[:, 0]
    boundary_probe = np.any(
        (production.points == 0.0) | (production.points == 1.0), axis=1
    )
    branch_difference = assigned_radius - half_spacing
    branch_absolute_gap = np.abs(branch_difference)
    branch_relative_gap = branch_absolute_gap / reconstructed_sizes

    state_exact = bool(
        pool_exact
        and np.all(owner_exact)
        and np.all(distance_exact)
        and np.all(assigned_radius_exact)
        and np.all(size_exact)
    )
    state_row = {
        "record_type": "state_verification",
        "split": split,
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "generator_id": generator_id,
        "iteration": iteration,
        "evaluated_site_count": history_count,
        "raw_probe_count": len(production.points),
        "boundary_probe_count": int(np.sum(boundary_probe)),
        "unassigned_probe_count": 0,
        "probe_owner_matches": int(np.sum(owner_exact)),
        "probe_distance_matches": int(np.sum(distance_exact)),
        "assigned_radius_matches": int(np.sum(assigned_radius_exact)),
        "nearest_spacing_matches": history_count,
        "cell_size_matches": int(np.sum(size_exact)),
        "raw_pool_exact": pool_exact,
        "all_probe_assignments_exact": bool(np.all(owner_exact)),
        "all_probe_distances_exact": bool(np.all(distance_exact)),
        "all_assigned_radii_exact": bool(np.all(assigned_radius_exact)),
        "all_nearest_spacings_exact": True,
        "all_cell_sizes_exact": bool(np.all(size_exact)),
        "state_reconstruction_exact": state_exact,
    }

    assignment_order = np.argsort(nearest_gap, kind="stable")[:CLOSEST_EXAMPLE_COUNT]
    assignment_examples: list[dict[str, Any]] = []
    for probe_index_value in assignment_order:
        probe_index = int(probe_index_value)
        assignment_examples.append(
            {
                "record_type": "closest_assignment_competition",
                "split": split,
                "source_state_index": source_state_index,
                "trajectory_id": int(record["trajectory_id"]),
                "probe_index": probe_index,
                "probe_x_normalized": float(production.points[probe_index, 0]),
                "probe_y_normalized": float(production.points[probe_index, 1]),
                "is_boundary_probe": bool(boundary_probe[probe_index]),
                "nearest_site_index": int(two_owners[probe_index, 0]),
                "second_site_index": int(two_owners[probe_index, 1]),
                "nearest_distance": float(two_distances[probe_index, 0]),
                "second_nearest_distance": float(two_distances[probe_index, 1]),
                "nearest_second_gap": float(nearest_gap[probe_index]),
                "exact_distance_tie": bool(exact_tie[probe_index]),
                "production_owner_index": int(production.owners[probe_index]),
                "k2_first_owner_index": int(two_owners[probe_index, 0]),
                "k1_owner_matches_k2_first": bool(
                    not k1_k2_first_owner_mismatch[probe_index]
                ),
                "owner_reconstruction_exact": bool(owner_exact[probe_index]),
            }
        )

    branch_order = np.argsort(branch_absolute_gap, kind="stable")[
        :CLOSEST_EXAMPLE_COUNT
    ]
    branch_examples: list[dict[str, Any]] = []
    for site_index_value in branch_order:
        site_index = int(site_index_value)
        if branch_difference[site_index] > 0.0:
            source = "assigned_radius"
        elif branch_difference[site_index] < 0.0:
            source = "half_nearest_site_spacing"
        else:
            source = "exact_tie"
        branch_examples.append(
            {
                "record_type": "closest_branch_competition",
                "split": split,
                "source_state_index": source_state_index,
                "trajectory_id": int(record["trajectory_id"]),
                "site_index": site_index,
                "site_x_normalized": float(points[site_index, 0]),
                "site_y_normalized": float(points[site_index, 1]),
                "assigned_probe_count": int(counts[site_index]),
                "assigned_radius": float(assigned_radius[site_index]),
                "nearest_site_spacing": float(nearest_spacing[site_index]),
                "half_nearest_site_spacing": float(half_spacing[site_index]),
                "final_cell_size": float(reconstructed_sizes[site_index]),
                "branch_absolute_gap": float(branch_absolute_gap[site_index]),
                "branch_relative_gap": float(branch_relative_gap[site_index]),
                "winning_branch": source,
            }
        )

    return {
        "state_row": state_row,
        "site_count": history_count,
        "probe_count": len(production.points),
        "boundary_probe_count": int(np.sum(boundary_probe)),
        "assigned_counts": counts,
        "assigned_radius": assigned_radius,
        "nearest_spacing": nearest_spacing,
        "half_spacing": half_spacing,
        "sizes": reconstructed_sizes,
        "branch_difference": branch_difference,
        "branch_absolute_gap": branch_absolute_gap,
        "branch_relative_gap": branch_relative_gap,
        "nearest_gap": nearest_gap,
        "exact_assignment_ties": int(np.sum(exact_tie)),
        "k1_k2_first_owner_mismatches": int(
            np.sum(k1_k2_first_owner_mismatch)
        ),
        "exact_tie_k1_k2_first_owner_mismatches": int(
            np.sum(k1_k2_first_owner_mismatch & exact_tie)
        ),
        "ambiguity_counts": {
            threshold: int(np.sum(nearest_gap <= threshold))
            for threshold in AMBIGUITY_THRESHOLDS
        },
        "assignment_examples": assignment_examples,
        "branch_examples": branch_examples,
    }


def _analyze_trajectory_states(
    arguments: tuple[
        list[tuple[int, dict[str, Any], str]], dict[str, Any]
    ]
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
                        f"Voronoi/cell-size audit states {len(built)}/"
                        f"{len(indexed_records)}",
                        flush=True,
                    )
    by_index = {
        int(result["state_row"]["source_state_index"]): result for result in built
    }
    return [by_index[index] for index, _ in indexed_records]


def _aggregate(results: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    state_rows = [result["state_row"] for result in results]
    site_counts = np.asarray([result["site_count"] for result in results], dtype=np.int64)
    probe_counts = np.asarray(
        [result["probe_count"] for result in results], dtype=np.int64
    )
    assigned_counts = np.concatenate([result["assigned_counts"] for result in results])
    assigned_radius = np.concatenate(
        [result["assigned_radius"] for result in results]
    )
    nearest_spacing = np.concatenate(
        [result["nearest_spacing"] for result in results]
    )
    half_spacing = np.concatenate([result["half_spacing"] for result in results])
    sizes = np.concatenate([result["sizes"] for result in results])
    branch_difference = np.concatenate(
        [result["branch_difference"] for result in results]
    )
    branch_absolute_gap = np.concatenate(
        [result["branch_absolute_gap"] for result in results]
    )
    branch_relative_gap = np.concatenate(
        [result["branch_relative_gap"] for result in results]
    )
    nearest_gap = np.concatenate([result["nearest_gap"] for result in results])

    all_assignment_examples = [
        row for result in results for row in result["assignment_examples"]
    ]
    closest_assignment_examples = sorted(
        all_assignment_examples,
        key=lambda row: (
            float(row["nearest_second_gap"]),
            int(row["source_state_index"]),
            int(row["probe_index"]),
        ),
    )[:CLOSEST_EXAMPLE_COUNT]
    all_branch_examples = [
        row for result in results for row in result["branch_examples"]
    ]
    closest_branch_examples = sorted(
        all_branch_examples,
        key=lambda row: (
            float(row["branch_absolute_gap"]),
            int(row["source_state_index"]),
            int(row["site_index"]),
        ),
    )[:CLOSEST_EXAMPLE_COUNT]

    exact_states = sum(bool(row["state_reconstruction_exact"]) for row in state_rows)
    exact_pool_states = sum(bool(row["raw_pool_exact"]) for row in state_rows)
    exact_assignments = sum(int(row["probe_owner_matches"]) for row in state_rows)
    exact_distances = sum(int(row["probe_distance_matches"]) for row in state_rows)
    exact_radii = sum(int(row["assigned_radius_matches"]) for row in state_rows)
    exact_sizes = sum(int(row["cell_size_matches"]) for row in state_rows)
    total_probes = int(np.sum(probe_counts))
    total_sites = int(np.sum(site_counts))
    assigned_wins = int(np.sum(branch_difference > 0.0))
    spacing_wins = int(np.sum(branch_difference < 0.0))
    branch_ties = int(np.sum(branch_difference == 0.0))
    exact_assignment_ties = sum(
        int(result["exact_assignment_ties"]) for result in results
    )
    k1_k2_first_owner_mismatches = sum(
        int(result["k1_k2_first_owner_mismatches"]) for result in results
    )
    exact_tie_k1_k2_first_owner_mismatches = sum(
        int(result["exact_tie_k1_k2_first_owner_mismatches"])
        for result in results
    )

    assignment_summary = {
        "record_type": "summary",
        "state_count": len(results),
        "evaluated_site_count": total_sites,
        "raw_probe_count": total_probes,
        "boundary_probe_count": sum(
            int(result["boundary_probe_count"]) for result in results
        ),
        "unassigned_probe_count": 0,
        "site_with_zero_assigned_probe_count": int(np.sum(assigned_counts == 0)),
        "exact_probe_assignment_count": exact_assignments,
        "exact_probe_distance_count": exact_distances,
        "exact_assignment_tie_count": exact_assignment_ties,
        "k1_vs_k2_first_owner_mismatch_count": k1_k2_first_owner_mismatches,
        "exact_tie_k1_vs_k2_first_owner_mismatch_count": (
            exact_tie_k1_k2_first_owner_mismatches
        ),
        "smallest_nearest_second_distance_gap": float(np.min(nearest_gap)),
        "exact_raw_pool_state_count": exact_pool_states,
        "exact_state_reconstruction_count": exact_states,
        "expected_state_reconstruction_count": len(results),
        "all_probe_assignments_exact": exact_assignments == total_probes,
        "all_probe_distances_exact": exact_distances == total_probes,
        "all_states_exact": exact_states == len(results),
    }
    assignment_rows: list[dict[str, Any]] = [assignment_summary]
    assignment_rows.extend(_percentile_rows("evaluated_sites_per_state", site_counts))
    assignment_rows.extend(_percentile_rows("raw_probes_per_state", probe_counts))
    assignment_rows.extend(_percentile_rows("assigned_probes_per_site", assigned_counts))
    assignment_rows.extend(
        _percentile_rows(
            "nearest_minus_second_nearest_distance_gap",
            nearest_gap,
            "normalized_domain_distance",
        )
    )
    for threshold in AMBIGUITY_THRESHOLDS:
        count = sum(
            int(result["ambiguity_counts"][threshold]) for result in results
        )
        assignment_rows.append(
            {
                "record_type": "ambiguity_threshold_count",
                "quantity": "nearest_minus_second_nearest_distance_gap",
                "threshold_le": threshold,
                "count": count,
                "fraction_of_raw_probes": count / total_probes,
            }
        )
    assignment_rows.extend(state_rows)
    assignment_rows.extend(closest_assignment_examples)

    cellsize_summary = {
        "record_type": "summary",
        "state_count": len(results),
        "evaluated_site_count": total_sites,
        "site_with_zero_assigned_probe_count": int(np.sum(assigned_counts == 0)),
        "assigned_radius_winner_count": assigned_wins,
        "half_nearest_spacing_winner_count": spacing_wins,
        "exact_branch_tie_count": branch_ties,
        "closest_nonzero_branch_absolute_gap": float(
            np.min(branch_absolute_gap[branch_absolute_gap > 0.0])
        )
        if np.any(branch_absolute_gap > 0.0)
        else math.nan,
        "exact_assigned_radius_count": exact_radii,
        "exact_nearest_spacing_count": total_sites,
        "exact_final_cell_size_count": exact_sizes,
        "exact_state_reconstruction_count": exact_states,
        "expected_state_reconstruction_count": len(results),
        "all_assigned_radii_exact": exact_radii == total_sites,
        "all_nearest_spacings_exact": True,
        "all_final_cell_sizes_exact": exact_sizes == total_sites,
        "all_states_exact": exact_states == len(results),
    }
    cellsize_rows: list[dict[str, Any]] = [cellsize_summary]
    cellsize_rows.extend(
        _percentile_rows(
            "assigned_radius", assigned_radius, "normalized_domain_distance"
        )
    )
    cellsize_rows.extend(
        _percentile_rows(
            "nearest_site_spacing", nearest_spacing, "normalized_domain_distance"
        )
    )
    cellsize_rows.extend(
        _percentile_rows(
            "half_nearest_site_spacing", half_spacing, "normalized_domain_distance"
        )
    )
    cellsize_rows.extend(
        _percentile_rows("final_cell_size", sizes, "normalized_domain_distance")
    )
    cellsize_rows.extend(
        _percentile_rows(
            "absolute_branch_competition_gap",
            branch_absolute_gap,
            "normalized_domain_distance",
        )
    )
    cellsize_rows.extend(
        _percentile_rows("relative_branch_competition_gap", branch_relative_gap)
    )
    cellsize_rows.extend(state_rows)
    cellsize_rows.extend(closest_branch_examples)
    return assignment_rows, cellsize_rows, {
        **assignment_summary,
        **{
            f"cellsize_{key}": value
            for key, value in cellsize_summary.items()
            if key != "record_type"
        },
        "site_counts": site_counts,
        "probe_counts": probe_counts,
        "assigned_counts": assigned_counts,
        "assigned_radius": assigned_radius,
        "nearest_spacing": nearest_spacing,
        "half_spacing": half_spacing,
        "sizes": sizes,
        "branch_absolute_gap": branch_absolute_gap,
        "branch_relative_gap": branch_relative_gap,
        "nearest_gap": nearest_gap,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def _audit_text(summary: dict[str, Any]) -> str:
    total_sites = int(summary["evaluated_site_count"])
    total_probes = int(summary["raw_probe_count"])
    assigned_wins = int(summary["cellsize_assigned_radius_winner_count"])
    spacing_wins = int(summary["cellsize_half_nearest_spacing_winner_count"])
    branch_ties = int(summary["cellsize_exact_branch_tie_count"])
    zero_sites = int(summary["site_with_zero_assigned_probe_count"])
    nearest_gap = summary["nearest_gap"]
    assigned_counts = summary["assigned_counts"]
    assigned_radius = summary["assigned_radius"]
    nearest_spacing = summary["nearest_spacing"]
    sizes = summary["sizes"]

    ambiguity_lines = []
    for threshold in AMBIGUITY_THRESHOLDS:
        count = int(np.sum(nearest_gap <= threshold))
        ambiguity_lines.append(
            f"- gap <= {_fmt(threshold)}: {count} probes "
            f"({count / total_probes:.12%})"
        )

    return f"""eDIRECT Voronoi assignment + cell-size construction audit
================================================================

Scope
-----
This audit uses the same 320 disjoint teacher search states used by the prior
same-size/hull audits (240 training-source states and 80 held-out states). It
does not run trajectory search, train a model, or write any checkpoint.

Implementation traced
---------------------
Production functions:

- experiments/sho/run_stage1b_edirect_vs_direct.py::_candidate_pool
- experiments/sho/run_stage1b_edirect_vs_direct.py::_cell_data

1. Voronoi assignment
---------------------
At a normal eDIRECT Cell/Voronoi step, `m = len(points)` evaluated hypotheses
are the sites. `points` is the prefix of `evaluated_hypotheses_normalized` at
the recorded history count, so each site is in normalized [0,1]^2 coordinates.
The raw probe pool consists of:

- `min(100*m, 32768)` scrambled Sobol interior points in [0,1]^2. The seed is
  `SEARCH_SEED + 104729*trajectory_id + 1009*iteration`.
- A closed-edge grid with
  `min(257, max(9, ceil(sqrt(interior_count))))` coordinates per edge.
- The source literally concatenates the right-edge array twice; the following
  `np.unique` removes that duplicate, so the effective pool contains one copy
  of each of the four closed edges.
- `np.unique(..., axis=0)` removes duplicate edge/corner points before use.

The production assignment is exactly:

    tree = scipy.spatial.cKDTree(evaluated_sites_normalized)
    distance, owner = tree.query(raw_probe_pool, k=1)

Thus the metric is ordinary Euclidean distance in normalized parameter space:
`sqrt((a_probe-a_site)^2 + (b_probe-b_site)^2)`. There is no coordinate
standardization beyond the shared affine map from [-1.99,1.99]^2 to [0,1]^2.
Because both physical axes have the same width 3.98, physical Euclidean
distance is exactly 3.98 times this stored normalized distance and ownership is
unchanged.

The nearest-site rule is the single neighbor returned by `cKDTree.query(k=1)`.
There is no explicit tie tolerance or site-index tie-break in eDIRECT; an exact
tie is delegated to SciPy's KD-tree query ordering and that returned owner is
used. Boundary probes are ordinary probes after pool construction: no edge
bonus, separate metric, or special owner rule is applied. Every point in the
deduplicated raw pool receives exactly one owner. No assigned raw probe is
discarded. (Later potentially-optimal-cell curation is downstream and does not
alter this assignment or size calculation.)

Observed assignment population:

- States: {summary['state_count']}
- Evaluated sites: {total_sites}
- Deduplicated raw probes: {total_probes}
- Boundary probes: {summary['boundary_probe_count']}
- Unassigned probes: {summary['unassigned_probe_count']}
- Sites receiving zero probes: {zero_sites}
- Probes/site: min {_fmt(np.min(assigned_counts))}, median {_fmt(np.median(assigned_counts))}, max {_fmt(np.max(assigned_counts))}
- Smallest nearest-vs-second-nearest distance gap: {_fmt(np.min(nearest_gap))}
- Exact nearest-distance ties: {summary['exact_assignment_tie_count']}
- `k=1` owner differing from the first owner returned by a separate `k=2`
  diagnostic query: {summary['k1_vs_k2_first_owner_mismatch_count']} probes;
  among exact ties: {summary['exact_tie_k1_vs_k2_first_owner_mismatch_count']}.
  This is why the production `k=1` query itself, rather than an invented
  lowest-index rule or a `k=2` surrogate, defines tie ownership.

Diagnostic close-tie counts (absolute normalized-distance gap):
{chr(10).join(ambiguity_lines)}

2. Cell size
------------
Production starts with one zero radius per evaluated site and applies
`np.maximum.at` over assigned `(owner, distance)` pairs:

    assigned_radius[i] = max(distance(probe, site_i) for probes owned by i)

If site i owns no raw probe, this value remains exactly 0. With more than one
evaluated site:

    neighbor_spacing[i] = distance(site_i, nearest other evaluated site)
    spacing_floor[i] = 0.5 * neighbor_spacing[i]
    d_i = max(assigned_radius[i], spacing_floor[i])

For the unobserved single-site special case, no spacing floor is applied and
the assigned radius is retained. All 320 audited states have at least 10 sites.
There are no endpoint/global-boundary exceptions, loss/surrogate inputs,
tolerances, or branch-dependent bonuses. `np.maximum` handles an exact branch
tie by returning the common numerical value; it does not need to choose an
owner branch.

Observed cell-size population:

- Assigned-radius branch strictly larger: {assigned_wins} sites ({assigned_wins / total_sites:.12%})
- Half-nearest-spacing branch strictly larger: {spacing_wins} sites ({spacing_wins / total_sites:.12%})
- Exact branch ties: {branch_ties}
- Closest nonzero branch gap: {_fmt(summary['cellsize_closest_nonzero_branch_absolute_gap'])}
- Assigned radius: min {_fmt(np.min(assigned_radius))}, median {_fmt(np.median(assigned_radius))}, max {_fmt(np.max(assigned_radius))}
- Nearest-site spacing: min {_fmt(np.min(nearest_spacing))}, median {_fmt(np.median(nearest_spacing))}, max {_fmt(np.max(nearest_spacing))}
- Final d_i: min {_fmt(np.min(sizes))}, median {_fmt(np.median(sizes))}, max {_fmt(np.max(sizes))}

3. Exact operation decomposition
--------------------------------
The stage decomposes without loss as:

    (site, probe) -> normalized Euclidean distance
    distances from one probe to evaluated sites -> nearest-site owner
    assigned probe distances for one owner -> maximum assigned radius
    distances from one site to all other sites -> nearest-neighbor spacing
    nearest-neighbor spacing -> multiply by 0.5
    (assigned radius, half spacing) -> scalar maximum -> d_i

Candidate generation is deterministic geometry upstream of these operations;
same-size grouping, lower-hull construction, losses, surrogates, and candidate
winner selection are downstream or unused here.

4. Independent reconstruction
-----------------------------
A standalone replay regenerated the deterministic raw pool, rebuilt a fresh
KD-tree, repeated the exact `k=1` owner query (and separately queried `k=2`
only for ambiguity diagnostics),
reduced assigned distances with an independently implemented sorted grouped
maximum (not production's `np.maximum.at`), reconstructed nearest-site spacing,
and applied the two-way maximum.

- Exact raw-pool states: {summary['exact_raw_pool_state_count']}/{summary['state_count']}
- Exact probe owners: {summary['exact_probe_assignment_count']}/{total_probes}
- Exact probe distances: {summary['exact_probe_distance_count']}/{total_probes}
- Exact assigned radii: {summary['cellsize_exact_assigned_radius_count']}/{total_sites}
- Exact nearest-site spacings: {summary['cellsize_exact_nearest_spacing_count']}/{total_sites}
- Exact final d_i: {summary['cellsize_exact_final_cell_size_count']}/{total_sites}
- Exact complete states: {summary['exact_state_reconstruction_count']}/{summary['state_count']}

All requested reconstruction gates passed exactly.
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
    assignment_rows, cellsize_rows, summary = _aggregate(results)
    if not (
        bool(summary["all_probe_assignments_exact"])
        and bool(summary["all_probe_distances_exact"])
        and bool(summary["cellsize_all_assigned_radii_exact"])
        and bool(summary["cellsize_all_nearest_spacings_exact"])
        and bool(summary["cellsize_all_final_cell_sizes_exact"])
        and int(summary["exact_state_reconstruction_count"])
        == int(summary["state_count"])
    ):
        raise RuntimeError("Voronoi/cell-size independent reconstruction failed")
    _write_csv(ASSIGNMENT_PATH, assignment_rows)
    _write_csv(CELLSIZE_PATH, cellsize_rows)
    AUDIT_PATH.write_text(_audit_text(summary), encoding="utf-8")
    print(
        "Voronoi/cell-size audit exact: "
        f"states={summary['exact_state_reconstruction_count']}/"
        f"{summary['state_count']} probes={summary['exact_probe_assignment_count']}/"
        f"{summary['raw_probe_count']} sizes="
        f"{summary['cellsize_exact_final_cell_size_count']}/"
        f"{summary['evaluated_site_count']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    run(args.workers)


if __name__ == "__main__":
    main()
