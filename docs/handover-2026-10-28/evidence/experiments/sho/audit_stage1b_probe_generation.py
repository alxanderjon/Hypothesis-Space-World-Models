#!/usr/bin/env python3
"""Read-only probe generation reconstruction and source-removal audit."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc

import experiments.sho.audit_stage1b_voronoi_cellsize as prior
import experiments.sho.audit_stage1b_voronoi_tie_rule as downstream
import experiments.sho.validate_stage1b_learned_cell_size as locked_cells

base = locked_cells.base
edirect = prior.edirect
OUT = Path("outputs/sho/stage1B/probe_generation_audit")
REPORT = OUT / "probe_generation_audit.txt"
COUNTS = OUT / "probe_generation_counts.csv"
ROLES = OUT / "probe_generation_downstream_role.csv"


def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def row_view(a):
    return np.ascontiguousarray(a).view(np.dtype([("x", "<f8"), ("y", "<f8")])).reshape(-1)


def teacher_blocks(n, generator, iteration):
    captured = {}
    code = edirect._candidate_pool.__code__
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is code:
            for name in ("interior", "boundaries"):
                captured[name] = frame.f_locals[name].copy()
    try:
        sys.setprofile(observe)
        final = edirect._candidate_pool(n, generator, iteration)
    finally:
        sys.setprofile(previous)
    return final, captured["interior"], captured["boundaries"]


def independent(n, generator, iteration):
    requested = min(100*n, 32768)
    power = (max(2, requested)-1).bit_length()
    seed = 8730000 + 104729*generator + 1009*iteration
    # Independent recipe/control flow, retaining the exact installed Sobol
    # primitive. random(2**power) is an alternate API to random_base2(power).
    engine = qmc.Sobol(d=2, scramble=True, seed=seed)
    sobol = engine.random(n=1 << power)[:requested]
    k = min(257, max(9, math.isqrt(requested)+(math.isqrt(requested)**2 < requested)))
    grid = np.arange(k, dtype=np.float64)*(1.0/(k-1))
    grid[-1] = 1.0
    edges = np.empty((4*k, 2), dtype=np.float64)
    edges[:k, 0], edges[:k, 1] = grid, 0.0
    edges[k:2*k, 0], edges[k:2*k, 1] = grid, 1.0
    edges[2*k:3*k, 0], edges[2*k:3*k, 1] = 0.0, grid
    edges[3*k:, 0], edges[3*k:, 1] = 1.0, grid
    raw = np.concatenate((sobol, edges))
    tags = np.concatenate((np.ones(requested, dtype=np.uint8), np.repeat(np.array([2,4,8,16], dtype=np.uint8), k)))
    order = np.lexsort((np.arange(len(raw)), raw[:, 1], raw[:, 0]))
    ordered = raw[order]
    starts = np.r_[0, np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1))+1]
    first = order[starts]
    final = ordered[starts]
    membership = np.bitwise_or.reduceat(tags[order], starts)
    return final, sobol, edges, tags[first], membership, first, dict(
        requested_sobol=requested, generated_sobol_block=1 << power, discarded_sobol_tail=(1 << power)-requested,
        sobol_power=power, seed=seed, edge_points_per_edge=k, edge_step=1.0/(k-1),
        rng_type=type(engine.rng).__name__, bit_generator=type(engine.rng.bit_generator).__name__, sobol_bits=engine.bits)


def trace(sizes, losses):
    grouped = downstream.grouping_audit._teacher_trace(sizes, losses)
    hull = downstream._hull_site_ids(grouped)
    po = edirect._potentially_optimal_cells(sizes, losses)
    # Canonical member sets avoid falsely calling within-group ordering shifts
    # a change to same-size group membership.
    groups = [tuple(sorted(g["members"])) for g in grouped["groups"]]
    return grouped, hull, po, groups


def roles_for_state(state, split, sites, probes, mask, losses):
    tree = cKDTree(sites)
    distance, owners = tree.query(probes, k=1)
    spacing = tree.query(sites, k=2)[0][:, 1]
    n = len(sites)
    radius = np.zeros(n)
    np.maximum.at(radius, owners, distance)
    half = 0.5*spacing
    sizes = np.maximum(radius, half)
    baseline, hull, po, groups = trace(sizes, losses)
    counts = np.bincount(owners, minlength=n)
    farthest = distance == radius[owners]
    multiplicity = np.bincount(owners[farthest], minlength=n)
    witnesses = np.full(n, -1, dtype=np.int64)
    for q in np.flatnonzero(farthest):
        if witnesses[owners[q]] < 0:
            witnesses[owners[q]] = q
    # Individual deletion can change a radius only for a unique farthest probe.
    runner = np.zeros(n)
    nonmax = ~farthest
    np.maximum.at(runner, owners[nonmax], distance[nonmax])
    individual_radius_critical = farthest & (multiplicity[owners] == 1) & (distance > 0)
    individual_size_critical = individual_radius_critical & (np.maximum(runner[owners], half[owners]) < sizes[owners])
    boundary = (mask & 30) != 0
    corner = np.all((probes == 0) | (probes == 1), axis=1)
    sources = [("sobol", (mask & 1) != 0), ("boundary", boundary),
               ("bottom", (mask & 2) != 0), ("top", (mask & 4) != 0),
               ("left", (mask & 8) != 0), ("right", (mask & 16) != 0), ("corners", corner)]
    rows = []
    for name, source in sources:
        kept = ~source
        alternative_radius = np.zeros(n)
        np.maximum.at(alternative_radius, owners[kept], distance[kept])
        alternative_sizes = np.maximum(alternative_radius, half)
        changed = alternative_sizes != sizes
        alternative, alt_hull, alt_po, alt_groups = trace(alternative_sizes, losses)
        witness_source = source[witnesses[witnesses >= 0]]
        row = dict(record_type="source_state", source_state_index=state, split=split, source=name,
            probe_count=int(source.sum()), assigned_probe_count=int(source.sum()), unassigned_probe_count=0,
            cells_receiving_source=int(np.unique(owners[source]).size),
            co_max_probe_count=int(np.sum(source & farthest)), canonical_farthest_witness_count=int(witness_source.sum()),
            positive_radius_co_max_probe_count=int(np.sum(source & farthest & (distance > 0))),
            individually_radius_critical_probes=int(np.sum(source & individual_radius_critical)),
            individually_cell_size_critical_probes=int(np.sum(source & individual_size_critical)),
            co_max_probes_in_radius_dominant_cells=int(np.sum(source & farthest & (radius[owners] > half[owners]))),
            co_max_probes_in_radius_spacing_tied_cells=int(np.sum(source & farthest & (radius[owners] == half[owners]))),
            source_removal_radius_changes=int(np.sum(alternative_radius != radius)),
            source_removal_cell_size_changes=int(changed.sum()),
            source_removal_max_radius_error=float(np.max(np.abs(alternative_radius-radius))),
            source_removal_max_cell_size_error=float(np.max(np.abs(alternative_sizes-sizes))),
            source_removal_group_set_changes=len(set(groups)^set(alt_groups)),
            source_removal_group_membership_index_changes=int(np.sum(baseline["membership"] != alternative["membership"])),
            source_removal_representative_ordered_changes=downstream._ordered_position_change_count(baseline["representatives"], alternative["representatives"]),
            source_removal_hull_ordered_changes=downstream._ordered_position_change_count(hull, alt_hull),
            source_removal_hull_set_changes=downstream._symmetric_difference_count(hull, alt_hull),
            source_removal_PO_ordered_changes=downstream._ordered_position_change_count(po, alt_po),
            source_removal_PO_set_changes=downstream._symmetric_difference_count(po, alt_po))
        rows.append(row)
    return rows


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not any(p.exists() for p in (REPORT, COUNTS, ROLES))
    with locked_cells.HASHES.open() as f:
        protected = [Path(r["checkpoint_path"]) for r in csv.DictReader(f) if r["checkpoint_path"]]
    protected += [Path(inspect.getsourcefile(edirect)), Path(inspect.getsourcefile(qmc.Sobol))]
    hashes = {str(p):base._sha256(p) for p in protected}
    payload = torch.load(base.turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = base.ordinary_data._trajectory_map(payload)
    train = base.turn._stratified_state_indices(payload["states"], "train", 240, prior.smoke.DATA_SEED)
    held = base.turn._stratified_state_indices(payload["states"], "validation", 80, prior.smoke.DATA_SEED+1)
    indexed = [(i,"training") for i in train]+[(i,"held_out") for i in held]
    indexed.sort(key=lambda x:(payload["states"][x[0]]["generator_id"],payload["states"][x[0]]["iteration"]))
    count_rows, role_rows = [], []
    previous = None
    for position, (state, split) in enumerate(indexed, 1):
        record = payload["states"][state]
        trajectory = trajectories[int(record["trajectory_id"])]
        n, generator, iteration = int(record["history_count"]), int(record["generator_id"]), int(record["iteration"])
        sites = np.asarray(trajectory["evaluated_hypotheses_normalized"][:n], dtype=np.float64)
        losses = np.asarray(trajectory["evaluated_prefix_losses"][:n], dtype=np.float64)
        teacher, teacher_sobol, teacher_edges = teacher_blocks(n, generator, iteration)
        final, sobol, edges, first_tag, membership, first, recipe = independent(n, generator, iteration)
        raw = np.concatenate((teacher_sobol, teacher_edges))
        k = recipe["edge_points_per_edge"]
        raw_tags = np.r_[np.ones(len(teacher_sobol), dtype=np.uint8), np.repeat(np.array([2,4,8,16],dtype=np.uint8),k)]
        teacher_unique, teacher_first, teacher_inverse = np.unique(raw, axis=0, return_index=True, return_inverse=True)
        teacher_mask = np.zeros(len(teacher_unique), dtype=np.uint8)
        np.bitwise_or.at(teacher_mask, teacher_inverse, raw_tags)
        exact = (np.array_equal(final, teacher) and np.array_equal(sobol, teacher_sobol)
            and np.array_equal(edges, teacher_edges) and np.array_equal(teacher_mask, membership)
            and np.array_equal(first_tag, raw_tags[teacher_first]) and np.array_equal(first, teacher_first))
        assert exact, f"probe reconstruction failed in state {state}"
        # Bytes additionally verify ordering and exact float representation.
        assert final.tobytes() == teacher.tobytes()
        boundary = np.any((final == 0) | (final == 1), axis=1)
        corners = np.all((final == 0) | (final == 1), axis=1)
        matching = np.isin(row_view(final), row_view(sites))
        row = dict(record_type="state", split=split, source_state_index=state,
            trajectory_id=int(record["trajectory_id"]), generator_id=generator, iteration=iteration, evaluated_sites=n,
            **recipe, edge_grid_raw_count=len(edges), edge_unique_count=len(np.unique(edges,axis=0)),
            raw_combined_count=len(raw), duplicates_removed=len(raw)-len(final), final_probe_count=len(final),
            final_sobol_provenance_count=int(np.sum((membership&1)!=0)),
            final_edge_provenance_count=int(np.sum((membership&30)!=0)),
            cross_source_overlap_count=int(np.sum(((membership&1)!=0)&((membership&30)!=0))),
            canonical_sobol_fraction=float(np.mean(first_tag==1)), canonical_edge_fraction=float(np.mean(first_tag!=1)),
            raw_sobol_fraction=len(sobol)/len(raw), raw_edge_fraction=len(edges)/len(raw),
            boundary_probe_count=int(boundary.sum()), corner_probe_count=int(corners.sum()),
            sobol_on_boundary_count=int(np.sum(np.any((sobol==0)|(sobol==1),axis=1))),
            exactly_matching_evaluated_site_probes=int(matching.sum()),
            evaluated_sites_exactly_in_probe_pool=int(np.sum(np.isin(row_view(sites),row_view(final)))),
            ordering_changed_by_dedup=not np.all(first[1:] > first[:-1]), reconstruction_exact=exact,
            coordinates_order_sha256=digest_array(final), source_membership_sha256=digest_array(membership),
            first_source_label_sha256=digest_array(first_tag))
        if previous is not None and previous[0] == generator:
            _, previous_iteration, old_sobol, old_edges = previous
            row.update(previous_audited_iteration=previous_iteration,
                sobol_overlap_with_previous_audited_state=len(np.intersect1d(row_view(sobol),row_view(old_sobol))),
                edge_overlap_with_previous_audited_state=len(np.intersect1d(row_view(edges),row_view(old_edges))))
        previous = (generator, iteration, sobol, edges)
        count_rows.append(row)
        role_rows.extend(roles_for_state(state, split, sites, final, membership, losses))
        if position%40 == 0:
            print(f"probe generation audit {position}/320 states elapsed={time.monotonic()-started:.1f}s", flush=True)
    assert all(base._sha256(Path(p))==h for p,h in hashes.items())
    totals = []
    for split in ("all", "training", "held_out"):
        selected = count_rows if split=="all" else [r for r in count_rows if r["split"]==split]
        row = dict(record_type="summary",split=split,state_count=len(selected),exact_reconstruction_states=sum(r["reconstruction_exact"] for r in selected))
        for key in ("evaluated_sites","requested_sobol","generated_sobol_block","discarded_sobol_tail","edge_grid_raw_count",
                    "raw_combined_count","duplicates_removed","final_probe_count","final_sobol_provenance_count",
                    "final_edge_provenance_count","cross_source_overlap_count","boundary_probe_count","corner_probe_count",
                    "sobol_on_boundary_count","exactly_matching_evaluated_site_probes"):
            row[key] = sum(r[key] for r in selected)
        for key in ("requested_sobol","edge_points_per_edge","edge_grid_raw_count","final_probe_count","canonical_sobol_fraction"):
            values = np.asarray([r[key] for r in selected])
            row[key+"_min"],row[key+"_median"],row[key+"_max"] = float(values.min()),float(np.median(values)),float(values.max())
        row["final_sobol_fraction"] = row["final_sobol_provenance_count"]/row["final_probe_count"]
        row["final_edge_fraction"] = row["final_edge_provenance_count"]/row["final_probe_count"]
        row["ordering_changed_state_count"] = sum(r["ordering_changed_by_dedup"] for r in selected)
        repeated = [r for r in selected if "previous_audited_iteration" in r]
        row["adjacent_audited_state_pairs"] = len(repeated)
        row["observed_repeated_sobol_points"] = sum(r["sobol_overlap_with_previous_audited_state"] for r in repeated)
        row["observed_repeated_edge_points"] = sum(r["edge_overlap_with_previous_audited_state"] for r in repeated)
        totals.append(row)
    iteration_rows = []
    buckets = defaultdict(list)
    for r in count_rows:
        buckets[(r["split"],r["iteration"])].append(r["final_probe_count"])
    for (split,iteration),values in sorted(buckets.items()):
        iteration_rows.append(dict(record_type="iteration_distribution",split=split,iteration=iteration,state_count=len(values),
            unique_probe_count_min=min(values),unique_probe_count_median=float(np.median(values)),
            unique_probe_count_mean=float(np.mean(values)),unique_probe_count_max=max(values)))
    role_summaries = []
    for split in ("all","training","held_out"):
        for source in ("sobol","boundary","bottom","top","left","right","corners"):
            selected = [r for r in role_rows if r["source"]==source and (split=="all" or r["split"]==split)]
            row = dict(record_type="source_summary",split=split,source=source,state_count=len(selected))
            for key in selected[0]:
                if key not in ("record_type","split","source","source_state_index"):
                    row[key] = max(r[key] for r in selected) if key.startswith("source_removal_max_") else sum(r[key] for r in selected)
            for key in ("source_removal_cell_size_changes","source_removal_group_set_changes","source_removal_hull_set_changes","source_removal_PO_set_changes"):
                row[key+"_states"] = sum(r[key]>0 for r in selected)
            role_summaries.append(row)
    base._write_csv(COUNTS,totals+iteration_rows+count_rows)
    base._write_csv(ROLES,role_summaries+role_rows)
    detail = [r for r in role_summaries if r["split"]=="all" and r["source"] in ("sobol","boundary")]
    lines = ["RAW GEOMETRIC PROBE GENERATION AUDIT", "", "Scope and exact reconstruction",
        "Same 320 existing normal-eDIRECT cell states used by the Voronoi/cell-size audit: 240 training and 80 held-out. No search, training, or checkpoint mutation. Greedy and rescue variants are traced below but are not mixed into this normal-cell population.",
        f"Installed SciPy {scipy.__version__}; NumPy {np.__version__}. Protected inputs unchanged: {len(hashes)}.",
        "Production entry point: run_stage1b_edirect_vs_direct.py::_candidate_pool(history_count,generator_id,iteration), called by _cell_data before Voronoi assignment. The parameter named trajectory_id receives record['generator_id'], which can differ from the reporting trajectory_id.",
        "", "Sobol generation",
        "scipy.stats.qmc.Sobol(d=2,scramble=True,seed=seed).random_base2(exponent)[:requested].",
        "requested=min(100*number_of_evaluated_sites,32768); exponent=ceil(log2(max(2,requested))); actually draw 2**exponent points, retain the first requested and discard the tail.",
        "seed=8730000+104729*generator_id+1009*iteration. No modulo, global RNG draws, fast-forward, skipping, or persistent engine is used by this function. Every invocation starts a NEW engine at sequence index zero; every iteration changes its scramble seed.",
        "Current library defaults: bits=30, no optimization, float64 output in [0,1)^2. The installed seed path constructs a NumPy Generator/PCG64; Sobol uses direction numbers with LMS scrambling plus a digital shift. The shifted first point is included; points appear in the installed Sobol sequence order before merging.",
        "Repeating an identical (site_count,generator_id,iteration) call regenerates the same set. With the same seed and different count, retained Sobol points form prefixes; normal iteration changes seed so it does not continue the previous sequence. The schedule performs no explicit point reuse or caching. Observed coordinate overlaps between adjacent audited states are reported separately, not interpreted as continuation.",
        "", "Boundary probes",
        "k=min(257,max(9,ceil(sqrt(requested)))); t=np.linspace(0,1,k,dtype=float64), endpoints included, mathematical spacing 1/(k-1) rounded by float64 arithmetic.",
        "Raw edge block order: bottom (t,0), top (t,1), left (0,t), right (1,t); t ascends in each block. Every corner occurs twice in the raw edge blocks. There are 4*k raw edge probes and 4*k-4 distinct boundary coordinates.",
        "With current requested<=32768 the effective k range in these states is 32..182; the nominal 257 upper bound is not reached. Edge density changes through evaluated-site count until the Sobol cap, not directly through iteration, best point/loss, geometry, or surrogate.",
        "Normalized box is [0,1]^2. Normal-mode physical mapping outside this function is -1.99+3.98*x on each axis. Exact boundaries/corners are added regardless of current best location.",
        "", "Merge, equality, ordering and provenance",
        "Concatenate Sobol first, then the four edge blocks; production calls np.unique(combined,axis=0). Equality is exact coordinate-wise float64 value equality on these finite arrays, not a tolerance. The final rows are lexicographically sorted by x then y; this destroys the original Sobol/block ordering.",
        "Production returns coordinate VALUES only; it stores no duplicate identity or source labels. Equal rows are numerically indistinguishable. This audit uses np.unique(return_index=True)'s first original occurrence as canonical provenance, and independently retains ALL source memberships as a bitmask (Sobol=1,bottom=2,top=4,left=8,right=16). Corners therefore have multiple edge-source labels. If Sobol overlaps an edge, canonical first provenance is Sobol, while the membership records both.",
        "No evaluated-site exclusion, clipping, projection, or tolerance deduplication occurs in _candidate_pool. Coincident evaluated sites remain in the probe set and are counted exactly. The later proposal filter uses distance>1e-12 to avoid evaluating old sites; that is not raw-probe deduplication.",
        "Independent reconstruction uses the alternate Sobol .random(2**exponent) API, integer power/ceil-sqrt arithmetic, explicit arange*step edge construction, and lexsort plus adjacent exact row equality instead of np.unique. It shares the installed Sobol engine, not an independent implementation of its direction-number tables/scrambler.",
        "Teacher raw Sobol and edge blocks were observed read-only at function return. Reconstructed blocks, final coordinates, order, first-source provenance and full source-membership masks all matched. Source-label equality is against this instrumented derivation; production has no native persisted source labels to compare.",
        "Coordinates AND ordering were checked byte-for-byte on all 320 states including all 80 held-out states. Counts CSV includes coordinate-order and provenance SHA256 hashes per state.",
        "", "Counts and distributions", json.dumps(totals,indent=2),
        "Per-state counts, generated-versus-retained Sobol counts, exact overlaps with evaluated sites, and per-split/per-iteration unique-count distributions are in probe_generation_counts.csv.",
        "", "State dependence",
        "Iteration -> scramble seed; evaluated-site COUNT -> requested Sobol prefix, power-of-two draw size and boundary resolution; generator ID -> scramble seed. These are deterministic schedules conditioned on counters/identity, not geometric or loss-driven adaptation.",
        "Current best point/loss, site coordinates/Voronoi shape, and surrogate do not enter the normal raw generator. Search decisions can indirectly alter site count at a given iteration, but the recipe does not inspect those decisions.",
        "Each fresh Sobol engine consumes its newly seeded internal RNG for scrambling; no RNG/Sobol state survives between calls.",
        "Raw teacher traces are explicitly built from mode=='normal_edirect'. Boundary-rescue mode is outside this audited population. The related _run_boundary_rescue caller chooses an adaptive physical strip from the best point/local spacing, renormalizes in-strip history, and invokes _run_edirect_mode with iteration reset to zero. The SAME normalized raw-generator recipe remains unchanged; physical bounds/history can be adaptive outside it.",
        "The separate greedy stage calls _candidate_pool with iteration+1000000 and adds loss/geometry-dependent local candidates with clipping before another unique. These are NOT the raw probes used upstream of the audited Voronoi stage. The enhanced_edirect_boundary_rescue module also has a different generator; it is not the active recipe identified by these trace records.",
        "", "Downstream source roles and controlled source-removal ablations",
        "Every final probe receives one nearest-site owner. Roles count co-max probes (including equal maxima), canonical first farthest witnesses, unique individually radius-critical probes, and unique probes whose individual deletion would lower final d_i. Single-probe numeric effects are computed using the next remaining distance and unchanged half-spacing.",
        "For grouping/representative/hull/PO influence, remove ALL probes carrying one source membership, keep evaluated sites/losses/nearest spacing fixed, recompute radii/final sizes and replay the existing deterministic downstream teacher. Assignment of surviving probes cannot change since evaluated sites are fixed. These are whole-source counterfactuals, NOT per-probe leave-one-out causal counts.",
        "Sobol and boundary are the two main source classes. Bottom/top/left/right/corners rows are overlapping subsets of boundary (corners belong to two edges); do not sum all seven source categories. Co-max responsibility and individual-deletion necessity are distinct when maxima tie or the spacing floor dominates.",
        "Same-size group-set changes compare canonical member sets; membership-index changes also include positional shifts. Hull/PO set changes and ordered changes are reported separately. All downstream tolerances are the existing production rules; no new probe-equality tolerance is introduced.",
        json.dumps(detail,indent=2),
        "", "Smallest pieces (no architectures proposed)",
        "1. (site count, generator ID, iteration) -> requested count, draw exponent, scramble seed: deterministic schedule and arbitrary constants/bookkeeping.",
        "2. (dimension, direction-number tables, sequence index, scramble state) -> unit-box Sobol coordinates: sequence generation with mathematical low-discrepancy structure plus library RNG/table conventions.",
        "3. (closed box, edge resolution, grid index) -> edge coordinates and corners: mathematical geometry; density cap, edge enumeration and endpoint ordering are conventions.",
        "4. Source blocks -> concatenated rows/provenance: arbitrary ordering/bookkeeping.",
        "5. Two coordinate rows -> exact duplicate predicate: numerical coordinate equality; choosing tolerance-free float64 equality is the current numerical contract.",
        "6. Multiset of rows -> one copy per coordinate and lexicographic ordered unique set: deduplication, sorting, first-occurrence provenance conventions; provenance is audit metadata rather than production state.",
        "7. Optional normalized-to-physical bounds mapping (outside normal raw generation): geometric affine mapping; rescue bounds selection is a separate adaptive stage.",
        "No new component was trained, saved, or locked. Audit stops here.",
        f"Elapsed seconds: {time.monotonic()-started:.3f}", "", "Protected input hashes"]
    lines += [f"{h}  {p}" for p,h in sorted(hashes.items())]
    REPORT.write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(counts=totals[0],roles=detail),indent=2),flush=True)


if __name__ == "__main__":
    run()
