#!/usr/bin/env python3
"""Validate only recovered-pool -> production Voronoi candidate assignment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as production
import experiments.sho.run_stage1b_nearest_margin as nearest_base
import experiments.sho.run_stage1b_nearest_margin_symmetry as nearest_margin
import experiments.sho.run_stage1b_nearest_threshold as nearest_threshold
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data
import experiments.sho.validate_stage1b_nearest_argmin as nearest_locked
from experiments.sho.fast_nearest_argmin import FastNearestArgmin
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol, schedule


BASE = Path("outputs/sho/stage1B")
OUT = BASE / "candidate_voronoi_assignment"
COUNTS = BASE / "probe_generation_audit/probe_generation_counts.csv"
RAW_TRACE = nearest_base.turn.RAW_TRACE_PATH


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def recovered_pool(sample_count: int, generator_id: int, iteration: int,
                   sobol: LearnedSobol) -> np.ndarray:
    requested = min(production.MC_FACTOR_PER_SITE * sample_count,
                    production.MC_CANDIDATE_CAP)
    exponent, _ = schedule(requested)
    seed = production.SEARCH_SEED + 104729 * generator_id + 1009 * iteration
    block, _, _, _, _ = sobol.full_block(2, exponent, seed)
    interior = block[:requested]
    edge_count = min(257, max(9, int(math.ceil(math.sqrt(requested)))))
    t = np.linspace(0.0, 1.0, edge_count, dtype=np.float64)
    z, o = np.zeros_like(t), np.ones_like(t)
    boundary = np.concatenate((np.column_stack((t, z)), np.column_stack((t, o)),
                               np.column_stack((z, t)), np.column_stack((o, t))))
    return np.unique(np.concatenate((interior, boundary)), axis=0)


def load_fast() -> FastNearestArgmin:
    distance, _ = nearest_base._load_locked_distance()
    op = torch.load(nearest_locked.OPERATOR, map_location="cpu", weights_only=False)
    margin = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    margin.load_state_dict(torch.load(op["margin_checkpoint"], map_location="cpu",
                                      weights_only=False)["state_dict"])
    threshold = nearest_threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(op["threshold_checkpoint"], map_location="cpu",
                                         weights_only=False)["state_dict"])
    for model in (distance, margin, threshold):
        model.eval().requires_grad_(False)
    return FastNearestArgmin(distance, margin, threshold, half=None)


def memberships(owners: np.ndarray, site_count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    counts = np.bincount(owners, minlength=site_count).astype(np.int64)
    # This is the exact logical representation used downstream by
    # np.flatnonzero(owners == site_index): original candidate order is retained.
    members = [np.flatnonzero(owners == site).astype(np.int64) for site in range(site_count)]
    return counts, members


def validate_case(kind: str, case_id: str, sites: np.ndarray, probes: np.ndarray,
                  fast: FastNearestArgmin) -> dict[str, object]:
    sites = np.asarray(sites, dtype=np.float64)
    probes = np.asarray(probes, dtype=np.float64)
    tree = cKDTree(sites)
    teacher_distance, teacher_owner = tree.query(probes, k=1)
    two_distance = tree.query(probes, k=2)[0]
    got = fast.nearest(sites, probes)
    owner = got["probe_owner"]
    distance = got["probe_distance"]
    teacher_counts, teacher_members = memberships(teacher_owner, len(sites))
    counts, members = memberships(owner, len(sites))
    member_errors = sum(not np.array_equal(a, b) for a, b in zip(members, teacher_members))
    empty_errors = sum((len(a) == 0) != (len(b) == 0)
                       for a, b in zip(members, teacher_members))
    row: dict[str, object] = dict(
        kind=kind, case_id=case_id, site_count=len(sites), candidate_count=len(probes),
        assignment_errors=int(np.sum(owner != teacher_owner)),
        distance_errors=int(np.sum(distance != teacher_distance)),
        count_errors=int(np.sum(counts != teacher_counts)),
        membership_list_errors=int(member_errors), empty_cell_errors=int(empty_errors),
        empty_cells=int(np.sum(counts == 0)),
        exact_ties=int(np.sum(two_distance[:, 0] == two_distance[:, 1])),
        one_ulp_nearest_competitions=0,
        downstream_owner_input_errors=int(np.sum(owner != teacher_owner)),
        downstream_distance_input_errors=int(np.sum(distance != teacher_distance)),
        owner_dtype=str(owner.dtype), count_dtype=str(counts.dtype),
        member_dtype="int64", tie_scaffold_calls=int(got["stats"]["kd_query_calls"] - 2),
    )
    row["passed"] = not any(int(row[key]) for key in (
        "assignment_errors", "distance_errors", "count_errors", "membership_list_errors",
        "empty_cell_errors", "downstream_owner_input_errors", "downstream_distance_input_errors"))
    if not row["passed"]:
        raise AssertionError(f"assignment mismatch: {row}")
    return row


def targeted_controls(fast: FastNearestArgmin) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    half = np.float64(0.5)
    left = np.nextafter(half, np.float64(0.0))
    right = np.nextafter(half, np.float64(1.0))
    patterns = [
        ("horizontal", np.array([[0., 0.], [1., 0.], [.2, .8]]),
         np.array([[half, 0.], [left, 0.], [right, 0.], [half, .25]])),
        ("horizontal_reversed", np.array([[1., 0.], [0., 0.], [.2, .8]]),
         np.array([[half, 0.], [left, 0.], [right, 0.], [half, .25]])),
        ("vertical", np.array([[0., 0.], [0., 1.], [.8, .2]]),
         np.array([[0., half], [0., left], [0., right], [.25, half]])),
        ("diagonal", np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]),
         np.array([[half, half], [left, half], [right, half], [half, left], [half, right]])),
    ]
    tie_count = one_ulp_count = 0
    for name, sites, probes in patterns:
        row = validate_case("targeted_control", name, sites, probes, fast)
        tree = cKDTree(sites)
        d = tree.query(probes, k=2)[0]
        tie_count += int(np.sum(d[:, 0] == d[:, 1]))
        # These are intentionally one coordinate ULP to either side of a bisector.
        one_ulp = int(np.sum(np.any((probes == left) | (probes == right), axis=1)))
        one_ulp_count += one_ulp
        row["one_ulp_nearest_competitions"] = one_ulp
        rows.append(row)
    summary = dict(cases=len(rows), candidates=sum(int(r["candidate_count"]) for r in rows),
                   exact_ties=tie_count, one_ulp_neighbor_controls=one_ulp_count,
                   assignment_errors=sum(int(r["assignment_errors"]) for r in rows),
                   distance_errors=sum(int(r["distance_errors"]) for r in rows))
    return rows, summary


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "candidate_voronoi_assignment_locked.json"
    if lock.exists():
        raise RuntimeError("Refusing to overwrite existing lock")
    torch.set_num_threads(1)
    protected = [nearest_locked.OPERATOR,
                 BASE / "nearest_argmin_assembled/nearest_argmin_operator_locked.sha256",
                 nearest_base.LOCKED_DISTANCE_PATH,
                 nearest_margin.CANDIDATE_PATH,
                 BASE / "nearest_argmin_assembled/nearest_distance_threshold_frozen_candidate.pt",
                 BASE / "n1_boundary_merge_dedup/n1_boundary_merge_dedup_locked.json",
                 Path(production.__file__)]
    before = {str(path): sha(path) for path in protected}
    dump("protected_before.json", before)

    fast = load_fast()
    sobol = LearnedSobol()
    payload = torch.load(RAW_TRACE, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    with COUNTS.open() as handle:
        all_rows = [row for row in csv.DictReader(handle) if row.get("source_state_index")]

    def choose(split: str, target: int) -> dict[str, str]:
        candidates = [row for row in all_rows if row["split"] == split]
        return min(candidates, key=lambda row: abs(int(row["requested_sobol"]) - target))

    cases: list[dict[str, object]] = []
    for split in ("training", "held_out"):
        for target in (1000, 8600, 32768):
            record = choose(split, target)
            state = int(record["source_state_index"])
            source = payload["states"][state]
            n = int(source["history_count"])
            trajectory = trajectories[int(source["trajectory_id"])]
            sites = np.asarray(trajectory["evaluated_hypotheses_normalized"][:n], dtype=np.float64)
            probes = recovered_pool(n, int(source["generator_id"]), int(source["iteration"]), sobol)
            teacher_pool = production._candidate_pool(n, int(source["generator_id"]), int(source["iteration"]))
            if not np.array_equal(probes.view(np.uint64), teacher_pool.view(np.uint64)):
                raise AssertionError("recovered candidate pool differs from production")
            cases.append(validate_case("archived_" + split, str(state), sites, probes, fast))

    rng = np.random.default_rng(29092026)
    for index, n in enumerate((6, 10, 23, 41, 64)):
        sites = rng.random((n, 2), dtype=np.float64)
        # Retain deterministic boundary sites in several fresh states.
        if index % 2 == 0:
            sites[:4] = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        generator_id, iteration = 970001 + index, 31 + index
        probes = recovered_pool(n, generator_id, iteration, sobol)
        teacher_pool = production._candidate_pool(n, generator_id, iteration)
        if not np.array_equal(probes.view(np.uint64), teacher_pool.view(np.uint64)):
            raise AssertionError("fresh recovered candidate pool differs from production")
        cases.append(validate_case("fresh", f"fresh_{index}", sites, probes, fast))

    controls, control_summary = targeted_controls(fast)
    cases.extend(controls)

    # Immutable prior full-population and held-out learned-operator evidence.
    with (BASE / "voronoi_cellsize_audit/voronoi_assignment_distribution.csv").open() as handle:
        historical_summary = next(row for row in csv.DictReader(handle)
                                  if row["record_type"] == "summary")
    with (BASE / "nearest_argmin_assembled/nearest_argmin_assembled.csv").open() as handle:
        nearest_rows = [row for row in csv.DictReader(handle) if row["record_type"] == "scan_summary"]
    held_probe = next(row for row in nearest_rows if row["kind"] == "probe")

    after = {str(path): sha(path) for path in protected}
    changes = [path for path, digest in before.items() if after[path] != digest]
    if changes:
        raise AssertionError(f"protected files changed: {changes}")
    totals = dict(
        current_replay_cases=len(cases),
        current_replay_candidates=sum(int(row["candidate_count"]) for row in cases),
        current_replay_sites=sum(int(row["site_count"]) for row in cases),
        current_assignment_errors=sum(int(row["assignment_errors"]) for row in cases),
        current_distance_errors=sum(int(row["distance_errors"]) for row in cases),
        current_membership_errors=sum(int(row["membership_list_errors"]) for row in cases),
        current_count_errors=sum(int(row["count_errors"]) for row in cases),
        current_empty_cell_errors=sum(int(row["empty_cell_errors"]) for row in cases),
        current_downstream_input_errors=sum(int(row["downstream_owner_input_errors"]) +
                                            int(row["downstream_distance_input_errors"])
                                            for row in cases),
        historical_states=int(historical_summary["state_count"]),
        historical_candidates=int(historical_summary["raw_probe_count"]),
        historical_exact_assignments=int(historical_summary["exact_probe_assignment_count"]),
        historical_exact_ties=int(historical_summary["exact_assignment_tie_count"]),
        historical_empty_cells=int(historical_summary["site_with_zero_assigned_probe_count"]),
        held_out_learned_candidates=int(float(held_probe["query_count"])),
        held_out_learned_exact_owners=int(float(held_probe["exact_owners"])),
        held_out_learned_owner_errors=int(float(held_probe["final_argmin_mistakes"])),
        targeted_exact_ties=control_summary["exact_ties"],
        targeted_one_ulp_neighbors=control_summary["one_ulp_neighbor_controls"],
    )
    passed = (all(value == 0 for key, value in totals.items() if key.endswith("errors"))
              and totals["historical_exact_assignments"] == totals["historical_candidates"]
              and totals["held_out_learned_exact_owners"] == totals["held_out_learned_candidates"]
              and not changes)
    result = dict(
        status="LOCKED_ASSEMBLY" if passed else "FAILED",
        name="candidate_voronoi_assignment",
        new_parameters=0,
        production_rule="scipy.spatial.cKDTree(float64 sites).query(float64 candidates,k=1)",
        metric="ordinary normalized-coordinate Euclidean distance; cKDTree compiled evaluation order",
        tie_rule="exact ties inherit the owner returned by the production k=1 cKDTree query",
        ordering="site indices are input row positions; candidate order is recovered-pool order",
        membership="owners int64 array; cell i members are flatnonzero(owners==i), preserving candidate order",
        empty_cell="count 0 and empty int64 member-index array",
        reused_locked_components=dict(
            distance_operator=str(nearest_base.LOCKED_DISTANCE_PATH),
            nearest_argmin_operator=str(nearest_locked.OPERATOR),
            nearest_argmin_sha256=sha(nearest_locked.OPERATOR),
        ),
        deterministic_scaffolding=dict(
            exact_tie_detection="equality of k=2 cKDTree distances",
            exact_tie_owner="conditional production k=1 query already declared by locked nearest operator",
            membership_indexing="np.bincount + np.flatnonzero in original candidate order",
            note="No new tie router was trained; changing this internal contract would modify a locked operator.",
        ),
        totals=totals, protected_changes=changes, seconds=time.time() - started,
        next_operation="same-size grouping / representative construction (not run)",
    )
    if not passed:
        raise AssertionError(result)
    dump("case_validation.json", cases)
    dump("targeted_controls.json", control_summary)
    dump("results.json", result)
    lock.write_text(json.dumps(dict(
        status="LOCKED_ASSEMBLY", name="candidate_voronoi_assignment",
        result_sha256=sha(OUT / "results.json"), implementation_sha256=sha(Path(__file__)),
        upstream_pool_manifest=str(BASE / "n1_boundary_merge_dedup/n1_boundary_merge_dedup_locked.json"),
        nearest_operator=str(nearest_locked.OPERATOR), nearest_operator_sha256=sha(nearest_locked.OPERATOR),
        new_checkpoint=None,
    ), indent=2) + "\n")
    print("PASS", json.dumps(totals, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
