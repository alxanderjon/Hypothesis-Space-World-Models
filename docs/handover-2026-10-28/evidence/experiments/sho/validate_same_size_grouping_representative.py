#!/usr/bin/env python3
"""Validate only production same-size grouping and representative construction."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiments.sho.audit_stage1b_same_size_grouping as teacher
import experiments.sho.run_stage1b_edirect_vs_direct as production
import experiments.sho.run_stage1b_nearest_margin as nearest_base
import experiments.sho.run_stage1b_nearest_margin_symmetry as nearest_margin
import experiments.sho.run_stage1b_nearest_threshold as nearest_threshold
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data
import experiments.sho.run_stage1b_representative_margin as representative_margin
import experiments.sho.run_stage1b_representative_threshold as representative_threshold
import experiments.sho.run_stage1b_same_size_margin as same_margin
import experiments.sho.run_stage1b_same_size_threshold as same_threshold
import experiments.sho.validate_stage1b_distance_assembled as learned
import experiments.sho.validate_stage1b_nearest_argmin as nearest_locked
from experiments.sho.fast_nearest_argmin import FastNearestArgmin
from experiments.sho.validate_candidate_voronoi_assignment import recovered_pool
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol


BASE = Path("outputs/sho/stage1B")
OUT = BASE / "same_size_grouping_representative"
COUNTS = BASE / "probe_generation_audit/probe_generation_counts.csv"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def load_fast_cells() -> FastNearestArgmin:
    distance, _ = nearest_base._load_locked_distance()
    op = torch.load(nearest_locked.OPERATOR, map_location="cpu", weights_only=False)
    margin = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    margin.load_state_dict(torch.load(op["margin_checkpoint"], map_location="cpu",
                                      weights_only=False)["state_dict"])
    threshold = nearest_threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(op["threshold_checkpoint"], map_location="cpu",
                                         weights_only=False)["state_dict"])
    half_checkpoint = torch.load(BASE / "learned_cell_size_assembled/half_spacing_operator_locked.pt",
                                 map_location="cpu", weights_only=False)
    half = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    half.load_state_dict(half_checkpoint["state_dict"])
    for model in (distance, margin, threshold, half):
        model.eval().requires_grad_(False)
    return FastNearestArgmin(distance, margin, threshold, half)


def load_grouping() -> tuple[dict[str, object], dict[str, object]]:
    sm = torch.load(same_margin.CANDIDATE_PATH, map_location="cpu", weights_only=False)
    st = torch.load(same_threshold.THRESHOLD_PATH, map_location="cpu", weights_only=False)
    rm = torch.load(representative_margin.CANDIDATE_PATH, map_location="cpu", weights_only=False)
    rt = torch.load(representative_threshold.THRESHOLD_PATH, map_location="cpu", weights_only=False)
    assert sm["status"] == "FROZEN_CANDIDATE"
    assert st["status"] == "LOCKED_COMPONENT" and st["complete_same_size_grouping_operator_locked"]
    assert rm["status"] == "FROZEN_CANDIDATE"
    assert rt["status"] == "LOCKED_COMPONENT"
    assert rt["representative_selection_operator_locked"]
    assert rt["same_size_grouping_and_representative_selection_locked"]
    locked = dict(
        same_margin_weight=float(sm["state_dict"]["weight"].item()),
        same_margin_bias=float(sm["state_dict"]["bias"].item()),
        same_theta=float(st["learned_theta"].item()),
        representative_margin_weight=rm["state_dict"]["weight"].detach().cpu().numpy()[0].copy(),
        representative_theta=float(rt["learned_theta"].item()),
    )
    status = dict(
        same_size_margin=dict(path=str(same_margin.CANDIDATE_PATH), status=sm["status"],
                              sha256=sha(same_margin.CANDIDATE_PATH)),
        same_size_threshold=dict(path=str(same_threshold.THRESHOLD_PATH), status=st["status"],
                                 sha256=sha(same_threshold.THRESHOLD_PATH)),
        representative_margin=dict(path=str(representative_margin.CANDIDATE_PATH), status=rm["status"],
                                   sha256=sha(representative_margin.CANDIDATE_PATH)),
        representative_threshold=dict(path=str(representative_threshold.THRESHOLD_PATH), status=rt["status"],
                                      sha256=sha(representative_threshold.THRESHOLD_PATH)),
        composed_status="LOCKED",
    )
    return locked, status


def validate_values(kind: str, case_id: str, sizes: np.ndarray, losses: np.ndarray,
                    empty: np.ndarray, locked: dict[str, object]) -> dict[str, object]:
    truth = teacher._teacher_trace(sizes, losses)
    got = learned._learned_group_and_represent(sizes, losses, locked)
    fields = ("order", "membership", "grouped_sizes", "grouped_losses", "representatives")
    errors = {name + "_errors": int(np.sum(got[name] != truth[name])) for name in fields}
    # Group identity is the member set. Encounter order is already checked by
    # the exact global `order` array; flatnonzero intentionally returns site-index order.
    truth_groups = [tuple(sorted(int(x) for x in group["members"])) for group in truth["groups"]]
    got_groups = [tuple(int(x) for x in np.flatnonzero(got["membership"] == group))
                  for group in range(len(got["grouped_sizes"]))]
    group_identity_errors = sum(a != b for a, b in zip(got_groups, truth_groups)) + abs(len(got_groups)-len(truth_groups))
    empty_membership_errors = int(np.sum(got["membership"][empty] != truth["membership"][empty]))
    downstream_errors = (errors["grouped_sizes_errors"] + errors["grouped_losses_errors"] +
                         errors["representatives_errors"])
    row: dict[str, object] = dict(
        kind=kind, case_id=case_id, cell_count=len(sizes), group_count=len(truth_groups),
        singleton_groups=sum(len(group) == 1 for group in truth_groups),
        multicell_groups=sum(len(group) > 1 for group in truth_groups),
        empty_cell_count=int(np.sum(empty)), empty_cell_membership_errors=empty_membership_errors,
        group_identity_errors=group_identity_errors, downstream_input_errors=downstream_errors,
        **errors,
    )
    if errors["representatives_errors"]:
        row["teacher_representatives"] = truth["representatives"].tolist()
        row["learned_representatives"] = got["representatives"].tolist()
    row["passed"] = not any(int(value) for key, value in row.items() if key.endswith("_errors"))
    return row


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "same_size_grouping_representative_locked.json"
    if lock_path.exists():
        raise RuntimeError("Refusing to overwrite existing lock")
    torch.set_num_threads(1)
    locked, checkpoint_status = load_grouping()
    protected = [Path(item["path"]) for key, item in checkpoint_status.items() if isinstance(item, dict)]
    protected += [nearest_locked.OPERATOR,
                  BASE / "candidate_voronoi_assignment/candidate_voronoi_assignment_locked.json",
                  BASE / "learned_cell_size_assembled/cell_size_operator_locked.pt",
                  Path(production.__file__)]
    before = {str(path): sha(path) for path in protected}
    dump("protected_before.json", before)

    fast = load_fast_cells()
    sobol = LearnedSobol()
    payload = torch.load(nearest_base.turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    with COUNTS.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("source_state_index")]

    def choose(split: str, target: int) -> dict[str, str]:
        values = [row for row in rows if row["split"] == split]
        return min(values, key=lambda row: abs(int(row["requested_sobol"]) - target))

    results: list[dict[str, object]] = []
    for split in ("training", "held_out"):
        for target in (1000, 8600, 32768):
            source_row = choose(split, target)
            state_index = int(source_row["source_state_index"])
            state = payload["states"][state_index]
            n = int(state["history_count"])
            trajectory = trajectories[int(state["trajectory_id"])]
            sites = np.asarray(trajectory["evaluated_hypotheses_normalized"][:n], dtype=np.float64)
            losses = np.asarray(trajectory["evaluated_prefix_losses"][:n], dtype=np.float64)
            probes = recovered_pool(n, int(state["generator_id"]), int(state["iteration"]), sobol)
            cells = fast.cells(sites, probes)
            counts = np.bincount(cells["probe_owner"], minlength=n)
            # The locked upstream path must be bit-exact before grouping is assessed.
            teacher_cell = production._cell_data(sites, losses, int(state["generator_id"]), int(state["iteration"]))
            assert np.array_equal(cells["probe_owner"], teacher_cell.owners)
            assert np.array_equal(cells["sizes"], teacher_cell.sizes)
            results.append(validate_values("archived_" + split, str(state_index), cells["sizes"],
                                           losses, counts == 0, locked))

    rng = np.random.default_rng(30092026)
    for index, n in enumerate((6, 10, 23, 41, 64)):
        sites = rng.random((n, 2), dtype=np.float64)
        if index % 2 == 0:
            sites[:4] = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        losses = ((sites[:, 0] - .31) ** 2 + 1.7 * (sites[:, 1] - .63) ** 2 +
                  np.arange(n, dtype=np.float64) * 1e-12 + 1e-6)
        generator_id, iteration = 980001 + index, 41 + index
        probes = recovered_pool(n, generator_id, iteration, sobol)
        cells = fast.cells(sites, probes)
        counts = np.bincount(cells["probe_owner"], minlength=n)
        tree_cell = production._cell_data(sites, losses, generator_id, iteration)
        assert np.array_equal(cells["probe_owner"], tree_cell.owners)
        assert np.array_equal(cells["sizes"], tree_cell.sizes)
        results.append(validate_values("fresh", f"fresh_{index}", cells["sizes"], losses,
                                       counts == 0, locked))

    # Target exact threshold sides, anchor chaining, and representative loss ties.
    tol = np.float64(1e-14)
    synthetic = [
        ("same_boundary", np.array([.25, .25 + tol]), np.array([2., 1.])),
        ("same_inside", np.array([.25, np.nextafter(.25 + tol, .25)]), np.array([2., 1.])),
        ("new_outside", np.array([.25, np.nextafter(.25 + tol, 1.)]), np.array([2., 1.])),
        ("anchor_no_chain", np.array([.25, .25 + .75*tol, .25 + 1.5*tol]), np.array([3., 2., 1.])),
        ("loss_tie", np.array([.4, .4, .4]), np.array([1., 1., 1.])),
    ]
    for name, sizes, losses in synthetic:
        results.append(validate_values("targeted_control", name, sizes, losses,
                                       np.zeros(len(sizes), dtype=bool), locked))

    # Prior immutable population summaries establish full archived coverage.
    with (BASE / "same_size_grouping_audit/same_size_grouping_distribution.csv").open() as handle:
        audit_summary = next(row for row in csv.DictReader(handle) if row["record_type"] == "overall_summary")
    with (BASE / "representative_selection_threshold/representative_threshold_full_pool.csv").open() as handle:
        representative_summary = next(row for row in csv.DictReader(handle) if row["record_type"] == "summary")

    after = {str(path): sha(path) for path in protected}
    changes = [path for path, digest in before.items() if after[path] != digest]
    totals = dict(
        current_cases=len(results), current_cells=sum(int(row["cell_count"]) for row in results),
        current_groups=sum(int(row["group_count"]) for row in results),
        current_empty_cells=sum(int(row["empty_cell_count"]) for row in results),
        current_group_identity_errors=sum(int(row["group_identity_errors"]) for row in results),
        current_group_order_errors=sum(int(row["grouped_sizes_errors"]) for row in results),
        current_representative_errors=sum(int(row["representatives_errors"]) for row in results),
        current_empty_cell_errors=sum(int(row["empty_cell_membership_errors"]) for row in results),
        current_downstream_input_errors=sum(int(row["downstream_input_errors"]) for row in results),
        historical_states=int(audit_summary["state_count"]),
        historical_cells=int(audit_summary["total_cells_entering_stage"]),
        historical_groups=int(audit_summary["total_same_size_groups"]),
        historical_exact_states=int(audit_summary["state_level_exact_reconstruction"]),
        historical_exact_representative_groups=int(float(representative_summary["all_exact_final_representative_groups"])),
        historical_representative_groups=int(float(representative_summary["all_group_count"])),
        historical_representative_decision_errors=int(float(representative_summary["all_replacement_decision_changes"])),
    )
    errors = [value for key, value in totals.items() if key.endswith("errors")]
    passed = (not any(errors) and totals["historical_states"] == totals["historical_exact_states"]
              and totals["historical_exact_representative_groups"] == totals["historical_representative_groups"]
              and not changes)
    result = dict(
        status="LOCKED_ASSEMBLY" if passed else "FAILED", name="same_size_grouping_representative", new_parameters=0,
        checkpoint_status=checkpoint_status,
        teacher_rule=dict(
            ordering="np.lexsort((original_site_index, loss, size)): increasing size, then loss, then site index",
            same_size="abs(size-anchor_size) <= 1e-14; anchor is first cell in current group",
            representative="strictly lowest float64 loss; equality keeps first cell in sorted order",
            empty_cells="no special case; every evaluated site participates using its locked final cell size",
            hull_interface="group anchor sizes, representative losses, representative original site indices",
        ),
        totals=totals, protected_changes=changes, seconds=time.time()-started,
        next_operation="lower-hull construction (not run)",
    )
    dump("case_validation.json", results)
    dump("results.json", result)
    if not passed:
        failures = [row for row in results if not row["passed"]]
        dump("failures.json", failures)
        print("FAILED", json.dumps(dict(totals=totals, failures=failures), sort_keys=True), flush=True)
        return
    lock_path.write_text(json.dumps(dict(
        status="LOCKED_ASSEMBLY", name="same_size_grouping_representative",
        result_sha256=sha(OUT / "results.json"), implementation_sha256=sha(Path(__file__)),
        grouping_threshold_sha256=sha(same_threshold.THRESHOLD_PATH),
        representative_threshold_sha256=sha(representative_threshold.THRESHOLD_PATH),
        upstream_assignment_manifest=str(BASE / "candidate_voronoi_assignment/candidate_voronoi_assignment_locked.json"),
        new_checkpoint=None,
    ), indent=2) + "\n")
    print("PASS", json.dumps(totals, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
