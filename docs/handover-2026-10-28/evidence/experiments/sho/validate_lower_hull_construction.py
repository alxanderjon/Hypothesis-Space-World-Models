#!/usr/bin/env python3
"""Validate the existing locked lower-hull assembly on exact upstream inputs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import experiments.sho.audit_stage1b_same_size_grouping as grouping_teacher
import experiments.sho.run_stage1b_edirect_vs_direct as production
import experiments.sho.run_stage1b_hull_keep_pop_threshold as hull_operator
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
from experiments.sho.validate_candidate_voronoi_assignment import recovered_pool
from experiments.sho.validate_representative_tie_selector import StrictLossSelector, fixed_grouping
from experiments.sho.validate_same_size_grouping_representative import load_fast_cells, load_grouping
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol


BASE = Path("outputs/sho/stage1B")
OUT = BASE / "lower_hull_construction"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_hull() -> dict[str, object]:
    skew, skew_hash = hull_operator._load_skew()
    homogeneous = torch.load(hull_operator.HOMOGENEOUS_CHECKPOINT_PATH,
                             map_location="cpu", weights_only=False)
    threshold = torch.load(hull_operator.THRESHOLD_CHECKPOINT_PATH,
                           map_location="cpu", weights_only=False)
    assert homogeneous["lock_status"] == "LOCKED"
    assert threshold["lock_status"] == "LOCKED"
    return {
        "skew": skew,
        "skew_path": hull_operator.SKEW_CHECKPOINT_PATH,
        "skew_hash": skew_hash,
        "homogeneous": homogeneous["effective_weight"].numpy().copy(),
        "homogeneous_path": hull_operator.HOMOGENEOUS_CHECKPOINT_PATH,
        "homogeneous_hash": sha(hull_operator.HOMOGENEOUS_CHECKPOINT_PATH),
        "theta": float(threshold["learned_theta"]),
        "alpha": float(threshold["learned_alpha"]),
        "threshold_path": hull_operator.THRESHOLD_CHECKPOINT_PATH,
        "threshold_hash": sha(hull_operator.THRESHOLD_CHECKPOINT_PATH),
    }


def teacher_hull(sizes: np.ndarray, losses: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # This helper is an exact extraction of production's stack loop, including
    # cross-product evaluation order and equality KEEP behavior.
    return turn._teacher_hull_and_triples(sizes, losses)


def learned_hull(sizes: np.ndarray, losses: np.ndarray,
                 locked: dict[str, object]) -> np.ndarray:
    return hull_operator._threshold_hull(
        locked["skew"], locked["homogeneous"], locked["theta"],
        locked["alpha"], sizes, losses)


def check_hull(kind: str, identity: str, sizes: np.ndarray, losses: np.ndarray,
               representatives: np.ndarray, locked: dict[str, object]) -> dict[str, object]:
    expected, triples = teacher_hull(sizes, losses)
    actual = learned_hull(sizes, losses, locked)
    expected_sites = representatives[expected]
    actual_sites = representatives[actual]
    # These are the complete inputs passed to the improvement stage. The stage
    # itself is deliberately not evaluated here.
    expected_payload = (sizes[expected], losses[expected], expected_sites)
    actual_payload = (sizes[actual], losses[actual], actual_sites)
    membership_exact = np.array_equal(actual, expected)
    sites_exact = np.array_equal(actual_sites, expected_sites)
    payload_exact = membership_exact and all(
        np.array_equal(a, b) for a, b in zip(actual_payload, expected_payload)
    )
    return {
        "kind": kind, "case_id": identity, "representatives": len(sizes),
        "teacher_hull_vertices": len(expected), "learned_hull_vertices": len(actual),
        "turn_decisions": len(triples["label"]),
        "teacher_keep": int(np.sum(triples["label"])),
        "teacher_pop": int(np.sum(1 - triples["label"])),
        "exact_collinear_turns": int(np.sum(triples["exact_collinear"])),
        "near_collinear_turns": int(np.sum(triples["near_collinear"])),
        "hull_membership_and_order_exact": membership_exact,
        "representative_site_indices_exact": sites_exact,
        "downstream_improvement_inputs_exact": payload_exact,
        "teacher_hull": expected.tolist(), "learned_hull": actual.tolist(),
        "passed": membership_exact and sites_exact and payload_exact,
    }


def controls(grouping: dict[str, object], selector: StrictLossSelector,
             hull: dict[str, object]) -> list[dict[str, object]]:
    # All cases pass through the locked grouping/representative assembly first,
    # so duplicate size coordinates have production semantics before the hull.
    u = np.nextafter
    cases = [
        ("exact_horizontal_collinear", [0., .25, .5, .75], [1., 1., 1., 1.]),
        ("exact_positive_equal_slope", [0., .25, .5, .75], [0., .25, .5, .75]),
        ("exact_negative_equal_slope", [0., .25, .5, .75], [1., .75, .5, .25]),
        ("duplicate_coordinates_values", [.1, .1, .3, .3, .7], [2., 2., 1., 1., .5]),
        ("duplicate_coordinate_strict_loss", [.1, .1, .3, .6], [2., 1., 1., .5]),
        ("duplicate_values_unique_coordinates", [.1, .2, .4, .8], [1., 1., 1., .5]),
        ("nextafter_above_collinear", [0., .25, .5], [0., .25, u(.5, np.inf)]),
        ("nextafter_below_collinear", [0., .25, .5], [0., .25, u(.5, -np.inf)]),
        ("equal_slope_large_scale", [0., 1., 2., 3.], [1e100, 1e100+1e84, 1e100+2e84, 1e100+3e84]),
    ]
    rows: list[dict[str, object]] = []
    for name, s, l in cases:
        sizes = np.asarray(s, np.float64); losses = np.asarray(l, np.float64)
        grouped, _ = fixed_grouping(sizes, losses, grouping, selector)
        row = check_hull("targeted_control", name, grouped["grouped_sizes"],
                         grouped["grouped_losses"], grouped["representatives"], hull)
        row["raw_cells"] = len(sizes)
        rows.append(row)
    return rows


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "lower_hull_construction_locked.json"
    if lock_path.exists():
        raise RuntimeError("Refusing to overwrite lower-hull lock")
    torch.set_num_threads(1)
    grouping, grouping_status = load_grouping()
    selector = StrictLossSelector()
    hull = load_hull()
    protected = [
        Path(hull["skew_path"]), Path(hull["homogeneous_path"]),
        Path(hull["threshold_path"]), selector.sub_path, selector.zero_path,
        BASE / "same_size_grouping_representative/same_size_grouping_representative_locked.json",
        BASE / "candidate_voronoi_assignment/candidate_voronoi_assignment_locked.json",
        BASE / "learned_cell_size_assembled/cell_size_operator_locked.pt",
        Path(production.__file__),
    ]
    before = {str(p): sha(p) for p in protected}
    dump(OUT / "protected_before.json", before)

    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_ids = turn._stratified_state_indices(payload["states"], "train",
                                                turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED)
    held_ids = turn._stratified_state_indices(payload["states"], "validation",
                                               turn.VALIDATION_SOURCE_STATES, smoke.DATA_SEED + 1)
    indexed = [(i, "archived_training") for i in train_ids] + [(i, "held_out") for i in held_ids]
    states = grouping_teacher._build_states(payload, indexed, workers=8)
    rows: list[dict[str, object]] = []
    for state in states:
        grouped, _ = fixed_grouping(state["sizes"], state["losses"], grouping, selector)
        rows.append(check_hull(state["split"], str(state["source_state_index"]),
                               grouped["grouped_sizes"], grouped["grouped_losses"],
                               grouped["representatives"], hull))

    fast = load_fast_cells(); sobol = LearnedSobol(); rng = np.random.default_rng(11112026)
    for index, n in enumerate((6, 10, 23, 41, 64)):
        sites = rng.random((n, 2), dtype=np.float64)
        if index % 2 == 0:
            sites[:4] = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        losses = ((sites[:, 0] - .31) ** 2 + 1.7 * (sites[:, 1] - .61) ** 2
                  + np.arange(n) * 1e-12 + 1e-8)
        generator, iteration = 995001 + index, 71 + index
        probes = recovered_pool(n, generator, iteration, sobol)
        cell = fast.cells(sites, probes)
        truth_cell = production._cell_data(sites, losses, generator, iteration)
        assert np.array_equal(cell["probe_owner"], truth_cell.owners)
        assert np.array_equal(cell["sizes"], truth_cell.sizes)
        grouped, _ = fixed_grouping(cell["sizes"], losses, grouping, selector)
        rows.append(check_hull("fresh", f"fresh_{index}", grouped["grouped_sizes"],
                               grouped["grouped_losses"], grouped["representatives"], hull))
    rows.extend(controls(grouping, selector, hull))

    after = {str(p): sha(p) for p in protected}
    changes = [p for p in before if before[p] != after[p]]
    kinds = ("archived_training", "held_out", "fresh", "targeted_control")
    summary = {}
    for kind in kinds:
        subset = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            "states": len(subset),
            "representatives": sum(int(r["representatives"]) for r in subset),
            "hull_vertices": sum(int(r["teacher_hull_vertices"]) for r in subset),
            "turn_decisions": sum(int(r["turn_decisions"]) for r in subset),
            "exact_collinear_turns": sum(int(r["exact_collinear_turns"]) for r in subset),
            "near_collinear_turns": sum(int(r["near_collinear_turns"]) for r in subset),
            "errors": sum(not bool(r["passed"]) for r in subset),
        }
    passed = all(v["errors"] == 0 for v in summary.values()) and not changes
    result = {
        "status": "LOCKED_ASSEMBLY" if passed else "FAILED",
        "name": "lower_hull_construction", "new_parameters": 0,
        "teacher_rule": {
            "input_order": "increasing grouped size; inherited canonical representative order",
            "orientation": "lower-right hull",
            "turn": "cross=(d2-d1)*(f3-f1)-(f2-f1)*(d3-d1)",
            "pop": "cross < -5e-14*max(1,abs(f1),abs(f2),abs(f3))",
            "keep": "equality/collinearity and all other turns",
            "stack_order": "append increasing group index; pop middle until KEEP",
            "provenance": "hull group index maps to locked representative site index",
        },
        "components": {
            "bilinear_turn": {"status": "LOCKED", "path": str(hull["skew_path"]), "sha256": hull["skew_hash"]},
            "homogeneous_margin": {"status": "LOCKED", "path": str(hull["homogeneous_path"]), "sha256": hull["homogeneous_hash"]},
            "keep_pop_threshold": {"status": "LOCKED", "path": str(hull["threshold_path"]), "sha256": hull["threshold_hash"]},
        },
        "summary": summary, "protected_changes": changes,
        "seconds": time.time() - started,
        "next_operation": "improvement-condition evaluation (not run)",
    }
    dump(OUT / "case_validation.json", rows)
    dump(OUT / "results.json", result)
    if not passed:
        raise AssertionError(result)
    lock_path.write_text(json.dumps({
        "status": "LOCKED_ASSEMBLY", "name": "lower_hull_construction",
        "new_checkpoint": None, "new_parameters": 0,
        "results_sha256": sha(OUT / "results.json"),
        "implementation_sha256": sha(Path(__file__)),
        "component_hashes": {k: v["sha256"] for k, v in result["components"].items()},
    }, indent=2) + "\n")
    print("PASS", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
