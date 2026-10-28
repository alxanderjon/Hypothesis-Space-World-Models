#!/usr/bin/env python3
"""Fix only representative strict-loss comparison via locked subtraction/sign."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import experiments.sho.audit_stage1b_same_size_grouping as grouping_teacher
import experiments.sho.run_stage1b_edirect_vs_direct as production
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_nearest_margin as nearest_base
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data
import experiments.sho.validate_stage1b_distance_assembled as learned_grouping
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_subtraction_operator import Subtraction
from experiments.sho.validate_candidate_voronoi_assignment import recovered_pool
from experiments.sho.validate_same_size_grouping_representative import load_fast_cells, load_grouping
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol


BASE = Path("outputs/sho/stage1B")
OUT = BASE / "representative_tie_selector"
COMPOSED = BASE / "same_size_grouping_representative"
SUB_MANIFEST = BASE / "subtraction_operator_training/subtraction_operator_locked.json"
ZERO_MANIFEST = BASE / "stationary_magnitude_threshold_smoke/zero_anchor_converged/candidate_manifest.json"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


class StrictLossSelector:
    """REPLACE exactly when locked subtraction returns a negative state."""
    def __init__(self) -> None:
        sub_manifest = json.loads(SUB_MANIFEST.read_text())
        zero_manifest = json.loads(ZERO_MANIFEST.read_text())
        assert sub_manifest["status"] == "LOCKED" and sub_manifest["passed"]
        assert zero_manifest["status"] == "FROZEN_CANDIDATE" and zero_manifest["passed"]
        self.sub_path = Path(sub_manifest["checkpoint"])
        self.zero_path = Path(zero_manifest["checkpoint"])
        assert sha(self.sub_path) == sub_manifest["sha256"]
        assert sha(self.zero_path) == zero_manifest["sha256"]
        self.subtraction = Subtraction()
        self.subtraction.load_state_dict(torch.load(self.sub_path, map_location="cpu",
                                                    weights_only=False)["state_dict"])
        self.threshold = MagnitudeThreshold()
        self.threshold.load_state_dict(torch.load(self.zero_path, map_location="cpu",
                                                  weights_only=False)["state_dict"])
        self.subtraction.eval().requires_grad_(False)
        self.threshold.eval().requires_grad_(False)

    def decide(self, best: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pairs = np.column_stack((candidate, best)).astype(np.float64, copy=False)
        with torch.no_grad():
            delta = self.subtraction(torch.from_numpy(pairs))
            state = self.threshold.decision(delta)
        return state.numpy() < 0, delta.numpy(), state.numpy()


def fixed_grouping(sizes: np.ndarray, losses: np.ndarray, locked: dict[str, object],
                   selector: StrictLossSelector) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    order = np.lexsort((np.arange(len(sizes)), losses, sizes))
    groups: list[dict[str, object]] = []
    membership = np.full(len(sizes), -1, dtype=np.int64)
    comparisons: list[dict[str, object]] = []
    for value in order:
        index = int(value); size = float(sizes[index]); loss = float(losses[index])
        if not groups:
            groups.append(dict(anchor=size, representative=index, representative_loss=loss))
            membership[index] = 0
            continue
        group = groups[-1]
        normalized_gap = abs(size - float(group["anchor"])) / 1e-14
        margin = locked["same_margin_weight"] * normalized_gap + locked["same_margin_bias"]
        if margin < locked["same_theta"]:
            groups.append(dict(anchor=size, representative=index, representative_loss=loss))
            membership[index] = len(groups) - 1
            continue
        membership[index] = len(groups) - 1
        best_index = int(group["representative"]); best = float(losses[best_index])
        replace, delta, state = selector.decide(np.array([best]), np.array([loss]))
        teacher_replace = loss < best
        comparisons.append(dict(best=best, candidate=loss, delta=float(delta[0]),
                                state=int(state[0]), learned_replace=bool(replace[0]),
                                teacher_replace=teacher_replace,
                                exact=bool(replace[0]) == teacher_replace))
        if replace[0]:
            group["representative"] = index; group["representative_loss"] = loss
    result = dict(
        order=order.astype(np.int64), membership=membership,
        grouped_sizes=np.asarray([float(g["anchor"]) for g in groups], dtype=np.float64),
        grouped_losses=np.asarray([float(g["representative_loss"]) for g in groups], dtype=np.float64),
        representatives=np.asarray([int(g["representative"]) for g in groups], dtype=np.int64),
    )
    return result, comparisons


def validate_state(kind: str, identity: str, sizes: np.ndarray, losses: np.ndarray,
                   empty: np.ndarray, locked: dict[str, object], selector: StrictLossSelector) -> tuple[dict[str, object], list[dict[str, object]]]:
    truth = grouping_teacher._teacher_trace(sizes, losses)
    got, comparisons = fixed_grouping(sizes, losses, locked, selector)
    fields = ("order", "membership", "grouped_sizes", "grouped_losses", "representatives")
    errors = {field + "_errors": int(np.sum(got[field] != truth[field])) for field in fields}
    row: dict[str, object] = dict(
        kind=kind, case_id=identity, cells=len(sizes), groups=len(truth["groups"]),
        comparisons=len(comparisons), replaces=sum(c["teacher_replace"] for c in comparisons),
        exact_tie_comparisons=sum(c["candidate"] == c["best"] for c in comparisons),
        comparison_errors=sum(not c["exact"] for c in comparisons), empty_cells=int(np.sum(empty)),
        empty_membership_errors=int(np.sum(got["membership"][empty] != truth["membership"][empty])),
        downstream_hull_input_errors=(errors["grouped_sizes_errors"] + errors["grouped_losses_errors"] +
                                      errors["representatives_errors"]), **errors)
    row["passed"] = not any(int(v) for k, v in row.items() if k.endswith("_errors"))
    if not row["passed"]: raise AssertionError(row)
    return row, comparisons


def boundary_controls(selector: StrictLossSelector) -> dict[str, object]:
    tiny = np.nextafter(np.float64(0), np.float64(1))
    values = np.array([0., -0., tiny, -tiny, 1e-300, -1e-300, 1., -1., 1e100, -1e100,
                       np.finfo(float).max / 2, -np.finfo(float).max / 2], dtype=np.float64)
    best, candidate, labels = [], [], []
    for value in values:
        for item in (np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf)):
            if np.isfinite(item):
                best.append(value); candidate.append(item); labels.append(item < value)
    rng = np.random.default_rng(30102026)
    broad = np.ldexp(rng.uniform(1, 2, 50000), rng.integers(-1000, 1000, 50000))
    broad *= rng.choice([-1., 1.], len(broad))
    broad_candidate = broad.copy()
    direction = rng.choice([-np.inf, np.inf], len(broad))
    steps = rng.integers(0, 5, len(broad))
    for _ in range(4):
        mask = steps > _
        broad_candidate[mask] = np.nextafter(broad_candidate[mask], direction[mask])
    finite = np.isfinite(broad_candidate)
    best = np.r_[best, broad[finite]]; candidate = np.r_[candidate, broad_candidate[finite]]
    labels = np.r_[labels, broad_candidate[finite] < broad[finite]]
    prediction, delta, state = selector.decide(np.asarray(best), np.asarray(candidate))
    native = np.asarray(candidate) - np.asarray(best)
    errors = prediction != labels
    equality = np.asarray(candidate) == np.asarray(best)
    one_ulp = (~equality) & ((np.asarray(candidate) == np.nextafter(np.asarray(best), np.inf)) |
                             (np.asarray(candidate) == np.nextafter(np.asarray(best), -np.inf)))
    return dict(cases=len(prediction), errors=int(errors.sum()), equality_cases=int(equality.sum()),
                equality_errors=int(np.sum(errors & equality)), one_ulp_cases=int(one_ulp.sum()),
                one_ulp_errors=int(np.sum(errors & one_ulp)), subtraction_bit_errors=int(np.sum(delta.view('u8') != native.view('u8'))),
                zero_state_errors=int(np.sum(equality & (state != 0))), negative_state_errors=int(np.sum((native < 0) & (state != -1))),
                positive_state_errors=int(np.sum((native > 0) & (state != 1))))


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); COMPOSED.mkdir(parents=True, exist_ok=True)
    selector_lock = OUT / "representative_tie_selector_locked.json"
    composed_lock = COMPOSED / "same_size_grouping_representative_locked.json"
    if selector_lock.exists() or composed_lock.exists(): raise RuntimeError("Refusing to overwrite lock")
    torch.set_num_threads(1)
    selector = StrictLossSelector(); locked, checkpoint_status = load_grouping()
    protected = [selector.sub_path, selector.zero_path,
                 Path(checkpoint_status["same_size_margin"]["path"]), Path(checkpoint_status["same_size_threshold"]["path"]),
                 Path(checkpoint_status["representative_margin"]["path"]), Path(checkpoint_status["representative_threshold"]["path"]),
                 BASE / "candidate_voronoi_assignment/candidate_voronoi_assignment_locked.json",
                 BASE / "learned_cell_size_assembled/cell_size_operator_locked.pt", Path(production.__file__)]
    before = {str(path): sha(path) for path in protected}; dump(OUT / "protected_before.json", before)
    # The immediately preceding failed composition replay used exact upstream
    # candidate ownership/counts and explicitly tracked empty cells. Preserve
    # that immutable evidence because the full 320-state audit stores sizes but
    # not the discarded pre-floor empty-cell count.
    prior_rows_path = COMPOSED / "case_validation.json"
    prior_rows = json.loads(prior_rows_path.read_text())
    empty_evidence = dict(
        states=sum(row["kind"] in ("archived_training", "archived_held_out", "fresh") for row in prior_rows),
        empty_cells=sum(int(row["empty_cell_count"]) for row in prior_rows
                        if row["kind"] in ("archived_training", "archived_held_out", "fresh")),
        empty_membership_errors=sum(int(row["empty_cell_membership_errors"]) for row in prior_rows
                                    if row["kind"] in ("archived_training", "archived_held_out", "fresh")),
        source=str(prior_rows_path), source_sha256=sha(prior_rows_path),
    )

    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = turn._stratified_state_indices(payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED)
    held_indices = turn._stratified_state_indices(payload["states"], "validation", turn.VALIDATION_SOURCE_STATES, smoke.DATA_SEED + 1)
    indexed = [(index, "archived_training") for index in train_indices] + [(index, "held_out") for index in held_indices]
    states = grouping_teacher._build_states(payload, [(i, k) for i, k in indexed], workers=8)
    rows: list[dict[str, object]] = []; comparisons: list[dict[str, object]] = []
    for state in states:
        row, comp = validate_state(state["split"], str(state["source_state_index"]), state["sizes"], state["losses"],
                                   np.zeros(len(state["sizes"]), dtype=bool), locked, selector)
        rows.append(row); comparisons.extend(comp)

    # Fresh states pass through the newly locked pool, assignment, and cell-size path.
    fast = load_fast_cells(); sobol = LearnedSobol(); rng = np.random.default_rng(31102026)
    for index, n in enumerate((6, 10, 23, 41, 64)):
        sites = rng.random((n, 2), dtype=np.float64)
        if index % 2 == 0: sites[:4] = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        losses = (sites[:, 0] - .37) ** 2 + 1.4 * (sites[:, 1] - .59) ** 2 + np.arange(n) * 1e-12 + 1e-8
        generator, iteration = 990001 + index, 51 + index
        probes = recovered_pool(n, generator, iteration, sobol); cells = fast.cells(sites, probes)
        truth_cell = production._cell_data(sites, losses, generator, iteration)
        assert np.array_equal(cells["probe_owner"], truth_cell.owners) and np.array_equal(cells["sizes"], truth_cell.sizes)
        counts = np.bincount(cells["probe_owner"], minlength=n)
        row, comp = validate_state("fresh", f"fresh_{index}", cells["sizes"], losses, counts == 0, locked, selector)
        rows.append(row); comparisons.extend(comp)

    # Exact downstream control that failed before the fix.
    control_sizes = np.array([.4, .4, .4]); control_losses = np.array([1., 1., 1.])
    row, comp = validate_state("exact_tie_control", "three_equal", control_sizes, control_losses,
                               np.zeros(3, bool), locked, selector)
    rows.append(row); comparisons.extend(comp)
    controls = boundary_controls(selector)

    after = {str(path): sha(path) for path in protected}; changes = [p for p, h in before.items() if after[p] != h]
    def aggregate(kind: str) -> dict[str, int]:
        selected = [row for row in rows if row["kind"] == kind]
        return dict(states=len(selected), cells=sum(int(r["cells"]) for r in selected), groups=sum(int(r["groups"]) for r in selected),
                    comparisons=sum(int(r["comparisons"]) for r in selected), exact_ties=sum(int(r["exact_tie_comparisons"]) for r in selected),
                    errors=sum(int(r["comparison_errors"]) + int(r["order_errors"]) + int(r["membership_errors"]) +
                               int(r["grouped_sizes_errors"]) + int(r["grouped_losses_errors"]) + int(r["representatives_errors"]) +
                               int(r["empty_membership_errors"]) + int(r["downstream_hull_input_errors"]) for r in selected))
    metrics = {kind: aggregate(kind) for kind in ("archived_training", "held_out", "fresh", "exact_tie_control")}
    passed = (all(item["errors"] == 0 for item in metrics.values())
              and all(value == 0 for key, value in controls.items() if key.endswith("errors"))
              and empty_evidence["empty_cells"] > 0 and empty_evidence["empty_membership_errors"] == 0
              and not changes)
    if not passed: raise AssertionError(dict(metrics=metrics, controls=controls, changes=changes))
    result = dict(status="LOCKED_ASSEMBLY", name="representative_tie_selector", new_parameters=0,
                  teacher_rule="REPLACE iff candidate_loss < current_best_loss; exact equality KEEP",
                  assembly=["locked subtraction(candidate,current)", "frozen zero-anchored sign threshold", "REPLACE only NEGATIVE; ZERO/POSITIVE KEEP"],
                  subtraction=dict(path=str(selector.sub_path), sha256=sha(selector.sub_path), status="LOCKED"),
                  zero_threshold=dict(path=str(selector.zero_path), sha256=sha(selector.zero_path), status="FROZEN_CANDIDATE reused by locked sign operator"),
                  metrics=metrics, boundary_controls=controls, empty_cell_validation=empty_evidence,
                  total_comparisons=len(comparisons),
                  total_comparison_errors=sum(not c["exact"] for c in comparisons), protected_changes=changes,
                  seconds=time.time()-started, next_operation="lower-hull construction (not run)")
    dump(OUT / "case_validation.json", rows); dump(OUT / "results.json", result)
    selector_lock.write_text(json.dumps(dict(status="LOCKED_ASSEMBLY", name="representative_tie_selector",
        result_sha256=sha(OUT / "results.json"), implementation_sha256=sha(Path(__file__)), new_checkpoint=None,
        subtraction_sha256=sha(selector.sub_path), zero_threshold_sha256=sha(selector.zero_path)), indent=2) + "\n")
    composed = dict(status="LOCKED_ASSEMBLY", name="same_size_grouping_representative", new_parameters=0,
                    grouping_components=checkpoint_status, representative_tie_selector=str(selector_lock),
                    representative_tie_selector_sha256=sha(selector_lock), validation=result,
                    next_operation="lower-hull construction (not run)")
    dump(COMPOSED / "fixed_results.json", composed)
    composed_lock.write_text(json.dumps(dict(status="LOCKED_ASSEMBLY", name="same_size_grouping_representative",
        fixed_results_sha256=sha(COMPOSED / "fixed_results.json"), implementation_sha256=sha(Path(__file__)),
        representative_tie_selector_sha256=sha(selector_lock), new_checkpoint=None), indent=2) + "\n")
    print("PASS", json.dumps(dict(metrics=metrics, controls=controls, comparisons=len(comparisons)), sort_keys=True), flush=True)


if __name__ == "__main__": main()
