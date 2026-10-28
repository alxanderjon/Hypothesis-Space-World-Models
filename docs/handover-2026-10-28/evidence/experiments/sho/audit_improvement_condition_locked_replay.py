#!/usr/bin/env python3
"""Replay the locked improvement assembly and stop on its first boundary mismatch."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiments.sho.audit_stage1b_same_size_grouping as grouping_teacher
import experiments.sho.run_stage1b_edirect_vs_direct as production
import experiments.sho.run_stage1b_feasibility_threshold as feasibility_threshold
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_improvement_and as improvement
from experiments.sho.validate_candidate_voronoi_assignment import recovered_pool
from experiments.sho.validate_lower_hull_construction import learned_hull, load_hull
from experiments.sho.validate_representative_tie_selector import StrictLossSelector, fixed_grouping
from experiments.sho.validate_same_size_grouping_representative import load_fast_cells, load_grouping
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol


BASE = Path("outputs/sho/stage1B")
OUT = BASE / "improvement_condition_production_replay"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_and() -> nn.Linear:
    checkpoint = torch.load(improvement.AND_PATH, map_location="cpu", weights_only=False)
    assert checkpoint["status"] == "LOCKED_COMPONENT"
    model = nn.Linear(2, 1, bias=True).to(torch.float64)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)


def replay(kind: str, states: list[dict[str, object]], model: nn.Linear) -> dict[str, object]:
    pool = improvement._build_pool(states, kind)
    metrics = improvement._metrics(model, pool["inputs"], pool["teacher_pass"])
    prediction = metrics["prediction"]
    selected_errors = rejected_errors = payload_errors = 0
    fallback_states = 0
    for position, state in enumerate(states):
        mask = pool["state_position"] == position
        local_positions = pool["hull_position"][mask]
        local_prediction = prediction[mask]
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = np.asarray(state["sizes"], dtype=np.float64)
        losses = np.asarray(state["losses"], dtype=np.float64)
        reps = np.asarray(state["representatives"], dtype=np.int64)
        picked = [int(hull[int(p)]) for p, keep in zip(local_positions, local_prediction) if keep]
        if not picked:
            fallback_states += 1
            picked.append(int(np.argmax(sizes)))
        got = np.asarray(sorted(picked, key=lambda i: (sizes[i], losses[i], i)), dtype=np.int64)
        expected = production._potentially_optimal_cells(sizes, losses)
        if not np.array_equal(got, expected):
            selected_errors += 1
        expected_rejected = np.setdiff1d(hull, expected, assume_unique=True)
        got_rejected = np.setdiff1d(hull, got, assume_unique=True)
        if not np.array_equal(got_rejected, expected_rejected):
            rejected_errors += 1
        got_payload = (sizes[got], losses[got], reps[got])
        expected_payload = (sizes[expected], losses[expected], reps[expected])
        if not all(np.array_equal(a, b) for a, b in zip(got_payload, expected_payload)):
            payload_errors += 1
    return {
        "states": len(states), "events": len(pool["teacher_pass"]),
        "pass_events": int(np.sum(pool["teacher_pass"])),
        "fail_events": int(np.sum(~pool["teacher_pass"])),
        "endpoint_events": len(pool["endpoint_indices"]),
        "decision_errors": int(metrics["mistake_count"]),
        "selected_vertex_state_errors": selected_errors,
        "rejected_vertex_state_errors": rejected_errors,
        "downstream_scoring_input_state_errors": payload_errors,
        "fallback_states": fallback_states,
    }


def boundary_controls(model: nn.Linear) -> dict[str, object]:
    tau = turn.HULL_TOLERANCE
    upper = np.float64(1.0)
    lower_equal = np.float64(upper + tau)
    lower_values = np.asarray([
        np.nextafter(lower_equal, -np.inf),
        lower_equal,
        np.nextafter(lower_equal, np.inf),
    ])
    names = ("nextafter_feasible", "exact_equality", "nextafter_infeasible")
    scale = np.maximum.reduce((np.full(3, abs(upper)), np.abs(lower_values),
                               np.full(3, tau)))
    raw_margin = (upper + tau) - lower_values
    inputs = np.column_stack((np.full(3, upper) / scale,
                              lower_values / scale, np.full(3, tau) / scale))
    learned_margin = feasibility_threshold._frozen_margin({"inputs": inputs})
    checkpoint = torch.load(feasibility_threshold.THRESHOLD_PATH,
                            map_location="cpu", weights_only=False)
    theta = float(checkpoint["learned_theta"]); alpha = float(checkpoint["learned_alpha"])
    learned_feasible = alpha * (learned_margin - theta) >= 0.0
    teacher_feasible = raw_margin >= 0.0
    records = []
    for i, name in enumerate(names):
        records.append({
            "case": name, "K_upper": float(upper), "K_lower": float(lower_values[i]),
            "K_lower_bits": int(lower_values.view(np.uint64)[i]), "tau": tau,
            "raw_feasibility_margin": float(raw_margin[i]),
            "teacher_feasible": bool(teacher_feasible[i]),
            "learned_normalized_margin": float(learned_margin[i]),
            "learned_theta": theta, "learned_feasible": bool(learned_feasible[i]),
            "exact": bool(teacher_feasible[i] == learned_feasible[i]),
        })
    # The final learned AND is also checked exhaustively. Positivity is strict:
    # zero is represented by categorical false before entering this head.
    truth_inputs = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    with torch.no_grad():
        logits = model(torch.from_numpy(truth_inputs)).squeeze(-1).numpy()
    truth = np.asarray([False, False, False, True])
    return {
        "feasibility": records,
        "feasibility_errors": int(np.sum(learned_feasible != teacher_feasible)),
        "exact_equality_errors": int(records[1]["exact"] is False),
        "nextafter_errors": int(not records[0]["exact"]) + int(not records[2]["exact"]),
        "and_truth_table_errors": int(np.sum((logits >= 0.0) != truth)),
        "strict_positivity_contract": "K_upper==0 -> false; K_upper>0 -> true",
        "endpoint_contract": "+inf K_upper is positive and feasible; last hull endpoint passes",
    }


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    grouping, _ = load_grouping(); selector = StrictLossSelector(); hull = load_hull(); model = load_and()
    protected = [
        improvement.AND_PATH, feasibility_threshold.THRESHOLD_PATH,
        improvement.FEASIBILITY_MARGIN_PATH, improvement.SLOPE_SIGN_PATH,
        Path(hull["skew_path"]), Path(hull["homogeneous_path"]), Path(hull["threshold_path"]),
        BASE / "lower_hull_construction/lower_hull_construction_locked.json",
        BASE / "same_size_grouping_representative/same_size_grouping_representative_locked.json",
        Path(production.__file__),
    ]
    before = {str(p): sha(p) for p in protected}; dump(OUT / "protected_before.json", before)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_ids = turn._stratified_state_indices(payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED)
    held_ids = turn._stratified_state_indices(payload["states"], "validation", turn.VALIDATION_SOURCE_STATES, smoke.DATA_SEED + 1)
    raw_states = grouping_teacher._build_states(payload,
        [(i, "archived_training") for i in train_ids] + [(i, "held_out") for i in held_ids], workers=8)
    by_kind: dict[str, list[dict[str, object]]] = {"archived_training": [], "held_out": [], "fresh": []}
    for raw in raw_states:
        grouped, _ = fixed_grouping(raw["sizes"], raw["losses"], grouping, selector)
        h = learned_hull(grouped["grouped_sizes"], grouped["grouped_losses"], hull)
        by_kind[raw["split"]].append({
            "source_state_index": raw["source_state_index"], "trajectory_id": raw["trajectory_id"],
            "iteration": raw["iteration"], "state_bin": turn._state_bin(len(raw["sizes"])),
            "sizes": grouped["grouped_sizes"], "losses": grouped["grouped_losses"],
            "representatives": grouped["representatives"], "teacher_hull": h,
        })
    fast = load_fast_cells(); sobol = LearnedSobol(); rng = np.random.default_rng(12112026)
    for i, n in enumerate((6, 10, 23, 41, 64)):
        sites = rng.random((n, 2), dtype=np.float64)
        if i % 2 == 0: sites[:4] = np.asarray([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        losses = (sites[:,0]-.29)**2 + 1.3*(sites[:,1]-.67)**2 + np.arange(n)*1e-12 + 1e-8
        gid, iteration = 996001+i, 81+i
        probes = recovered_pool(n, gid, iteration, sobol); cell = fast.cells(sites, probes)
        truth = production._cell_data(sites, losses, gid, iteration)
        assert np.array_equal(cell["probe_owner"], truth.owners) and np.array_equal(cell["sizes"], truth.sizes)
        grouped, _ = fixed_grouping(cell["sizes"], losses, grouping, selector)
        h = learned_hull(grouped["grouped_sizes"], grouped["grouped_losses"], hull)
        by_kind["fresh"].append({"source_state_index": 900000+i, "trajectory_id": gid,
            "iteration": iteration, "state_bin": turn._state_bin(n), "sizes": grouped["grouped_sizes"],
            "losses": grouped["grouped_losses"], "representatives": grouped["representatives"], "teacher_hull": h})
    replay_results = {kind: replay(kind, states, model) for kind, states in by_kind.items()}
    controls = boundary_controls(model)
    after = {str(p): sha(p) for p in protected}; changes = [p for p in before if before[p] != after[p]]
    real_errors = sum(sum(v[k] for k in ("decision_errors", "selected_vertex_state_errors",
        "rejected_vertex_state_errors", "downstream_scoring_input_state_errors")) for v in replay_results.values())
    passed = real_errors == 0 and controls["feasibility_errors"] == 0 and controls["and_truth_table_errors"] == 0 and not changes
    result = {
        "status": "LOCKED_ASSEMBLY" if passed else "STOPPED_BOUNDARY_MISMATCH",
        "teacher_rule": {
            "target": "f_min - 1e-4*abs(f_min)",
            "K_prev": "-inf at first vertex, else (f_i-f_prev)/(d_i-d_prev)",
            "K_next": "+inf at last vertex, else (f_next-f_i)/(d_next-d_i)",
            "K_required": "(f_i-target)/d_i if d_i>0 else +inf",
            "K_lower": "max(K_prev,K_required,0)",
            "pass": "(K_next > 0) AND (K_lower <= K_next + 5e-14)",
            "fallback": "if no PASS, first np.argmax(size); then sort by (size,loss,index)",
        },
        "existing_operator": {"status": "LOCKED", "and_checkpoint": str(improvement.AND_PATH),
            "and_sha256": sha(improvement.AND_PATH), "new_training": False},
        "replay": replay_results, "boundary_controls": controls,
        "protected_changes": changes, "seconds": time.time()-started,
        "promotion_written": False,
        "next_operation": "STOP: repair/review feasibility boundary before cell scoring",
    }
    dump(OUT / "results.json", result)
    print(result["status"], json.dumps({"replay": replay_results, "controls": controls}, sort_keys=True))


if __name__ == "__main__": main()
