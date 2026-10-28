"""Generate offline E-DIRECT handoff teacher data at 100-evaluation checkpoints.

Teacher parameters are used only to label containment and stop this offline
teacher-data run. They are never passed to proposal generation or scoring.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as base
from experiments.sho.deterministic_full_k_framework import deterministic_finish, hashes, prior


OUT = Path("outputs/sho/stage1B/checkpointed_resource_edirect")
CASES = OUT / "cases"
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
FULL_SOURCE = Path("outputs/sho/stage1B/resource_limited_edirect")
EARLY_SOURCE = Path("outputs/sho/stage1B/continuous_surrogate_edirect")
BLOCK = 100
CAP = 5_000
LOSS_GATE = 1.0e-4


def _atomic_text(path: Path, value: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _outside_distance(teacher, lower, upper):
    coordinate = np.maximum(np.maximum(lower - teacher, teacher - upper), 0.0)
    return coordinate, float(np.linalg.norm(coordinate))


def _checkpoint(points, losses, teacher, lower, upper, case_id, iteration, traces):
    count = len(losses)
    best = int(np.argmin(losses))
    theta = lower + (upper - lower) * points[best]
    radius = (upper - lower) / count ** (1.0 / len(lower))
    box_lower = np.maximum(lower, theta - radius)
    box_upper = np.minimum(upper, theta + radius)
    outside, outside_l2 = _outside_distance(teacher, box_lower, box_upper)
    loss_ok = bool(losses[best] <= LOSS_GATE)
    containment_ok = bool(outside_l2 == 0.0)

    api = base.composition(points.shape[1])
    cell = api["_cell_data"](points, losses, case_id, iteration)
    owned = cell.points[cell.owners == best]
    support = np.vstack((points[best], owned))
    return dict(
        evaluation=count, iteration=iteration, best_index=best,
        best_theta=theta.tolist(), best_loss=float(losses[best]),
        loss_ok=loss_ok, containment_ok=containment_ok,
        both_ok=bool(loss_ok and containment_ok),
        box_lower=box_lower.tolist(), box_upper=box_upper.tolist(),
        teacher_outside_coordinate_distance=outside.tolist(),
        teacher_outside_l2=outside_l2,
        incumbent_cell_extent=float(cell.sizes[best]),
        incumbent_owned_probe_count=int(len(owned)),
        incumbent_is_potentially_optimal=bool(best in cell.selected),
        potentially_optimal_cell_count=int(len(cell.selected)),
        candidate_pool_count=int(len(cell.points)),
        incumbent_support_lower=support.min(axis=0).tolist(),
        incumbent_support_upper=support.max(axis=0).tolist(),
        continuous_calls=len(traces),
        continuous_duplicate_rejections=int(sum(t["exact_duplicate_rejected"] for t in traces)),
        continuous_surrogate_evaluations=int(sum(
            start["nfev"] for trace in traces for start in trace["starts"]
        )),
    )


def _first_loss_crossing(events):
    best = math.inf
    for event in events:
        best = min(best, float(event["loss"]))
        if best <= LOSS_GATE:
            return dict(
                evaluation=int(event["evaluation"]), iteration=int(event["iteration"]),
                best_loss=best,
            )
    return None


def _case_objectives(states):
    x = states[:-1]
    y = states[1:]
    torch_x = torch.from_numpy(x.copy())
    torch_y = torch.from_numpy(y.copy())

    def objective(theta):
        matrix = np.asarray(theta).reshape(2, 2)
        return float(np.mean((np.einsum("ij,nj->ni", matrix, x) - y) ** 2))

    def torch_objective(theta):
        return torch.mean((torch_x @ theta.reshape(2, 2).T - torch_y) ** 2)

    return objective, torch_objective


def _save_resume(case_dir, points, losses, events, traces, iteration, phase,
                 pending, next_pending, checkpoints):
    npz_tmp = case_dir / "resume_state.tmp.npz"
    np.savez_compressed(
        npz_tmp, points=np.asarray(points), losses=np.asarray(losses),
        pending=np.asarray(pending, dtype=np.float64),
    )
    npz_tmp.replace(case_dir / "resume_state.npz")
    _atomic_text(case_dir / "resume_state.json", json.dumps(dict(
        iteration=iteration, phase=phase, next_pending=next_pending,
        events=events, continuous_traces=traces, checkpoints=checkpoints,
    )))


def _save_final_state(case_dir, points, losses, events, traces):
    np.savez_compressed(
        case_dir / "stopping_state.npz",
        points=np.asarray(points), losses=np.asarray(losses),
    )
    _atomic_text(case_dir / "events.json", json.dumps(events))
    _atomic_text(case_dir / "continuous.json", json.dumps(traces))


def _finish_result(name, index, category, teacher, states, points, losses,
                   events, traces, checkpoints, provenance, started):
    case_dir = CASES / name
    first_loss = _first_loss_crossing(events)
    first_containment = next((c["evaluation"] for c in checkpoints if c["containment_ok"]), None)
    first_both = next((c["evaluation"] for c in checkpoints if c["both_ok"]), None)
    hit_cap = first_both is None and len(losses) >= CAP
    result = dict(
        case=name, case_id=index, category=category, provenance=provenance,
        first_loss_crossing=first_loss,
        first_containment_checkpoint=first_containment,
        first_both_checkpoint=first_both,
        total_n1_evaluations=int(len(losses)), hit_cap_without_both=hit_cap,
        checkpoints=checkpoints, checkpoint_count=len(checkpoints),
        continuous_calls=len(traces),
        continuous_duplicate_rejections=int(sum(t["exact_duplicate_rejected"] for t in traces)),
        continuous_surrogate_evaluations=int(sum(
            start["nfev"] for trace in traces for start in trace["starts"]
        )),
        seconds=time.time() - started,
    )
    if checkpoints:
        checkpoint = (
            next(c for c in checkpoints if c["evaluation"] == first_both)
            if first_both is not None else checkpoints[-1]
        )
        objective, torch_objective = _case_objectives(states)
        start = dict(
            theta=np.asarray(checkpoint["best_theta"]),
            loss=checkpoint["best_loss"],
            lower=np.asarray(checkpoint["box_lower"]),
            upper=np.asarray(checkpoint["box_upper"]),
        )
        result.update(deterministic_finish(objective, torch_objective, start))
        final = np.asarray(result["final_theta"]).reshape(2, 2)
        delta = final - teacher.reshape(2, 2)
        true_state = states[-1].copy()
        predicted_state = true_state.copy()
        rollout_errors = []
        for _ in range(4096):
            true_state = teacher.reshape(2, 2) @ true_state
            predicted_state = final @ predicted_state
            rollout_errors.append(predicted_state - true_state)
        rollout_errors = np.asarray(rollout_errors)
        result.update(
            n2_n3_verified=True, matrix_error=float(np.linalg.norm(delta)),
            matrix_max_error=float(np.max(np.abs(delta))),
            rollout_max_error=float(np.max(np.abs(rollout_errors))),
            rollout_mse=float(np.mean(rollout_errors ** 2)),
            handoff_checkpoint=int(checkpoint["evaluation"]),
            handoff_loss_ok=bool(checkpoint["loss_ok"]),
            handoff_containment_ok=bool(checkpoint["containment_ok"]),
            n2_improved_n1=bool(result["n2_loss"] < checkpoint["best_loss"]),
            n3_recovered=bool(result["final_loss"] < 1.0e-20),
        )
    else:
        result["n2_n3_verified"] = False
    _save_final_state(case_dir, points, losses, events, traces)
    _atomic_text(case_dir / "result.json", json.dumps(result, indent=2))
    return result


def _reuse_source(name, index, category, teacher, states, source):
    sample_path = source / f"{name}_samples.npz"
    event_path = source / f"{name}_events.json"
    trace_path = source / f"{name}_continuous.json"
    if not (sample_path.exists() and event_path.exists() and trace_path.exists()):
        return None
    started = time.time()
    samples = np.load(sample_path)
    all_points = samples["points"]
    all_losses = samples["losses"]
    all_events = json.loads(event_path.read_text())
    all_traces = json.loads(trace_path.read_text())
    checkpoints = []
    stop = None
    for count in range(BLOCK, min(CAP, len(all_losses)) + 1, BLOCK):
        event = all_events[count - 1]
        trace_count = sum(e["source"] == "greedy" for e in all_events[:count])
        checkpoint = _checkpoint(
            all_points[:count], all_losses[:count], teacher, prior.LOWER,
            prior.UPPER, index, int(event["iteration"]), all_traces[:trace_count],
        )
        checkpoints.append(checkpoint)
        if checkpoint["both_ok"]:
            stop = count
            break
    if stop is None:
        return None
    case_dir = CASES / name
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_count = sum(e["source"] == "greedy" for e in all_events[:stop])
    provenance = "reused_full_trace" if source == FULL_SOURCE else "reused_early_trace"
    return _finish_result(
        name, index, category, teacher, states,
        all_points[:stop], all_losses[:stop], all_events[:stop],
        all_traces[:trace_count], checkpoints, provenance, started,
    )


def _run_case(index):
    torch.set_num_threads(1)
    with np.load(DATA) as data:
        name = str(data["names"][index])
        category = str(data["categories"][index])
        teacher = data["teachers"][index].ravel()
        states = data[f"prefix_{index}"].copy()
    case_dir = CASES / name
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("n2_n3_verified", False):
            return existing
        stopping_path = case_dir / "stopping_state.npz"
        event_path = case_dir / "events.json"
        trace_path = case_dir / "continuous.json"
        if stopping_path.exists() and event_path.exists() and trace_path.exists():
            stopping = np.load(stopping_path)
            return _finish_result(
                name, index, category, teacher, states,
                stopping["points"], stopping["losses"],
                json.loads(event_path.read_text()),
                json.loads(trace_path.read_text()),
                existing["checkpoints"], existing["provenance"], time.time(),
            )
        return existing

    for source in (FULL_SOURCE, EARLY_SOURCE):
        reused = _reuse_source(name, index, category, teacher, states, source)
        if reused is not None:
            return reused

    started = time.time()
    objective, _ = _case_objectives(states)
    traces = []
    api = base.composition(4)
    api["_greedy_proposal"] = (
        lambda points, losses, current_case, iteration:
        continuous.proposal(points, losses, current_case, iteration, traces)
    )
    points = []
    losses = []
    events = []
    checkpoints = []
    iteration = 0
    phase = "initial"
    pending = np.empty((0, 4), dtype=np.float64)
    next_pending = 0
    had_incomplete_source = any(
        (source / f"{name}_samples.npz").exists()
        for source in (FULL_SOURCE, EARLY_SOURCE)
    )
    provenance = "restarted_incomplete_trace" if had_incomplete_source else "new_run"

    resume_npz = case_dir / "resume_state.npz"
    resume_json = case_dir / "resume_state.json"
    if resume_npz.exists() and resume_json.exists():
        saved = np.load(resume_npz)
        points = [p.copy() for p in saved["points"]]
        losses = saved["losses"].tolist()
        pending = saved["pending"].copy()
        meta = json.loads(resume_json.read_text())
        events = meta["events"]
        traces = meta["continuous_traces"]
        checkpoints = meta["checkpoints"]
        iteration = int(meta["iteration"])
        phase = meta["phase"]
        next_pending = int(meta["next_pending"])
        provenance = "resumed_exact_checkpoint"

    def checkpoint_if_due(current_phase, current_pending, current_next):
        if not losses or len(losses) % BLOCK:
            return False
        record = _checkpoint(
            np.asarray(points), np.asarray(losses), teacher, prior.LOWER,
            prior.UPPER, index, iteration, traces,
        )
        checkpoints.append(record)
        _save_resume(
            case_dir, points, losses, events, traces, iteration,
            current_phase, current_pending, current_next, checkpoints,
        )
        return record["both_ok"] or len(losses) >= CAP

    def evaluate(unit_point, source):
        value = float(objective(prior.LOWER + (prior.UPPER - prior.LOWER) * unit_point))
        if not np.isfinite(value):
            raise ValueError("nonfinite objective")
        points.append(np.asarray(unit_point).copy())
        losses.append(value)
        events.append(dict(
            evaluation=len(losses), iteration=iteration, source=source,
            point=np.asarray(unit_point).tolist(), loss=value,
            best_loss=float(min(losses)),
            boundary=bool(np.any((unit_point == 0.0) | (unit_point == 1.0))),
        ))

    if phase == "initial":
        design = base.initial_design(4)
        for unit_point in design[len(points):]:
            evaluate(unit_point, "initial")
        phase = "next_round"

    stop = False
    while len(losses) < CAP and not stop:
        if phase == "next_round":
            pending = api["_cell_proposals"](
                np.asarray(points), np.asarray(losses), index, iteration
            )
            next_pending = 0
            phase = "cell"
        if phase == "cell":
            while next_pending < len(pending) and len(losses) < CAP:
                unit_point = pending[next_pending]
                next_pending += 1
                evaluate(unit_point, "cell")
                if checkpoint_if_due("cell", pending, next_pending):
                    stop = True
                    break
            if stop:
                break
            phase = "greedy"
        if phase == "greedy" and len(losses) < CAP:
            unit_point = api["_greedy_proposal"](
                np.asarray(points), np.asarray(losses), index, iteration
            )
            if unit_point is not None:
                evaluate(unit_point, "greedy")
            iteration += 1
            phase = "next_round"
            pending = np.empty((0, 4), dtype=np.float64)
            next_pending = 0
            if checkpoint_if_due("next_round", pending, 0):
                stop = True

    return _finish_result(
        name, index, category, teacher, states,
        np.asarray(points), np.asarray(losses), events, traces,
        checkpoints, provenance, started,
    )


def _write_teacher_table(results):
    rows = []
    for result in results:
        for checkpoint in result["checkpoints"]:
            rows.append(dict(
                case=result["case"], case_id=result["case_id"],
                category=result["category"], provenance=result["provenance"],
                **checkpoint,
            ))
    path = OUT / "teacher_data.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_classification_table(results):
    fields = [
        "case", "category", "N1_loss_ok", "containment_ok", "BOTH_ok",
        "N2_gate_ok", "N3_recovered", "N1_evals", "N2_evals",
        "final_matrix_entry_error", "final_matrix_error", "rollout_error",
    ]
    with (OUT / "classification.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(dict(
                case=result["case"], category=result["category"],
                N1_loss_ok=result["first_loss_crossing"] is not None,
                containment_ok=result["first_containment_checkpoint"] is not None,
                BOTH_ok=result["first_both_checkpoint"] is not None,
                N2_gate_ok=bool(result.get("n2_gate", False)),
                N3_recovered=bool(result.get("final_loss", math.inf) < 1.0e-20),
                N1_evals=result["total_n1_evaluations"],
                N2_evals=result.get("n2_evaluations"),
                final_matrix_entry_error=result.get("matrix_max_error"),
                final_matrix_error=result.get("matrix_error"),
                rollout_error=result.get("rollout_max_error"),
            ))


def _summary(results, before, after, workers, started):
    successful = [r for r in results if r["first_both_checkpoint"] is not None]
    categories = {}
    for category in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == category]
        passed = [r for r in rows if r["first_both_checkpoint"] is not None]
        categories[category] = dict(
            cases=len(rows), both_successes=len(passed), cap_failures=sum(r["hit_cap_without_both"] for r in rows),
            median_evaluations_to_both=(float(np.median([r["first_both_checkpoint"] for r in passed])) if passed else None),
            n3_recoveries=sum(r.get("final_loss", math.inf) < 1.0e-20 for r in rows),
            recovery_rate=(sum(r.get("final_loss", math.inf) < 1.0e-20 for r in rows) / len(rows)),
        )
    def distribution(values):
        finite = [int(value) for value in values if value is not None]
        if not finite:
            return dict(count=0, missing=len(values), median=None, p90=None, p99=None, maximum=None)
        data = np.asarray(finite, dtype=float)
        return dict(
            count=len(finite), missing=len(values) - len(finite),
            median=float(np.median(data)),
            p90=float(np.quantile(data, 0.90, method="higher")),
            p99=float(np.quantile(data, 0.99, method="higher")),
            maximum=int(np.max(data)),
        )
    loss_evaluations = [
        r["first_loss_crossing"]["evaluation"] if r["first_loss_crossing"] else None
        for r in results
    ]
    containment_evaluations = [r["first_containment_checkpoint"] for r in results]
    both_evaluations = [r["first_both_checkpoint"] for r in results]
    provenance = {name: sum(r["provenance"] == name for r in results) for name in (
        "reused_full_trace", "reused_early_trace", "resumed_exact_checkpoint",
        "restarted_incomplete_trace", "new_run"
    )}
    n1_loss_failures = [r for r in results if r["first_loss_crossing"] is None]
    n2_gate_failures = [r for r in results if not r.get("n2_gate", False)]
    matrix_errors = [r["matrix_error"] for r in results if "matrix_error" in r]
    entry_errors = [r["matrix_max_error"] for r in results if "matrix_max_error" in r]
    rollout_errors = [r["rollout_max_error"] for r in results if "rollout_max_error" in r]
    return dict(
        status=("COMPLETE" if len(results) == 256 else "INCOMPLETE"),
        cases=len(results), both_successes=len(successful),
        success_rate=len(successful) / len(results) if results else 0.0,
        n1_loss_successes=sum(r["first_loss_crossing"] is not None for r in results),
        containment_successes=sum(r["first_containment_checkpoint"] is not None for r in results),
        failures_at_cap=sum(r["hit_cap_without_both"] for r in results),
        reused_without_rerun=provenance["reused_full_trace"] + provenance["reused_early_trace"],
        resumed=provenance["resumed_exact_checkpoint"],
        restarted=provenance["restarted_incomplete_trace"],
        newly_completed=provenance["restarted_incomplete_trace"] + provenance["new_run"],
        provenance=provenance,
        evaluations_to_loss=distribution(loss_evaluations),
        evaluations_to_containment=distribution(containment_evaluations),
        evaluations_to_both=distribution(both_evaluations),
        categories=categories,
        n2_n3_verifications=sum(r["n2_n3_verified"] for r in results),
        n2_successes=sum(r.get("n2_gate", False) for r in results),
        n3_successes=sum(r.get("final_loss", math.inf) < 1e-20 for r in results),
        overall_recovery_rate=(
            sum(r.get("final_loss", math.inf) < 1e-20 for r in results) / len(results)
            if results else 0.0
        ),
        n2_failures_or_caps=sum(not r.get("n2_gate", False) for r in results),
        n2_evaluation_cap_hits=sum(r.get("n2_evaluations", 0) >= 1500 for r in results),
        n2_gate_failures_at_cap=sum(
            (not r.get("n2_gate", False)) and r.get("n2_evaluations", 0) >= 1500
            for r in results
        ),
        n2_improved_n1_loss_failures=sum(
            r.get("n2_improved_n1", False) for r in n1_loss_failures
        ),
        n3_repaired_n1_loss_failures=sum(
            r.get("final_loss", math.inf) < 1.0e-20 for r in n1_loss_failures
        ),
        n3_repaired_n2_gate_failures=sum(
            r.get("final_loss", math.inf) < 1.0e-20 for r in n2_gate_failures
        ),
        worst_matrix_entry_error=max(entry_errors, default=None),
        worst_matrix_error=max(matrix_errors, default=None),
        worst_rollout_error=max(rollout_errors, default=None),
        workers=workers,
        machine=dict(logical_cores=10, physical_cores=10, memory_bytes=17179869184,
                     numerical_threads_per_worker=1),
        checkpoint_hashes_unchanged=before == after,
        elapsed_seconds=time.time() - started,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case-index", type=int, default=None)
    args = parser.parse_args()
    if args.case_index is not None:
        result = _run_case(args.case_index)
        print(json.dumps(dict(
            case=result["case"], provenance=result["provenance"],
            first_both_checkpoint=result["first_both_checkpoint"],
            hit_cap_without_both=result["hit_cap_without_both"],
        ), sort_keys=True), flush=True)
        return
    workers = max(1, min(args.workers, 6))
    started = time.time()
    CASES.mkdir(parents=True, exist_ok=True)
    before = hashes()
    _atomic_text(OUT / "checkpoint_hashes_before.json", json.dumps(before, indent=2))
    with np.load(DATA) as data:
        case_names = [str(name) for name in data["names"]]
        count = len(case_names)
    results = []
    for offset in range(0, count, workers):
        indices = list(range(offset, min(offset + workers, count)))
        child_env = os.environ.copy()
        children = []
        for index in indices:
            command = [
                sys.executable, "-m",
                "experiments.sho.generate_checkpointed_edirect_teacher_data",
                "--case-index", str(index),
            ]
            children.append((index, subprocess.Popen(command, env=child_env)))
        failures = []
        for index, child in children:
            return_code = child.wait()
            if return_code != 0:
                failures.append((index, return_code))
        if failures:
            raise RuntimeError(f"case subprocess failures: {failures}")
        batch = [
            json.loads((CASES / case_names[index] / "result.json").read_text())
            for index in indices
        ]
        results.extend(batch)
        results.sort(key=lambda row: row["case_id"])
        _write_teacher_table(results)
        _write_classification_table(results)
        _atomic_text(OUT / "results_partial.json", json.dumps(results, indent=2))
        for row in batch:
            print(
                row["case"], row["provenance"], "loss",
                row["first_loss_crossing"]["evaluation"] if row["first_loss_crossing"] else None,
                "contain", row["first_containment_checkpoint"],
                "both", row["first_both_checkpoint"], flush=True,
            )
    after = hashes()
    _atomic_text(OUT / "checkpoint_hashes_after.json", json.dumps(after, indent=2))
    report = _summary(results, before, after, workers, started)
    _atomic_text(OUT / "report.json", json.dumps(report, indent=2))
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
