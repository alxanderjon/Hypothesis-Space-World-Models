"""Checkpoint-aware continuation of the interrupted generic SHO screen.

This module changes persistence only.  All optimization decisions delegate to
the same assemblies used by ``recover_generic_update_targeted``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3CurvatureRouteBinding,
    HybridN3DampedMatrixRHSBinding,
    HybridN3DampingBinding,
    Log,
    run_n2_with_best_ever_assembly,
)
from experiments.sho import recover_affine_bias_targeted as parent
from experiments.sho import recover_generic_update_targeted as original
from experiments.sho.assemble_n3_learned_polisher import ControlAssembly
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.validate_generic_update_network_adapter import protected


CASE = "companion_6"
N1_COUNT = 4
ARCHIVE_INDEX = 3
STEM = "n00004_i00003"
ROOT = original.OUT / CASE
AUTHORITATIVE = ROOT / f"{STEM}_n3.jsonl"
CONTINUATION = ROOT / f"{STEM}_n3_continuation.jsonl"
RESUME_STATE = ROOT / f"{STEM}_resume_state.json"
VALIDATION = ROOT / f"{STEM}_resume_validation.json"
RESUME_RESULT = ROOT / f"{STEM}_resume_result.json"
REPORT = original.OUT / "resume_report.json"
EXPECTED_STEP = 26600
EXPECTED_LOSS = np.float64(5.671415362759253e-15)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bits(value):
    return np.asarray(value, np.float64).view(np.uint64)


def bit_equal(left, right) -> bool:
    return bool(np.array_equal(bits(left), bits(right)))


def exact_tree(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            exact_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            exact_tree(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bit_equal(left, right)
    return left == right


def append_durable(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(original.clean(value), allow_nan=False) + "\n"
    with path.open("a") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class Runtime:
    """The unchanged N3 operators with restorable reporting counters."""

    def __init__(self, adapter, trajectory, log):
        self.adapter = adapter
        self.trajectory = trajectory
        self.log = log
        self.lower = torch.from_numpy(adapter.lower.copy())
        self.upper = torch.from_numpy(adapter.upper.copy())
        self.curvature = HybridN3CurvatureRouteBinding()
        self.damping = HybridN3DampingBinding(self.curvature)
        self.matrix_rhs = HybridN3DampedMatrixRHSBinding()
        self.controls = ControlAssembly()
        _, self.direction_oracle, _ = parent.staged._make_n3(
            lambda value: value.sum() * 0.0, adapter.lower, adapter.upper
        )
        self.objective_evaluations = 0

    def scalar_loss(self, theta) -> float:
        self.objective_evaluations += 1
        with torch.no_grad():
            return float(self.adapter.objective(theta, self.trajectory))

    def derivatives_and_direction(self, theta, count_route=True):
        result = self.adapter.derivatives(theta.numpy(), self.trajectory)
        gradient = result.gradient
        hessian = result.symmetric_hessian
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        pre = self.controls.pre_step_route(0, gradient_norm)
        if pre != "CONTINUE":
            return result, gradient, hessian, gradient_norm, pre, None, None, None
        spectrum, route = self.curvature.evaluate(gradient, hessian, self.log)
        initial = self.damping.initial(
            spectrum.scale, spectrum.minimum_eigenvalue, route["direct"], self.log
        )
        for attempt in range(original.n3_production.MAX_DAMPING_TRIALS):
            self.damping.value(
                spectrum.scale, initial, route["direct"], attempt, self.log
            )
        direction, qp = dimension_general_box_qp_direction(
            theta, gradient, hessian, self.lower, self.upper, self.direction_oracle
        )
        return result, gradient, hessian, gradient_norm, pre, route, direction, qp


def jsonl_rows(path: Path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL record {path}:{line_number}") from exc


def reconstruct(runtime: Runtime) -> dict:
    if not AUTHORITATIVE.exists():
        raise RuntimeError("GENERIC_NETWORK_RESUME_STATE_INCOMPLETE: authoritative JSONL missing")
    previous = None
    last = None
    expected = 0
    stagnation = 0
    routes = Counter()
    objective_evaluations = 1
    backtracking_rejections = 0
    records = 0
    authoritative_boundary = None
    sources = [AUTHORITATIVE]
    if CONTINUATION.exists() and CONTINUATION.stat().st_size:
        sources.append(CONTINUATION)
    for source in sources:
        for row in jsonl_rows(source):
            step = int(row["diagnostics"]["accepted_before"])
            if step != expected:
                raise RuntimeError(
                    f"GENERIC_NETWORK_RESUME_STATE_INCOMPLETE: expected step {expected}, got {step}"
                )
            if not bit_equal(row["history"]["theta"], row["diagnostics"]["phi"]):
                raise RuntimeError(f"phi disagreement at accepted step {step}")
            if not bit_equal(row["history"]["loss"], row["diagnostics"]["loss"]):
                raise RuntimeError(f"loss disagreement at accepted step {step}")
            routes[row["diagnostics"]["curvature_route"]] += 1
            if previous is not None:
                backtrack = int(row["history"]["backtrack"])
                backtracking_rejections += backtrack
                old_theta = torch.from_numpy(
                    np.asarray(previous["history"]["theta"], np.float64)
                )
                old_direction = torch.from_numpy(
                    np.asarray(previous["diagnostics"]["direction"], np.float64)
                )
                for attempt in range(backtrack + 1):
                    candidate = runtime.controls.candidate(old_theta, old_direction, attempt)
                    if runtime.controls.in_bounds(
                        candidate, runtime.lower.numpy(), runtime.upper.numpy()
                    ):
                        objective_evaluations += 1
                full_predicted = -float(
                    previous["diagnostics"]["qp"]["quadratic_value"]
                )
                stagnation = original.n3_production._updated_float64_stagnation_count(
                    stagnation,
                    float(previous["history"]["loss"]),
                    float(row["history"]["loss"]),
                    full_predicted,
                )
            previous = row
            last = row
            expected += 1
            records += 1
        if source == AUTHORITATIVE and last is not None:
            authoritative_boundary = {
                "accepted_steps": int(last["diagnostics"]["accepted_before"]),
                "loss": float(last["history"]["loss"]),
            }
    if last is None:
        raise RuntimeError("GENERIC_NETWORK_RESUME_STATE_INCOMPLETE: no accepted state")
    return {
        "row": last,
        "accepted_steps": int(last["diagnostics"]["accepted_before"]),
        "stagnation_count": stagnation,
        "route_counts": routes,
        "objective_evaluations": objective_evaluations,
        "backtracking_rejections": backtracking_rejections,
        "records": records,
        "sources": [str(path) for path in sources],
        "authoritative_boundary": authoritative_boundary,
    }


def load_case(log):
    corpus = np.load(parent.DATA)
    legacy = json.loads((parent.OLD / CASE / "result.json").read_text())
    case_id = int(legacy["case_id"])
    trajectory = corpus[f"prefix_{case_id}"].copy()
    adapter = GenericSHOUpdateNetworkAdapter()
    objective = original.Objective(adapter, trajectory)
    runner = original.GenericArchiveN1(objective, adapter, case_id, log)
    events = json.loads((ROOT / "n1_history.json").read_text())
    if len(events) != N1_COUNT:
        raise RuntimeError("GENERIC_NETWORK_RESUME_STATE_INCOMPLETE: wrong N1 count")
    for offset, event in enumerate(events):
        if event["evaluation"] != offset + 1 or event["source"] != "initial":
            raise RuntimeError("GENERIC_NETWORK_RESUME_STATE_INCOMPLETE: unexpected N1 phase")
        if not bit_equal(event["point"], runner._design[offset]):
            raise RuntimeError("GENERIC_NETWORK_RESUME_STATE_INCOMPLETE: initial point mismatch")
    runner.points = [np.asarray(event["point"], np.float64) for event in events]
    runner.losses = [float(event["loss"]) for event in events]
    runner.events = events
    runner.iteration = 0
    runner.phase = "initial"
    runner.next_pending = N1_COUNT
    runner.pending = np.empty((0, adapter.dimension), np.float64)
    runner.traces = []
    return corpus, case_id, trajectory, adapter, objective, runner


def validate_restoration(runtime, state, adapter, objective, runner, case_id) -> dict:
    row = state["row"]
    theta = torch.from_numpy(np.asarray(row["history"]["theta"], np.float64).copy())
    saved_loss = float(row["history"]["loss"])
    with torch.no_grad():
        live_loss = float(adapter.objective(theta.numpy(), runtime.trajectory))
    result, gradient, hessian, gradient_norm, pre, route, direction, qp = (
        runtime.derivatives_and_direction(theta)
    )
    checks = {
        "branch_identity": [N1_COUNT, ARCHIVE_INDEX] == [4, 3],
        "accepted_step": state["accepted_steps"] >= EXPECTED_STEP,
        "authoritative_boundary_step":
            state["authoritative_boundary"]["accepted_steps"] == EXPECTED_STEP,
        "authoritative_boundary_loss": bit_equal(
            state["authoritative_boundary"]["loss"], EXPECTED_LOSS
        ),
        "phi_bits": bit_equal(theta.numpy(), row["diagnostics"]["phi"]),
        "objective_loss_bits": bit_equal(live_loss, saved_loss),
        "derivative_loss_bits": bit_equal(float(result.loss), saved_loss),
        "gradient_bits": bit_equal(gradient.numpy(), row["diagnostics"]["gradient"]),
        "hessian_bits": bit_equal(hessian.numpy(), row["diagnostics"]["hessian"]),
        "gradient_norm_bits": bit_equal(gradient_norm, row["diagnostics"]["gradient_norm"]),
        "pre_step_route": pre == "CONTINUE",
        "curvature_route": route["route"] == row["diagnostics"]["curvature_route"],
        "direction_bits": bit_equal(direction.numpy(), row["diagnostics"]["direction"]),
        "qp_state": exact_tree(qp, row["diagnostics"]["qp"]),
        "bounds": bit_equal(adapter.lower, np.full(22, -1.99))
        and bit_equal(adapter.upper, np.full(22, 1.99)),
        "dimension": adapter.dimension == 22,
        "success_threshold": bit_equal(original.FINAL_GATE, 1.0e-20),
        "n1_count": len(runner.losses) == N1_COUNT,
        "n1_phase": runner.phase == "initial" and runner.next_pending == N1_COUNT,
        "case_id": case_id == int(json.loads((parent.OLD / CASE / "result.json").read_text())["case_id"]),
        "stagnation_state": state["stagnation_count"] == 0,
    }
    report = {
        "status": "RESUME_STATE_VALIDATED" if all(checks.values()) else "RESUME_VALIDATION_FAILED",
        "checks": checks,
        "authoritative_jsonl": str(AUTHORITATIVE),
        "continuation_jsonl": str(CONTINUATION),
        "authoritative_sha256": sha256(AUTHORITATIVE),
        "accepted_steps": state["accepted_steps"],
        "loss": saved_loss,
        "stagnation_count": state["stagnation_count"],
        "route_counts": dict(state["route_counts"]),
        "objective_evaluations_reconstructed": state["objective_evaluations"],
        "backtracking_rejections_reconstructed": state["backtracking_rejections"],
        "active_face": row["diagnostics"]["qp"]["winning_face_status"],
        "runner_source_sha256": sha256(Path(original.__file__)),
        "adapter_source_sha256": sha256(Path("experiments/sho/generic_update_network_adapter.py")),
        "architecture": "Linear(2,4)->SiLU->Linear(4,2)",
        "derivative_oracle": adapter.derivative_oracle,
    }
    original.atomic(VALIDATION, report)
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"resume validation failed: {failed}")
    return report


def diagnostic(theta, current_loss, accepted_steps, gradient, hessian,
               gradient_norm, route, qp, direction):
    return {
        "accepted_before": accepted_steps,
        "loss": current_loss,
        "gradient": gradient.numpy().tolist(),
        "hessian": hessian.numpy().tolist(),
        "gradient_norm": gradient_norm,
        "curvature_route": route["route"],
        "qp": qp,
        "phi": theta.numpy().tolist(),
        "direction": None if direction is None else direction.numpy().tolist(),
    }


def continue_n3(runtime: Runtime, state: dict) -> dict:
    row = state["row"]
    theta = torch.from_numpy(np.asarray(row["history"]["theta"], np.float64).copy())
    current_loss = float(row["history"]["loss"])
    accepted_steps = int(state["accepted_steps"])
    resumed_at = accepted_steps
    stagnation_count = int(state["stagnation_count"])
    route_counts = Counter(state["route_counts"])
    runtime.objective_evaluations = int(state["objective_evaluations"])
    backtracking_rejections = int(state["backtracking_rejections"])
    already_logged = True
    stop_reason = "not_started"
    last_printed = -1
    while True:
        result, gradient, hessian, gradient_norm, pre, route, direction, qp = (
            runtime.derivatives_and_direction(theta)
        )
        derivative_loss = float(result.loss)
        if not bit_equal(derivative_loss, current_loss):
            raise RuntimeError("generic forward and derivative objectives diverged")
        if pre != "CONTINUE":
            stop_reason = pre
            break
        if not already_logged:
            route_counts[route["route"]] += 1
            record = {
                "history": {
                    "theta": theta.numpy().tolist(),
                    "loss": current_loss,
                    "backtrack": accepted_backtrack,
                    "route": accepted_route,
                },
                "diagnostics": diagnostic(
                    theta, current_loss, accepted_steps, gradient, hessian,
                    gradient_norm, route, qp, direction,
                ),
            }
            append_durable(CONTINUATION, record)
            if accepted_steps % 50 == 0:
                original.atomic(RESUME_STATE, {
                    "status": "N3_RUNNING",
                    "case": CASE,
                    "n1_count": N1_COUNT,
                    "archive_index": ARCHIVE_INDEX,
                    "accepted_steps": accepted_steps,
                    "theta": theta.numpy(),
                    "loss": current_loss,
                    "stagnation_count": stagnation_count,
                    "objective_evaluations": runtime.objective_evaluations,
                    "backtracking_rejections": backtracking_rejections,
                    "route_counts": dict(route_counts),
                    "authoritative_jsonl": str(AUTHORITATIVE),
                    "continuation_jsonl": str(CONTINUATION),
                })
        else:
            # The authoritative last row already counted and logged this state.
            already_logged = False
        if accepted_steps % 25 == 0 and accepted_steps != last_printed:
            print(
                f"RESUMED N3 {STEM} steps={accepted_steps} loss={current_loss:.17g}",
                flush=True,
            )
            last_printed = accepted_steps
        if direction is None:
            stop_reason = qp["route_status"]
            break
        runtime.matrix_rhs.construct(hessian, gradient, float(qp["damping"]), runtime.log)
        descent, dot, _ = runtime.controls.descent.check(gradient, direction)
        if not descent:
            stop_reason = "no_descent_direction"
            break
        full_predicted = -float(qp["quadratic_value"])
        accepted = False
        unchanged = False
        for backtrack in range(original.n3_production.MAX_BACKTRACKS + 1):
            candidate = runtime.controls.candidate(theta, direction, backtrack)
            if bool(torch.equal(candidate, theta)):
                unchanged = True
                break
            if not runtime.controls.in_bounds(
                candidate, runtime.lower.numpy(), runtime.upper.numpy()
            ):
                backtracking_rejections += 1
                continue
            candidate_loss = runtime.scalar_loss(candidate.numpy())
            if runtime.controls.accepts(candidate_loss, current_loss):
                stagnation_count = original.n3_production._updated_float64_stagnation_count(
                    stagnation_count, current_loss, candidate_loss, full_predicted
                )
                theta = candidate.detach()
                current_loss = candidate_loss
                accepted_steps += 1
                accepted_backtrack = backtrack
                accepted_route = route["route"]
                accepted = True
                break
            backtracking_rejections += 1
        if accepted:
            if stagnation_count >= original.n3_production.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
                stop_reason = original.n3_production.FLOAT64_STAGNATION_STOP_REASON
                break
            continue
        numerical_stop, _, _ = runtime.controls.no_further_improvement(
            unchanged, dot, current_loss
        )
        stop_reason = (
            "floating_point_no_further_improvement"
            if numerical_stop else "backtracking_no_improvement"
        )
        break
    result = {
        "theta": theta.numpy().copy(),
        "loss": current_loss,
        "accepted_steps": accepted_steps,
        "resumed_at_accepted_step": resumed_at,
        "continued_accepted_steps": accepted_steps - resumed_at,
        "stop_reason": stop_reason,
        "objective_evaluations": runtime.objective_evaluations,
        "backtracking_rejections": backtracking_rejections,
        "route_counts": dict(route_counts),
        "stagnation_count": stagnation_count,
        "authoritative_jsonl": str(AUTHORITATIVE),
        "continuation_jsonl": str(CONTINUATION),
    }
    original.atomic(RESUME_RESULT, result)
    return result


def fresh_branch(adapter, trajectory, objective, runner, log, branches, n, i):
    phi = adapter.lower + (adapter.upper - adapter.lower) * runner.points[i]
    lower, upper = adapter.archive_box(phi, n)
    stem = f"n{n:05d}_i{i:05d}"
    print(f"{CASE} {stem} start loss={runner.losses[i]:.17g}", flush=True)
    original.atomic(ROOT / "progress.json", {
        "stage": "N2", "evaluation_count": n, "archive_index": i,
    })
    objective.capture = []
    n2 = run_n2_with_best_ever_assembly(
        objective, phi, runner.losses[i], lower, upper, log
    )
    evaluations = objective.capture
    objective.capture = None
    n2_record = {key: value for key, value in n2.items() if key not in ("tracker", "powell")}
    original.atomic(ROOT / f"{stem}_n2.json", {
        "box_lower": lower, "box_upper": upper,
        "result": n2_record, "history": evaluations,
    })
    solver = original.GenericN3(adapter, trajectory, log, ROOT / f"{stem}_n3.json")
    n3 = solver.run(n2["theta"])
    success = bool(n3["loss"] < original.FINAL_GATE)
    original.atomic(ROOT / f"{stem}_n3.json", {
        "stage": "N3_COMPLETE", "result": n3, "diagnostics": solver.diagnostics,
    })
    branches.append({
        "evaluation_count": n, "archive_index": i, "N2": n2_record,
        "N3": {key: value for key, value in n3.items() if key != "history"},
        "valid_handoff": success,
    })
    original.atomic(ROOT / "branches.json", branches)
    print(
        f"{CASE} {stem} N3 {n3['stop_reason']} steps={n3['accepted_steps']} "
        f"loss={n3['loss']:.17g} handoff={success}", flush=True,
    )
    return n2_record, n3, success


def finish_companion(case_id, corpus, adapter, runner, branches, n2_record, n3, success, i):
    rollout = original.rollout_diagnostic(
        adapter, np.load(parent.DATA)[f"prefix_{case_id}"].copy(),
        corpus["teachers"][case_id], n3["theta"],
    )
    final_success = bool(success and rollout["rollout_finite"])
    result = {
        "case": CASE, "case_id": case_id,
        "prefix_length": len(np.load(parent.DATA)[f"prefix_{case_id}"]),
        "dimension": adapter.dimension,
        "first_evaluated_site": runner.events[0],
        "N1_evaluations": len(runner.losses),
        "branches_tested": len(branches),
        "archive_handoff_index": i if success else None,
        "handoff_is_newest": (i == len(runner.losses) - 1) if success else None,
        "handoff_is_incumbent": (i == int(np.argmin(runner.losses))) if success else None,
        "N2": n2_record,
        "N3": {key: value for key, value in n3.items() if key != "history"},
        "recovered_phi": n3["theta"],
        "final_transition_mse": n3["loss"],
        "rollout": rollout,
        "existing_recovery_status": final_success,
        "failure_stage": None if final_success else ("rollout" if success else "N1_RESOURCE_INCOMPLETE"),
        "adapter_invocations": adapter.calls,
        "resumed": True,
    }
    original.atomic(ROOT / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    protected_before = protected()
    checkpoints_before = hashes()
    log = Log()
    corpus, case_id, trajectory, adapter, objective, runner = load_case(log)
    runtime = Runtime(adapter, trajectory, log)
    state = reconstruct(runtime)
    validation = validate_restoration(
        runtime, state, adapter, objective, runner, case_id
    )
    print(json.dumps(original.clean(validation), indent=2), flush=True)
    if args.validate_only:
        return
    forbidden = Counter()

    def profile(frame, event, arg):
        if event != "call":
            return
        instance = frame.f_locals.get("self")
        if instance is not None and type(instance).__name__ in (
            "AffineSHOUpdateAssembly", "AffineN3", "SHOExactSolutionDecoder",
            "FullKSHOHessianAssembly",
        ):
            forbidden[type(instance).__name__] += 1
            raise RuntimeError("forbidden affine physical path")

    rows = []
    report = {"status": "RUNNING", "cases": rows}
    try:
        sys.setprofile(profile)
        branches = json.loads((ROOT / "branches.json").read_text())
        if any(
            row["evaluation_count"] == N1_COUNT and row["archive_index"] == ARCHIVE_INDEX
            for row in branches
        ):
            raise RuntimeError("active branch already recorded complete")
        n2_payload = json.loads((ROOT / f"{STEM}_n2.json").read_text())
        n2_record = n2_payload["result"]
        n3 = continue_n3(runtime, state)
        success = bool(n3["loss"] < original.FINAL_GATE)
        branches.append({
            "evaluation_count": N1_COUNT,
            "archive_index": ARCHIVE_INDEX,
            "N2": n2_record,
            "N3": n3,
            "valid_handoff": success,
            "resumed": True,
        })
        original.atomic(ROOT / "branches.json", branches)
        print(
            f"{CASE} {STEM} resumed N3 {n3['stop_reason']} "
            f"steps={n3['accepted_steps']} loss={n3['loss']:.17g} handoff={success}",
            flush=True,
        )
        last_n2 = n2_record
        last_i = ARCHIVE_INDEX
        if not success:
            for i in range(N1_COUNT - 1):
                last_n2, n3, success = fresh_branch(
                    adapter, trajectory, objective, runner, log, branches, N1_COUNT, i
                )
                last_i = i
                if success:
                    break
        while not success and len(runner.losses) < 5000:
            runner.advance_one()
            n = len(runner.losses)
            original.atomic(ROOT / "n1_history.json", runner.events)
            for i in [n - 1] + list(range(n - 1)):
                last_n2, n3, success = fresh_branch(
                    adapter, trajectory, objective, runner, log, branches, n, i
                )
                last_i = i
                if success:
                    break
        first = finish_companion(
            case_id, corpus, adapter, runner, branches,
            last_n2, n3, success, last_i,
        )
        rows.append(first)
        original.atomic(original.OUT / "partial_report.json", {"cases": rows})
        if first["existing_recovery_status"]:
            for name in original.CASES[1:]:
                row = original.run_case(name, corpus, log)
                rows.append(row)
                original.atomic(original.OUT / "partial_report.json", {"cases": rows})
                if not row["existing_recovery_status"]:
                    break
        report = {
            "status": (
                "GENERIC_UPDATE_NETWORK_RECOVERY_TARGETED"
                if len(rows) == 4 and all(row["existing_recovery_status"] for row in rows)
                else "FIRST_TARGETED_RECOVERY_FAILURE"
            ),
            "cases": rows,
        }
    except Exception as exc:
        report = {
            "status": "FAIL_CLOSED", "cases": rows,
            "exception_type": type(exc).__name__, "exception": str(exc),
        }
        raise
    finally:
        sys.setprofile(None)
        report.update({
            "forbidden_affine_calls": dict(forbidden),
            "forbidden_affine_call_total": sum(forbidden.values()),
            "protected_unchanged": protected_before == protected(),
            "checkpoint_count": len(checkpoints_before),
            "checkpoint_hashes_unchanged": checkpoints_before == hashes(),
            "full_corpus_run": False,
            "new_baseline_frozen": False,
            "resume_validation": str(VALIDATION),
            "continuation_jsonl": str(CONTINUATION),
            "resume_entrypoint_sha256": sha256(Path(__file__)),
        })
        original.atomic(REPORT, report)
    print(json.dumps(original.clean(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
