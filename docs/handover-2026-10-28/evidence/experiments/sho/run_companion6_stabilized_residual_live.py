"""One checkpoint-aware companion_6 live run using opt-in stabilized N3.

This never runs N1 or N2 and never changes the production N3 default.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from experiments.resume_hybrid_student_after_ckdtree import Log
from experiments.sho import recover_affine_bias_targeted as parent
from experiments.sho import recover_generic_update_targeted as original
from experiments.sho import run_stage1b_n3_gradient_polishing as production_n3
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import (
    DEFAULT_DIRECTION_MODE,
    EXISTING_FULL_HESSIAN,
    STABILIZED_RESIDUAL,
    OptInResidualN3Runtime,
)
from experiments.sho.validate_generic_update_network_adapter import protected


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/companion6_stabilized_residual_n3_live")
TRACE = OUT / "accepted_states.jsonl"
CHECKPOINT = OUT / "resume_state.json"
RESULT = OUT / "result.json"
MANIFEST = OUT / "run_manifest.json"
N2_SOURCE = Path(
    "outputs/sho/generic_update_network_targeted_recovery/companion_6/"
    "n00005_i00004_n2.json"
)
HISTORICAL_EXISTING = Path(
    "outputs/sho/generic_update_network_targeted_recovery/companion_6/"
    "n00005_i00004_n3.jsonl"
)
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
CASE = "companion_6"
CASE_ID = 6
BRANCH = [5, 4]
HANDOFF = np.float64(1.0e-20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bits_equal(left, right) -> bool:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def clean(value):
    if isinstance(value, torch.Tensor):
        return clean(value.detach().numpy())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_durable(value) -> None:
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), allow_nan=False) + "\n"
    with TRACE.open("a") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def last_valid_record():
    if not TRACE.exists():
        return None
    last = None
    with TRACE.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"partial/invalid checkpoint at line {line_number}") from exc
            if row.get("record_type") == "accepted_state":
                last = row
    return last


def historical_summary():
    first = None
    last = None
    count = 0
    with HISTORICAL_EXISTING.open() as stream:
        for line in stream:
            row = json.loads(line)
            if first is None:
                first = row
            last = row
            count += 1
    return {
        "recorded_states": count,
        "accepted_step": int(last["diagnostics"]["accepted_before"]),
        "loss": float(last["history"]["loss"]),
        "terminal": False,
        "source_sha256": sha256(HISTORICAL_EXISTING),
    }


def checkpoint_payload(theta, loss, accepted, stagnation, runtime, rejections, routes, path_length):
    return {
        "status": "N3_RUNNING",
        "case": CASE,
        "case_id": CASE_ID,
        "branch": BRANCH,
        "direction_mode": STABILIZED_RESIDUAL,
        "accepted_steps": accepted,
        "theta": theta,
        "loss": loss,
        "stagnation_count": stagnation,
        "objective_evaluations": runtime.objective_evaluations,
        "backtracking_rejections": rejections,
        "route_counts": dict(routes),
        "parameter_path_length": path_length,
        "authoritative_jsonl": str(TRACE),
        "jsonl_sha256": sha256(TRACE),
    }


def main():
    if RESULT.exists():
        existing = json.loads(RESULT.read_text())
        if existing.get("terminal"):
            raise RuntimeError("terminal live result already exists; refusing a second run")

    if DEFAULT_DIRECTION_MODE != EXISTING_FULL_HESSIAN:
        raise RuntimeError("production/default direction contract changed")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_before = protected()
    checkpoints_before = hashes()
    n2_document = json.loads(N2_SOURCE.read_text())
    n2 = n2_document["result"]
    trajectory = torch.from_numpy(np.load(DATA)["prefix_6"].copy())
    adapter = GenericSHOUpdateNetworkAdapter()
    runtime = OptInResidualN3Runtime(
        adapter, trajectory, Log(), direction_mode=STABILIZED_RESIDUAL
    )

    theta0 = torch.tensor(n2["theta"])
    if int(n2["evaluations"]) != 1500:
        raise RuntimeError("wrong saved N2 evaluation count")
    if not bits_equal(n2["loss"], 0.0018503643453732527):
        raise RuntimeError("wrong saved N2 loss")
    live_start_loss = runtime.scalar_loss(theta0.numpy())
    if not bits_equal(live_start_loss, n2["loss"]):
        raise RuntimeError("saved N2 endpoint objective does not replay exactly")

    last = last_valid_record()
    if last is None:
        theta = theta0.clone()
        loss = live_start_loss
        accepted = 0
        stagnation = 0
        rejections = 0
        routes = Counter()
        path_length = 0.0
        append_durable({
            "record_type": "accepted_state",
            "case": CASE,
            "branch": BRANCH,
            "accepted_steps": 0,
            "theta": theta,
            "loss": loss,
            "stagnation_count": stagnation,
            "objective_evaluations": runtime.objective_evaluations,
            "backtracking_rejections": rejections,
            "route_counts": {},
            "parameter_path_length": path_length,
            "start_state": True,
        })
    else:
        if last["case"] != CASE or last["branch"] != BRANCH:
            raise RuntimeError("checkpoint identity mismatch")
        theta = torch.tensor(last["theta"])
        loss = float(last["loss"])
        accepted = int(last["accepted_steps"])
        stagnation = int(last["stagnation_count"])
        runtime.objective_evaluations = int(last["objective_evaluations"])
        rejections = int(last["backtracking_rejections"])
        routes = Counter(last["route_counts"])
        path_length = float(last["parameter_path_length"])
        replay_loss = runtime.scalar_loss(theta.numpy())
        runtime.objective_evaluations -= 1
        if not bits_equal(replay_loss, loss):
            raise RuntimeError("resume checkpoint objective mismatch")

    manifest = {
        "case": CASE,
        "case_id": CASE_ID,
        "branch": BRANCH,
        "n2_source": str(N2_SOURCE),
        "n2_source_sha256": sha256(N2_SOURCE),
        "n2_evaluations": 1500,
        "start_loss": float(n2["loss"]),
        "network": "Linear(2,4)->SiLU->Linear(4,2)",
        "dimension": 22,
        "bounds": [-1.99, 1.99],
        "direction_mode": STABILIZED_RESIDUAL,
        "default_direction_mode": DEFAULT_DIRECTION_MODE,
        "handoff_loss": float(HANDOFF),
        "accepted_step_cap": None,
        "runner_source_sha256": sha256(Path(__file__)),
        "opt_in_source_sha256": sha256(Path("experiments/sho/generic_update_residual_n3_opt_in.py")),
        "frozen_artifact_count": len(checkpoints_before),
        "resume_at_step": accepted,
        "created_at_unix": time.time(),
    }
    atomic(MANIFEST, manifest)
    atomic(CHECKPOINT, checkpoint_payload(theta, loss, accepted, stagnation, runtime, rejections, routes, path_length))

    interrupted = False
    stop_reason = None
    started = time.monotonic()
    last_printed = -1
    try:
        while True:
            if loss < HANDOFF:
                stop_reason = "HANDOFF_THRESHOLD_REACHED"
                break
            result, gradient, hessian, gradient_norm, pre, route, direction, qp = (
                runtime.derivatives_and_direction(theta)
            )
            if not bits_equal(result.loss, loss):
                raise RuntimeError("objective/derivative loss divergence")
            if pre != "CONTINUE":
                stop_reason = pre
                break
            routes[route["route"]] += 1
            if direction is None:
                stop_reason = qp["route_status"]
                break

            projected = gradient.clone()
            projected[((theta == runtime.lower) & (gradient >= 0)) | ((theta == runtime.upper) & (gradient <= 0))] = 0
            mu = float(qp["damping"])
            runtime.matrix_rhs.construct(hessian, gradient, mu, runtime.log)
            descent, dot, _ = runtime.controls.descent.check(gradient, direction)
            if not descent:
                stop_reason = "no_descent_direction"
                break
            full_predicted = -float(qp["quadratic_value"])
            old_theta = theta
            old_loss = loss
            took_step = False
            unchanged = False
            for backtrack in range(production_n3.MAX_BACKTRACKS + 1):
                candidate = runtime.controls.candidate(theta, direction, backtrack)
                if torch.equal(candidate, theta):
                    unchanged = True
                    break
                if not runtime.controls.in_bounds(candidate, runtime.lower.numpy(), runtime.upper.numpy()):
                    rejections += 1
                    continue
                candidate_loss = runtime.scalar_loss(candidate.numpy())
                if runtime.controls.accepts(candidate_loss, loss):
                    displacement = candidate - theta
                    displacement_norm = float(displacement.norm())
                    actual_decrease = loss - candidate_loss
                    stagnation = production_n3._updated_float64_stagnation_count(
                        stagnation, loss, candidate_loss, full_predicted
                    )
                    theta = candidate.detach()
                    loss = candidate_loss
                    accepted += 1
                    path_length += displacement_norm
                    took_step = True
                    append_durable({
                        "record_type": "accepted_state",
                        "case": CASE,
                        "branch": BRANCH,
                        "direction_mode": STABILIZED_RESIDUAL,
                        "accepted_steps": accepted,
                        "theta": theta,
                        "loss": loss,
                        "stagnation_count": stagnation,
                        "objective_evaluations": runtime.objective_evaluations,
                        "backtracking_rejections": rejections,
                        "route_counts": dict(routes),
                        "parameter_path_length": path_length,
                        "update": {
                            "previous_loss": old_loss,
                            "gradient_norm": gradient_norm,
                            "projected_gradient_norm": float(projected.norm()),
                            "damping": mu,
                            "route": route["route"],
                            "backtrack": backtrack,
                            "active_bounds_before": torch.where((old_theta == runtime.lower) | (old_theta == runtime.upper))[0],
                            "active_bounds_after": torch.where((theta == runtime.lower) | (theta == runtime.upper))[0],
                            "direction_norm": float(direction.norm()),
                            "accepted_displacement_norm": displacement_norm,
                            "full_predicted_decrease": full_predicted,
                            "actual_loss_decrease": actual_decrease,
                        },
                    })
                    break
                rejections += 1
            if not took_step:
                numerical, _, _ = runtime.controls.no_further_improvement(unchanged, dot, loss)
                stop_reason = (
                    "floating_point_no_further_improvement"
                    if numerical else "backtracking_no_improvement"
                )
                break
            if accepted % 50 == 0:
                atomic(CHECKPOINT, checkpoint_payload(theta, loss, accepted, stagnation, runtime, rejections, routes, path_length))
            if accepted % 25 == 0 and accepted != last_printed:
                print(
                    f"STABILIZED LIVE steps={accepted} loss={loss:.17g} "
                    f"objective_evaluations={runtime.objective_evaluations} backtrack_rejections={rejections}",
                    flush=True,
                )
                last_printed = accepted
            if stagnation >= production_n3.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
                stop_reason = production_n3.FLOAT64_STAGNATION_STOP_REASON
                break
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "EXTERNAL_INTERRUPTION"

    atomic(CHECKPOINT, checkpoint_payload(theta, loss, accepted, stagnation, runtime, rejections, routes, path_length))
    handoff = bool(loss < HANDOFF)
    rollout = None
    behavioral_recovery = False
    if handoff:
        corpus = np.load(parent.DATA)
        rollout = original.rollout_diagnostic(
            adapter,
            trajectory.numpy(),
            corpus["teachers"][CASE_ID],
            theta.numpy(),
        )
        behavioral_recovery = bool(rollout["rollout_finite"])

    frozen_after = protected()
    checkpoints_after = hashes()
    if frozen_before != frozen_after or checkpoints_before != checkpoints_after:
        raise RuntimeError("frozen artifacts/checkpoints changed")
    if DEFAULT_DIRECTION_MODE != EXISTING_FULL_HESSIAN:
        raise RuntimeError("default route changed during live test")
    if runtime.binding_counters.existing_direction_outputs != 0:
        raise RuntimeError("opt-in run emitted an existing-direction output")

    status = (
        "GENERIC_UPDATE_NETWORK_FIRST_CASE_RECOVERED_WITH_STABILIZED_N3"
        if handoff and behavioral_recovery
        else "STABILIZED_N3_EXTERNAL_INTERRUPTION_RESUMABLE"
        if interrupted
        else "STABILIZED_RESIDUAL_N3_TERMINATED_ABOVE_HANDOFF"
    )
    final = {
        "status": status,
        "terminal": not interrupted,
        "case": CASE,
        "case_id": CASE_ID,
        "branch": BRANCH,
        "direction_mode": STABILIZED_RESIDUAL,
        "accepted_steps": accepted,
        "objective_evaluations": runtime.objective_evaluations,
        "backtracking_rejections": rejections,
        "route_counts": dict(routes),
        "parameter_path_length": path_length,
        "theta": theta,
        "loss": loss,
        "handoff": handoff,
        "rollout": rollout,
        "behavioral_recovery": behavioral_recovery,
        "stop_reason": stop_reason,
        "stagnation_count": stagnation,
        "active_bounds": torch.where((theta == runtime.lower) | (theta == runtime.upper))[0],
        "elapsed_seconds": time.monotonic() - started,
        "binding_counters": vars(runtime.binding_counters),
        "historical_existing_n3": historical_summary(),
        "newest_safe_resume": checkpoint_payload(theta, loss, accepted, stagnation, runtime, rejections, routes, path_length),
        "accepted_states_sha256": sha256(TRACE),
        "manifest_sha256": sha256(MANIFEST),
        "frozen_integrity_unchanged": True,
        "production_default": DEFAULT_DIRECTION_MODE,
        "no_affine_or_teacher_initialization": True,
    }
    atomic(RESULT, final)
    print(json.dumps(clean(final), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
