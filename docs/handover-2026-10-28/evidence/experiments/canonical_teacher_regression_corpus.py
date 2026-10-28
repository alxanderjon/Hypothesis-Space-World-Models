#!/usr/bin/env python3
"""Generate and load the only authoritative 822-case teacher corpus.

Every artifact is fail-closed against the frozen teacher manifest. Historical
artifacts are never inputs to generation and are not accepted by the loader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
import torch

from experiments import expand_chronological_archive_handoff_dataset as archive_data
from experiments import validate_d5_production_pipeline as production
from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "canonical_frozen_teacher_regression_822"
CASES = OUT / "cases"
SPEC_MANIFEST = OUT / "case_manifest.json"
CORPUS_MANIFEST = OUT / "corpus_manifest.json"
CANONICAL_POINTER = OUT / "CANONICAL"
FREEZE = ROOT / "outputs" / "deterministic_teacher_frozen" / "freeze_manifest.json"
CHECKPOINT_REFERENCE = ROOT / "outputs" / "deterministic_teacher_readiness" / "checkpoint_hashes_after.json"
SHO_LEGACY_INPUT = ROOT / "outputs" / "sho" / "stage1B" / "dimension_general_edirect" / "validation_cases.npz"
ARCHIVE_MANIFEST = ROOT / "outputs" / "chronological_archive_handoff_supervision_expanded" / "manifest.json"
ARCHIVE_STATES = ROOT / "outputs" / "chronological_archive_handoff_supervision_expanded" / "expanded_archive_states.jsonl"
READINESS_MANIFEST = ROOT / "outputs" / "deterministic_teacher_readiness" / "manifest.json"
REQUIRED_FREEZE_SHA256 = "66cfe5f07347bdadfb11baef61f4212918c42102cf48accf0d6563ab96c88759"
FINAL_GATE = 1.0e-20
N1_CAP = 5_000
SCHEMA_VERSION = 1


class ProvenanceError(RuntimeError):
    """An artifact is not an authoritative frozen-teacher target."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode() + b"\n")


def environment() -> dict:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
    }


@lru_cache(maxsize=1)
def frozen_context() -> dict:
    if sha256_file(FREEZE) != REQUIRED_FREEZE_SHA256:
        raise ProvenanceError("freeze-manifest hash mismatch")
    freeze = json.loads(FREEZE.read_text())
    for relative, expected in freeze["sources"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProvenanceError(f"frozen source mismatch: {relative}")
    actual_environment = environment()
    for key, expected in freeze["environment"].items():
        if actual_environment.get(key) != expected:
            raise ProvenanceError(
                f"environment mismatch {key}: {actual_environment.get(key)!r} != {expected!r}"
            )
    if sha256_file(CHECKPOINT_REFERENCE) != freeze["checkpoint_manifest_sha256"]:
        raise ProvenanceError("checkpoint-manifest file hash mismatch")
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected_checkpoints:
        raise ProvenanceError("checkpoint/source inventory changed")
    return {
        "freeze_manifest_sha256": REQUIRED_FREEZE_SHA256,
        "frozen_source_hashes": freeze["sources"],
        "environment": actual_environment,
        "checkpoint_manifest_sha256": freeze["checkpoint_manifest_sha256"],
    }


def seal(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("artifact_hash", None)
    value["artifact_hash"] = sha256_bytes(canonical_bytes(value))
    return value


def verify_seal(value: dict) -> None:
    observed = value.get("artifact_hash")
    if not isinstance(observed, str):
        raise ProvenanceError("artifact hash missing")
    payload = copy.deepcopy(value); payload.pop("artifact_hash", None)
    expected = sha256_bytes(canonical_bytes(payload))
    if observed != expected:
        raise ProvenanceError("artifact hash mismatch")


def case_key(spec: dict) -> str:
    return str(spec["canonical_case_id"])


def case_dir(spec: dict) -> Path:
    return CASES / spec["system"] / case_key(spec).replace(":", "__")


def artifact_path(spec: dict) -> Path:
    return case_dir(spec) / "artifact.json"


def _positive_archive_case_ids(system: str) -> list[int]:
    values = set()
    for line in ARCHIVE_STATES.read_text().splitlines():
        row = json.loads(line)
        if row["system"] == system and row["target_any_good"]:
            values.add(int(row["case_id"]))
    return sorted(values)


def build_specs() -> list[dict]:
    specs = []
    for index in range(256):
        specs.append({
            "canonical_case_id": f"full_k_sho:legacy_prefix:{index:03d}",
            "system": "full_k_sho", "family": "legacy_prefix",
            "case_id": index, "sobol_index": None,
            "n1_seed": 8_100_000 + index,
        })
    for index in range(64):
        specs.append({
            "canonical_case_id": f"full_k_sho:sobol:{index:03d}",
            "system": "full_k_sho", "family": "readiness_sobol",
            "case_id": index, "sobol_index": index,
            "n1_seed": 8_110_000 + index,
        })
    for index in _positive_archive_case_ids("reflection_ball"):
        specs.append({
            "canonical_case_id": f"reflection_ball:sobol:{index:05d}",
            "system": "reflection_ball", "family": "sobol_identifiable",
            "case_id": index, "sobol_index": index,
            "n1_seed": 8_200_000 + index,
        })
    readiness_specs = json.loads(READINESS_MANIFEST.read_text())["cases"]
    for row in readiness_specs:
        if row["system"] != "reflection_ball":
            continue
        index = int(row["sobol_index"])
        specs.append({
            "canonical_case_id": f"reflection_ball:sobol:{index:05d}",
            "system": "reflection_ball", "family": "sobol_identifiable",
            "case_id": index, "sobol_index": index,
            "n1_seed": 8_200_000 + index,
            "identifiability": row.get("identifiability"),
        })
    for index in range(256):
        specs.append({
            "canonical_case_id": f"kepler5:sobol:{index:05d}",
            "system": "kepler5", "family": "sobol",
            "case_id": index, "sobol_index": index,
            "n1_seed": 8_300_000 + index,
        })
    keys = [case_key(row) for row in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate canonical case identity")
    counts = Counter(row["system"] for row in specs)
    if counts != Counter({"full_k_sho": 320, "reflection_ball": 246, "kepler5": 256}):
        raise RuntimeError(f"wrong canonical case counts: {counts}")
    return specs


def ensure_spec_manifest() -> list[dict]:
    context = frozen_context()
    specs = build_specs()
    payload = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": "canonical_teacher_case_manifest",
        "provenance": context,
        "counts": dict(Counter(row["system"] for row in specs)),
        "cases": specs,
    })
    if SPEC_MANIFEST.exists():
        existing = json.loads(SPEC_MANIFEST.read_text())
        verify_seal(existing)
        if existing != payload:
            raise ProvenanceError("canonical case manifest changed")
    else:
        atomic_json(SPEC_MANIFEST, payload)
    return specs


def case_data(spec: dict):
    system = spec["system"]
    if system == "full_k_sho" and spec["family"] == "legacy_prefix":
        data = np.load(SHO_LEGACY_INPUT)
        index = int(spec["case_id"])
        trajectory = np.asarray(data[f"prefix_{index}"], np.float64)
        teacher = np.asarray(data["teachers"][index], np.float64).ravel()
        initial = {
            "source": str(SHO_LEGACY_INPUT.relative_to(ROOT)),
            "prefix_index": index, "prefix_length": int(len(trajectory)),
        }
        objective, torch_objective = readiness.sho_objectives(trajectory)
        lower = np.full(4, -1.99, np.float64)
        upper = np.full(4, 1.99, np.float64)
        return trajectory, initial, objective, torch_objective, lower, upper, teacher
    if system == "full_k_sho":
        trajectory, initial, teacher = readiness.sho_case(int(spec["sobol_index"]))
        objective, torch_objective = readiness.sho_objectives(trajectory)
        lower = np.full(4, -1.99, np.float64)
        upper = np.full(4, 1.99, np.float64)
        return trajectory, initial, objective, torch_objective, lower, upper, teacher
    if system == "reflection_ball":
        trajectory, initial = archive_data.ball_case(int(spec["sobol_index"]))
    else:
        trajectory, initial = archive_data.kepler_case(int(spec["sobol_index"]))
    objective, torch_objective = production._objectives(system, trajectory)
    lower, upper, teacher = production._config(system)
    return trajectory, initial, objective, torch_objective, lower, upper, teacher


def _npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def n3_diagnostics(result, derivatives, direction_function, lower, upper) -> dict:
    gradients = []; hessians = []; directions = []; routes = []
    conditions = []; dampings = []; alphas = []; predicted = []
    active_iterations = []; blocking_events = []; release_events = []
    for step, value in enumerate(result.theta_history):
        theta = torch.from_numpy(np.asarray(value, np.float64).copy())
        _, gradient, hessian = derivatives(theta, torch.empty(0))
        direction, info = dimension_general_box_qp_direction(
            theta, gradient, hessian, lower, upper, direction_function
        )
        gradient_np = gradient.detach().cpu().numpy().copy()
        hessian_np = hessian.detach().cpu().numpy().copy()
        direction_np = (np.full(len(value), np.nan, np.float64) if direction is None
                        else direction.detach().cpu().numpy().copy())
        gradients.append(gradient_np); hessians.append(hessian_np)
        directions.append(direction_np); routes.append(str(info["route_status"]))
        conditions.append(float(info["condition"])); dampings.append(float(info["damping"]))
        active_iterations.append(int(info["active_set_iterations"]))
        blocking_events.append(int(info["blocking_events"])); release_events.append(int(info["release_events"]))
        if direction is None:
            predicted.append(math.nan)
        else:
            model = -(torch.dot(gradient, direction) + 0.5 * torch.dot(
                direction, hessian @ direction))
            predicted.append(float(model))
        if step + 1 < len(result.theta_history) and direction is not None:
            displacement = np.asarray(result.theta_history[step + 1]) - np.asarray(value)
            nonzero = np.flatnonzero(direction_np != 0.0)
            alpha = (float(displacement[nonzero[0]] / direction_np[nonzero[0]])
                     if len(nonzero) else 0.0)
            alphas.append(alpha)
        else:
            alphas.append(math.nan)
    return {
        "gradient": np.asarray(gradients, np.float64),
        "hessian": np.asarray(hessians, np.float64),
        "direction": np.asarray(directions, np.float64),
        "route": np.asarray(routes), "condition": np.asarray(conditions, np.float64),
        "damping": np.asarray(dampings, np.float64), "accepted_alpha": np.asarray(alphas, np.float64),
        "full_predicted_decrease": np.asarray(predicted, np.float64),
        "active_set_iterations": np.asarray(active_iterations, np.int64),
        "blocking_events": np.asarray(blocking_events, np.int64),
        "release_events": np.asarray(release_events, np.int64),
    }


def branch_paths(spec: dict, n: int, index: int) -> tuple[Path, Path]:
    root = case_dir(spec) / "branches"
    stem = f"n{n:05d}_i{index + 1:05d}"
    return root / f"{stem}.json", root / f"{stem}.npz"


def load_authoritative_artifact(path: Path, *, expected_case_id: str | None = None) -> dict:
    """Load an authoritative artifact or fail closed on any provenance defect."""
    if not path.is_file() or OUT not in path.parents:
        raise ProvenanceError("artifact is outside the canonical corpus")
    value = json.loads(path.read_text())
    verify_seal(value)
    context = frozen_context()
    provenance = value.get("provenance")
    if provenance != context:
        raise ProvenanceError("artifact provenance does not match frozen context")
    identity = value.get("case_identity", {})
    if expected_case_id is not None and identity.get("canonical_case_id") != expected_case_id:
        raise ProvenanceError("case identity mismatch")
    for trace in value.get("trace_files", []):
        trace_path = OUT / trace["relative_path"]
        if not trace_path.is_file() or sha256_file(trace_path) != trace["sha256"]:
            raise ProvenanceError(f"trace hash mismatch: {trace_path}")
    return value


def _validated_case_branches(case: dict) -> list[tuple[Path, dict]]:
    """Validate every branch and its exact chronological references."""
    spec = case["case_identity"]
    expected_id = case_key(spec)
    rows = []
    by_location = {}
    for path in sorted((case_dir(spec) / "branches").glob("*.json")):
        branch = load_authoritative_artifact(path, expected_case_id=expected_id)
        trace_path = OUT / branch["trace_files"][0]["relative_path"]
        with np.load(trace_path, allow_pickle=False) as trace:
            required = {
                "n2_theta", "n2_loss", "n3_theta", "n3_loss", "n3_gradient",
                "n3_hessian", "n3_direction", "n3_route", "n3_condition",
                "n3_damping", "n3_accepted_alpha", "n3_full_predicted_decrease",
                "n3_active_set_iterations", "n3_blocking_events", "n3_release_events",
            }
            if not required.issubset(trace.files):
                raise ProvenanceError(f"incomplete branch trace: {trace_path}")
            if len(trace["n2_loss"]) != int(branch["n2"]["evaluations"]):
                raise ProvenanceError(f"N2 trace length mismatch: {trace_path}")
            if len(trace["n3_theta"]) != int(branch["n3"]["accepted_steps"]) + 1:
                raise ProvenanceError(f"N3 trace length mismatch: {trace_path}")
            if any(str(value) == "EMITTED_SUMMARY" for value in trace["n3_route"]):
                raise ProvenanceError(f"compact N3 route remains: {trace_path}")
        location = (int(branch["current_evaluation_count"]),
                    int(branch["archive_index"]))
        if location in by_location:
            raise ProvenanceError(f"duplicate branch location for {expected_id}: {location}")
        by_location[location] = branch
        rows.append((path, branch))

    chronological = []
    for state in case["archive_states"]:
        n = int(state["evaluation_count"])
        expected_indexes = ([n - 1] + list(range(n - 1)))[
            :len(state["branch_artifact_hashes_in_probe_order"])
        ]
        actual = []
        for index in expected_indexes:
            branch = by_location.get((n, index))
            if branch is None:
                raise ProvenanceError(
                    f"missing branch for {expected_id} at {(n, index)}"
                )
            actual.append(branch["artifact_hash"])
        if actual != state["branch_artifact_hashes_in_probe_order"]:
            raise ProvenanceError(f"archive branch order mismatch for {expected_id}")
        chronological.extend(actual)
    if chronological != case["branch_order"]:
        raise ProvenanceError(f"case branch order mismatch for {expected_id}")
    if len(rows) != len(chronological):
        raise ProvenanceError(f"unreferenced branch artifact for {expected_id}")
    successful = [branch["artifact_hash"] for _, branch in rows
                  if branch["existing_recovery_criterion"]]
    expected_success = case["successful_handoff_artifact_hash"]
    if expected_success is None:
        if successful:
            raise ProvenanceError(f"unrecorded successful branch for {expected_id}")
    elif expected_success not in successful:
        raise ProvenanceError(f"successful branch mismatch for {expected_id}")
    return rows


def enrich_incomplete_branch_diagnostics(specs: list[dict]) -> int:
    """Fill compact branch traces and reseal their owning cases.

    N1, N2, and N3 are not rerun. The frozen derivative and box-QP diagnostic
    path is replayed on the already saved N3 theta trajectory so every branch
    has the complete routing trace required by the canonical corpus.
    """
    upgraded = 0
    for spec in specs:
        case_path = artifact_path(spec)
        case = load_authoritative_artifact(case_path, expected_case_id=case_key(spec))
        _, _, _, torch_objective, lower, upper, _ = case_data(spec)
        derivatives = direction_function = None
        by_location = {}
        for path in sorted((case_dir(spec) / "branches").glob("*.json")):
            branch = load_authoritative_artifact(path, expected_case_id=case_key(spec))
            trace_path = OUT / branch["trace_files"][0]["relative_path"]
            with np.load(trace_path, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            routes = arrays.get("n3_route")
            compact = routes is not None and any(
                str(value) == "EMITTED_SUMMARY" for value in routes
            )
            if compact:
                if derivatives is None:
                    derivatives, direction_function, _ = staged._make_n3(
                        torch_objective, lower, upper
                    )
                result = SimpleNamespace(theta_history=arrays["n3_theta"])
                diagnostic = n3_diagnostics(
                    result, derivatives, direction_function, lower, upper
                )
                for key in list(arrays):
                    if key.startswith("n3_") and key not in ("n3_theta", "n3_loss"):
                        arrays.pop(key)
                arrays.update({f"n3_{key}": value
                               for key, value in diagnostic.items()})
                _npz_atomic(trace_path, **arrays)
                branch["trace_files"][0]["sha256"] = sha256_file(trace_path)
                branch = seal(branch)
                atomic_json(path, branch)
                upgraded += 1
            by_location[(int(branch["current_evaluation_count"]),
                         int(branch["archive_index"]))] = branch

        chronological = []
        for state in case["archive_states"]:
            n = int(state["evaluation_count"])
            count = len(state["branch_artifact_hashes_in_probe_order"])
            indexes = ([n - 1] + list(range(n - 1)))[:count]
            hashes = [by_location[(n, index)]["artifact_hash"] for index in indexes]
            state["branch_artifact_hashes_in_probe_order"] = hashes
            chronological.extend(hashes)
        case["branch_order"] = chronological
        identity = case.get("successful_handoff_identity")
        if identity is not None:
            location = (int(identity["evaluation_count"]),
                        int(identity["archive_index"]))
            case["successful_handoff_artifact_hash"] = by_location[location]["artifact_hash"]
        case = seal(case)
        atomic_json(case_path, case)
        _validated_case_branches(case)
    return upgraded


def run_branch(spec, trajectory, objective, torch_objective, lower, upper,
               teacher, runner, n: int, index: int) -> dict:
    json_path, trace_path = branch_paths(spec, n, index)
    if json_path.exists():
        return load_authoritative_artifact(json_path, expected_case_id=case_key(spec))
    started = time.monotonic()
    theta = lower + (upper - lower) * runner.points[index]
    radius = (upper - lower) / float(n) ** (1.0 / len(lower))
    box_lower = np.maximum(lower, theta - radius)
    box_upper = np.minimum(upper, theta + radius)
    n2 = staged._run_n2(objective, theta, float(runner.losses[index]), box_lower, box_upper)
    derivatives, direction_function, run_n3 = staged._make_n3(torch_objective, lower, upper)
    endpoint = polish.N2Endpoint(
        0, n2["theta"].copy(), float(n2["loss"]),
        "exact_learned_assembly_powell", "canonical_frozen_teacher", False,
    )
    n3 = run_n3(endpoint, torch.empty(0))
    final = np.asarray(n3.theta_history[-1], np.float64)
    rollout = readiness.rollout(spec["system"], trajectory, final, teacher)
    recovered = bool(n3.loss_history[-1] < FINAL_GATE and rollout["rollout_finite"])
    # The successful handoff receives the complete derivative/route replay.
    # Failed archive probes retain the complete executed N3 trajectory and the
    # diagnostics emitted by N3 itself without a second derivative replay.
    if recovered:
        diagnostic = n3_diagnostics(
            n3, derivatives, direction_function, lower, upper
        )
    else:
        count = len(n3.theta_history)
        diagnostic = {
            "gradient_norm": np.asarray(n3.gradient_norms, np.float64),
            "condition": np.asarray(n3.condition_numbers, np.float64),
            "damping": np.asarray(n3.damping_values, np.float64),
            "route": np.asarray(["EMITTED_SUMMARY"] * count),
        }
    _npz_atomic(
        trace_path,
        n2_theta=np.asarray([row["theta"] for row in n2["trace"]], np.float64),
        n2_loss=np.asarray([row["loss"] for row in n2["trace"]], np.float64),
        n3_theta=np.asarray(n3.theta_history, np.float64),
        n3_loss=np.asarray(n3.loss_history, np.float64),
        **{f"n3_{key}": value for key, value in diagnostic.items()},
    )
    context = frozen_context()
    parameter_error = final - teacher
    payload = seal({
        "schema_version": SCHEMA_VERSION, "kind": "canonical_teacher_branch",
        "provenance": context, "case_identity": spec,
        "current_evaluation_count": n, "archive_index": index,
        "insertion_evaluation": index + 1,
        "archive_theta": theta.tolist(), "archive_loss": float(runner.losses[index]),
        "n2_box_lower": box_lower.tolist(), "n2_box_upper": box_upper.tolist(),
        "n2": {"evaluations": int(n2["evaluations"]), "stop_reason": n2["message"],
               "endpoint": n2["theta"].tolist(), "loss": float(n2["loss"])},
        "n3": {"accepted_steps": int(n3.accepted_steps),
               "objective_evaluations": int(n3.actual_loss_evaluations),
               "backtracking_rejections": int(n3.backtracking_rejections),
               "accepted_with_backtracking": int(n3.accepted_with_backtracking),
               "damping_steps": int(n3.damping_steps), "stop_reason": n3.stop_reason,
               "final_theta": final.tolist(), "final_loss": float(n3.loss_history[-1])},
        "offline": {"parameter_error": parameter_error.tolist(),
                    "max_parameter_error": float(np.max(np.abs(parameter_error))),
                    "l2_parameter_error": float(np.linalg.norm(parameter_error)), **rollout},
        "existing_recovery_criterion": recovered,
        "trace_files": [{"relative_path": str(trace_path.relative_to(OUT)),
                         "sha256": sha256_file(trace_path)}],
        "elapsed_seconds": float(time.monotonic() - started),
    })
    atomic_json(json_path, payload)
    return payload


def run_case(spec: dict) -> dict:
    torch.set_num_threads(1)
    target = artifact_path(spec)
    if target.exists():
        return load_authoritative_artifact(target, expected_case_id=case_key(spec))
    started = time.monotonic()
    trajectory, initial, objective, torch_objective, lower, upper, teacher = case_data(spec)
    runner = archive_data.ArchiveN1Runner(
        objective, lower, upper, int(spec["n1_seed"])
    )
    branches = []; archive_states = []; success = None
    while len(runner.losses) < N1_CAP and success is None:
        runner.advance_one(); n = len(runner.losses)
        state_branches = []
        for index in [n - 1] + list(range(n - 1)):
            branch = run_branch(spec, trajectory, objective, torch_objective,
                                lower, upper, teacher, runner, n, index)
            branches.append(branch); state_branches.append(branch["artifact_hash"])
            if branch["existing_recovery_criterion"]:
                success = branch
                break
        archive_states.append({
            "evaluation_count": n, "site_count": n,
            "points": [np.asarray(row).tolist() for row in runner.points],
            "losses": list(map(float, runner.losses)),
            "branch_artifact_hashes_in_probe_order": state_branches,
            "success": success is not None,
        })
    if success is None:
        classification = "RESOURCE_INCOMPLETE"
    else:
        classification = "RECOVERED"
    trace_path = case_dir(spec) / "n1_archive_trace.npz"
    _npz_atomic(
        trace_path, normalized_points=np.asarray(runner.points, np.float64),
        losses=np.asarray(runner.losses, np.float64), lower=lower, upper=upper,
        teacher=teacher, trajectory=trajectory,
    )
    payload = seal({
        "schema_version": SCHEMA_VERSION, "kind": "canonical_teacher_case",
        "provenance": frozen_context(), "case_identity": spec,
        "input": {"initial_condition": initial,
                  "trajectory_sha256": sha256_bytes(np.ascontiguousarray(trajectory).tobytes()),
                  "teacher_parameters": teacher.tolist()},
        "teacher_pipeline": (
            "N1 -> chronological archive -> newest-first/older retries -> exact Powell N2 "
            "-> box-QP uncapped N3 -> FLOAT64_STAGNATION_TAIL -> absorbing success"
        ),
        "n1": {"evaluations": len(runner.losses), "events": runner.events,
               "continuous_traces": runner.traces},
        "archive_states": archive_states,
        "branch_order": [row["artifact_hash"] for row in branches],
        "successful_handoff_artifact_hash": (
            None if success is None else success["artifact_hash"]),
        "successful_handoff_identity": (None if success is None else {
            "evaluation_count": success["current_evaluation_count"],
            "archive_index": success["archive_index"],
            "insertion_evaluation": success["insertion_evaluation"],
        }),
        "classification": classification,
        "final": (None if success is None else {
            "parameters": success["n3"]["final_theta"],
            "loss": success["n3"]["final_loss"],
            "n3_stop_reason": success["n3"]["stop_reason"],
            "n3_accepted_steps": success["n3"]["accepted_steps"],
            **success["offline"],
        }),
        "trace_files": [{"relative_path": str(trace_path.relative_to(OUT)),
                         "sha256": sha256_file(trace_path)}],
        "elapsed_seconds": float(time.monotonic() - started),
    })
    atomic_json(target, payload)
    return payload


def load_canonical_corpus() -> list[dict]:
    """The sole supported regression-target loader."""
    if not CORPUS_MANIFEST.is_file() or not CANONICAL_POINTER.is_file():
        raise ProvenanceError("canonical corpus is incomplete")
    manifest = json.loads(CORPUS_MANIFEST.read_text()); verify_seal(manifest)
    if CANONICAL_POINTER.read_text().strip() != manifest["artifact_hash"]:
        raise ProvenanceError("canonical pointer does not match corpus manifest")
    if manifest.get("provenance") != frozen_context():
        raise ProvenanceError("corpus manifest provenance mismatch")
    records = []
    for row in manifest["artifacts"]:
        path = OUT / row["relative_path"]
        if sha256_file(path) != row["file_sha256"]:
            raise ProvenanceError(f"case artifact file hash mismatch: {path}")
        artifact = load_authoritative_artifact(
            path, expected_case_id=row["canonical_case_id"]
        )
        if artifact["artifact_hash"] != row["artifact_hash"]:
            raise ProvenanceError("case artifact payload hash mismatch")
        branches = _validated_case_branches(artifact)
        expected_branches = manifest.get("branch_artifacts", {}).get(
            row["canonical_case_id"]
        )
        if expected_branches is None:
            raise ProvenanceError("corpus manifest is missing branch provenance")
        actual_branches = [{
            "relative_path": str(path.relative_to(OUT)),
            "artifact_hash": branch["artifact_hash"],
            "file_sha256": sha256_file(path),
        } for path, branch in branches]
        if actual_branches != expected_branches:
            raise ProvenanceError("branch artifact manifest mismatch")
        records.append(artifact)
    counts = Counter(row["case_identity"]["system"] for row in records)
    if len(records) != 822 or counts != Counter(
        {"full_k_sho": 320, "reflection_ball": 246, "kepler5": 256}
    ):
        raise ProvenanceError("canonical corpus case counts mismatch")
    return records


def load_case_branch(case: dict, artifact_hash: str) -> dict:
    """Resolve a branch only inside its guarded canonical case directory."""
    spec = case["case_identity"]
    for path in sorted((case_dir(spec) / "branches").glob("*.json")):
        value = load_authoritative_artifact(path, expected_case_id=case_key(spec))
        if value["artifact_hash"] == artifact_hash:
            return value
    raise ProvenanceError(
        f"canonical branch {artifact_hash} missing for {case_key(spec)}"
    )


def provenance_guard_self_test() -> dict:
    """Exercise every required fail-closed provenance field."""
    manifest = json.loads(CORPUS_MANIFEST.read_text())
    source_path = OUT / manifest["artifacts"][0]["relative_path"]
    source = json.loads(source_path.read_text())
    expected_id = source["case_identity"]["canonical_case_id"]
    scratch = OUT / "_provenance_guard_self_test"
    scratch.mkdir(parents=True, exist_ok=True)
    tests = []

    def expect_rejection(name: str, payload: dict, *, seal_payload: bool = True,
                         expected_case_id: str = expected_id) -> None:
        path = scratch / f"{name}.json"
        value = seal(payload) if seal_payload else payload
        atomic_json(path, value)
        try:
            load_authoritative_artifact(path, expected_case_id=expected_case_id)
        except ProvenanceError as error:
            tests.append({"name": name, "rejected": True, "reason": str(error)})
        else:
            tests.append({"name": name, "rejected": False, "reason": None})
        path.unlink()

    value = copy.deepcopy(source); value.pop("provenance", None)
    expect_rejection("missing_provenance", value)
    value = copy.deepcopy(source)
    value["provenance"]["freeze_manifest_sha256"] = "0" * 64
    expect_rejection("mismatched_freeze_manifest", value)
    value = copy.deepcopy(source)
    first_source = next(iter(value["provenance"]["frozen_source_hashes"]))
    value["provenance"]["frozen_source_hashes"][first_source] = "0" * 64
    expect_rejection("mismatched_frozen_source", value)
    value = copy.deepcopy(source)
    value["provenance"]["environment"].pop("torch", None)
    expect_rejection("missing_environment_metadata", value)
    value = copy.deepcopy(source)
    value["case_identity"]["canonical_case_id"] = "wrong:case"
    expect_rejection("mismatched_case_identity", value)
    value = copy.deepcopy(source)
    value["classification"] = "TAMPERED_WITHOUT_RESEAL"
    expect_rejection("mismatched_artifact_hash", value, seal_payload=False)
    value = copy.deepcopy(source)
    value["trace_files"][0]["sha256"] = "0" * 64
    expect_rejection("mismatched_trace_hash", value)
    try:
        load_authoritative_artifact(ARCHIVE_STATES)
    except ProvenanceError as error:
        tests.append({"name": "historical_artifact_outside_canonical_namespace",
                      "rejected": True, "reason": str(error)})
    else:
        tests.append({"name": "historical_artifact_outside_canonical_namespace",
                      "rejected": False, "reason": None})
    scratch.rmdir()
    if not all(row["rejected"] for row in tests):
        raise ProvenanceError("provenance guard self-test accepted an invalid artifact")
    report = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": "canonical_provenance_guard_validation",
        "provenance": frozen_context(),
        "tests": tests,
        "rejected": len(tests),
        "accepted": 0,
    })
    atomic_json(OUT / "provenance_guard_validation.json", report)
    return report


def distribution(values) -> dict:
    a = np.asarray(values, np.float64)
    return {"count": int(len(a)), "minimum": float(np.min(a)),
            "median": float(np.median(a)),
            "p90": float(np.quantile(a, .9, method="higher")),
            "p99": float(np.quantile(a, .99, method="higher")),
            "maximum": float(np.max(a))}


def finalize(specs: list[dict]) -> dict:
    artifacts = []
    branch_artifacts = {}
    loaded = []
    for spec in specs:
        path = artifact_path(spec)
        value = load_authoritative_artifact(path, expected_case_id=case_key(spec))
        loaded.append(value)
        artifacts.append({
            "canonical_case_id": case_key(spec),
            "relative_path": str(path.relative_to(OUT)),
            "artifact_hash": value["artifact_hash"],
            "file_sha256": sha256_file(path),
        })
        branch_artifacts[case_key(spec)] = [{
            "relative_path": str(branch_path.relative_to(OUT)),
            "artifact_hash": branch["artifact_hash"],
            "file_sha256": sha256_file(branch_path),
        } for branch_path, branch in _validated_case_branches(value)]
    by_system = {}
    for system in ("full_k_sho", "reflection_ball", "kepler5"):
        rows = [row for row in loaded if row["case_identity"]["system"] == system]
        recovered = [row for row in rows if row["classification"] == "RECOVERED"]
        by_system[system] = {
            "cases": len(rows), "recoveries": len(recovered),
            "parameter_max_error": distribution([row["final"]["max_parameter_error"] for row in recovered]),
            "parameter_l2_error": distribution([row["final"]["l2_parameter_error"] for row in recovered]),
            "rollout_max_error": distribution([row["final"]["continuation_max_error"] for row in recovered]),
            "final_loss": distribution([row["final"]["loss"] for row in recovered]),
            "n1_evaluations": distribution([row["n1"]["evaluations"] for row in recovered]),
            "n3_stop_reasons": dict(Counter(row["final"]["n3_stop_reason"] for row in recovered)),
        }
    manifest = seal({
        "schema_version": SCHEMA_VERSION, "kind": "canonical_teacher_corpus",
        "status": "CANONICAL_FROZEN_TEACHER_REGRESSION_CORPUS",
        "provenance": frozen_context(), "case_manifest_sha256": sha256_file(SPEC_MANIFEST),
        "counts": dict(Counter(row["case_identity"]["system"] for row in loaded)),
        "total_cases": len(loaded),
        "total_archive_branches": sum(map(len, branch_artifacts.values())),
        "complete_branch_traces_verified": True,
        "frozen_source_and_checkpoint_integrity_verified": True,
        "systems": by_system,
        "legacy_or_missing_provenance_accepted": 0,
        "artifacts": artifacts, "branch_artifacts": branch_artifacts,
    })
    atomic_json(CORPUS_MANIFEST, manifest)
    atomic_bytes(CANONICAL_POINTER, (manifest["artifact_hash"] + "\n").encode())
    # Load through the public fail-closed path as the final validation.
    verified = load_canonical_corpus()
    guard = provenance_guard_self_test()
    report = {
        "status": "CANONICAL_CORPUS_VALIDATED", "cases_verified": len(verified),
        "archive_branches_verified": manifest["total_archive_branches"],
        "complete_branch_traces_verified": True,
        "frozen_source_and_checkpoint_integrity_verified": True,
        "systems": by_system, "corpus_manifest": str(CORPUS_MANIFEST),
        "corpus_manifest_file_sha256": sha256_file(CORPUS_MANIFEST),
        "corpus_payload_hash": manifest["artifact_hash"],
        "legacy_or_missing_provenance_accepted": 0,
        "provenance_guard_invalid_artifacts_rejected": guard["rejected"],
    }
    atomic_json(OUT / "report.json", report)
    return report


def worker(spec: dict) -> tuple[str, str]:
    value = run_case(spec)
    return case_key(spec), value["classification"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    specs = ensure_spec_manifest()
    if args.verify_only:
        rows = load_canonical_corpus(); print(f"verified {len(rows)} cases")
        return
    if args.case:
        spec = next(row for row in specs if case_key(row) == args.case)
        print(json.dumps(run_case(spec), indent=2)); return
    pending = [row for row in specs if not artifact_path(row).exists()]
    completed = len(specs) - len(pending)
    print(json.dumps({"total": len(specs), "completed": completed,
                      "pending": len(pending)}, indent=2), flush=True)
    workers = max(1, min(args.workers, 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, row): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            key, status = future.result()
            completed += 1
            if index % 10 == 0 or index == len(pending):
                print(json.dumps({"completed": completed, "total": len(specs),
                                  "last": key, "status": status}), flush=True)
    print(json.dumps(finalize(specs), indent=2), flush=True)


if __name__ == "__main__":
    main()
