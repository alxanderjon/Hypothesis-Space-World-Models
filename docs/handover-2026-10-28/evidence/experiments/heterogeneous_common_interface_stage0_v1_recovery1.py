#!/usr/bin/env python3
"""Additive recovery for Stage-0 path-resolution failure.

Attempt 1 stopped before its first trajectory read because it guessed a
directory stem from a provenance-family label.  This recovery resolves every
artifact through the authoritative corpus manifest.  The original namespace
is left untouched and its zero trajectory reads remain recorded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/canonical_frozen_teacher_regression_822"
ATTEMPT1 = ROOT / "outputs/heterogeneous_common_interface_stage0_v1"
OUT = ROOT / "outputs/heterogeneous_common_interface_stage0_v1_recovery1"
MAX_TRAJECTORY_READS = 822
MAX_STATE_ROWS = 500_000
MAX_WALL_SECONDS = 60.0
MAX_RSS_BYTES = 1 << 30
MAX_ARTIFACT_BYTES = 1 << 20


def load_base():
    path = ROOT / "experiments/heterogeneous_common_interface_stage0_v1.py"
    spec = importlib.util.spec_from_file_location("heterogeneous_stage0_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def main() -> None:
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(f"write-once output exists: {OUT}")
    if not (ATTEMPT1 / "plan.json").is_file() or (ATTEMPT1 / "result.json").exists():
        raise RuntimeError("attempt1 evidence does not match path-resolution stop")
    OUT.mkdir(parents=True)
    base = load_base()
    plan = {
        "status": "RECOVERY1_PREFLIGHT_PRECHARGED",
        "reason": "authoritative artifact-map resolution replaces guessed directory stem",
        "attempt1": {
            "namespace": str(ATTEMPT1.relative_to(ROOT)),
            "plan_sha256": sha256(ATTEMPT1 / "plan.json"),
            "trajectory_reads": 0,
            "scientific_operations": 0,
            "outcome": "PATH_RESOLUTION_STOP_BEFORE_FIRST_TRAJECTORY_READ",
        },
        "recovery_trajectory_reads_ceiling": MAX_TRAJECTORY_READS,
        "cumulative_trajectory_reads_ceiling": MAX_TRAJECTORY_READS,
        "state_rows_ceiling": MAX_STATE_ROWS,
        "wall_seconds_ceiling": MAX_WALL_SECONDS,
        "rss_bytes_ceiling": MAX_RSS_BYTES,
        "artifact_bytes_ceiling": MAX_ARTIFACT_BYTES,
        "driver_sha256": sha256(Path(__file__)),
        "base_driver_sha256": sha256(ROOT / "experiments/heterogeneous_common_interface_stage0_v1.py"),
        "prohibited": ["simulation", "fitting", "prediction", "scoring", "teacher_parameter_export"],
    }
    atomic_json(OUT / "plan.json", plan)

    corpus = json.loads((SOURCE / "corpus_manifest.json").read_text())
    cases = json.loads((SOURCE / "case_manifest.json").read_text())
    artifacts = {row["canonical_case_id"]: row for row in corpus["artifacts"]}
    if len(artifacts) != 822 or len(cases["cases"]) != 822:
        raise RuntimeError("canonical artifact inventory mismatch")

    counts = {key: 0 for key in base.EXPECTED_SYSTEMS}
    lengths = {key: set() for key in base.EXPECTED_SYSTEMS}
    shapes = {key: set() for key in base.EXPECTED_SYSTEMS}
    masks = {key: set() for key in base.EXPECTED_SYSTEMS}
    reads = rows = finite_values = 0
    roundtrip_max_error = 0.0
    for case in cases["cases"]:
        system = case["system"]
        entry = artifacts[case["canonical_case_id"]]
        artifact_path = SOURCE / entry["relative_path"]
        if sha256(artifact_path) != entry["file_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {artifact_path}")
        artifact = json.loads(artifact_path.read_text())
        trace = artifact["trace_files"][0]
        trace_path = SOURCE / trace["relative_path"]
        if sha256(trace_path) != trace["sha256"]:
            raise RuntimeError(f"trace hash mismatch: {trace_path}")
        if reads >= MAX_TRAJECTORY_READS:
            raise RuntimeError("trajectory-read ceiling would be exceeded")
        with np.load(trace_path, allow_pickle=False) as values:
            trajectory = np.asarray(values["trajectory"], dtype=np.float64)
        reads += 1
        rows += len(trajectory)
        if rows > MAX_STATE_ROWS:
            raise RuntimeError("state-row ceiling exceeded")
        if not np.isfinite(trajectory).all():
            raise RuntimeError(f"nonfinite trajectory: {trace_path}")
        neutral, entity_mask, channel_mask = base.to_neutral(system, trajectory)
        recovered = base.from_neutral(system, neutral)
        error = float(np.max(np.abs(recovered - trajectory)))
        if error != 0.0:
            raise RuntimeError(f"non-exact round trip: {trace_path}: {error}")
        roundtrip_max_error = max(roundtrip_max_error, error)
        finite_values += int(trajectory.size)
        counts[system] += 1
        lengths[system].add(int(len(trajectory)))
        shapes[system].add(tuple(int(x) for x in trajectory.shape))
        masks[system].add(hashlib.sha256(entity_mask.tobytes() + channel_mask.tobytes()).hexdigest())

    if counts != base.EXPECTED_SYSTEMS:
        raise RuntimeError(f"system count mismatch: {counts}")
    elapsed = time.monotonic() - started
    resources = {"wall_seconds": elapsed, "peak_rss_bytes": rss_bytes()}
    if elapsed > MAX_WALL_SECONDS or resources["peak_rss_bytes"] > MAX_RSS_BYTES:
        raise RuntimeError(f"resource ceiling exceeded: {resources}")
    result = {
        "status": "HETEROGENEOUS_COMMON_INTERFACE_STAGE0_RECOVERY1_COMPLETE",
        "classification": "INTERFACE_FEASIBLE_NO_SCIENTIFIC_RESULT",
        "counts": {
            "attempt1_scientific_operations": 0,
            "recovery_scientific_operations": 0,
            "attempt1_trajectory_reads": 0,
            "recovery_trajectory_reads": reads,
            "cumulative_trajectory_reads": reads,
            "state_rows": rows,
            "finite_scalar_values": finite_values,
            "systems": counts,
        },
        "source_shapes": {key: sorted(list(value)) for key, value in shapes.items()},
        "trajectory_lengths": {key: sorted(value) for key, value in lengths.items()},
        "neutral_shape": [2, 4],
        "mask_hashes": {key: sorted(value) for key, value in masks.items()},
        "roundtrip_max_abs_error": roundtrip_max_error,
        "quarantined_fields_not_exported": [
            "system", "family", "teacher_parameters", "initial_condition", "simulator_source",
            "mass", "G", "wall_bounds", "integrator", "force_parameters",
        ],
        "resources": resources,
        "qualification": [
            "This validates only a lossless neutral tensor schema over opened cached trajectories.",
            "It performs no learning and supplies no evidence of cross-mechanism transfer.",
            "Ball and Kepler cases still vary initial conditions around one teacher rule each.",
        ],
    }
    atomic_json(OUT / "result.json", result)
    artifact_bytes = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    if artifact_bytes > MAX_ARTIFACT_BYTES:
        raise RuntimeError("artifact ceiling exceeded")
    atomic_json(OUT / "completion.json", {
        "status": "COMPLETE",
        "result_sha256": sha256(OUT / "result.json"),
        "artifact_bytes_before_completion": artifact_bytes,
        "counts": result["counts"],
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
