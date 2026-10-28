#!/usr/bin/env python3
"""Read-only trajectory/schema preflight for the heterogeneous-interface pilot.

This program performs no simulation, fitting, prediction, or scoring.  It reads
the already frozen canonical trajectories, maps each state into a neutral
entity/channel tensor, verifies exact round trips, and writes only aggregate
shape/hash evidence into a new write-once namespace.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/canonical_frozen_teacher_regression_822"
OUT = ROOT / "outputs/heterogeneous_common_interface_stage0_v1"
EXPECTED_SYSTEMS = {"full_k_sho": 320, "reflection_ball": 246, "kepler5": 256}
MAX_TRAJECTORY_READS = 822
MAX_STATE_ROWS = 500_000
MAX_WALL_SECONDS = 60.0
MAX_RSS_BYTES = 1 << 30
MAX_ARTIFACT_BYTES = 1 << 20


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


def to_neutral(system: str, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return states[T,E=2,C=4], entity mask and channel mask."""
    rows = len(trajectory)
    states = np.zeros((rows, 2, 4), dtype=np.float64)
    entities = np.zeros((2,), dtype=np.bool_)
    channels = np.zeros((2, 4), dtype=np.bool_)
    if system == "full_k_sho":
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(f"unexpected SHO shape {trajectory.shape}")
        states[:, 0, :2] = trajectory
        entities[0] = True
        channels[0, :2] = True
    elif system == "reflection_ball":
        if trajectory.ndim != 2 or trajectory.shape[1] != 4:
            raise ValueError(f"unexpected ball shape {trajectory.shape}")
        states[:, 0, :] = trajectory
        entities[0] = True
        channels[0, :] = True
    elif system == "kepler5":
        if trajectory.ndim != 2 or trajectory.shape[1] != 8:
            raise ValueError(f"unexpected Kepler shape {trajectory.shape}")
        states[:, 0, :] = trajectory[:, :4]
        states[:, 1, :] = trajectory[:, 4:]
        entities[:] = True
        channels[:, :] = True
    else:
        raise ValueError(system)
    return states, entities, channels


def from_neutral(system: str, states: np.ndarray) -> np.ndarray:
    if system == "full_k_sho":
        return states[:, 0, :2].copy()
    if system == "reflection_ball":
        return states[:, 0, :].copy()
    if system == "kepler5":
        return np.concatenate((states[:, 0, :], states[:, 1, :]), axis=1)
    raise ValueError(system)


def main() -> None:
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(f"write-once output exists: {OUT}")
    OUT.mkdir(parents=True)
    plan = {
        "status": "PREFLIGHT_PRECHARGED",
        "scientific_operations": 0,
        "trajectory_reads_ceiling": MAX_TRAJECTORY_READS,
        "state_rows_ceiling": MAX_STATE_ROWS,
        "wall_seconds_ceiling": MAX_WALL_SECONDS,
        "rss_bytes_ceiling": MAX_RSS_BYTES,
        "artifact_bytes_ceiling": MAX_ARTIFACT_BYTES,
        "source_manifest_sha256": sha256(SOURCE / "corpus_manifest.json"),
        "case_manifest_sha256": sha256(SOURCE / "case_manifest.json"),
        "driver_sha256": sha256(Path(__file__)),
        "prohibited": ["simulation", "fitting", "prediction", "scoring", "teacher_parameter_export"],
    }
    atomic_json(OUT / "plan.json", plan)

    case_manifest = json.loads((SOURCE / "case_manifest.json").read_text())
    if case_manifest["counts"] != EXPECTED_SYSTEMS or len(case_manifest["cases"]) != 822:
        raise RuntimeError("canonical case inventory mismatch")

    counts = {key: 0 for key in EXPECTED_SYSTEMS}
    length_sets: dict[str, set[int]] = {key: set() for key in EXPECTED_SYSTEMS}
    shape_sets: dict[str, set[tuple[int, ...]]] = {key: set() for key in EXPECTED_SYSTEMS}
    mask_hashes: dict[str, set[str]] = {key: set() for key in EXPECTED_SYSTEMS}
    trajectory_reads = 0
    state_rows = 0
    finite_values = 0
    roundtrip_max_error = 0.0

    for spec in case_manifest["cases"]:
        system = spec["system"]
        family = spec["family"]
        case_id = int(spec["case_id"])
        stem = f"{system}__{family}__{case_id:03d}"
        artifact_path = SOURCE / "cases" / system / stem / "artifact.json"
        artifact = json.loads(artifact_path.read_text())
        trace_path = SOURCE / artifact["trace_files"][0]["relative_path"]
        if sha256(trace_path) != artifact["trace_files"][0]["sha256"]:
            raise RuntimeError(f"trace hash mismatch: {trace_path}")
        trajectory_reads += 1
        if trajectory_reads > MAX_TRAJECTORY_READS:
            raise RuntimeError("trajectory-read ceiling exceeded before next read")
        with np.load(trace_path, allow_pickle=False) as values:
            trajectory = np.asarray(values["trajectory"], dtype=np.float64)
        state_rows += len(trajectory)
        if state_rows > MAX_STATE_ROWS:
            raise RuntimeError("state-row ceiling exceeded")
        if not np.isfinite(trajectory).all():
            raise RuntimeError(f"nonfinite trajectory: {trace_path}")
        finite_values += int(trajectory.size)
        neutral, entity_mask, channel_mask = to_neutral(system, trajectory)
        recovered = from_neutral(system, neutral)
        error = float(np.max(np.abs(recovered - trajectory)))
        roundtrip_max_error = max(roundtrip_max_error, error)
        if error != 0.0:
            raise RuntimeError(f"non-exact round trip: {trace_path}: {error}")
        mask_digest = hashlib.sha256(entity_mask.tobytes() + channel_mask.tobytes()).hexdigest()
        counts[system] += 1
        length_sets[system].add(int(len(trajectory)))
        shape_sets[system].add(tuple(int(x) for x in trajectory.shape))
        mask_hashes[system].add(mask_digest)

    if counts != EXPECTED_SYSTEMS:
        raise RuntimeError(f"system count mismatch: {counts}")
    elapsed = time.monotonic() - started
    resources = {"wall_seconds": elapsed, "peak_rss_bytes": rss_bytes()}
    if elapsed > MAX_WALL_SECONDS or resources["peak_rss_bytes"] > MAX_RSS_BYTES:
        raise RuntimeError(f"resource ceiling exceeded: {resources}")
    result = {
        "status": "HETEROGENEOUS_COMMON_INTERFACE_STAGE0_COMPLETE",
        "classification": "INTERFACE_FEASIBLE_NO_SCIENTIFIC_RESULT",
        "counts": {
            "scientific_operations": 0,
            "trajectory_reads": trajectory_reads,
            "state_rows": state_rows,
            "finite_scalar_values": finite_values,
            "systems": counts,
        },
        "source_shapes": {key: sorted(list(value)) for key, value in shape_sets.items()},
        "trajectory_lengths": {key: sorted(value) for key, value in length_sets.items()},
        "neutral_shape": [2, 4],
        "mask_hashes": {key: sorted(value) for key, value in mask_hashes.items()},
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
