#!/usr/bin/env python3
"""Prospectively sealed law-diversity acquisition for the heterogeneous pilot.

The script creates 32 ball-adapter laws and 32 Kepler laws.  Each law supplies
one passive 32-transition context.  Evaluation inputs are 64 fixed,
law-independent states per mechanism, so the query distribution cannot encode
the law.  Teacher parameters and mechanism identities are stored only in a
quarantined manifest, never in learner arrays.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.stats import qmc

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.kepler import deterministic_teacher as kepler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/heterogeneous_rule_diversity_acquisition_v1"
BALL_SEED = 2026102901
KEPLER_SEED = 2026102902
ORDER_SEED = 2026102903
BALL_QUERY_SEED = 2026102904
KEPLER_QUERY_SEED = 2026102905
LAW_COUNT = 32
CONTEXT_STEPS = 32
QUERY_COUNT = 64
EXPECTED_TRANSITIONS = 6_144
EXPECTED_PASSIVE_ROLLOUT_STEPS = 2_048
EXPECTED_QUERY_STEPS = 4_096
EXPECTED_BALL_STEPS = 3_072
EXPECTED_BALL_EVENT_SCANS = 3_072
EXPECTED_KEPLER_STEPS = 3_072
EXPECTED_KEPLER_ACCELERATION_EVALUATIONS = 6_144
WALL_SECONDS = 120.0
RSS_BYTES = 1 << 30
ARTIFACT_BYTES = 32 << 20


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def sobol(count: int, dimension: int, seed: int) -> np.ndarray:
    exponent = int(math.ceil(math.log2(count)))
    return qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(exponent)[:count]


def neutral(system: str, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float64)
    output = np.zeros(values.shape[:-1] + (2, 4), dtype=np.float64)
    entity_mask = np.zeros((2,), dtype=np.bool_)
    channel_mask = np.zeros((2, 4), dtype=np.bool_)
    if system == "ball":
        output[..., 0, :] = values
        entity_mask[0] = True
        channel_mask[0, :] = True
    elif system == "kepler":
        output[..., 0, :] = values[..., :4]
        output[..., 1, :] = values[..., 4:]
        entity_mask[:] = True
        channel_mask[:, :] = True
    else:
        raise ValueError(system)
    return output, entity_mask, channel_mask


def ball_query_states() -> np.ndarray:
    unit = sobol(QUERY_COUNT, 4, BALL_QUERY_SEED)
    position = -4.9 + 9.8 * unit[:, :2]
    velocity = -3.0 + 6.0 * unit[:, 2:]
    return np.concatenate((position, velocity), axis=1).astype(np.float64)


def kepler_query_states() -> np.ndarray:
    unit = sobol(QUERY_COUNT, 3, KEPLER_QUERY_SEED)
    output = []
    for row in unit:
        eccentricity = 0.18 + 0.50 * row[0]
        periapsis = 0.72 + 0.80 * row[1]
        orientation = 2.0 * np.pi * row[2]
        output.append(kepler.initial_state(eccentricity, periapsis, orientation))
    return np.asarray(output, dtype=np.float64)


def main() -> None:
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(f"write-once output exists: {OUT}")
    OUT.mkdir(parents=True)

    ball_unit = sobol(LAW_COUNT, 2, BALL_SEED)
    ball_laws = ball.LOWER + (ball.UPPER - ball.LOWER) * ball_unit
    kepler_unit = sobol(LAW_COUNT, 2, KEPLER_SEED)
    kepler_laws = kepler.LOWER + (kepler.UPPER - kepler.LOWER) * kepler_unit
    if len({tuple(row) for row in ball_laws}) != LAW_COUNT:
        raise RuntimeError("duplicate ball laws")
    if len({tuple(row) for row in kepler_laws}) != LAW_COUNT:
        raise RuntimeError("duplicate Kepler laws")

    ball_context_initial = np.asarray((4.4, -2.7, 1.8, 1.2), dtype=np.float64)
    kepler_context_initial = kepler.initial_state(0.35, 1.0, 0.0)
    ball_queries = ball_query_states()
    kepler_queries = kepler_query_states()
    order = np.random.default_rng(ORDER_SEED).permutation(2 * LAW_COUNT)
    entries = []
    for index, theta in enumerate(ball_laws):
        entries.append({"pre_shuffle_index": index, "mechanism": "ball", "law_ordinal": index,
                        "role": "training" if index < 24 else "opened_development",
                        "teacher_parameters": theta.tolist()})
    for index, theta in enumerate(kepler_laws):
        entries.append({"pre_shuffle_index": LAW_COUNT + index, "mechanism": "kepler",
                        "law_ordinal": index,
                        "role": "training" if index < 24 else "opened_development",
                        "teacher_parameters": theta.tolist()})
    shuffled_entries = []
    for learner_index, source_index in enumerate(order):
        row = dict(entries[int(source_index)])
        row["learner_index"] = learner_index
        shuffled_entries.append(row)

    schedule = {
        "status": "SCHEDULE_PERSISTED_BEFORE_RESPONSES",
        "laws": shuffled_entries,
        "ball_context_initial": ball_context_initial.tolist(),
        "kepler_context_initial": kepler_context_initial.tolist(),
        "ball_query_states": ball_queries.tolist(),
        "kepler_query_states": kepler_queries.tolist(),
        "query_states_are_law_independent": True,
        "seeds": {"ball_laws": BALL_SEED, "kepler_laws": KEPLER_SEED,
                  "learner_order": ORDER_SEED, "ball_queries": BALL_QUERY_SEED,
                  "kepler_queries": KEPLER_QUERY_SEED},
        "counts": {
            "laws": 64, "context_transitions_per_law": CONTEXT_STEPS,
            "query_transitions_per_law": QUERY_COUNT,
            "passive_rollout_steps": EXPECTED_PASSIVE_ROLLOUT_STEPS,
            "one_step_query_evaluations": EXPECTED_QUERY_STEPS,
            "total_transitions": EXPECTED_TRANSITIONS,
            "ball_steps": EXPECTED_BALL_STEPS,
            "ball_event_signature_scans": EXPECTED_BALL_EVENT_SCANS,
            "kepler_steps": EXPECTED_KEPLER_STEPS,
            "kepler_acceleration_evaluations": EXPECTED_KEPLER_ACCELERATION_EVALUATIONS,
        },
        "source_hashes": {
            "driver": sha256(Path(__file__)),
            "ball": sha256(ROOT / "experiments/bouncing_ball/run_first_transfer.py"),
            "kepler": sha256(ROOT / "experiments/kepler/deterministic_teacher.py"),
            "interface_plan": sha256(ROOT / "coordination/HETEROGENEOUS_COMMON_INTERFACE_PILOT_V1.md"),
        },
        "ceilings": {"wall_seconds": WALL_SECONDS, "peak_rss_bytes": RSS_BYTES,
                     "artifact_bytes": ARTIFACT_BYTES},
        "learner_quarantine": ["mechanism", "law_ordinal", "teacher_parameters", "role",
                               "pre_shuffle_index"],
    }
    atomic_json(OUT / "schedule_before_responses.json", schedule)

    context_x = np.zeros((64, CONTEXT_STEPS, 2, 4), dtype=np.float64)
    context_dx = np.zeros_like(context_x)
    query_x = np.zeros((64, QUERY_COUNT, 2, 4), dtype=np.float64)
    query_dx = np.zeros_like(query_x)
    entity_mask = np.zeros((64, 2), dtype=np.bool_)
    channel_mask = np.zeros((64, 2, 4), dtype=np.bool_)
    delta_t = np.zeros((64,), dtype=np.float64)
    ball_collision_events = 0
    ball_event_scans = 0
    ball_steps = 0
    kepler_steps = 0

    generated: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, float]] = {}
    for index, theta in enumerate(ball_laws):
        states = [ball_context_initial.copy()]
        for _ in range(CONTEXT_STEPS):
            signature = ball.numpy_k_event_signature(states[-1], theta)
            ball_event_scans += 1
            ball_collision_events += len(signature)
            states.append(ball.numpy_k_step(states[-1], theta))
            ball_steps += 1
        states = np.asarray(states)
        cx, em, cm = neutral("ball", states[:-1])
        cy, _, _ = neutral("ball", states[1:])
        qx, _, _ = neutral("ball", ball_queries)
        qnext = []
        for state in ball_queries:
            signature = ball.numpy_k_event_signature(state, theta)
            ball_event_scans += 1
            ball_collision_events += len(signature)
            qnext.append(ball.numpy_k_step(state, theta))
            ball_steps += 1
        qnext, _, _ = neutral("ball", np.asarray(qnext))
        generated[index] = (cx, cy - cx, qx, qnext - qx, em, cm, float(ball.DT))

    for index, theta in enumerate(kepler_laws):
        states = [kepler_context_initial.copy()]
        for _ in range(CONTEXT_STEPS):
            states.append(kepler.numpy_step(states[-1], theta))
            kepler_steps += 1
        states = np.asarray(states)
        cx, em, cm = neutral("kepler", states[:-1])
        cy, _, _ = neutral("kepler", states[1:])
        qx, _, _ = neutral("kepler", kepler_queries)
        qnext = []
        for state in kepler_queries:
            qnext.append(kepler.numpy_step(state, theta))
            kepler_steps += 1
        qnext, _, _ = neutral("kepler", np.asarray(qnext))
        generated[LAW_COUNT + index] = (cx, cy - cx, qx, qnext - qx, em, cm, float(kepler.DT))

    for learner_index, source_index in enumerate(order):
        cx, cdx, qx, qdx, em, cm, dt = generated[int(source_index)]
        context_x[learner_index] = cx
        context_dx[learner_index] = cdx
        query_x[learner_index] = qx
        query_dx[learner_index] = qdx
        entity_mask[learner_index] = em
        channel_mask[learner_index] = cm
        delta_t[learner_index] = dt

    if ball_steps != EXPECTED_BALL_STEPS or ball_event_scans != EXPECTED_BALL_EVENT_SCANS:
        raise RuntimeError("ball work-count mismatch")
    if kepler_steps != EXPECTED_KEPLER_STEPS:
        raise RuntimeError("Kepler work-count mismatch")
    if ball_steps + kepler_steps != EXPECTED_TRANSITIONS:
        raise RuntimeError("total transition-count mismatch")
    for name, value in (("context_x", context_x), ("context_dx", context_dx),
                        ("query_x", query_x), ("query_dx", query_dx),
                        ("delta_t", delta_t)):
        if not np.isfinite(value).all():
            raise RuntimeError(f"nonfinite {name}")

    # A different parameter vector must implement a different transition map
    # on the fixed query bank.  This is a source-validity check, not a selection
    # threshold and not learner-visible supervision.
    ball_signatures = query_dx[np.flatnonzero(delta_t == float(ball.DT))].reshape(LAW_COUNT, -1)
    kepler_signatures = query_dx[np.flatnonzero(delta_t == float(kepler.DT))].reshape(LAW_COUNT, -1)
    def minimum_pairwise(values: np.ndarray) -> float:
        answer = np.inf
        for i in range(len(values)):
            for j in range(i):
                answer = min(answer, float(np.max(np.abs(values[i] - values[j]))))
        return answer
    ball_minimum = minimum_pairwise(ball_signatures)
    kepler_minimum = minimum_pairwise(kepler_signatures)
    if not ball_minimum > 0.0 or not kepler_minimum > 0.0:
        raise RuntimeError("distinct parameters did not define distinct fixed-bank transition maps")

    # Context inputs and learner-visible metadata are separated from protected
    # query labels.  A later evaluator must persist predictions before reading
    # protected_query_targets.npz.
    atomic_npz(OUT / "learner_contexts_and_queries.npz",
               context_x=context_x, context_dx=context_dx, query_x=query_x,
               entity_mask=entity_mask, channel_mask=channel_mask, delta_t=delta_t)
    atomic_npz(OUT / "protected_query_targets.npz", query_dx=query_dx)
    barrier = {
        "status": "ACQUISITION_COMPLETE_QUERY_TARGETS_QUARANTINED",
        "learner_file_sha256": sha256(OUT / "learner_contexts_and_queries.npz"),
        "protected_query_target_sha256": sha256(OUT / "protected_query_targets.npz"),
        "prediction_before_label_required": True,
    }
    atomic_json(OUT / "prediction_barrier.json", barrier)

    elapsed = time.monotonic() - started
    resources = {"wall_seconds": elapsed, "peak_rss_bytes": rss_bytes()}
    artifact_bytes = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    resources["artifact_bytes_before_result"] = artifact_bytes
    if elapsed > WALL_SECONDS or resources["peak_rss_bytes"] > RSS_BYTES or artifact_bytes > ARTIFACT_BYTES:
        raise RuntimeError(f"resource ceiling exceeded: {resources}")
    counts = {
        "laws": 64,
        "passive_rollout_steps": LAW_COUNT * 2 * CONTEXT_STEPS,
        "one_step_query_evaluations": LAW_COUNT * 2 * QUERY_COUNT,
        "total_transition_steps": ball_steps + kepler_steps,
        "ball_steps": ball_steps,
        "ball_event_signature_scans": ball_event_scans,
        "ball_collision_events": ball_collision_events,
        "kepler_steps": kepler_steps,
        "kepler_acceleration_evaluations": 2 * kepler_steps,
        "learner_context_rows": 64 * CONTEXT_STEPS,
        "protected_query_rows": 64 * QUERY_COUNT,
    }
    result = {
        "status": "HETEROGENEOUS_RULE_DIVERSITY_ACQUISITION_V1_COMPLETE",
        "classification": "DEVELOPMENT_DATA_ACQUIRED_NO_LEARNER_RESULT",
        "counts": counts,
        "distinct_transition_map_check": {
            "ball_minimum_pairwise_fixed_bank_max_abs_difference": ball_minimum,
            "kepler_minimum_pairwise_fixed_bank_max_abs_difference": kepler_minimum,
            "all_32_maps_distinct_per_mechanism": True,
        },
        "resources": resources,
        "learner_visible": ["context_x", "context_dx", "query_x", "entity_mask",
                            "channel_mask", "delta_t"],
        "quarantined": ["mechanism", "law_ordinal", "teacher_parameters", "role",
                        "query_dx", "simulator metadata and source"],
        "supplied_semantics": [
            "fully observed Markov states", "body order", "position/velocity channel grouping",
            "entity and channel masks", "elapsed time", "one or two exchangeable entities",
        ],
        "qualification": [
            "Ball K(a,b) is the repository transport/reflection adapter, not a material model.",
            "Kepler retains fixed masses, G, planar bodies and velocity-Verlet generation.",
            "All 64 laws are development data; no untouched mechanism confirmation was opened.",
            "Different fixed-bank maps establish rule variation, not identifiability from 32 transitions.",
        ],
    }
    atomic_json(OUT / "result.json", result)
    atomic_json(OUT / "completion.json", {
        "status": "COMPLETE", "result_sha256": sha256(OUT / "result.json"),
        "counts": counts, "barrier_sha256": sha256(OUT / "prediction_barrier.json"),
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
