#!/usr/bin/env python3
"""Build a fold-safe SHO/ball/Kepler package for the heterogeneous pilot.

Ball and Kepler arrays are copied from the sealed acquisition.  Thirty-two
cached SHO laws contribute passive32 contexts; their fixed query responses are
the only new calculations (2,048 matrix-vector products).  Training and opened
development targets are stored in separate files so the learner can persist
predictions before opening development labels.
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


ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "outputs/canonical_frozen_teacher_regression_822"
ACQ = ROOT / "outputs/heterogeneous_rule_diversity_acquisition_v1"
OUT = ROOT / "outputs/heterogeneous_three_group_package_v1"
SHO_QUERY_SEED = 2026102906
ORDER_SEED = 2026102907
LAW_COUNT = 32
CONTEXT_STEPS = 32
QUERY_COUNT = 64
SHO_QUERY_EVALUATIONS = LAW_COUNT * QUERY_COUNT
WALL_SECONDS = 60.0
RSS_BYTES = 1 << 30
ARTIFACT_BYTES = 64 << 20


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
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def main() -> None:
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    acq_completion = json.loads((ACQ / "completion.json").read_text())
    if acq_completion["status"] != "COMPLETE":
        raise RuntimeError("acquisition incomplete")
    if sha256(ACQ / "result.json") != acq_completion["result_sha256"]:
        raise RuntimeError("acquisition result hash mismatch")

    corpus = json.loads((CANON / "corpus_manifest.json").read_text())
    artifacts = {row["canonical_case_id"]: row for row in corpus["artifacts"]}
    sho_ids = [f"full_k_sho:sobol:{index:03d}" for index in range(LAW_COUNT)]
    unit = qmc.Sobol(2, scramble=True, seed=SHO_QUERY_SEED).random_base2(6)[:QUERY_COUNT]
    sho_query_raw = -1.0 + 2.0 * unit
    sho_context_x = np.zeros((LAW_COUNT, CONTEXT_STEPS, 2, 4), np.float64)
    sho_context_dx = np.zeros_like(sho_context_x)
    sho_query_x = np.zeros((LAW_COUNT, QUERY_COUNT, 2, 4), np.float64)
    sho_query_dx = np.zeros_like(sho_query_x)
    sho_teachers = []
    sho_query_evaluations = 0
    for index, case_id in enumerate(sho_ids):
        entry = artifacts[case_id]
        artifact_path = CANON / entry["relative_path"]
        if sha256(artifact_path) != entry["file_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {case_id}")
        artifact = json.loads(artifact_path.read_text())
        trace = artifact["trace_files"][0]
        trace_path = CANON / trace["relative_path"]
        if sha256(trace_path) != trace["sha256"]:
            raise RuntimeError(f"trace hash mismatch: {case_id}")
        with np.load(trace_path, allow_pickle=False) as values:
            trajectory = np.asarray(values["trajectory"], np.float64)
        if len(trajectory) < CONTEXT_STEPS + 1:
            raise RuntimeError(f"short SHO trajectory: {case_id}")
        theta = np.asarray(artifact["input"]["teacher_parameters"], np.float64).reshape(2, 2)
        sho_teachers.append(theta.ravel().tolist())
        sho_context_x[index, :, 0, :2] = trajectory[:CONTEXT_STEPS]
        sho_context_dx[index, :, 0, :2] = trajectory[1:CONTEXT_STEPS + 1] - trajectory[:CONTEXT_STEPS]
        sho_query_x[index, :, 0, :2] = sho_query_raw
        next_state = sho_query_raw @ theta.T
        sho_query_dx[index, :, 0, :2] = next_state - sho_query_raw
        sho_query_evaluations += QUERY_COUNT
    if sho_query_evaluations != SHO_QUERY_EVALUATIONS:
        raise RuntimeError("SHO query-count mismatch")

    with np.load(ACQ / "learner_contexts_and_queries.npz", allow_pickle=False) as values:
        acq_context_x = np.asarray(values["context_x"])
        acq_context_dx = np.asarray(values["context_dx"])
        acq_query_x = np.asarray(values["query_x"])
        acq_entity_mask = np.asarray(values["entity_mask"])
        acq_channel_mask = np.asarray(values["channel_mask"])
        acq_delta_t = np.asarray(values["delta_t"])
    with np.load(ACQ / "protected_query_targets.npz", allow_pickle=False) as values:
        acq_query_dx = np.asarray(values["query_dx"])
    acq_schedule = json.loads((ACQ / "schedule_before_responses.json").read_text())
    acq_by_index = {int(row["learner_index"]): row for row in acq_schedule["laws"]}

    records = []
    source_arrays = []
    sho_entity = np.asarray((True, False))
    sho_channels = np.asarray(((True, True, False, False), (False, False, False, False)))
    for index in range(LAW_COUNT):
        records.append({"group": "sho", "law_ordinal": index,
                        "role": "training" if index < 24 else "opened_development",
                        "teacher_parameters": sho_teachers[index]})
        source_arrays.append((sho_context_x[index], sho_context_dx[index], sho_query_x[index],
                              sho_query_dx[index], sho_entity, sho_channels, 1.0))
    for index in range(64):
        meta = acq_by_index[index]
        records.append({"group": meta["mechanism"], "law_ordinal": meta["law_ordinal"],
                        "role": meta["role"], "teacher_parameters": meta["teacher_parameters"]})
        source_arrays.append((acq_context_x[index], acq_context_dx[index], acq_query_x[index],
                              acq_query_dx[index], acq_entity_mask[index], acq_channel_mask[index],
                              float(acq_delta_t[index])))

    order = np.random.default_rng(ORDER_SEED).permutation(96)
    context_x = np.zeros((96, CONTEXT_STEPS, 2, 4), np.float64)
    context_dx = np.zeros_like(context_x)
    query_x = np.zeros((96, QUERY_COUNT, 2, 4), np.float64)
    entity_mask = np.zeros((96, 2), np.bool_)
    channel_mask = np.zeros((96, 2, 4), np.bool_)
    delta_t = np.zeros((96,), np.float64)
    train_index = []
    train_target = []
    dev_index = []
    dev_target = []
    shuffled_records = []
    for learner_index, source_index in enumerate(order):
        cx, cdx, qx, qdx, em, cm, dt = source_arrays[int(source_index)]
        context_x[learner_index] = cx; context_dx[learner_index] = cdx
        query_x[learner_index] = qx; entity_mask[learner_index] = em
        channel_mask[learner_index] = cm; delta_t[learner_index] = dt
        record = dict(records[int(source_index)]); record["learner_index"] = learner_index
        shuffled_records.append(record)
        if record["role"] == "training":
            train_index.append(learner_index); train_target.append(qdx)
        else:
            dev_index.append(learner_index); dev_target.append(qdx)
    atomic_json(OUT / "quarantined_split_manifest.json", {
        "status": "SPLITS_PERSISTED_BEFORE_LEARNER_EXECUTION",
        "records": shuffled_records,
        "seeds": {"sho_queries": SHO_QUERY_SEED, "learner_order": ORDER_SEED},
        "learner_prohibited_fields": ["group", "law_ordinal", "role", "teacher_parameters"],
        "whole_group_folds": ["sho", "ball", "kepler"],
    })
    atomic_npz(OUT / "learner_inputs.npz", context_x=context_x, context_dx=context_dx,
               query_x=query_x, entity_mask=entity_mask, channel_mask=channel_mask,
               delta_t=delta_t)
    atomic_npz(OUT / "training_query_targets.npz",
               learner_index=np.asarray(train_index, np.int64), query_dx=np.asarray(train_target))
    atomic_npz(OUT / "protected_development_query_targets.npz",
               learner_index=np.asarray(dev_index, np.int64), query_dx=np.asarray(dev_target))
    barrier = {
        "status": "PACKAGE_COMPLETE_DEVELOPMENT_TARGETS_QUARANTINED",
        "inputs_sha256": sha256(OUT / "learner_inputs.npz"),
        "training_targets_sha256": sha256(OUT / "training_query_targets.npz"),
        "development_targets_sha256": sha256(OUT / "protected_development_query_targets.npz"),
        "prediction_before_development_label_required": True,
    }
    atomic_json(OUT / "prediction_barrier.json", barrier)
    elapsed = time.monotonic() - started
    resources = {"wall_seconds": elapsed, "peak_rss_bytes": rss_bytes()}
    artifact_bytes = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    resources["artifact_bytes_before_result"] = artifact_bytes
    if elapsed > WALL_SECONDS or resources["peak_rss_bytes"] > RSS_BYTES or artifact_bytes > ARTIFACT_BYTES:
        raise RuntimeError(f"resource ceiling exceeded: {resources}")
    result = {
        "status": "HETEROGENEOUS_THREE_GROUP_PACKAGE_V1_COMPLETE",
        "classification": "DEVELOPMENT_PACKAGE_READY_NO_LEARNER_RESULT",
        "counts": {"systems": 96, "groups": {"sho": 32, "ball": 32, "kepler": 32},
                   "training_systems": len(train_index), "opened_development_systems": len(dev_index),
                   "context_rows": 96 * CONTEXT_STEPS, "query_rows": 96 * QUERY_COUNT,
                   "new_sho_matrix_vector_evaluations": sho_query_evaluations,
                   "copied_ball_kepler_transition_rows": 64 * (CONTEXT_STEPS + QUERY_COUNT)},
        "resources": resources,
        "qualification": [
            "SHO query labels are new fixed-bank matrix-vector evaluations; ball/Kepler rows are copied.",
            "Group identity and teacher parameters are quarantined orchestration metadata.",
            "Masks, channel groupings and delta_t remain learner-visible supplied semantics.",
            "All three groups are development evidence; no confirmation mechanism is opened.",
        ],
    }
    atomic_json(OUT / "result.json", result)
    atomic_json(OUT / "completion.json", {"status": "COMPLETE",
        "result_sha256": sha256(OUT / "result.json"), "barrier_sha256": sha256(OUT / "prediction_barrier.json"),
        "counts": result["counts"]})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
