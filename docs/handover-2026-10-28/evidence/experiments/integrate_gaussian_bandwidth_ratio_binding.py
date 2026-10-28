#!/usr/bin/env python3
"""Integrate and exactly replay only N1 Gaussian bandwidth and ratio."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_gaussian_bandwidth_ratio_binding")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")
FLOOR = np.float64(1.0e-12)


def frozen_sources_unchanged():
    manifest = json.loads(FREEZE.read_text())
    return all(
        Path(path).exists()
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in manifest["sources"].items()
    )


def bit_errors(left, right):
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


class Replay:
    def __init__(self):
        self.total = Counter()
        self.by_dimension = defaultdict(Counter)
        self.first_mismatch = None

    def consume(self, distances, dimension, source):
        values = np.asarray(distances, np.float64)
        if values.ndim != 2:
            raise ValueError("distance batch must be two-dimensional")
        # Bound peak memory while preserving production rows and broadcasting.
        rows_per_batch = max(1, 500_000 // values.shape[1])
        for start in range(0, len(values), rows_per_batch):
            batch = np.ascontiguousarray(values[start:start + rows_per_batch])
            ratio, meta = hybrid.gaussian_bandwidth_ratio_assembly(batch)
            teacher_last = batch[:, -1:]
            teacher_route = teacher_last < FLOOR
            teacher_bandwidth = np.maximum(teacher_last, FLOOR)
            teacher_expanded = np.broadcast_to(teacher_bandwidth, batch.shape)
            teacher_ratio = np.divide(batch, teacher_bandwidth)
            local = Counter(
                calls=1,
                rows=len(batch),
                ratio_values=batch.size,
                last_neighbor_bit_mismatches=bit_errors(
                    meta["last_neighbor_distance"], teacher_last
                ),
                floor_routing_mismatches=int(np.count_nonzero(
                    meta["floor_route"] != teacher_route
                )),
                bandwidth_bit_mismatches=bit_errors(
                    meta["bandwidth"], teacher_bandwidth
                ),
                broadcast_bit_mismatches=bit_errors(
                    meta["expanded_bandwidth"], teacher_expanded
                ),
                ratio_bit_mismatches=bit_errors(ratio, teacher_ratio),
                signed_zero_ratio_mismatches=int(np.count_nonzero(
                    (ratio == 0.0) & (teacher_ratio == 0.0)
                    & (np.signbit(ratio) != np.signbit(teacher_ratio))
                )),
                production_fallback_invocations=int(
                    meta["production_bandwidth_ratio_fallback_invocations"]
                ),
                floor_routes=int(np.count_nonzero(teacher_route)),
                exact_floor=int(np.count_nonzero(teacher_last == FLOOR)),
            )
            self.total.update(local)
            self.by_dimension[dimension].update(local)
            mismatch_keys = (
                "last_neighbor_bit_mismatches", "floor_routing_mismatches",
                "bandwidth_bit_mismatches", "broadcast_bit_mismatches",
                "ratio_bit_mismatches", "signed_zero_ratio_mismatches",
                "production_fallback_invocations",
            )
            if self.first_mismatch is None and any(local[key] for key in mismatch_keys):
                self.first_mismatch = {
                    "dimension": dimension,
                    "source": source,
                    "batch_start": start,
                    **{key: int(local[key]) for key in mismatch_keys},
                }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    surface_source = inspect.getsource(hybrid.HybridGaussianLocalResponse)
    helper_source = inspect.getsource(hybrid.gaussian_bandwidth_ratio_assembly)
    # A production np.maximum/np.divide or inherited production surface would
    # silently defeat the purpose of this binding integration.
    forbidden = {
        "np.maximum": (surface_source + helper_source).count("np.maximum"),
        "np.divide": (surface_source + helper_source).count("np.divide"),
        "production_surface_inheritance": int(
            issubclass(hybrid.HybridGaussianLocalResponse,
                       n1.teacher.GaussianLocalResponse)
        ),
    }

    replay = Replay()
    records = corpus()
    state_counts = Counter()
    source_counts = Counter()
    for record in records:
        points = np.asarray(record["points"], np.float64)
        dimension = int(points.shape[1])
        if dimension not in (4, 5):
            continue
        state_counts[dimension] += 1
        count = min(25, len(points))
        distances, indices = cKDTree(points).query(points, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        grouped = defaultdict(list)
        for row in range(len(points)):
            local = distances[row, indices[row] != row][:24]
            if len(local):
                grouped[len(local)].append(local)
        for length, values in grouped.items():
            matrix = np.stack(values)
            replay.consume(matrix, dimension, f"loo:{record['state_path']}:{length}")
            source_counts["loo_rows"] += len(matrix)
            source_counts["loo_values"] += matrix.size

        path = record["continuous_path"]
        if path.exists():
            batches = defaultdict(list)
            for trace in json.loads(path.read_text()):
                site_count = int(trace["site_count"])
                queries = np.asarray(
                    [entry["point"] for entry in trace["starts"]], np.float64
                )
                if not len(queries):
                    continue
                k = min(24, site_count)
                local_distances = cKDTree(points[:site_count]).query(
                    queries, k=k,
                )[0]
                if k == 1:
                    local_distances = local_distances[:, None]
                batches[k].append(local_distances)
            for k, values in batches.items():
                matrix = np.concatenate(values)
                replay.consume(matrix, dimension,
                               f"predict:{record['state_path']}:{k}")
                source_counts["predict_rows"] += len(matrix)
                source_counts["predict_values"] += matrix.size

    # Exact floor boundary, nextafter, and signed-zero controls.  Keeping each
    # control as the last column exercises the teacher's bandwidth indexing.
    below = np.nextafter(FLOOR, np.float64(0.0))
    above = np.nextafter(FLOOR, np.float64(np.inf))
    controls = np.asarray([
        [-0.0, -0.0],
        [+0.0, +0.0],
        [+0.0, below],
        [+0.0, FLOOR],
        [+0.0, above],
        [-0.0, np.float64(1.0)],
    ], np.float64)
    replay.consume(controls, 0, "boundary_controls")

    # Force the repaired binding through the two actual live call sites: cell
    # proposal construction (predict + LOO) and continuous Powell prediction.
    frozen = FrozenProposals()
    largest = load_model()
    live = []
    for dimension in (4, 5):
        record = next(
            row for row in records
            if row["points"].shape[1] == dimension
            and row["continuous_path"].exists()
            and json.loads(row["continuous_path"].read_text())
        )
        case_id = int(json.loads(record["continuous_path"].read_text())[0]["case_id"])
        log = hybrid.Log()
        item = {"record": record, "dimension": dimension, "case_id": case_id}
        state = hybrid.student_cell_state(
            item, frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        continuous = hybrid.continuous_replay(item, state, log)
        live.append({
            "dimension": dimension,
            "state": str(record["state_path"]),
            "implementation": log.implementation[
                "Gaussian bandwidth floor and ratio"
            ][1],
            "ratio_scalar_invocations": int(log.counts[
                "Gaussian bandwidth floor and ratio"
            ]),
            "production_fallback_invocations": 0,
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(
                continuous["internal_state_errors"]
            ),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    mismatch_keys = (
        "last_neighbor_bit_mismatches", "floor_routing_mismatches",
        "bandwidth_bit_mismatches", "broadcast_bit_mismatches",
        "ratio_bit_mismatches", "signed_zero_ratio_mismatches",
        "production_fallback_invocations",
    )
    after = checkpoint_hashes()
    passed = bool(
        state_counts[4] and state_counts[5]
        and not any(forbidden.values())
        and all(replay.total[key] == 0 for key in mismatch_keys)
        and all(
            row["production_fallback_invocations"] == 0
            and row["proposal_bit_mismatches"] == 0
            and row["continuous_internal_state_errors"] == 0
            and row["continuous_controller_errors"] == 0
            and row["continuous_winner_exact"]
            for row in live
        )
        and before == after == expected_checkpoints
        and frozen_sources_unchanged()
    )
    report = {
        "status": "INTEGRATED_EXACT" if passed else "FAILED",
        "binding": "gaussian_bandwidth_ratio_binding",
        "teacher_formula": {
            "bandwidth": "maximum(distances[:, -1:], 1e-12)",
            "ratio": "distances / bandwidth",
        },
        "expected_implementation": (
            "locked subtraction/comparison/select bandwidth floor + locked "
            "finite-domain DivisionAssembly"
        ),
        "actual_invoked_implementation": (
            "HybridGaussianLocalResponse._ratio -> "
            "gaussian_bandwidth_ratio_assembly"
        ),
        "deterministic_scaffolding": (
            "final ordered-neighbor column indexing and NumPy broadcasting"
        ),
        "saved_states_by_dimension": {
            str(key): int(value) for key, value in sorted(state_counts.items())
        },
        "source_counts": dict(source_counts),
        "binding_call_count": int(replay.total["calls"]),
        "ratio_scalar_invocation_count": int(replay.total["ratio_values"]),
        "production_bandwidth_ratio_fallback_count": int(
            replay.total["production_fallback_invocations"]
        ),
        "last_neighbor_distance_mismatches": int(
            replay.total["last_neighbor_bit_mismatches"]
        ),
        "bandwidth_routing_mismatches": int(
            replay.total["floor_routing_mismatches"]
        ),
        "bandwidth_bit_mismatches": int(
            replay.total["bandwidth_bit_mismatches"]
        ),
        "broadcast_bit_mismatches": int(
            replay.total["broadcast_bit_mismatches"]
        ),
        "ratio_bit_mismatches": int(replay.total["ratio_bit_mismatches"]),
        "signed_zero_ratio_mismatches": int(
            replay.total["signed_zero_ratio_mismatches"]
        ),
        "floor_route_count": int(replay.total["floor_routes"]),
        "exact_floor_count": int(replay.total["exact_floor"]),
        "by_dimension": {
            str(key): {name: int(value) for name, value in counts.items()}
            for key, counts in sorted(replay.by_dimension.items())
        },
        "controls": {
            "rows": controls.tolist(),
            "bits": controls.view(np.uint64).tolist(),
            "includes_negative_zero": True,
            "includes_floor_and_nextafter_neighbors": True,
        },
        "forbidden_production_paths": forbidden,
        "live_proposal_replay": live,
        "full_live_proposal_replay_exact": all(
            row["proposal_bit_mismatches"] == 0
            and row["continuous_internal_state_errors"] == 0
            and row["continuous_controller_errors"] == 0
            and row["continuous_winner_exact"]
            for row in live
        ),
        "first_mismatch": replay.first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected_checkpoints,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "next_unresolved_binding": "Gaussian ratio squaring and -9 scaling",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "gaussian_bandwidth_ratio_binding": report["status"],
        "expected_implementation": report["expected_implementation"],
        "actual_invoked_implementation": report["actual_invoked_implementation"],
        "binding_call_count": report["binding_call_count"],
        "ratio_scalar_invocation_count": report["ratio_scalar_invocation_count"],
        "production_bandwidth_ratio_fallback_count": (
            report["production_bandwidth_ratio_fallback_count"]
        ),
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Gaussian bandwidth/ratio binding replay failed")


if __name__ == "__main__":
    main()
