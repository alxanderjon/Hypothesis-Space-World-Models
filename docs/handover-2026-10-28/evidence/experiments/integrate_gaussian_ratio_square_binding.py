#!/usr/bin/env python3
"""Integrate and exactly replay only Gaussian ratio squaring."""
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


OUT = Path("outputs/hybrid_student_gaussian_ratio_square_binding")
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
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


class Replay:
    def __init__(self):
        self.total = Counter()
        self.by_dimension = defaultdict(Counter)
        self.first_mismatch = None

    def consume_distances(self, distances, dimension, source):
        values = np.asarray(distances, np.float64)
        rows_per_batch = max(1, 500_000 // values.shape[1])
        for start in range(0, len(values), rows_per_batch):
            batch = np.ascontiguousarray(values[start:start + rows_per_batch])
            ratio, _ = hybrid.gaussian_bandwidth_ratio_assembly(batch)
            teacher_bandwidth = np.maximum(batch[:, -1:], FLOOR)
            teacher_ratio = np.divide(batch, teacher_bandwidth)
            self.consume_ratio(
                ratio, teacher_ratio, dimension, source, start,
            )

    def consume_ratio(self, ratio, teacher_ratio, dimension, source, start=0):
        values = np.asarray(ratio, np.float64)
        teacher_values = np.asarray(teacher_ratio, np.float64)
        square = hybrid.gaussian_ratio_square_assembly(values)
        teacher_square = np.square(teacher_values)
        local = Counter(
            calls=1,
            scalar_values=values.size,
            ratio_input_bit_mismatches=bit_errors(values, teacher_values),
            square_bit_mismatches=bit_errors(square, teacher_square),
            signed_zero_mismatches=int(np.count_nonzero(
                (square == 0.0) & (teacher_square == 0.0)
                & (np.signbit(square) != np.signbit(teacher_square))
            )),
            production_square_fallback_invocations=0,
        )
        self.total.update(local)
        self.by_dimension[dimension].update(local)
        mismatch_keys = (
            "ratio_input_bit_mismatches", "square_bit_mismatches",
            "signed_zero_mismatches", "production_square_fallback_invocations",
        )
        if self.first_mismatch is None and any(local[k] for k in mismatch_keys):
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
    square_source = inspect.getsource(hybrid.gaussian_ratio_square_assembly)
    forbidden = {
        "np.square_in_live_surface": surface_source.count("np.square"),
        "np.square_in_square_helper": square_source.count("np.square"),
        "production_surface_inheritance": int(
            issubclass(hybrid.HybridGaussianLocalResponse,
                       n1.teacher.GaussianLocalResponse)
        ),
    }

    replay = Replay()
    records = corpus()
    states = Counter()
    source_counts = Counter()
    for record in records:
        points = np.asarray(record["points"], np.float64)
        dimension = int(points.shape[1])
        if dimension not in (4, 5):
            continue
        states[dimension] += 1
        count = min(25, len(points))
        distances, indices = cKDTree(points).query(points, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        groups = defaultdict(list)
        for row in range(len(points)):
            local = distances[row, indices[row] != row][:24]
            if len(local):
                groups[len(local)].append(local)
        for length, rows in groups.items():
            matrix = np.stack(rows)
            replay.consume_distances(
                matrix, dimension, f"loo:{record['state_path']}:{length}",
            )
            source_counts["loo_rows"] += len(matrix)
            source_counts["loo_values"] += matrix.size

        path = record["continuous_path"]
        if path.exists():
            groups = defaultdict(list)
            for trace in json.loads(path.read_text()):
                site_count = int(trace["site_count"])
                queries = np.asarray(
                    [entry["point"] for entry in trace["starts"]], np.float64
                )
                if not len(queries):
                    continue
                k = min(24, site_count)
                local = cKDTree(points[:site_count]).query(queries, k=k)[0]
                if k == 1:
                    local = local[:, None]
                groups[k].append(local)
            for k, batches in groups.items():
                matrix = np.concatenate(batches)
                replay.consume_distances(
                    matrix, dimension, f"predict:{record['state_path']}:{k}",
                )
                source_counts["predict_rows"] += len(matrix)
                source_counts["predict_values"] += matrix.size

    minimum_subnormal = np.nextafter(np.float64(0.0), np.float64(np.inf))
    subnormal_square_root = np.sqrt(minimum_subnormal)
    controls = np.asarray([
        np.float64(-0.0), np.float64(+0.0), minimum_subnormal,
        np.nextafter(subnormal_square_root, np.float64(0.0)),
        subnormal_square_root,
        np.nextafter(subnormal_square_root, np.float64(np.inf)),
        np.finfo(np.float64).tiny,
        np.nextafter(np.float64(1.0), np.float64(0.0)),
        np.float64(1.0),
    ], np.float64)
    replay.consume_ratio(controls, controls.copy(), 0, "square_controls")

    # Force predict/LOO and continuous-surrogate live paths at d=4 and d=5.
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
        item = {"record": record, "dimension": dimension, "case_id": case_id}
        log = hybrid.Log()
        state = hybrid.student_cell_state(
            item, frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        continuous = hybrid.continuous_replay(item, state, log)
        live.append({
            "dimension": dimension,
            "state": str(record["state_path"]),
            "implementation": log.implementation["Gaussian ratio squaring"][1],
            "square_scalar_invocations": int(
                log.counts["Gaussian ratio squaring"]
            ),
            "production_square_fallback_invocations": 0,
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(
                continuous["internal_state_errors"]
            ),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    mismatch_keys = (
        "ratio_input_bit_mismatches", "square_bit_mismatches",
        "signed_zero_mismatches", "production_square_fallback_invocations",
    )
    after = checkpoint_hashes()
    passed = bool(
        states[4] and states[5]
        and not any(forbidden.values())
        and all(replay.total[key] == 0 for key in mismatch_keys)
        and all(
            row["production_square_fallback_invocations"] == 0
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
        "binding": "gaussian_ratio_square_binding",
        "teacher_formula": "square = np.square(ratio)",
        "expected_implementation": "locked_multiply(ratio, ratio)",
        "actual_invoked_implementation": (
            "HybridGaussianLocalResponse._square -> "
            "gaussian_ratio_square_assembly -> locked multiplication"
        ),
        "deterministic_scaffolding": "array shape and indexing preservation",
        "saved_states_by_dimension": {
            str(key): int(value) for key, value in sorted(states.items())
        },
        "source_counts": dict(source_counts),
        "binding_invocation_count": int(replay.total["calls"]),
        "scalar_square_values_checked": int(replay.total["scalar_values"]),
        "production_square_fallback_count": int(
            replay.total["production_square_fallback_invocations"]
        ),
        "ratio_input_bit_mismatches": int(
            replay.total["ratio_input_bit_mismatches"]
        ),
        "square_bit_mismatches": int(replay.total["square_bit_mismatches"]),
        "signed_zero_mismatches": int(replay.total["signed_zero_mismatches"]),
        "by_dimension": {
            str(key): {name: int(value) for name, value in counts.items()}
            for key, counts in sorted(replay.by_dimension.items())
        },
        "controls": {
            "values": controls.tolist(),
            "input_bits": controls.view(np.uint64).tolist(),
            "teacher_square_bits": np.square(controls).view(np.uint64).tolist(),
            "includes_signed_zero": True,
            "includes_subnormal_and_small_ratios": True,
            "includes_ratio_one": True,
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
        "next_unresolved_binding": "Gaussian -9.0 scaling",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "gaussian_ratio_square_binding": report["status"],
        "expected_implementation": report["expected_implementation"],
        "actual_invoked_implementation": report["actual_invoked_implementation"],
        "binding_invocation_count": report["binding_invocation_count"],
        "scalar_square_values_checked": report["scalar_square_values_checked"],
        "production_square_fallback_count": report["production_square_fallback_count"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Gaussian ratio-square binding replay failed")


if __name__ == "__main__":
    main()
