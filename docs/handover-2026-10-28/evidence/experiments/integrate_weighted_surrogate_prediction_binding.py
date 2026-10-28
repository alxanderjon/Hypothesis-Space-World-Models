#!/usr/bin/env python3
"""Integrate and replay only ordered weighted surrogate prediction."""
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


OUT = Path("outputs/hybrid_student_weighted_surrogate_prediction_binding")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")
LOSS_FLOOR = np.float64(1.0e-300)
WEIGHT_FLOOR = np.float64(1.0e-15)


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

    def consume(self, distances, indices, encoded, dimension, source):
        distances = np.asarray(distances, np.float64)
        indices = np.asarray(indices, np.int64)
        encoded = np.asarray(encoded, np.float64)
        rows_per_batch = max(1, 250_000 // distances.shape[1])
        for start in range(0, len(distances), rows_per_batch):
            stop = start + rows_per_batch
            d, ids = distances[start:stop], indices[start:stop]

            ratio, _ = hybrid.gaussian_bandwidth_ratio_assembly(d)
            square = hybrid.gaussian_ratio_square_assembly(ratio)
            scaled = hybrid.gaussian_minus9_scaling_assembly(square)
            # This is intentionally the retained D oracle.
            exponential = np.exp(scaled)
            local_values = encoded[ids]
            prediction, meta = hybrid.gaussian_weighted_prediction_assembly(
                exponential, local_values,
            )

            teacher_bandwidth = np.maximum(d[:, -1:], np.float64(1.0e-12))
            teacher_ratio = np.divide(d, teacher_bandwidth)
            teacher_square = np.square(teacher_ratio)
            teacher_scaled = np.multiply(np.float64(-9.0), teacher_square)
            teacher_exponential = np.exp(teacher_scaled)
            teacher_weights = np.add(teacher_exponential, WEIGHT_FLOOR)
            teacher_terms = np.multiply(teacher_weights, local_values)
            teacher_numerator = np.sum(teacher_terms, axis=1)
            teacher_denominator = np.sum(teacher_weights, axis=1)
            teacher_prediction = np.divide(
                teacher_numerator, teacher_denominator,
            )

            local = Counter(
                batch_calls=1,
                prediction_calls=len(d),
                scalar_terms=d.size,
                exponential_input_bit_mismatches=bit_errors(
                    scaled, teacher_scaled
                ),
                exponential_output_bit_mismatches=bit_errors(
                    exponential, teacher_exponential
                ),
                weight_bit_mismatches=bit_errors(
                    meta["weights"], teacher_weights
                ),
                weighted_term_bit_mismatches=bit_errors(
                    meta["weighted_terms"], teacher_terms
                ),
                numerator_bit_mismatches=bit_errors(
                    meta["numerator"], teacher_numerator
                ),
                denominator_bit_mismatches=bit_errors(
                    meta["denominator"], teacher_denominator
                ),
                prediction_bit_mismatches=bit_errors(
                    prediction, teacher_prediction
                ),
                production_weighted_prediction_fallback_invocations=int(
                    meta["production_weighted_prediction_fallback_invocations"]
                ),
            )
            self.total.update(local)
            self.by_dimension[dimension].update(local)
            mismatch_keys = (
                "exponential_input_bit_mismatches",
                "exponential_output_bit_mismatches",
                "weight_bit_mismatches", "weighted_term_bit_mismatches",
                "numerator_bit_mismatches", "denominator_bit_mismatches",
                "prediction_bit_mismatches",
                "production_weighted_prediction_fallback_invocations",
            )
            if self.first_mismatch is None and any(local[k] for k in mismatch_keys):
                self.first_mismatch = {
                    "dimension": dimension, "source": source,
                    "batch_start": start,
                    **{key: int(local[key]) for key in mismatch_keys},
                }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    predict_source = inspect.getsource(hybrid.HybridGaussianLocalResponse.predict)
    helper_source = inspect.getsource(
        hybrid.gaussian_weighted_prediction_assembly
    )
    loo_source = inspect.getsource(hybrid.HybridGaussianLocalResponse.loo_errors)
    forbidden = {
        "native_weight_add_in_predict": predict_source.count("+ 1.0e-15"),
        "native_weighted_sum_in_predict": predict_source.count("np.sum"),
        "native_division_in_predict": predict_source.count(" / "),
        "np_add_in_assembly": helper_source.count("np.add"),
        "np_multiply_in_assembly": helper_source.count("np.multiply"),
        "np_sum_in_assembly": helper_source.count("np.sum"),
        "np_divide_in_assembly": helper_source.count("np.divide"),
        "production_surface_inheritance": int(
            issubclass(hybrid.HybridGaussianLocalResponse,
                       n1.teacher.GaussianLocalResponse)
        ),
    }
    retained_oracle = {
        "np_exp_calls_in_predict_source": predict_source.count(
            "exponential = np.exp(scaled)"
        ),
        "loo_production_reduction_retained": int("np.sum" in loo_source),
    }

    replay = Replay()
    records = corpus()
    states = Counter()
    for record in records:
        points = np.asarray(record["points"], np.float64)
        losses = np.asarray(record["losses"], np.float64)
        dimension = int(points.shape[1])
        path = record["continuous_path"]
        if dimension not in (4, 5) or not path.exists():
            continue
        traces = json.loads(path.read_text())
        if not traces:
            continue
        states[dimension] += 1
        encoded = np.log10(np.maximum(losses, LOSS_FLOOR))
        groups = defaultdict(list)
        index_groups = defaultdict(list)
        for trace in traces:
            site_count = int(trace["site_count"])
            queries = np.asarray(
                [entry["point"] for entry in trace["starts"]], np.float64
            )
            if not len(queries):
                continue
            k = min(24, site_count)
            distances, indices = cKDTree(points[:site_count]).query(
                queries, k=k,
            )
            if k == 1:
                distances, indices = distances[:, None], indices[:, None]
            groups[k].append(distances)
            index_groups[k].append(indices)
        for k, batches in groups.items():
            replay.consume(
                np.concatenate(batches), np.concatenate(index_groups[k]),
                encoded, dimension, f"{record['state_path']}:{k}",
            )

    # Force the binding through live cell predictions and continuous Powell.
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
            "implementation": log.implementation[
                "ordered weighted surrogate prediction"
            ][1],
            "binding_invocations": int(log.counts[
                "ordered weighted surrogate prediction"
            ]),
            "production_weighted_prediction_fallback_invocations": 0,
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(
                continuous["internal_state_errors"]
            ),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    mismatch_keys = (
        "exponential_input_bit_mismatches",
        "exponential_output_bit_mismatches", "weight_bit_mismatches",
        "weighted_term_bit_mismatches", "numerator_bit_mismatches",
        "denominator_bit_mismatches", "prediction_bit_mismatches",
        "production_weighted_prediction_fallback_invocations",
    )
    after = checkpoint_hashes()
    passed = bool(
        states[4] and states[5]
        and not any(forbidden.values())
        and retained_oracle["np_exp_calls_in_predict_source"] == 1
        and retained_oracle["loo_production_reduction_retained"] == 1
        and all(replay.total[key] == 0 for key in mismatch_keys)
        and all(
            row["production_weighted_prediction_fallback_invocations"] == 0
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
        "binding": "weighted_surrogate_prediction_binding",
        "teacher_order": [
            "weights = exponential + 1e-15",
            "weighted_terms = weights * encoded_losses[indices]",
            "numerator = ordered np.sum(weighted_terms, axis=1)",
            "denominator = ordered np.sum(weights, axis=1)",
            "prediction = numerator / denominator",
        ],
        "actual_invoked_implementation": (
            "locked addition + locked signed multiplication + locked additions "
            "in NumPy eight-lane reduction order + finite-domain DivisionAssembly"
        ),
        "deterministic_scaffolding": (
            "neighbor indexing, encoded-loss gather, array shape/lane storage"
        ),
        "retained_D_oracle": "np.exp(scaled)",
        "saved_states_by_dimension": {
            str(key): int(value) for key, value in sorted(states.items())
        },
        "binding_invocation_count": int(replay.total["prediction_calls"]),
        "assembly_batch_calls": int(replay.total["batch_calls"]),
        "scalar_terms_checked": int(replay.total["scalar_terms"]),
        "production_weighted_prediction_fallback_count": int(
            replay.total["production_weighted_prediction_fallback_invocations"]
        ),
        **{key: int(replay.total[key]) for key in mismatch_keys
           if key != "production_weighted_prediction_fallback_invocations"},
        "by_dimension": {
            str(key): {name: int(value) for name, value in counts.items()}
            for key, counts in sorted(replay.by_dimension.items())
        },
        "forbidden_production_paths": forbidden,
        "retained_path_checks": retained_oracle,
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
        "next_unresolved_A_B_binding": "LOO residual and small-site fallback",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "weighted_surrogate_prediction_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "scalar_terms_checked": report["scalar_terms_checked"],
        "production_weighted_prediction_fallback_count": (
            report["production_weighted_prediction_fallback_count"]
        ),
        "retained_D_oracle": report["retained_D_oracle"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("weighted surrogate prediction binding replay failed")


if __name__ == "__main__":
    main()
