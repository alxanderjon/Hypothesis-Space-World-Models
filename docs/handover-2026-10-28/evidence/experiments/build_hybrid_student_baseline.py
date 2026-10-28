#!/usr/bin/env python3
"""Fail-closed integration of the frozen teacher's learned N1 prefix.

The integration is intentionally ordered exactly like the frozen inventory.  A
declared A/B operation is never replaced by its teacher implementation.  The
script stops the integration gate at the first same-input disagreement and
records which later operations were consequently not executed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from experiments.geometry.n1_sqrt_rounding_assembly import N1CorrectlyRoundedSqrt
from experiments.sho import assemble_n3_gradient_operator as arithmetic
from experiments.sho import dimension_general_edirect as production_n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol, schedule


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_baseline")
FREEZE = Path("outputs/deterministic_teacher_frozen")
INVENTORY = FREEZE / "operation_inventory.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InvocationLog:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.implementations: dict[str, dict] = {}

    def call(self, operation: str, category: str, implementation: str) -> None:
        if category not in ("A", "B", "D"):
            raise ValueError(category)
        self.counts[operation] += 1
        previous = self.implementations.setdefault(operation, {
            "category": category, "implementation": implementation,
            "teacher_fallback": category == "D",
        })
        if previous["category"] != category or previous["implementation"] != implementation:
            raise RuntimeError(f"inconsistent invocation registration for {operation}")

    def rows(self, inventory: list[dict]) -> list[dict]:
        output = []
        for item in inventory:
            if item["category"] not in ("A", "B"):
                continue
            metadata = self.implementations.get(item["operation"])
            output.append({
                "section": item["section"], "operation": item["operation"],
                "category": item["category"], "invocations": int(self.counts[item["operation"]]),
                "implementation": None if metadata is None else metadata["implementation"],
                "teacher_fallback": None if metadata is None else metadata["teacher_fallback"],
                "reached_before_gate_stop": bool(self.counts[item["operation"]]),
            })
        return output


class HybridN1Prefix:
    """Recovered Sobol/merge/distance path plus the declared D oracles."""

    def __init__(self, log: InvocationLog) -> None:
        self.log = log
        self.sobol = LearnedSobol()
        arithmetic.OPS = arithmetic.LearnedArithmetic()
        self.ops = arithmetic.OPS
        self.sqrt = N1CorrectlyRoundedSqrt()

    def pool(self, dimension: int, sample_count: int, case_id: int,
             iteration: int) -> np.ndarray:
        requested = min(
            production_n1.teacher.MC_FACTOR_PER_SITE * sample_count,
            production_n1.teacher.MC_CANDIDATE_CAP,
        )
        exponent, generated = schedule(requested)
        self.log.call("Sobol full-block sizing and prefix", "B",
                      "validate_sobol_batching_truncation.schedule + first-N slice")
        seed = (production_n1.teacher.SEARCH_SEED + 104729 * case_id
                + 1009 * iteration)
        block, _, _, _, _ = self.sobol.full_block(dimension, exponent, seed)
        if len(block) != generated:
            raise RuntimeError("learned Sobol block-size contract failed")
        interior = block[:requested]
        self.log.call("scrambled Sobol coordinate generation", "B",
                      "LearnedSobol.full_block locked scramble/XOR/scaling assembly")

        # Explicitly one of the six permitted current N1 D oracles.
        faces, _ = production_n1.boundary(dimension, requested, seed)
        self.log.call("dimension-general boundary-face support", "D",
                      "temporary production boundary oracle")

        raw = np.concatenate((interior, faces), axis=0)
        keys = tuple([np.arange(len(raw))]
                     + [raw[:, axis] for axis in range(dimension - 1, -1, -1)])
        order = np.lexsort(keys)
        ordered = raw[order]
        starts = np.r_[0, np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1]
        result = ordered[starts]
        self.log.call("pool merge, exact deduplication, lexicographic order", "B",
                      "n1_boundary_merge_dedup independent lexsort assembly")
        return result

    def distances(self, queries: np.ndarray, sites: np.ndarray) -> np.ndarray:
        """Arbitrary-d composition specified by the frozen distance inventory."""
        query = np.asarray(queries, dtype=np.float64)
        site = np.asarray(sites, dtype=np.float64)
        if query.shape != site.shape or query.ndim != 2:
            raise ValueError("paired [n,d] inputs required")
        q = torch.from_numpy(np.ascontiguousarray(query))
        p = torch.from_numpy(np.ascontiguousarray(site))
        delta = self.ops.subtract(q, p)
        square = self.ops.multiply(delta, delta)
        total = square[:, 0]
        for axis in range(1, square.shape[1]):
            total = self.ops.add(total, square[:, axis])
        result = self.sqrt(total.detach().cpu().numpy())
        self.log.call("Euclidean distance", "A",
                      "locked subtract/multiply/add + N1CorrectlyRoundedSqrt")
        return result


def representative_records() -> list[dict]:
    records = corpus()

    def find(path_fragment: str, dimension: int) -> dict:
        for row in records:
            if path_fragment in str(row["state_path"]) and row["points"].shape[1] == dimension:
                return row
        raise RuntimeError(f"missing representative state: {path_fragment}")

    return [
        {"system": "full_k_sho", "dimension": 4,
         "record": find("checkpointed_resource_edirect", 4), "case_id": 8_100_000},
        {"system": "reflection_ball", "dimension": 5,
         "record": find("bouncing_ball/reflection_peelback", 5), "case_id": 8_200_000},
        {"system": "kepler5", "dimension": 5,
         "record": find("d5_handoff_characterization/kepler_cases", 5), "case_id": 8_300_000},
    ]


def metric(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    left_bits, right_bits = left.view(np.uint64), right.view(np.uint64)
    different = left_bits != right_bits
    ordered_left = np.where(left_bits >> 63, ~left_bits,
                            left_bits ^ (np.uint64(1) << np.uint64(63)))
    ordered_right = np.where(right_bits >> 63, ~right_bits,
                             right_bits ^ (np.uint64(1) << np.uint64(63)))
    ulp = np.maximum(ordered_left, ordered_right) - np.minimum(ordered_left, ordered_right)
    return {
        "values": int(len(left)), "bit_mismatches": int(np.count_nonzero(different)),
        "numeric_mismatches": int(np.count_nonzero(left != right)),
        "max_absolute_error": float(np.max(np.abs(left - right))) if len(left) else 0.0,
        "max_ulp_error": int(np.max(ulp)) if len(left) else 0,
    }


def replay_prefix(hybrid: HybridN1Prefix) -> tuple[list[dict], dict | None]:
    rows = []
    first_divergence = None
    for item in representative_records():
        dimension = item["dimension"]
        record = item["record"]
        site_count = min(20, len(record["points"]))
        sites = np.asarray(record["points"][:site_count], np.float64)
        assembled_pool = hybrid.pool(dimension, site_count, item["case_id"], 0)
        teacher_pool = production_n1.pool(dimension, site_count, item["case_id"], 0)
        pool_exact = bool(np.array_equal(assembled_pool.view(np.uint64),
                                         teacher_pool.view(np.uint64)))
        if not pool_exact and first_divergence is None:
            first_divergence = {
                "operation": "Sobol/boundary/merge reservoir path",
                "system": item["system"], "dimension": dimension,
                "reason": "assembled reservoir differs from frozen teacher",
            }

        # k=2 candidate identities and exact ties remain cKDTree scaffolding/D.
        reference_distance, candidates = cKDTree(sites).query(teacher_pool, k=2)
        learned_columns = []
        for column in range(2):
            learned_columns.append(hybrid.distances(
                teacher_pool, sites[candidates[:, column]]))
        learned_distance = np.column_stack(learned_columns)
        distance_metric = metric(reference_distance, learned_distance)
        formula_distance = np.sqrt(np.sum(
            (teacher_pool[:, None, :] - sites[candidates]) ** 2, axis=2
        ))
        formula_vs_assembly = metric(formula_distance, learned_distance)
        formula_vs_ckdtree = metric(reference_distance, formula_distance)
        strict_reference = reference_distance[:, 0] < reference_distance[:, 1]
        strict_learned = learned_distance[:, 0] < learned_distance[:, 1]
        order_errors = int(np.count_nonzero(strict_reference != strict_learned))
        exact_ties = reference_distance[:, 0] == reference_distance[:, 1]
        row = {
            "system": item["system"], "dimension": dimension,
            "state_path": str(record["state_path"]), "site_count": site_count,
            "pool_values": int(teacher_pool.size), "pool_bit_exact": pool_exact,
            "distance": distance_metric, "strict_pair_order_errors": order_errors,
            "production_exact_ties": int(np.count_nonzero(exact_ties)),
            "distance_localization": {
                "learned_arithmetic_and_sqrt_vs_numpy_formula": formula_vs_assembly,
                "numpy_formula_vs_production_ckdtree": formula_vs_ckdtree,
            },
        }
        rows.append(row)
        if (distance_metric["bit_mismatches"] or order_errors) and first_divergence is None:
            first_divergence = {
                "operation": "Euclidean distance",
                "category": "A", "system": item["system"],
                "dimension": dimension,
                "reason": ("locked arbitrary-d subtract/square/repeated-add/sqrt assembly "
                           "does not reproduce production scipy.spatial.cKDTree distances"),
                "metrics": distance_metric,
                "strict_pair_order_errors": order_errors,
                "localized_suboperation": (
                    "production cKDTree distance kernel/evaluation order; the locked "
                    "arithmetic plus correctly-rounded sqrt exactly matches the explicit "
                    "np.sqrt(np.sum(delta*delta)) formula on these same inputs"
                ),
            }
    return rows, first_divergence


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    inventory_doc = json.loads(INVENTORY.read_text())
    inventory = inventory_doc["inventory"]
    frozen_hash_before = sha256(INVENTORY)
    frozen_checkpoints = json.loads(Path(
        "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
    ).read_text())
    current_checkpoints = checkpoint_hashes()
    if current_checkpoints != frozen_checkpoints:
        raise RuntimeError("checkpoint inventory changed after teacher freeze")
    freeze_doc = json.loads((FREEZE / "freeze_manifest.json").read_text())
    source_hashes_unchanged = all(
        Path(path).exists() and sha256(Path(path)) == expected
        for path, expected in freeze_doc["sources"].items()
    )
    if not source_hashes_unchanged:
        raise RuntimeError("frozen teacher source changed after teacher freeze")
    log = InvocationLog()
    hybrid = HybridN1Prefix(log)
    replays, divergence = replay_prefix(hybrid)

    # Fail closed: no downstream component or closed-loop run occurs after the
    # first substantive operation with a same-input mismatch.
    status = ("HYBRID_STUDENT_BASELINE" if divergence is None
              else "STOPPED_FIRST_A_B_DIVERGENCE")
    invocation_rows = log.rows(inventory)
    current_oracles = [row for row in inventory
                       if row["category"] == "D" and row["debt_scope"] == "current"]
    later_physics = [row for row in inventory
                     if row["category"] == "D" and row["debt_scope"] == "later_hypothesis"]
    report = {
        "status": status,
        "teacher_status": "DETERMINISTIC_TEACHER_VALIDATED",
        "freeze_manifest_sha256": sha256(FREEZE / "freeze_manifest.json"),
        "inventory_sha256_before": frozen_hash_before,
        "inventory_sha256_after": sha256(INVENTORY),
        "checkpoint_count": len(current_checkpoints),
        "checkpoints_modified": current_checkpoints != frozen_checkpoints,
        "frozen_teacher_sources_modified": not source_hashes_unchanged,
        "new_training": False,
        "same_input_component_replay": replays,
        "first_divergence": divergence,
        "closed_loop": {
            "executed": divergence is None,
            "reason_not_executed": (None if divergence is None else
                "required fail-closed stop at the first A/B component divergence"),
        },
        "invocations": invocation_rows,
        "active_A_B_components": [row for row in invocation_rows if row["invocations"] > 0],
        "not_reached_A_B_components": [row["operation"] for row in invocation_rows
                                         if row["invocations"] == 0],
        "current_D_oracles": [row["operation"] for row in current_oracles],
        "later_physical_D_untouched": [row["operation"] for row in later_physics],
        "hybrid_baseline_frozen": divergence is None,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "invocations.json").write_text(json.dumps(invocation_rows, indent=2) + "\n")
    lines = [
        "# Hybrid student integration gate", "", f"Status: `{status}`", "",
        "## Same-input replay", "",
    ]
    for row in replays:
        lines.append(
            f"- {row['system']} d={row['dimension']}: pool exact={row['pool_bit_exact']}; "
            f"distance mismatches={row['distance']['bit_mismatches']}/{row['distance']['values']}; "
            f"max ULP={row['distance']['max_ulp_error']}; order errors={row['strict_pair_order_errors']}"
        )
    lines.extend(["", "## First divergence", "", "```json",
                  json.dumps(divergence, indent=2), "```", "",
                  "Closed-loop replay was not run after the fail-closed gate.", ""])
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({
        "status": status, "first_divergence": divergence,
        "replays": replays, "output": str(OUT / "report.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
