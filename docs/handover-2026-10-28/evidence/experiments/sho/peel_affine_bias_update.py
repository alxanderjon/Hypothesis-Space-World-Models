#!/usr/bin/env python3
"""First post-baseline physical peel: free only the SHO affine bias."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.geometry.affine_sho_update_assembly import (
    AffineSHOUpdateAssembly,
)
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "sho" / "post_baseline_physical_peel_affine_bias"
LEGACY = (
    ROOT / "outputs" / "sho" / "stage1B" / "dimension_general_edirect"
    / "validation_cases.npz"
)
INVENTORY = ROOT / "outputs" / "deterministic_teacher_frozen" / "operation_inventory.json"
REPRESENTATIVE_LEGACY = (0, 51, 100, 200)
REPRESENTATIVE_SOBOL = (0, 17, 51)
PERTURBATION = np.asarray((0.1, -0.2, 0.3, -0.4, 0.05, -0.07), np.float64)


def metric(expected, actual) -> dict:
    left = np.asarray(expected, np.float64).reshape(-1)
    right = np.asarray(actual, np.float64).reshape(-1)
    unequal = left.view(np.uint64) != right.view(np.uint64)
    return {
        "values": int(left.size),
        "bit_mismatches": int(np.count_nonzero(unequal)),
        "sign_mismatches": int(np.count_nonzero(
            np.signbit(left) != np.signbit(right)
        )),
        "zero_nonzero_mismatches": int(np.count_nonzero(
            (left == 0.0) != (right == 0.0)
        )),
        "signed_zero_mismatches": int(np.count_nonzero(
            (left == 0.0) & (right == 0.0)
            & (np.signbit(left) != np.signbit(right))
        )),
        "max_absolute_error": (
            float(np.max(np.abs(left - right))) if left.size else 0.0
        ),
    }


def native_reference(trajectory: np.ndarray, theta: np.ndarray) -> dict:
    states = torch.from_numpy(np.asarray(trajectory, np.float64).copy())
    value = torch.from_numpy(np.asarray(theta, np.float64).copy()).requires_grad_(True)
    inputs, targets = states[:-1], states[1:]
    # Keep the peel's required operation boundary explicit: matmul first, then
    # broadcast bias addition, then residual/MSE.
    before_bias = inputs @ value[:4].reshape(2, 2).T
    prediction = before_bias + value[4:]
    residual = prediction - targets
    square = residual.square()
    loss = square.mean()
    gradient = torch.autograd.grad(
        loss, value, create_graph=True, retain_graph=True,
    )[0]
    rows = tuple(
        torch.autograd.grad(
            gradient[index], value, retain_graph=True,
        )[0]
        for index in range(6)
    )
    raw = torch.stack(rows)
    symmetric = 0.5 * (raw + raw.T)
    return {
        "before_bias": before_bias.detach().numpy(),
        "prediction": prediction.detach().numpy(),
        "residual": residual.detach().numpy(),
        "square": square.detach().numpy(),
        "loss": loss.detach().numpy(),
        "gradient": gradient.detach().numpy(),
        "raw_hessian": raw.detach().numpy(),
        "symmetric_hessian": symmetric.detach().numpy(),
    }


def replay_cases(assembly: AffineSHOUpdateAssembly, legacy) -> tuple[list[dict], dict | None]:
    rows = []
    first_mismatch = None
    cases = []
    for case_id in REPRESENTATIVE_LEGACY:
        cases.append((
            "legacy", case_id,
            np.asarray(legacy[f"prefix_{case_id}"], np.float64),
            np.asarray(legacy["teachers"][case_id], np.float64).reshape(-1),
        ))
    for case_id in REPRESENTATIVE_SOBOL:
        trajectory, _, teacher = readiness.sho_case(case_id)
        cases.append(("readiness_sobol", case_id, trajectory, teacher))

    for family, case_id, trajectory, teacher in cases:
        affine_teacher = np.concatenate((teacher, np.zeros(2, np.float64)))
        for theta_source, theta in (
            ("ground_truth_extension", affine_teacher),
            ("ordinary_perturbation", affine_teacher + PERTURBATION),
        ):
            result = assembly.evaluate(trajectory, theta)
            reference = native_reference(trajectory, theta)
            comparisons = {
                "prediction_before_bias": metric(
                    reference["before_bias"], result.prediction_before_bias.numpy()
                ),
                "prediction": metric(
                    reference["prediction"], result.prediction.numpy()
                ),
                "residual": metric(reference["residual"], result.residual.numpy()),
                "square": metric(reference["square"], result.square.numpy()),
                "loss": metric(reference["loss"], result.loss.numpy()),
                "gradient": metric(reference["gradient"], result.gradient.numpy()),
                "raw_hessian": metric(
                    reference["raw_hessian"], result.raw_hessian.numpy()
                ),
                "symmetric_hessian": metric(
                    reference["symmetric_hessian"],
                    result.symmetric_hessian.numpy(),
                ),
            }
            exact = not any(row["bit_mismatches"] for row in comparisons.values())
            record = {
                "family": family, "case_id": case_id,
                "prefix_states": int(len(trajectory)),
                "transition_count": int(len(trajectory) - 1),
                "theta_source": theta_source,
                "comparisons": comparisons, "exact": exact,
            }
            rows.append(record)
            if not exact and first_mismatch is None:
                first_mismatch = record
                return rows, first_mismatch
    return rows, first_mismatch


def matrix_rank_details(matrix: np.ndarray) -> tuple[int, list[float], float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    return rank, singular.tolist(), float(tolerance)


def identifiability(legacy) -> dict:
    rows = []
    coordinate_min = np.full(2, np.inf, np.float64)
    coordinate_max = np.full(2, -np.inf, np.float64)
    for case_id in range(256):
        rows.append((
            "legacy", case_id,
            np.asarray(legacy[f"prefix_{case_id}"], np.float64),
        ))
    for case_id in range(64):
        trajectory, _, _ = readiness.sho_case(case_id)
        rows.append(("readiness_sobol", case_id, trajectory))

    details = []
    ranks = Counter()
    conditions = []
    for family, case_id, trajectory in rows:
        coordinate_min = np.minimum(coordinate_min, trajectory.min(axis=0))
        coordinate_max = np.maximum(coordinate_max, trajectory.max(axis=0))
        design = np.column_stack((
            trajectory[:-1], np.ones(len(trajectory) - 1, np.float64),
        ))
        rank, singular, tolerance = matrix_rank_details(design)
        jacobian_rank = 2 * rank
        ranks[jacobian_rank] += 1
        condition = float(singular[0] / singular[-1])
        conditions.append(condition)
        if jacobian_rank != 6:
            details.append({
                "family": family, "case_id": case_id,
                "design_rank": rank, "jacobian_rank": jacobian_rank,
                "singular_values": singular, "rank_tolerance": tolerance,
            })
    return {
        "cases": len(rows),
        "prefix_length_counts": dict(Counter(
            len(trajectory) for _, _, trajectory in rows
        )),
        "jacobian_rank_counts": {str(key): value for key, value in ranks.items()},
        "rank_loss_cases": details,
        "condition": {
            "minimum": float(np.min(conditions)),
            "median": float(np.median(conditions)),
            "maximum": float(np.max(conditions)),
        },
        "observed_state_coordinate_minimum": coordinate_min.tolist(),
        "observed_state_coordinate_maximum": coordinate_max.tolist(),
    }


def search_domain_audit() -> dict:
    inventory = json.loads(INVENTORY.read_text())["inventory"]
    wanted = {
        "dimension-general initial evaluated design",
        "ordered history snapshot",
        "reservoir budget and deterministic seed",
        "Sobol full-block sizing and prefix",
        "scrambled Sobol coordinate generation",
        "dimension-general boundary-face support",
        "pool merge, exact deduplication, lexicographic order",
        "FIFO proposal queue",
        "observation/archive insertion",
        "round/cap/trace control",
        "chronological retention",
        "retest old sites when current-count box changes",
        "newest-first then older-site ordering",
        "branch execution",
        "current-count local box construction",
        "bounded Powell mathematical state machine",
        "Powell loops/vector state/counters",
        "best-ever strict-loss selection",
        "endpoint packaging to N3",
        "dimension-general box-QP active-set direction",
        "active-set loop/index/state",
        "valid-handoff/recovered-law oracle",
    }
    declared = [
        {
            "section": row["section"], "operation": row["operation"],
            "dimension_general": bool(row["dimension_general"]),
            "category": row["category"],
        }
        for row in inventory if row["operation"] in wanted
    ]
    return {
        "hypothesis_dimension": 6,
        "declared_components": declared,
        "all_audited_components_declared_dimension_general": bool(
            len(declared) == len(wanted)
            and all(row["dimension_general"] for row in declared)
        ),
        "post_baseline_updates": {
            "initial_design": (
                "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_CONTROL_STORAGE"
            ),
            "boundary_support": (
                "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_CONTROL_STORAGE"
            ),
            "note": (
                "the frozen 86-operation inventory retains their pre-peel D "
                "labels; post-baseline binding artifacts supersede only those "
                "two classifications without changing their dimension-general contracts"
            ),
        },
        "current_K_bounds": {
            "lower": [-1.99] * 4, "upper": [1.99] * 4,
            "source": "canonical_teacher_regression_corpus.case_data",
            "rule": "explicit four-entry full-K bound vectors",
        },
        "bias_bound_rule_found": False,
        "bias_bounds": None,
        "reason": (
            "No frozen inventory, SHO configuration, canonical case adapter, "
            "or post-baseline binding declares bounds for newly introduced "
            "additive bias coordinates. Existing bounds are parameter-specific, "
            "not a rule assigning [-1.99,1.99] to every future coordinate."
        ),
        "success_semantics": (
            "existing final-loss/finite-rollout criterion is reusable, but the "
            "physical rollout adapter must apply Kz+b; no threshold change is needed"
        ),
    }


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    legacy = np.load(LEGACY)
    assembly = AffineSHOUpdateAssembly()
    replay, first_mismatch = replay_cases(assembly, legacy)
    if first_mismatch is not None:
        status = "FIRST_AFFINE_ARITHMETIC_MISMATCH"
        ident = None
        domain = None
    else:
        ident = identifiability(legacy)
        domain = search_domain_audit()
        status = (
            "AFFINE_PEEL_REQUIRES_BIAS_BOUND_DECISION"
            if not domain["bias_bound_rule_found"]
            else "SEARCH_DOMAIN_COMPLETE"
        )
    after = hashes()
    report = {
        "status": status,
        "scope": "post-baseline SHO homogeneous-to-affine physical peel only",
        "law": {
            "parameter_order": list(AffineSHOUpdateAssembly.parameter_order),
            "forward_order": [
                "locked two-term Kz matmul",
                "locked coordinatewise bias addition",
                "locked residual/square/production-order MSE",
            ],
            "new_parameter_count": 2,
            "ground_truth_bias_used_only_for_evaluation": [0.0, 0.0],
        },
        "derivative_decomposition": [
            {"operation": "Kz forward", "class": "A", "basis": "locked exact matmul assembly"},
            {"operation": "broadcast bias addition", "class": "A", "basis": "locked addition"},
            {"operation": "residual/square/MSE", "class": "A", "basis": "locked prefix objective"},
            {"operation": "prediction adjoint", "class": "B", "basis": "existing exact residual/MSE reverse assembly"},
            {"operation": "K gradient", "class": "B", "basis": "recovered MmBackward dot assembly"},
            {"operation": "bias gradient broadcast reduction", "class": "C", "basis": "ATen exact implementation-order sum scaffold"},
            {"operation": "constant-one feature insertion", "class": "C", "basis": "deterministic storage/scaffolding"},
            {"operation": "six Hessian rows", "class": "B", "basis": "locked multiply + MmBackward + exact sum scaffold"},
            {"operation": "Hessian symmetrization", "class": "B", "basis": "locked add/multiply in production order"},
        ],
        "direct_replay": {
            "evaluations": len(replay), "rows": replay,
            "all_exact": first_mismatch is None,
            "first_mismatch": first_mismatch,
        },
        "identifiability": ident,
        "search_domain_audit": domain,
        "targeted_recovery": {
            "executed": False,
            "reason": (
                "missing declared bias bounds" if status ==
                "AFFINE_PEEL_REQUIRES_BIAS_BOUND_DECISION" else
                "stopped at earlier required gate"
            ),
        },
        "checkpoint_integrity": {
            "count": len(before), "unchanged": before == after,
        },
        "frozen_teacher_modified": False,
        "hybrid_student_baseline_modified": False,
        "canonical_corpus_modified": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "STATUS").write_text(status + "\n")
    print(json.dumps({
        "status": status,
        "direct_replay_evaluations": len(replay),
        "direct_replay_exact": first_mismatch is None,
        "identifiability": ident,
        "bias_bound_rule_found": None if domain is None else domain["bias_bound_rule_found"],
        "checkpoint_integrity": before == after,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    if first_mismatch is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
