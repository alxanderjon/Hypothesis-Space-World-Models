#!/usr/bin/env python3
"""Inventory-only audit of current A/B hybrid-student bindings."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from experiments.audit_hybrid_student_bindings import binding_map
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


OUT = Path("outputs/current_hybrid_student_ab_binding_inventory")
INVENTORY = Path("outputs/deterministic_teacher_frozen/operation_inventory.json")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")


OVERRIDES = {
    "Gaussian bandwidth floor and ratio": (
        "gaussian_bandwidth_ratio_assembly",
        "resume_hybrid_student_after_ckdtree.py:gaussian_bandwidth_ratio_assembly",
    ),
    "ordered weighted surrogate prediction": (
        "ordered_weighted_prediction_assembly",
        "resume_hybrid_student_after_ckdtree.py:ordered_weighted_prediction_assembly",
    ),
    "LOO residual and small-site fallback": (
        "HybridGaussianLocalResponse.loo_errors assembled path",
        "resume_hybrid_student_after_ckdtree.py:HybridGaussianLocalResponse.loo_errors",
    ),
    "guarded min-max normalization": (
        "guarded_normalization_assembly",
        "resume_hybrid_student_after_ckdtree.py:guarded_normalization_assembly",
    ),
    "selected-cell weight": (
        "selected_cell_weight_assembly",
        "resume_hybrid_student_after_ckdtree.py:selected_cell_weight_assembly",
    ),
    "ordinary-cell winner": (
        "ordinary_first_max_assembly",
        "resume_hybrid_student_after_ckdtree.py:ordinary_first_max_assembly",
    ),
    "history duplicate filter": (
        "proposal_duplicate_filter_assembly",
        "resume_hybrid_student_after_ckdtree.py:proposal_duplicate_filter_assembly",
    ),
    "empty-proposal global fallback": (
        "empty_proposal_fallback_assembly",
        "resume_hybrid_student_after_ckdtree.py:empty_proposal_fallback_assembly",
    ),
    "normalized-to-physical affine map": (
        "normalized_to_physical_assembly",
        "resume_hybrid_student_after_ckdtree.py:normalized_to_physical_assembly",
    ),
    "continuous-stage Sobol multistarts": (
        "continuous_sobol_starts_assembly",
        "resume_hybrid_student_after_ckdtree.py:continuous_sobol_starts_assembly",
    ),
    "continuous result guards/winner/duplicate route": (
        "continuous_result_guards_assembly",
        "resume_hybrid_student_after_ckdtree.py:continuous_result_guards_assembly",
    ),
    "absorbing success latch": (
        "ArchiveAbsorbingSuccessLatch",
        "resume_hybrid_student_after_ckdtree.py:ArchiveAbsorbingSuccessLatch",
    ),
    "best-ever strict-loss selection": (
        "N2BestEverStrictLossTracker",
        "resume_hybrid_student_after_ckdtree.py:N2BestEverStrictLossTracker",
    ),
    "gradient operator": (
        "HybridN3GradientBinding",
        "resume_hybrid_student_after_ckdtree.py:HybridN3GradientBinding",
    ),
    "symmetric Hessian operator": (
        "HybridN3HessianBinding",
        "resume_hybrid_student_after_ckdtree.py:HybridN3HessianBinding",
    ),
    "positive-base variable-power adjoints": (
        "assembled_variable_power inside exact gradient/Hessian bindings",
        "geometry/n3_gradient_binding_assembly.py and n3_hessian_binding_assembly.py",
    ),
    "DIRECT/DAMPED curvature route": (
        "HybridN3CurvatureRouteBinding",
        "resume_hybrid_student_after_ckdtree.py:HybridN3CurvatureRouteBinding",
    ),
    "damping arithmetic/retry values": (
        "HybridN3DampingBinding",
        "resume_hybrid_student_after_ckdtree.py:HybridN3DampingBinding",
    ),
    "damped matrix and RHS assembly": (
        "HybridN3DampedMatrixRHSBinding",
        "resume_hybrid_student_after_ckdtree.py:HybridN3DampedMatrixRHSBinding",
    ),
    "one-step prefix residual/MSE reduction": (
        "PrefixResidualMSEAssembly wired into live objective/gradient/Hessian paths",
        "geometry/prefix_residual_mse_assembly.py and n3_gradient_binding_assembly.py",
    ),
}


STATUS_FILES = {
    "Gaussian bandwidth floor and ratio": "outputs/hybrid_student_gaussian_bandwidth_ratio_binding/binding_status.json",
    "ordered weighted surrogate prediction": "outputs/hybrid_student_weighted_surrogate_prediction_binding/binding_status.json",
    "LOO residual and small-site fallback": "outputs/hybrid_student_loo_residual_binding/binding_status.json",
    "guarded min-max normalization": "outputs/hybrid_student_guarded_normalization_binding/binding_status.json",
    "selected-cell weight": "outputs/hybrid_student_selected_cell_weight_binding/binding_status.json",
    "ordinary-cell winner": "outputs/hybrid_student_ordinary_cell_winner_binding/binding_status.json",
    "history duplicate filter": "outputs/hybrid_student_duplicate_filter_binding/binding_status.json",
    "empty-proposal global fallback": "outputs/hybrid_student_empty_proposal_fallback_binding/binding_status.json",
    "normalized-to-physical affine map": "outputs/hybrid_student_normalized_to_physical_binding/binding_status.json",
    "continuous-stage Sobol multistarts": "outputs/hybrid_student_continuous_sobol_starts_binding/binding_status.json",
    "continuous result guards/winner/duplicate route": "outputs/hybrid_student_continuous_result_guards_binding/binding_status.json",
    "absorbing success latch": "outputs/hybrid_student_archive_absorbing_success_binding/binding_status.json",
    "best-ever strict-loss selection": "outputs/hybrid_student_n2_best_ever_selection_binding/binding_status.json",
    "gradient operator": "outputs/hybrid_student_n3_gradient_binding/binding_status.json",
    "symmetric Hessian operator": "outputs/hybrid_student_n3_hessian_binding/binding_status.json",
    "positive-base variable-power adjoints": "outputs/hybrid_student_n3_variable_power_binding/binding_status.json",
    "DIRECT/DAMPED curvature route": "outputs/hybrid_student_n3_curvature_route_binding/binding_status.json",
    "damping arithmetic/retry values": "outputs/hybrid_student_n3_damping_binding/binding_status.json",
    "damped matrix and RHS assembly": "outputs/hybrid_student_n3_damped_matrix_rhs_binding/binding_status.json",
    "one-step prefix residual/MSE reduction": "outputs/hybrid_student_prefix_residual_mse_binding/binding_status.json",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = [
        row for row in json.loads(INVENTORY.read_text())["inventory"]
        if row["category"] in ("A", "B")
    ]
    prior = binding_map()
    rows = []
    for item in inventory:
        operation = item["operation"]
        row = dict(prior[operation])
        if operation in OVERRIDES:
            implementation, source = OVERRIDES[operation]
            row.update(
                actual_implementation=implementation,
                source_file_function=source,
                status="CORRECT_BINDING",
            )
        status_file = STATUS_FILES.get(operation)
        artifact_exact = None
        if status_file is not None:
            payload = json.loads(Path(status_file).read_text())
            artifact_exact = any(
                value == "INTEGRATED_EXACT" for value in payload.values()
            )
            if not artifact_exact:
                row["status"] = "AMBIGUOUS / MULTIPLE_PATHS"
        row.update(
            section=item["section"], category=item["category"],
            required_implementation=item["reuse"],
            validation_artifact=status_file,
            validation_artifact_exact=artifact_exact,
        )
        rows.append(row)

    counts = Counter(row["status"] for row in rows)
    unresolved = [
        row for row in rows if row["status"] != "CORRECT_BINDING"
    ]
    expected = json.loads(CHECKPOINTS.read_text())
    current = checkpoint_hashes()
    report = {
        "status": "ALL_AB_BINDINGS_WIRED" if not unresolved else "AB_BINDINGS_REMAIN",
        "scope": "inventory-only; no closed-loop replay and no D-oracle peelback",
        "inventory_ab_operations": len(rows),
        "counts": dict(counts),
        "unwired_or_fallback_count": len(unresolved),
        "unwired_or_fallback_bindings": unresolved,
        "binding_table": rows,
        "retained_d_oracles_excluded": [
            "Hessian eigenspectrum and condition", "dense linear solve",
            "dimension-general box-QP active-set direction",
            "FLOAT64_STAGNATION_TAIL stop",
        ],
        "checkpoint_integrity": {
            "checkpoint_count": len(current),
            "unchanged": current == expected,
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "inventory_ab_operations": report["inventory_ab_operations"],
        "counts": report["counts"],
        "unwired_or_fallback_count": report["unwired_or_fallback_count"],
        "checkpoint_integrity": report["checkpoint_integrity"],
    }, indent=2))
    if unresolved or current != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
