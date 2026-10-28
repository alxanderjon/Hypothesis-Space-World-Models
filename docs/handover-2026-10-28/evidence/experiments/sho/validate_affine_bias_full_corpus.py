#!/usr/bin/env python3
"""Fail-closed 320-case validation of only the SHO homogeneity peel."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from experiments import canonical_teacher_regression_corpus as canonical
from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.recover_affine_bias_targeted import (
    AffineArchiveN1,
    AffineN3,
    AffineObjective,
    CASES as TARGETED_CASES,
    FINAL_GATE,
    LOWER,
    OUT as TARGETED_OUT,
    UPPER,
    branch,
    rollout,
)
from experiments.resume_hybrid_student_after_ckdtree import Log


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "sho" / "homogeneity_peeled_full_validation"
CASE_OUT = OUT / "cases"
FREEZE_OUT = ROOT / "outputs" / "sho" / "homogeneity_peeled_baseline"
LEGACY = ROOT / "outputs" / "sho" / "stage1B" / "dimension_general_edirect" / "validation_cases.npz"
BASELINE_MANIFEST = ROOT / "outputs" / "hybrid_student_baseline_frozen" / "freeze_manifest.json"
TARGETED_REPORT = TARGETED_OUT / "targeted_recovery_report.json"
BOUND_RULE = TARGETED_OUT / "coefficient_bound_rule.json"
AFFINE_SOURCE = ROOT / "experiments" / "geometry" / "affine_sho_update_assembly.py"
TARGETED_SOURCE = ROOT / "experiments" / "sho" / "recover_affine_bias_targeted.py"
THIS_SOURCE = Path(__file__).resolve()
EXPECTED_TEACHER_FREEZE = "66cfe5f07347bdadfb11baef61f4212918c42102cf48accf0d6563ab96c88759"
EXPECTED_CANONICAL_PAYLOAD = "51c5273ec3876c35db58f75e6fa0a831b3223ee5477082d8575cae2923d506dd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def payload_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def immutable_json(path: Path, value) -> None:
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def distribution(values) -> dict:
    data = np.asarray(values, np.float64)
    return {
        "minimum": float(np.min(data)), "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.9)), "maximum": float(np.max(data)),
    }


def source_context() -> dict:
    corpus = json.loads(canonical.CORPUS_MANIFEST.read_text())
    baseline = json.loads(BASELINE_MANIFEST.read_text())
    targeted = json.loads(TARGETED_REPORT.read_text())
    if sha256(canonical.FREEZE) != EXPECTED_TEACHER_FREEZE:
        raise RuntimeError("frozen teacher manifest changed")
    canonical.verify_seal(corpus)
    if corpus["artifact_hash"] != EXPECTED_CANONICAL_PAYLOAD:
        raise RuntimeError("canonical corpus payload changed")
    if canonical.CANONICAL_POINTER.read_text().strip() != EXPECTED_CANONICAL_PAYLOAD:
        raise RuntimeError("canonical corpus pointer changed")
    if baseline.get("status") != "HYBRID_STUDENT_BASELINE":
        raise RuntimeError("parent hybrid baseline is not frozen")
    if targeted.get("status") != "HOMOGENEITY_ASSUMPTION_REMOVED_TARGETED":
        raise RuntimeError("targeted affine screen is not complete")
    if targeted.get("checkpoint_hashes_unchanged") is not True:
        raise RuntimeError("targeted affine screen checkpoint integrity failed")
    if list(LOWER) != [-1.99] * 6 or list(UPPER) != [1.99] * 6:
        raise RuntimeError("six-dimensional affine bounds changed")
    # Full guarded loading proves every canonical target still matches its
    # manifest, sources, environment, case identity, and artifact hash.
    records = canonical.load_canonical_corpus()
    sho_records = [row for row in records
                   if row["case_identity"]["system"] == "full_k_sho"]
    if len(sho_records) != 320:
        raise RuntimeError("canonical SHO corpus is not 320 cases")
    return {
        "teacher_freeze_manifest_sha256": sha256(canonical.FREEZE),
        "canonical_corpus_payload_sha256": corpus["artifact_hash"],
        "canonical_corpus_file_sha256": sha256(canonical.CORPUS_MANIFEST),
        "canonical_case_manifest_file_sha256": sha256(canonical.SPEC_MANIFEST),
        "hybrid_student_baseline_manifest_sha256": sha256(BASELINE_MANIFEST),
        "checkpoint_manifest_sha256": baseline["checkpoint_manifest_sha256"],
        "checkpoint_count": baseline["checkpoint_count"],
        "affine_source_sha256": sha256(AFFINE_SOURCE),
        "targeted_runner_source_sha256": sha256(TARGETED_SOURCE),
        "full_validator_source_sha256": sha256(THIS_SOURCE),
        "targeted_report_sha256": sha256(TARGETED_REPORT),
        "coefficient_bound_rule_sha256": sha256(BOUND_RULE),
        "canonical_records_guarded": len(records),
        "canonical_sho_records_guarded": len(sho_records),
    }


def specs_and_reference():
    specs = [row for row in canonical.build_specs()
             if row["system"] == "full_k_sho"]
    records = canonical.load_canonical_corpus()
    reference = {row["case_identity"]["canonical_case_id"]: row
                 for row in records
                 if row["case_identity"]["system"] == "full_k_sho"}
    if [canonical.case_key(row) for row in specs] != list(reference):
        raise RuntimeError("canonical SHO ordering differs from guarded corpus")
    return specs, reference


def design_condition(trajectory) -> dict:
    design = np.column_stack((
        np.asarray(trajectory[:-1], np.float64),
        np.ones(len(trajectory) - 1, np.float64),
    ))
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = max(design.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        "design_rank": rank, "affine_jacobian_rank": 2 * rank,
        "singular_values": singular.tolist(),
        "rank_tolerance": float(tolerance),
        "condition_number": float(singular[0] / singular[-1]),
    }


def case_category(spec, legacy) -> str:
    if spec["family"] == "legacy_prefix":
        return str(legacy["categories"][int(spec["case_id"])])
    return "readiness_sobol"


def run_case(spec, original, legacy, log, signature):
    key = canonical.case_key(spec)
    path = CASE_OUT / (key.replace(":", "__") + ".json")
    if path.exists():
        row = json.loads(path.read_text())
        if row.get("run_signature") != signature or row.get("canonical_case_id") != key:
            raise RuntimeError(f"saved case provenance mismatch: {key}")
        return row, True

    trajectory, initial, _, _, _, _, teacher = canonical.case_data(spec)
    trajectory = np.asarray(trajectory, np.float64)
    teacher = np.asarray(teacher, np.float64).reshape(-1)
    condition = design_condition(trajectory)
    objective = AffineObjective(trajectory)
    runner = AffineArchiveN1(objective, int(spec["n1_seed"]), log)
    branches_tested = 0
    successful = None
    stage = "N1"
    while successful is None:
        runner.advance_one()
        n = len(runner.losses)
        for archive_index in [n - 1] + list(range(n - 1)):
            branches_tested += 1
            archive_theta = LOWER + (UPPER - LOWER) * runner.points[archive_index]
            stage = "N2/N3"
            n2, n3, success, box_lower, box_upper = branch(
                trajectory, objective, archive_theta,
                runner.losses[archive_index], n, log,
            )
            if success:
                successful = (
                    archive_index, archive_theta, n2, n3, box_lower, box_upper,
                )
                break
        if n >= 5000:
            row = {
                "run_signature": signature, "canonical_case_id": key,
                "family": spec["family"], "case_id": int(spec["case_id"]),
                "category": case_category(spec, legacy),
                "prefix_length": int(len(trajectory)), "condition": condition,
                "success": False, "failure_stage": "d=6 N1 behavior",
                "failure_detail": "existing N1 resource cap reached before a valid handoff",
            }
            atomic_json(path, row)
            return row, False

    index, archive_theta, n2, n3, box_lower, box_upper = successful
    final = np.asarray(n3["theta"], np.float64)
    truth6 = np.concatenate((teacher, np.zeros(2, np.float64)))
    parameter_error = final - truth6
    rollout_result = rollout(trajectory, teacher, final)
    success = bool(n3["loss"] < FINAL_GATE and rollout_result["rollout_finite"])
    old4 = np.asarray(original["final"]["parameters"], np.float64)
    row = {
        "run_signature": signature, "canonical_case_id": key,
        "family": spec["family"], "case_id": int(spec["case_id"]),
        "category": case_category(spec, legacy), "initial_condition": initial,
        "prefix_length": int(len(trajectory)),
        "transition_count": int(len(trajectory) - 1), "condition": condition,
        "ground_truth_K": teacher.tolist(), "recovered_K": final[:4].tolist(),
        "ground_truth_b": [0.0, 0.0], "recovered_b": final[4:].tolist(),
        "final_parameters": final.tolist(),
        "final_parameter_error": parameter_error.tolist(),
        "final_max_parameter_error": float(np.max(np.abs(parameter_error))),
        "final_l2_parameter_error": float(np.linalg.norm(parameter_error)),
        "K_max_error": float(np.max(np.abs(final[:4] - teacher))),
        "max_abs_bias": float(np.max(np.abs(final[4:]))),
        "original_four_parameter_recovery": old4.tolist(),
        "K_minus_original_four_parameter_recovery": (final[:4] - old4).tolist(),
        "K_vs_original_max_difference": float(np.max(np.abs(final[:4] - old4))),
        "final_loss": float(n3["loss"]), "rollout": rollout_result,
        "N1": {
            "stopping_evaluation": len(runner.losses),
            "stopping_phase": runner.phase,
            "archive_branches_tested": branches_tested,
            "handoff_archive_index": index,
            "handoff_insertion_evaluation": index + 1,
            "handoff_is_newest": index == len(runner.losses) - 1,
            "handoff_is_incumbent": index == int(np.argmin(runner.losses)),
            "handoff_theta": archive_theta.tolist(),
            "handoff_loss": float(runner.losses[index]),
            "evaluated_sources": [event["source"] for event in runner.events],
        },
        "N2": {
            "box_lower": box_lower.tolist(), "box_upper": box_upper.tolist(),
            "endpoint": np.asarray(n2["theta"], np.float64).tolist(),
            "loss": float(n2["loss"]), "evaluations": int(n2["evaluations"]),
            "stop_reason": n2["message"],
        },
        "N3": {
            "accepted_steps": int(n3["accepted_steps"]),
            "objective_evaluations": int(n3["objective_evaluations"]),
            "stop_reason": n3["stop_reason"],
            "route_counts": n3["route_counts"],
            "backtracking_rejections": int(n3["backtracking_rejections"]),
        },
        "success": success,
        "failure_stage": None if success else "N3 behavior",
        "failure_detail": None if success else (
            "completed route did not satisfy existing final-loss/finite-rollout semantics"
        ),
    }
    atomic_json(path, row)
    return row, False


def summarize(rows) -> dict:
    bias = np.asarray([row["max_abs_bias"] for row in rows])
    kerror = np.asarray([row["K_max_error"] for row in rows])
    olddiff = np.asarray([row["K_vs_original_max_difference"] for row in rows])
    loss = np.asarray([row["final_loss"] for row in rows])
    rollout_error = np.asarray([
        row["rollout"]["continuation_max_error"] for row in rows
    ])
    condition = np.asarray([row["condition"]["condition_number"] for row in rows])
    log_condition = np.log10(condition)
    log_bias = np.log10(np.maximum(bias, np.finfo(np.float64).tiny))
    log_kerror = np.log10(np.maximum(kerror, np.finfo(np.float64).tiny))
    return {
        "recovery_count": int(sum(row["success"] for row in rows)),
        "case_count": len(rows),
        "recovery_by_category": {
            key: {"recovered": int(sum(row["success"] for row in values)),
                  "total": len(values)}
            for key in sorted({row["category"] for row in rows})
            for values in [[row for row in rows if row["category"] == key]]
        },
        "max_abs_bias": distribution(bias),
        "K_max_error": distribution(kerror),
        "K_vs_original_four_parameter_max_difference": distribution(olddiff),
        "final_loss": distribution(loss),
        "rollout_max_error": distribution(rollout_error),
        "affine_design_condition": distribution(condition),
        "conditioning_relationship": {
            "pearson_log10_condition_vs_log10_max_abs_bias": float(np.corrcoef(
                log_condition, log_bias,
            )[0, 1]),
            "spearman_condition_vs_max_abs_bias": float(spearmanr(
                condition, bias,
            ).statistic),
            "pearson_log10_condition_vs_log10_K_error": float(np.corrcoef(
                log_condition, log_kerror,
            )[0, 1]),
            "spearman_condition_vs_K_error": float(spearmanr(
                condition, kerror,
            ).statistic),
            "descriptive_only_no_gate": True,
        },
        "route_statistics": {
            "N1_stopping_evaluation": dict(Counter(
                str(row["N1"]["stopping_evaluation"]) for row in rows
            )),
            "archive_handoff_newest": int(sum(row["N1"]["handoff_is_newest"] for row in rows)),
            "archive_handoff_incumbent": int(sum(row["N1"]["handoff_is_incumbent"] for row in rows)),
            "archive_branches_tested": distribution([
                row["N1"]["archive_branches_tested"] for row in rows
            ]),
            "N2_evaluations": distribution([row["N2"]["evaluations"] for row in rows]),
            "N2_stop_reasons": dict(Counter(row["N2"]["stop_reason"] for row in rows)),
            "N2_maxfev_routes_recovered": int(sum(
                row["success"] and "Maximum number" in row["N2"]["stop_reason"]
                for row in rows
            )),
            "N3_accepted_steps": distribution([
                row["N3"]["accepted_steps"] for row in rows
            ]),
            "N3_stop_reasons": dict(Counter(row["N3"]["stop_reason"] for row in rows)),
            "N3_route_totals": dict(sum(
                (Counter(row["N3"]["route_counts"]) for row in rows), Counter()
            )),
            "N3_backtracking_rejections": distribution([
                row["N3"]["backtracking_rejections"] for row in rows
            ]),
        },
    }


def freeze(report, context, checkpoint_after) -> dict:
    report_sha = sha256(OUT / "report.json")
    remaining_assumptions = [
        "two observed components are the complete state",
        "first-order Markov structure", "deterministic autonomous update",
        "linear dependence on state", "time invariance",
        "fixed two-dimensional observed state", "full 2x2 connectivity",
        "shared coefficients across transitions", "no nonlinear correction",
        "no latent state", "no exogenous input", "no observation model",
        "unit discrete transition / no separately inferred timestep",
    ]
    manifest = {
        "status": "SHO_HOMOGENEITY_PEELED_BASELINE",
        "scope": "ONLY homogeneity removed: z_next=Kz+b",
        "parent_hybrid_student_baseline_manifest_sha256": context[
            "hybrid_student_baseline_manifest_sha256"
        ],
        "teacher_freeze_manifest_sha256": context["teacher_freeze_manifest_sha256"],
        "canonical_corpus_payload_sha256": context["canonical_corpus_payload_sha256"],
        "affine_source_hashes": {
            "experiments/geometry/affine_sho_update_assembly.py": sha256(AFFINE_SOURCE),
            "experiments/sho/recover_affine_bias_targeted.py": sha256(TARGETED_SOURCE),
            "experiments/sho/validate_affine_bias_full_corpus.py": sha256(THIS_SOURCE),
        },
        "coefficient_bound_rule": json.loads(BOUND_RULE.read_text()),
        "parameter_order": ["K00", "K01", "K10", "K11", "b0", "b1"],
        "search_domain": {"lower": [-1.99] * 6, "upper": [1.99] * 6},
        "checkpoint_manifest_sha256": context["checkpoint_manifest_sha256"],
        "checkpoint_count": context["checkpoint_count"],
        "checkpoint_hashes_unchanged": checkpoint_after,
        "validation": {
            "artifact": "outputs/sho/homogeneity_peeled_full_validation/report.json",
            "sha256": report_sha, "cases": 320, "recovered": 320,
        },
        "remaining_supplied_physical_assumptions": remaining_assumptions,
        "remaining_deferred_D_oracles": [
            "cKDTree exact-tie ownership", "np.log10",
            "ordered top-k retrieval", "np.exp",
        ],
        "training_performed": False,
        "next_physical_peel_selected": False,
    }
    manifest["manifest_payload_sha256"] = payload_hash(manifest)
    immutable_json(FREEZE_OUT / "freeze_manifest.json", manifest)
    marker = manifest["manifest_payload_sha256"] + "\n"
    marker_path = FREEZE_OUT / "SHO_HOMOGENEITY_PEELED_BASELINE"
    if marker_path.exists() and marker_path.read_text() != marker:
        raise RuntimeError("immutable homogeneity baseline marker differs")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(marker)
    return manifest


def main():
    started = time.monotonic()
    context_before = source_context()
    checkpoint_before = hashes()
    specs, references = specs_and_reference()
    legacy = np.load(LEGACY)
    signature = payload_hash({
        "kind": "sho_affine_full_320", "context": context_before,
        "bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "parameter_order": ["K00", "K01", "K10", "K11", "b0", "b1"],
        "case_order": [canonical.case_key(row) for row in specs],
        "targeted_cases": list(TARGETED_CASES),
    })
    atomic_json(OUT / "preflight.json", {
        "status": "IMMUTABILITY_AND_LIVE_PATH_VERIFIED",
        "context": context_before, "run_signature": signature,
        "objective_order": [
            "locked Kz matmul", "separate locked coordinatewise +b",
            "locked residual/square/production-order MSE",
        ],
        "all_six_coordinates_free": True,
        "special_zero_bias_handling": False,
        "regularization": None, "ground_truth_available_to_search": False,
    })

    log = Log()
    rows = []
    resumed = 0
    first_failure = None
    for ordinal, spec in enumerate(specs):
        key = canonical.case_key(spec)
        try:
            row, reused = run_case(
                spec, references[key], legacy, log, signature,
            )
        except Exception as error:
            first_failure = {
                "ordinal": ordinal, "canonical_case_id": key,
                "classification": "another demonstrated cause",
                "first_demonstrated_stage": "execution",
                "exception_type": type(error).__name__, "exception": str(error),
            }
            break
        rows.append(row)
        resumed += int(reused)
        print(
            f"[{ordinal + 1:03d}/320] {key} success={row['success']} "
            f"loss={row.get('final_loss')} cond={row['condition']['condition_number']}",
            flush=True,
        )
        if not row["success"]:
            first_failure = {
                "ordinal": ordinal, "canonical_case_id": key,
                "classification": row["failure_stage"],
                "affine_design_condition_number": row["condition"]["condition_number"],
                "ground_truth_parameters": row["ground_truth_K"] + row["ground_truth_b"],
                "recovered_parameters": row["final_parameters"],
                "final_loss": row["final_loss"], "rollout": row["rollout"],
                "N1": row["N1"], "N2": row["N2"], "N3": row["N3"],
            }
            break
        atomic_json(OUT / "progress.json", {
            "run_signature": signature, "completed": len(rows),
            "last_case": key, "all_completed_successful": True,
        })

    checkpoint_after_values = hashes()
    context_after = source_context()
    immutable = context_before == context_after
    checkpoint_unchanged = checkpoint_before == checkpoint_after_values
    complete = len(rows) == 320 and first_failure is None
    report = {
        "status": (
            "HOMOGENEITY_ASSUMPTION_REMOVED_FULL_SHO" if complete
            else "FULL_SHO_AFFINE_VALIDATION_FAILED_CLOSED"
        ),
        "scope": "only SHO homogeneity peel z_next=Kz+b",
        "run_signature": signature, "canonical_case_count": 320,
        "completed_cases": len(rows), "resumed_cases": resumed,
        "first_failure": first_failure,
        "summary": summarize(rows) if complete else None,
        "case_artifact_directory": str(CASE_OUT.relative_to(ROOT)),
        "case_artifact_count": len(rows),
        "immutability": {
            "context_before": context_before, "context_after": context_after,
            "frozen_teacher_canonical_corpus_and_parent_baseline_unchanged": immutable,
            "checkpoint_hashes_unchanged": checkpoint_unchanged,
        },
        "live_path": {
            "objective_order": [
                "locked Kz matmul", "separate locked coordinatewise +b",
                "locked residual/square/production-order MSE",
            ],
            "all_six_coordinates_free": True,
            "bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
            "zero_bias_initialization_constraint_or_regularizer": False,
            "ground_truth_available_to_search": False,
            "deferred_D_oracles_unchanged": [
                "cKDTree exact-tie ownership", "np.log10",
                "ordered top-k retrieval", "np.exp",
            ],
        },
        "binding_invocations": dict(log.counts),
        "freeze_target": (
            "outputs/sho/homogeneity_peeled_baseline/freeze_manifest.json"
            if complete else None
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(OUT / "report.json", report)
    if complete:
        if not immutable or not checkpoint_unchanged:
            raise RuntimeError("immutability gate failed after complete validation")
        manifest = freeze(report, context_after, checkpoint_unchanged)
        atomic_json(OUT / "freeze_result.json", {
            "status": manifest["status"],
            "manifest": str((FREEZE_OUT / "freeze_manifest.json").relative_to(ROOT)),
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        })
    (OUT / "STATUS.txt").write_text(report["status"] + "\n")
    print(json.dumps({
        "status": report["status"], "completed": len(rows),
        "first_failure": first_failure,
        "checkpoint_hashes_unchanged": checkpoint_unchanged,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
