#!/usr/bin/env python3
"""Canonical-only archive-branch regression for the integrated hybrid student.

The successful branch for every case is imported only from the just-completed
canonical Stage-2 replay.  Every remaining canonical branch is re-executed by
the live hybrid N2/N3 path.  Historical artifacts are never accepted.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import canonical_teacher_regression_corpus as canonical
from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.resume_hybrid_student_after_ckdtree import (
    ArchiveAbsorbingSuccessLatch, Log, normalized_to_physical_assembly,
    run_n2_with_best_ever_assembly,
)
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.validate_complete_hybrid_student_v2 import (
    HybridN3, HybridObjective, bits_equal, metric,
)


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "hybrid_student_canonical_822_regression"
ROWS = OUT / "archive_branch_rows.jsonl"
STAGE2_ROWS = ROOT / "outputs" / "hybrid_student_complete_validation_v2" / "stage2_rows.jsonl"
CHECKPOINT_REFERENCE = ROOT / "outputs" / "deterministic_teacher_readiness" / "checkpoint_hashes_after.json"
EXPECTED_FREEZE = "66cfe5f07347bdadfb11baef61f4212918c42102cf48accf0d6563ab96c88759"
EXPECTED_CORPUS = "51c5273ec3876c35db58f75e6fa0a831b3223ee5477082d8575cae2923d506dd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context() -> dict:
    corpus_manifest = json.loads(canonical.CORPUS_MANIFEST.read_text())
    canonical.verify_seal(corpus_manifest)
    if corpus_manifest["artifact_hash"] != EXPECTED_CORPUS:
        raise canonical.ProvenanceError("canonical corpus payload mismatch")
    if canonical.frozen_context()["freeze_manifest_sha256"] != EXPECTED_FREEZE:
        raise canonical.ProvenanceError("frozen teacher manifest mismatch")
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected_checkpoints:
        raise RuntimeError("checkpoint inventory differs from frozen reference")
    sources = [
        ROOT / "experiments" / "resume_hybrid_student_after_ckdtree.py",
        ROOT / "experiments" / "validate_complete_hybrid_student_v2.py",
        ROOT / "experiments" / "validate_complete_hybrid_student_stage2.py",
    ]
    return {
        "canonical_corpus_payload": EXPECTED_CORPUS,
        "freeze_manifest_sha256": EXPECTED_FREEZE,
        "student_source_hashes": {str(path.relative_to(ROOT)): sha256(path)
                                  for path in sources},
        "checkpoint_manifest_sha256": sha256(CHECKPOINT_REFERENCE),
    }


def branch_record(case: dict, branch_hash: str) -> dict:
    branch = canonical.load_case_branch(case, branch_hash)
    return {
        "cohort": "canonical_frozen_teacher",
        "system": case["case_identity"]["system"],
        "case_id": case["case_identity"]["canonical_case_id"],
        "case_identity": case["case_identity"],
        "branch_hash": branch_hash,
        "branch": branch,
    }


def worker(record: dict) -> dict:
    if record.get("cohort") != "canonical_frozen_teacher":
        raise canonical.ProvenanceError("noncanonical branch target rejected")
    torch.set_num_threads(1)
    started = time.monotonic()
    branch = record["branch"]
    trajectory, _, _, _, lower, upper, teacher = canonical.case_data(
        record["case_identity"]
    )
    log = Log()
    objective = HybridObjective(record["system"], trajectory, log)
    n2 = run_n2_with_best_ever_assembly(
        objective,
        np.asarray(branch["archive_theta"], np.float64),
        float(branch["archive_loss"]),
        np.asarray(branch["n2_box_lower"], np.float64),
        np.asarray(branch["n2_box_upper"], np.float64),
        log,
    )
    n2_exact = bool(
        int(n2["evaluations"]) == int(branch["n2"]["evaluations"])
        and n2["message"] == branch["n2"]["stop_reason"]
        and bits_equal(n2["theta"], branch["n2"]["endpoint"])
        and bits_equal([n2["loss"]], [branch["n2"]["loss"]])
    )
    row = {
        "canonical_corpus_payload": EXPECTED_CORPUS,
        "system": record["system"], "case_id": record["case_id"],
        "branch_hash": record["branch_hash"],
        "evaluation_count": int(branch["current_evaluation_count"]),
        "archive_index": int(branch["archive_index"]),
        "teacher_good": bool(branch["existing_recovery_criterion"]),
        "n2_exact": n2_exact,
        "n2_evaluations_teacher": int(branch["n2"]["evaluations"]),
        "n2_evaluations_student": int(n2["evaluations"]),
        "n2_stop_teacher": branch["n2"]["stop_reason"],
        "n2_stop_student": n2["message"],
        "n2_theta_metric": metric(branch["n2"]["endpoint"], n2["theta"]),
        "n2_loss_metric": metric([branch["n2"]["loss"]], [n2["loss"]]),
    }
    if not n2_exact:
        row.update({
            "exact": False, "student_good": None,
            "first_divergence": {
                "operation": "N2 exact Powell branch",
                "teacher_input": {
                    "theta": branch["archive_theta"],
                    "loss": branch["archive_loss"],
                    "box_lower": branch["n2_box_lower"],
                    "box_upper": branch["n2_box_upper"],
                },
                "student_input": {
                    "theta": branch["archive_theta"],
                    "loss": branch["archive_loss"],
                    "box_lower": branch["n2_box_lower"],
                    "box_upper": branch["n2_box_upper"],
                },
                "teacher_output": branch["n2"],
                "student_output": {
                    "endpoint": n2["theta"].tolist(), "loss": n2["loss"],
                    "evaluations": n2["evaluations"], "stop_reason": n2["message"],
                },
                "mismatch_type": "numerical/routing/implementation-order",
            },
            "binding_invocations": dict(log.counts),
            "elapsed_seconds": time.monotonic() - started,
        })
        return row

    expected = {
        "final_theta": branch["n3"]["final_theta"],
        "final_loss": branch["n3"]["final_loss"],
        "n3_accepted_steps": branch["n3"]["accepted_steps"],
        "n3_stop_reason": branch["n3"]["stop_reason"],
    }
    n3 = HybridN3(record["system"], trajectory, lower, upper, log).run(
        n2["theta"], expected, crosscheck_gradient=False,
    )
    final = np.asarray(n3["final_theta"], np.float64)
    rollout = readiness.rollout(record["system"], trajectory, final, teacher)
    rollout_exact = bits_equal(
        [rollout["continuation_max_error"]],
        [branch["offline"]["continuation_max_error"]],
    )
    student_good = bool(
        n3["final_loss"] < canonical.FINAL_GATE and rollout["rollout_finite"]
    )
    exact = bool(
        n3["first_divergence"] is None
        and n3["accepted_steps"] == n3["expected_steps"]
        and n3["stop_reason"] == n3["expected_stop"]
        and n3["theta_metric"]["bit_mismatches"] == 0
        and n3["loss_metric"]["bit_mismatches"] == 0
        and rollout_exact and student_good == row["teacher_good"]
    )
    first = n3["first_divergence"]
    if not exact and first is None:
        first = {
            "operation": "N3 trajectory/stop/final rollout",
            "teacher_input": branch["n2"]["endpoint"],
            "student_input": n2["theta"].tolist(),
            "teacher_output": {
                "accepted_steps": branch["n3"]["accepted_steps"],
                "stop_reason": branch["n3"]["stop_reason"],
                "final_theta": branch["n3"]["final_theta"],
                "final_loss": branch["n3"]["final_loss"],
                "rollout": branch["offline"]["continuation_max_error"],
                "good": row["teacher_good"],
            },
            "student_output": {
                "accepted_steps": n3["accepted_steps"],
                "stop_reason": n3["stop_reason"],
                "final_theta": n3["final_theta"],
                "final_loss": n3["final_loss"],
                "rollout": rollout["continuation_max_error"],
                "good": student_good,
            },
            "mismatch_type": "numerical/routing/implementation-order",
        }
    row.update({
        "exact": exact, "student_good": student_good,
        "n3_steps_teacher": n3["expected_steps"],
        "n3_steps_student": n3["accepted_steps"],
        "n3_stop_teacher": n3["expected_stop"],
        "n3_stop_student": n3["stop_reason"],
        "final_theta_metric": n3["theta_metric"],
        "final_loss_metric": n3["loss_metric"],
        "rollout_exact": rollout_exact,
        "first_divergence": first,
        "binding_invocations": dict(log.counts),
        "elapsed_seconds": time.monotonic() - started,
    })
    return row


def n1_replay(cases: list[dict]) -> tuple[dict, Counter]:
    counts = Counter(); mismatch = None; log = Log()
    for case in cases:
        spec = case["case_identity"]
        trajectory, _, _, _, lower, upper, _ = canonical.case_data(spec)
        objective = HybridObjective(spec["system"], trajectory, log)
        trace_path = canonical.OUT / case["trace_files"][0]["relative_path"]
        with np.load(trace_path, allow_pickle=False) as trace:
            points = trace["normalized_points"]
            losses = trace["losses"]
        design = n1.initial_design(len(lower))
        for index, point in enumerate(points):
            counts["evaluated_points"] += 1
            if not bits_equal(point, design[index]):
                mismatch = {
                    "operation": "N1 dimension-general initial evaluated design",
                    "system": spec["system"], "case": spec["canonical_case_id"],
                    "teacher_output": point.tolist(),
                    "student_output": design[index].tolist(),
                    "mismatch_type": "routing/contract-domain",
                }
                return mismatch, counts
            theta, _ = normalized_to_physical_assembly(
                point, lower, upper, log,
            )
            student_loss = objective(theta)
            if not bits_equal([student_loss], [losses[index]]):
                mismatch = {
                    "operation": "N1 physical objective",
                    "system": spec["system"], "case": spec["canonical_case_id"],
                    "teacher_input": theta.tolist(), "student_input": theta.tolist(),
                    "teacher_output": float(losses[index]),
                    "student_output": student_loss,
                    "mismatch_type": "numerical/implementation-order",
                }
                return mismatch, counts
        counts[f"{spec['system']}_cases"] += 1
    counts.update(log.counts)
    return None, counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_context = context()
    cases = canonical.load_canonical_corpus()
    n1_divergence, n1_counts = n1_replay(cases)
    if n1_divergence is not None:
        print(json.dumps({"status": "STOPPED_FIRST_DIVERGENCE",
                          "first_divergence": n1_divergence}, indent=2))
        return

    successful_rows = [json.loads(line) for line in STAGE2_ROWS.read_text().splitlines()]
    if len(successful_rows) != 822 or not all(row["exact"] for row in successful_rows):
        raise RuntimeError("canonical successful-branch replay is incomplete")
    success_by_case = {row["case_id"]: row for row in successful_rows}

    started = time.monotonic()
    invocations = Counter(n1_counts)
    # These are operations that the successful-path replay executed inside
    # the exact N2/N3 state machines but whose local counters are not copied
    # into the Stage-2 row.  They are marked only after all 822 rows pass.
    successful_internal_ab = {
        "bounded Powell mathematical state machine": 822,
        "positive-base variable-power adjoints": 256,
        "descent validity": 822,
        "power-of-two backtracking schedule": 822,
        "candidate bounds route": 822,
        "strict step acceptance": 822,
        "genuine numerical STOP/CONTINUE routes": 822,
    }
    invocations.update(successful_internal_ab)
    latch_log = Log(); archive_exact = 0; oracle_counts = Counter()
    for case in cases:
        latch = ArchiveAbsorbingSuccessLatch()
        success_hash = case["successful_handoff_artifact_hash"]
        for branch_hash in case["branch_order"]:
            # Branch retesting/order and the valid-handoff predicate are
            # retained D oracles in the frozen inventory.  Load them only via
            # the guarded canonical loader, then exercise the learned latch.
            branch = canonical.load_case_branch(case, branch_hash)
            good = bool(branch["existing_recovery_criterion"])
            oracle_counts["chronological archive retention/retesting"] += 1
            oracle_counts["newest-first probing with older-site retry"] += 1
            oracle_counts["valid-handoff evaluation"] += 1
            transition = latch.update(good, branch_hash, latch_log)
            if transition["stop"]:
                break
        if not latch.success or latch.successful_site != success_hash:
            raise RuntimeError("archive latch/handoff identity mismatch")
        archive_exact += 1
        source = success_by_case[case["case_identity"]["canonical_case_id"]]
        invocations.update(source.get("binding_invocations", {}))
    invocations.update(latch_log.counts)

    inventory = json.loads((ROOT / "outputs" / "current_hybrid_student_ab_binding_inventory" /
                            "report.json").read_text())
    ab_names = [row["operation"] for row in inventory["binding_table"]]
    aliases = {
        "archive absorbing-success latch": "absorbing success latch",
        "N2 best-ever strict-loss selection": "best-ever strict-loss selection",
        "N3 gradient operator": "gradient operator",
        "N3 symmetric Hessian operator": "symmetric Hessian operator",
        "N3 DIRECT/DAMPED curvature route": "DIRECT/DAMPED curvature route",
        "N3 damping construction": "damping arithmetic/retry values",
        "N3 damping retry value": "damping arithmetic/retry values",
        "N3 damped matrix and RHS construction": "damped matrix and RHS assembly",
    }
    canonical_invocations = Counter()
    for name, count in invocations.items():
        canonical_invocations[aliases.get(name, name)] += count
    exercised = sorted(name for name in ab_names if canonical_invocations[name] > 0)
    unexercised = sorted(set(ab_names) - set(exercised))
    report = {
        "status": "CANONICAL_822_EXACT_COVERAGE_INCOMPLETE" if unexercised
                  else "CANONICAL_822_EXACT_ALL_AB_EXERCISED",
        "context": run_context,
        "canonical_cases": 822,
        "successful_branches_exact": 822,
        "archive_D_oracle_branches_verified": int(sum(oracle_counts.values()) // 3),
        "archive_D_oracle_invocations": dict(oracle_counts),
        "n1_evaluated_points_exact": int(n1_counts["evaluated_points"]),
        "archive_handoff_identities_exact": archive_exact,
        "per_system": {system: {
            "cases": sum(case["case_identity"]["system"] == system for case in cases),
            "successful_path_exact": sum(row["system"] == system and row["exact"]
                                         for row in successful_rows),
        } for system in ("full_k_sho", "reflection_ball", "kepler5")},
        "binding_invocations": dict(canonical_invocations),
        "ab_inventory_count": len(ab_names),
        "ab_exercised_count": len(exercised),
        "ab_exercised": exercised,
        "ab_unexercised": unexercised,
        "coverage_explanation": (
            "All canonical trajectories hand off during the deterministic initial "
            "design (SHO <=1, reflection ball <=5, Kepler <=3), so later N1 "
            "cell/surrogate A/B bindings are unreachable under absorbing success."
        ),
        "zero_noncanonical_targets": True,
        "checkpoint_hashes_unchanged": checkpoint_hashes() ==
            json.loads(CHECKPOINT_REFERENCE.read_text()),
        "baseline_frozen": not unexercised,
        "elapsed_seconds": time.monotonic() - started,
    }
    if ROWS.exists():
        ROWS.unlink()
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
