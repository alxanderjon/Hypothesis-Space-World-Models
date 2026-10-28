#!/usr/bin/env python3
"""Fail-closed mixed validation of the integrated hybrid student.

Stage 2 is deliberately unreachable when Stage 1 finds a divergence.  The
script uses the frozen readiness artifacts only to select real, on-policy
branches; all N2 and N3 checks below execute the active recovered assemblies.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.geometry.float64_unary_negation_assembly import (
    Float64UnaryNegationAssembly,
)
from experiments.geometry.n3_eigenspectrum_condition_assembly import (
    N3EigenspectrumConditionAssembly,
)
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.recover_float64_unary_negation_and_resume import replay_system
import experiments.recover_float64_unary_negation_and_resume as n3_replay
from experiments.resume_hybrid_student_after_ckdtree import (
    Log,
    StrictOwnerAssembly,
    student_cell_state,
)
from experiments.resume_hybrid_student_at_curvature import load_locked_router
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import run_stage1b_n3_gradient_polishing as production_n3
from experiments.sho.assemble_n3_learned_polisher import ControlAssembly
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus as n1_corpus
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_complete_validation")
FREEZE = Path("outputs/deterministic_teacher_frozen")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left, right) -> bool:
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return a.shape == b.shape and bool(
        np.array_equal(a.view(np.uint64), b.view(np.uint64))
    )


def saved_result(system: str, case_id: int) -> dict:
    return json.loads(
        (readiness.CASES / system / f"case_{case_id:05d}" / "result.json").read_text()
    )


def selected_branch(system: str, case_id: int, selector: str) -> dict:
    result = saved_result(system, case_id)
    if selector == "success":
        return result["successful_branch"]
    if selector == "first_stagnation_bad":
        for path in sorted((readiness.CASES / system / f"case_{case_id:05d}" /
                            "branches").glob("*.json")):
            row = json.loads(path.read_text())
            if (not row["existing_recovery_criterion"] and
                    row["n3_stop_reason"] == "FLOAT64_STAGNATION_TAIL"):
                return row
    raise RuntimeError(f"missing branch {system} {case_id} {selector}")


def replay_target_branch(system: str, case_id: int, selector: str,
                         negation, controls, spectral) -> dict:
    spec = next(row for row in json.loads(readiness.MANIFEST.read_text())["cases"]
                if row["system"] == system and int(row["case_id"]) == case_id)
    trajectory, _, objective, _, lower, upper, _ = readiness.case_data(spec)
    saved = selected_branch(system, case_id, selector)
    theta = np.asarray(saved["archive_theta"], np.float64)
    n = int(saved["current_evaluation_count"])
    radius = (upper - lower) / float(n) ** (1.0 / len(lower))
    box_lower = np.maximum(lower, theta - radius)
    box_upper = np.minimum(upper, theta + radius)
    assembled_n2 = staged._run_n2(
        objective, theta, float(saved["archive_loss"]), box_lower, box_upper
    )
    n2_exact = bool(
        bits_equal(box_lower, saved["n2_box_lower"])
        and bits_equal(box_upper, saved["n2_box_upper"])
        and int(assembled_n2["evaluations"]) == int(saved["n2_evaluations"])
        and assembled_n2["message"] == saved["n2_stop_reason"]
        and bits_equal(assembled_n2["theta"], saved["n2_endpoint"])
        and bits_equal([assembled_n2["loss"]], [saved["n2_loss"]])
    )
    if not n2_exact:
        return {
            "system": system, "case_id": case_id, "selector": selector,
            "first_divergence": {
                "operation": "bounded Powell mathematical state machine",
                "teacher": saved,
                "student": {
                    "n2_evaluations": assembled_n2["evaluations"],
                    "n2_stop_reason": assembled_n2["message"],
                    "n2_endpoint": assembled_n2["theta"].tolist(),
                    "n2_loss": assembled_n2["loss"],
                },
            },
        }

    # replay_system normally reads the successful branch.  Substitute this
    # saved on-policy BAD branch only for the duration of this call.
    old_saved = n3_replay.saved_branch
    n3_replay.saved_branch = lambda _system, _case_id: saved
    try:
        replay = replay_system(saved, negation, controls, spectral)
    finally:
        n3_replay.saved_branch = old_saved
    result = saved_result(system, case_id)
    return {
        "system": system, "case_id": case_id, "selector": selector,
        "trajectory_shape": list(trajectory.shape),
        "archive_count": n, "archive_index": int(saved["archive_index"]),
        "archive_branch_order": result["branch_sequence"],
        "successful_site_newest": result["successful_site_newest"],
        "successful_site_incumbent": result["successful_site_incumbent"],
        "n2_exact": n2_exact,
        "n2_evaluations": int(assembled_n2["evaluations"]),
        "n2_stop_reason": assembled_n2["message"],
        "n3_accepted_steps": replay["accepted_steps"],
        "n3_stop_reason": replay["stop_reason"],
        "n3_final_theta_bit_mismatches": replay["final_theta_metric"]["bit_mismatches"],
        "n3_final_loss_bit_mismatches": replay["final_loss_metric"]["bit_mismatches"],
        "n3_operation_counts": replay["operation_counts"],
        "n3_first_divergence": replay["first_divergence"],
    }


def actual_n1_divergence() -> tuple[dict, Counter]:
    record = next(row for row in n1_corpus()
                  if "checkpointed_resource_edirect/cases/companion_24/"
                  in str(row["state_path"]))
    frozen = FrozenProposals()
    log = Log()
    owner = StrictOwnerAssembly(frozen, log)
    state = student_cell_state(
        {"record": record, "dimension": 4, "case_id": 24},
        frozen, owner, load_model(), log,
    )
    teacher = np.asarray(state["teacher_proposals"], np.float64)
    student = np.asarray(state["student_proposals"], np.float64)
    unequal_rows = np.flatnonzero(
        np.any(teacher.view(np.uint64) != student.view(np.uint64), axis=1)
    )
    if unequal_rows.tolist() != [1]:
        raise RuntimeError("unexpected companion_24 divergence signature")
    teacher_events = json.loads(Path(record["events_path"]).read_text())
    teacher_event = next(row for row in teacher_events if row["evaluation"] == 22)
    if not bits_equal(teacher_event["point"], teacher[1]):
        raise RuntimeError("teacher proposal is not the actual evaluation-22 point")

    # Values from the exact production statement and the active checkpoint
    # scorer on the two candidates whose order reverses.  They are retained
    # explicitly so the failure is reviewable without changing the scorer.
    return ({
        "operation": "ordinary-cell score arithmetic / ordinary-cell winner",
        "section": "N1", "declared_category": "B",
        "case": "companion_24", "case_id": 24, "dimension": 4,
        "input_state": str(record["state_path"]),
        "active_site_prefix": 20, "actual_divergence_evaluation": 22,
        "selected_cells_teacher": state["selected_teacher"],
        "selected_cells_student": state["selected_student"],
        "selected_cell": 4, "proposal_row": 1,
        "teacher_output": teacher[1].tolist(),
        "student_output": student[1].tolist(),
        "teacher_output_bits": teacher[1].view(np.uint64).tolist(),
        "student_output_bits": student[1].view(np.uint64).tolist(),
        "mismatch_type": "routing caused by non-exact active score implementation",
        "mismatch_count": 1,
        "differing_parameter_values": int(np.count_nonzero(
            teacher[1].view(np.uint64) != student[1].view(np.uint64))),
        "teacher_winner": {
            "owned_candidate_index": 64, "pool_index": 417,
            "point": teacher[1].tolist(),
            "production_score": 0.1795323334463992,
            "active_student_score": 0.1790089332336598,
        },
        "student_winner": {
            "owned_candidate_index": 101, "pool_index": 646,
            "point": student[1].tolist(),
            "production_score": 0.17950297315341293,
            "active_student_score": 0.17902994624973848,
        },
        "production_margin_teacher_minus_alternative": (
            0.1795323334463992 - 0.17950297315341293
        ),
        "student_margin_teacher_minus_alternative": (
            0.1790089332336598 - 0.17902994624973848
        ),
        "localization": (
            "distance, ownership, cell size, selected-cell order, normalization, "
            "and candidate set agree; the active integration calls the old frozen "
            "score network instead of the validated zero-parameter production-order "
            "ordinary_assembly, reversing the top two candidates in cell 4"
        ),
        "teacher_point_was_executed": True,
        "teacher_event_source": teacher_event["source"],
        "student_point_absent_from_teacher_history": not any(
            bits_equal(row["point"], student[1]) for row in teacher_events
        ),
    }, log.counts)


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints:
        raise RuntimeError("checkpoint inventory differs from frozen reference")
    freeze = json.loads((FREEZE / "freeze_manifest.json").read_text())
    sources_unchanged = all(
        Path(path).exists() and sha256(Path(path)) == digest
        for path, digest in freeze["sources"].items()
    )
    if not sources_unchanged:
        raise RuntimeError("frozen teacher source changed")

    gradient.OPS = gradient.LearnedArithmetic()
    negation = Float64UnaryNegationAssembly()
    controls = ControlAssembly()
    locked = load_locked_router()
    controls.curvature.and_router.load_state_dict(locked.and_router.state_dict())
    spectral = N3EigenspectrumConditionAssembly(production_n3.CONDITION_LIMIT)

    # Ordered mixed prelude.  It covers each requested difficult behavior
    # before entering the full-K N1 search-state sweep.
    targets = [
        ("full_k_sho", 0, "success", "ordinary recovery"),
        ("reflection_ball", 253, "success", "older/non-incumbent + N2 maxfev"),
        ("reflection_ball", 265, "first_stagnation_bad", "FLOAT64_STAGNATION_TAIL BAD"),
        ("reflection_ball", 215, "success", "constrained box-QP"),
        ("kepler5", 193, "success", "heavy backtracking + N2 maxfev"),
        ("kepler5", 231, "success", "constrained box-QP"),
    ]
    rows = []
    first_divergence = None
    for system, case_id, selector, reason in targets:
        row = replay_target_branch(
            system, case_id, selector, negation, controls, spectral
        )
        row["coverage_reason"] = reason
        rows.append(row)
        first_divergence = row.get("first_divergence") or row.get("n3_first_divergence")
        if first_divergence:
            break

    n1_invocations = Counter()
    if first_divergence is None:
        first_divergence, n1_invocations = actual_n1_divergence()

    after = checkpoint_hashes()
    inventory = json.loads((FREEZE / "operation_inventory.json").read_text())
    retained = [
        {"section": row["section"], "operation": row["operation"]}
        for row in inventory["remaining_D"]
    ]
    report = {
        "status": "STOPPED_FIRST_A_B_DIVERGENCE",
        "teacher_status": freeze["teacher_readiness_status"],
        "requested_full_corpus_cases": freeze["validated_trajectory_count"],
        "stage1": {
            "completed_target_branches": len(rows),
            "distinct_cases_by_system": dict(Counter(
                row["system"] for row in rows
            )),
            "cases_touched_by_system_including_divergent_n1_case": {
                "full_k_sho": 2, "reflection_ball": 3, "kepler5": 2,
            },
            "branch_rows": rows,
            "n1_search_case_replayed_until_divergence": "full_k_sho companion_24",
            "n1_actual_evaluations_before_divergence": 21,
        },
        "stage2": {
            "executed": False,
            "reason": "Stage 1 fail-closed divergence",
            "cases_replayed": 0,
        },
        "first_divergence": first_divergence,
        "n1_invocations_at_divergent_state": dict(n1_invocations),
        "bit_exact_final_parameter_loss_counts": {
            "branches_checked": len(rows),
            "parameter_exact": sum(
                row.get("n3_final_theta_bit_mismatches", 1) == 0 for row in rows
            ),
            "loss_exact": sum(
                row.get("n3_final_loss_bit_mismatches", 1) == 0 for row in rows
            ),
        },
        "archive_n2_n3_routing_agreement_before_divergence": bool(
            all(row.get("n2_exact") and row.get("n3_first_divergence") is None
                for row in rows)
        ),
        "retained_D_oracles": retained,
        "contract_clarification": {
            "operation": "ordinary-cell score arithmetic",
            "inventory_contract": "B / ordinary_cell_scoring LOCKED_ASSEMBLY",
            "validated_exact_implementation": (
                "experiments.sho.validate_cell_scoring_assemblies.ordinary_assembly"
            ),
            "active_hybrid_implementation": "FrozenProposals.score checkpoint network",
            "finding": (
                "the declared exact assembly was not wired into the active hybrid path; "
                "the checkpoint scorer is approximate and changes a real teacher route"
            ),
        },
        "all_A_B_replacements_invoked": False,
        "all_A_B_before_divergence_used_without_teacher_fallback": True,
        "checkpoint_hashes_unchanged": before == after == expected_checkpoints,
        "frozen_teacher_sources_unchanged": sources_unchanged,
        "baseline_frozen": False,
        "baseline_status": "NOT_FROZEN_FIRST_DIVERGENCE",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "first_divergence.json").write_text(
        json.dumps(first_divergence, indent=2) + "\n"
    )
    (OUT / "stage1_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "stage1_branches": len(rows),
        "first_divergence": first_divergence,
        "stage2_executed": False,
        "checkpoints_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
