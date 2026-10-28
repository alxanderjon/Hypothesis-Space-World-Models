#!/usr/bin/env python3
"""Static binding audit plus focused ordinary-cell binding replay.

This deliberately does not execute a closed-loop recovery.  The executable
entry point audited is ``validate_complete_hybrid_student.py`` and the helpers
that it imports.  Rows are keyed to the frozen 86-operation inventory.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.resume_hybrid_student_after_ckdtree import (
    Log, StrictOwnerAssembly, student_cell_state,
)
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_cell_scoring_assemblies import ordinary_assembly
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


ROOT = Path("outputs/hybrid_student_binding_audit")
INVENTORY = Path("outputs/deterministic_teacher_frozen/operation_inventory.json")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")


def binding(operation, actual, source, status, conditional="always"):
    return dict(operation=operation, actual_implementation=actual,
                source_file_function=source, status=status,
                conditional_branch=conditional)


def binding_map() -> dict[str, dict]:
    # The statuses describe the post-repair tree.  ordinary-cell score also
    # carries its pre-repair status in the report generated below.
    rows = [
        binding("Sobol full-block sizing and prefix", "sobol_schedule + exact full_block[:requested] prefix", "resume_hybrid_student_after_ckdtree.py:scheduled_pool:233-247", "CORRECT_BINDING"),
        binding("scrambled Sobol coordinate generation", "locked scramble/parity/XOR/recurrence/scaling assembly", "resume_hybrid_student_after_ckdtree.py:scrambled_sobol_full_block:172-223", "CORRECT_BINDING"),
        binding("pool merge, exact deduplication, lexicographic order", "n1_boundary_merge_dedup independent lexsort assembly", "resume_hybrid_student_after_ckdtree.py:scheduled_pool:261-266", "CORRECT_BINDING"),
        binding("Euclidean distance", "CKDTreeEuclideanDistance exact FMA-lane assembly", "resume_hybrid_student_after_ckdtree.py:StrictOwnerAssembly.distance_matrix:77-85", "CORRECT_BINDING"),
        binding("nearest-owner routing away from exact ties", "FastNearestArgmin learned comparison route", "resume_hybrid_student_after_ckdtree.py:StrictOwnerAssembly.owners:87-126", "CORRECT_BINDING", "unique minimum; ties remain inventory-D oracle"),
        binding("assigned-radius reduction", "cells.assigned_radius locked operator", "resume_hybrid_student_after_ckdtree.py:student_cell_state:171-174", "CORRECT_BINDING"),
        binding("nearest-other-site spacing", "exact cKDTree-distance plus FastNearestArgmin repeated-min", "resume_hybrid_student_after_ckdtree.py:StrictOwnerAssembly.spacing:128-160", "CORRECT_BINDING", "unique neighbor; ties remain inventory-D oracle"),
        binding("cell extent", "cells.predict_half + cells.larger_candidate", "resume_hybrid_student_after_ckdtree.py:student_cell_state:175-180", "CORRECT_BINDING"),
        binding("same-size grouping", "_learned_group_and_represent locked grouping", "resume_hybrid_student_after_ckdtree.py:student_cell_state:182-185", "CORRECT_BINDING"),
        binding("group representative selection", "_learned_group_and_represent locked representative selector", "resume_hybrid_student_after_ckdtree.py:student_cell_state:182-185", "CORRECT_BINDING"),
        binding("lower-hull turn and margin", "downstream.hull_threshold._threshold_hull locked operators", "resume_hybrid_student_after_ckdtree.py:student_cell_state:186-190", "CORRECT_BINDING"),
        binding("sufficient-decrease target", "improvement_and._build_pool locked arithmetic chain", "resume_hybrid_student_after_ckdtree.py:student_cell_state:191-202", "CORRECT_BINDING"),
        binding("neighbor/required hull slopes", "improvement_and._build_pool locked slope chain", "resume_hybrid_student_after_ckdtree.py:student_cell_state:191-202", "CORRECT_BINDING"),
        binding("potentially-optimal feasibility", "locked and_model over locked feasibility inputs", "resume_hybrid_student_after_ckdtree.py:student_cell_state:191-202", "CORRECT_BINDING"),
        binding("empty-PO largest-cell fallback", "repaired_choice locked largest router", "resume_hybrid_student_after_ckdtree.py:student_cell_state:203-205", "CORRECT_BINDING", "only when learned PO set is empty"),
        binding("loss floor clamp", "assembled_clamp locked subtraction/comparison/select assembly", "resume_hybrid_student_after_ckdtree.py:gaussian_surface_with_clamp_assembly:288-308", "CORRECT_BINDING"),
        binding("Gaussian bandwidth floor and ratio", "GaussianLocalResponse production implementation", "resume_hybrid_student_after_ckdtree.py:student_cell_state:368-377 and production surface methods", "DETERMINISTIC_FALLBACK"),
        binding("ordered weighted surrogate prediction", "GaussianLocalResponse.predict production implementation", "resume_hybrid_student_after_ckdtree.py:student_cell_state:210-218,242-245", "DETERMINISTIC_FALLBACK"),
        binding("LOO residual and small-site fallback", "GaussianLocalResponse.loo_errors production implementation", "resume_hybrid_student_after_ckdtree.py:student_cell_state:217-218", "DETERMINISTIC_FALLBACK"),
        binding("guarded min-max normalization", "n1.teacher._normalize", "resume_hybrid_student_after_ckdtree.py:student_cell_state:217-218,243-244", "DETERMINISTIC_FALLBACK"),
        binding("selected-cell weight", "native NumPy 0.5*a + 0.5*b", "resume_hybrid_student_after_ckdtree.py:student_cell_state:217-218", "DETERMINISTIC_FALLBACK"),
        binding("largest-cell role", "repaired_choice locked largest-cell router", "resume_hybrid_student_after_ckdtree.py:student_cell_state:229-231", "CORRECT_BINDING"),
        binding("largest-cell candidate score/route", "repaired_choice locked candidate router", "resume_hybrid_student_after_ckdtree.py:student_cell_state:237-240", "CORRECT_BINDING", "selected largest cell"),
        binding("ordinary-cell score arithmetic", "ordinary_assembly exact production-order assembly", "resume_hybrid_student_after_ckdtree.py:student_cell_state:242-249", "CORRECT_BINDING", "selected non-largest cells"),
        binding("ordinary-cell winner", "np.argmax deterministic production fallback", "resume_hybrid_student_after_ckdtree.py:student_cell_state:247-249", "DETERMINISTIC_FALLBACK", "selected non-largest cells"),
        binding("history duplicate filter", "np.linalg.norm + np.min production fallback", "resume_hybrid_student_after_ckdtree.py:student_cell_state:250-258", "DETERMINISTIC_FALLBACK"),
        binding("empty-proposal global fallback", "np.argsort reverse traversal production fallback", "resume_hybrid_student_after_ckdtree.py:student_cell_state:253-257", "DETERMINISTIC_FALLBACK", "only when all selected-cell proposals are empty/duplicates"),
        binding("normalized-to-physical affine map", "no student call in current integration entry point", "validate_complete_hybrid_student.py (absent)", "NOT_WIRED"),
        binding("continuous-stage Sobol multistarts", "scipy.stats.qmc.Sobol production generator", "resume_hybrid_student_after_ckdtree.py:continuous_replay:283-293", "DETERMINISTIC_FALLBACK"),
        binding("bounded multistart Powell surrogate minimization", "powell_trace learned=True + NumpyDirectionController", "resume_hybrid_student_after_ckdtree.py:continuous_replay:294-310", "CORRECT_BINDING"),
        binding("continuous result guards/winner/duplicate route", "Python min tuple route + bits_equal", "resume_hybrid_student_after_ckdtree.py:continuous_replay:311-316", "DETERMINISTIC_FALLBACK"),
        binding("absorbing success latch", "saved teacher branch metadata; no student latch call", "validate_complete_hybrid_student.py:replay_target_branch:122-146", "NOT_WIRED"),
        binding("bounded Powell mathematical state machine", "minimize_powell_exact SciPy-1.13.0 assembly", "kepler/run_four_parameter_peelback.py:_run_n2:344-352", "CORRECT_BINDING"),
        binding("best-ever strict-loss selection", "native Python value < best_loss and assignment", "kepler/run_four_parameter_peelback.py:_run_n2.counted:335-342", "DETERMINISTIC_FALLBACK"),
        binding("gradient operator", "raw torch.autograd.grad in staged._make_n3", "kepler/run_four_parameter_peelback.py:_make_n3.derivatives:253-263", "DETERMINISTIC_FALLBACK"),
        binding("symmetric Hessian operator", "raw repeated torch.autograd.grad + native symmetrization", "kepler/run_four_parameter_peelback.py:_make_n3.derivatives:253-263", "DETERMINISTIC_FALLBACK"),
        binding("positive-base variable-power adjoints", "Torch PowBackward reached through raw Kepler objective", "kepler/run_four_parameter_peelback.py:_make_n3.derivatives:253-263", "DETERMINISTIC_FALLBACK", "Kepler only"),
        binding("DIRECT/DAMPED curvature route", "student_direct with locked curvature.and_router", "recover_float64_unary_negation_and_resume.py:replay_system:190-203", "CORRECT_BINDING"),
        binding("damping arithmetic/retry values", "ControlAssembly.damping.initial/value", "recover_float64_unary_negation_and_resume.py:replay_system:208-235", "CORRECT_BINDING", "DAMPED route/retry diagnostics"),
        binding("damped matrix and RHS assembly", "ControlAssembly.damping.matrix + Float64UnaryNegationAssembly", "recover_float64_unary_negation_and_resume.py:replay_system:247-267", "CORRECT_BINDING"),
        binding("descent validity", "ControlAssembly.descent.check", "recover_float64_unary_negation_and_resume.py:replay_system:269-285", "CORRECT_BINDING"),
        binding("power-of-two backtracking schedule", "ControlAssembly.candidate", "recover_float64_unary_negation_and_resume.py:replay_system:290-305", "CORRECT_BINDING"),
        binding("candidate bounds route", "ControlAssembly.in_bounds", "recover_float64_unary_negation_and_resume.py:replay_system:310-325", "CORRECT_BINDING"),
        binding("strict step acceptance", "ControlAssembly.accepts", "recover_float64_unary_negation_and_resume.py:replay_system:328-340", "CORRECT_BINDING"),
        binding("genuine numerical STOP/CONTINUE routes", "ControlAssembly.pre_step_route/no_further_improvement", "recover_float64_unary_negation_and_resume.py:replay_system:169-185,360-379", "CORRECT_BINDING"),
        binding("one-step prefix residual/MSE reduction", "native NumPy/Torch objectives", "validate_deterministic_teacher_readiness.py:sho_objectives:84-99 and system objective callbacks", "DETERMINISTIC_FALLBACK", "system objective callback"),
    ]
    return {row["operation"]: row for row in rows}


def bits_equal(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def ordinary_replay() -> dict:
    states = cells = candidates = 0
    score_errors = ordering_errors = winner_errors = 0
    first = None
    by_family = Counter()
    for item in corpus():
        if len(item["points"]) < 20 or not item["continuous_path"].exists():
            continue
        traces = json.loads(item["continuous_path"].read_text())
        if not traces:
            continue
        case_id = int(traces[0]["case_id"])
        points = np.asarray(item["points"][:20], np.float64)
        losses = np.asarray(item["losses"][:20], np.float64)
        api = n1.composition(points.shape[1])
        data = api["_cell_data"](points, losses, case_id, 0)
        surface = n1.teacher.GaussianLocalResponse(points, losses)
        selected_sizes = data.sizes[data.selected]
        weights = (0.5 * n1.teacher._normalize(selected_sizes) +
                   0.5 * n1.teacher._normalize(surface.loo_errors(data.selected)))
        largest = int(np.argmax(selected_sizes))
        weights[largest] = 1.0
        local_cells = 0
        for position, site in enumerate(data.selected):
            indices = np.flatnonzero(data.owners == site)
            if position == largest or not len(indices):
                continue
            features = np.column_stack((
                n1.teacher._normalize(data.distances[indices]),
                n1.teacher._normalize(surface.predict(data.points[indices])),
                np.full(len(indices), weights[position]),
            ))
            weight = features[:, 2]
            production = weight * features[:, 0] - (1.0 - weight) * features[:, 1]
            assembled = ordinary_assembly(features)
            score_bad = int(np.count_nonzero(
                production.view(np.uint64) != assembled.view(np.uint64)))
            production_order = np.argsort(-production, kind="stable")
            assembled_order = np.argsort(-assembled, kind="stable")
            order_bad = int(not np.array_equal(production_order, assembled_order))
            winner_bad = int(np.argmax(production) != np.argmax(assembled))
            score_errors += score_bad
            ordering_errors += order_bad
            winner_errors += winner_bad
            candidates += len(indices)
            cells += 1
            local_cells += 1
            if (score_bad or order_bad or winner_bad) and first is None:
                first = dict(state=str(item["state_path"]), cell=int(site),
                             score_errors=score_bad,
                             production_order=production_order.tolist(),
                             assembled_order=assembled_order.tolist())
        if local_cells:
            states += 1
            by_family[item["family"]] += 1

    # Force the active repaired call site on the previously failing state.
    record = next(row for row in corpus()
                  if "checkpointed_resource_edirect/cases/companion_24/"
                  in str(row["state_path"]))
    frozen = FrozenProposals()
    log = Log()
    state = student_cell_state(
        {"record": record, "dimension": 4, "case_id": 24}, frozen,
        StrictOwnerAssembly(frozen, log), load_model(), log,
    )
    companion = dict(
        proposal_rows=len(state["teacher_proposals"]),
        proposal_bit_exact=bits_equal(state["teacher_proposals"], state["student_proposals"]),
        proposal_bit_mismatches=int(state["proposal_bit_mismatches"]),
        evaluation_22_point=np.asarray(state["student_proposals"])[1].tolist(),
        ordinary_invocations=int(log.counts["ordinary-cell score arithmetic"]),
        implementation=log.implementation["ordinary-cell score arithmetic"][1],
    )
    return dict(states=states, cells=cells, candidate_scores=candidates,
                exact_candidate_scores=candidates-score_errors,
                score_bit_mismatches=score_errors,
                exact_candidate_orderings=cells-ordering_errors,
                ordering_mismatches=ordering_errors,
                exact_winners=cells-winner_errors,
                winner_mismatches=winner_errors,
                by_family=dict(by_family), first_mismatch=first,
                companion_24=companion)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    inventory_doc = json.loads(INVENTORY.read_text())
    inventory = [row for row in inventory_doc["inventory"]
                 if row["category"] in ("A", "B")]
    bindings = binding_map()
    if len(inventory) != 46 or set(bindings) != {row["operation"] for row in inventory}:
        raise RuntimeError("binding map does not exactly cover frozen A/B inventory")
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    rows = []
    for row in inventory:
        bound = dict(bindings[row["operation"]])
        bound.update(section=row["section"], category=row["category"],
                     teacher_operation=row["operation"],
                     required_implementation=row["reuse"])
        bound["pre_repair_status"] = (
            "STALE_APPROXIMATE_BINDING"
            if row["operation"] == "ordinary-cell score arithmetic"
            else bound["status"]
        )
        bound["pre_repair_actual_implementation"] = (
            "FrozenProposals.score approximate checkpoint"
            if row["operation"] == "ordinary-cell score arithmetic"
            else bound["actual_implementation"]
        )
        rows.append(bound)

    replay = ordinary_replay()
    if (replay["score_bit_mismatches"] or replay["ordering_mismatches"] or
            replay["winner_mismatches"] or
            not replay["companion_24"]["proposal_bit_exact"]):
        raise RuntimeError("ordinary-cell binding replay failed")
    after = checkpoint_hashes()
    pre_counts = Counter(row["pre_repair_status"] for row in rows)
    post_counts = Counter(row["status"] for row in rows)
    defects = [row for row in rows if row["pre_repair_status"] != "CORRECT_BINDING"]
    remaining = [row for row in rows if row["status"] != "CORRECT_BINDING"]
    report = dict(
        status="ORDINARY_CELL_SCORE_BINDING_INTEGRATED_EXACT",
        inventory_sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        inventory_ab_operations=len(rows), pre_repair_counts=dict(pre_counts),
        post_repair_counts=dict(post_counts), binding_table=rows,
        defects_discovered=defects, remaining_binding_defects=remaining,
        ordinary_score_replay=replay,
        ordinary_cell_score_binding="INTEGRATED_EXACT",
        next_binding_in_dependency_order=remaining[0]["teacher_operation"],
        checkpoint_hashes_unchanged=(before == after == expected_checkpoints),
    )
    (ROOT / "report.json").write_text(json.dumps(report, indent=2))
    with (ROOT / "binding_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, extrasaction="ignore", fieldnames=[
            "section", "category", "teacher_operation", "required_implementation",
            "pre_repair_actual_implementation", "pre_repair_status",
            "actual_implementation", "status", "source_file_function",
            "conditional_branch",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: report[k] for k in (
        "status", "inventory_ab_operations", "pre_repair_counts",
        "post_repair_counts", "ordinary_score_replay",
        "next_binding_in_dependency_order", "checkpoint_hashes_unchanged",
    )}, indent=2))


if __name__ == "__main__":
    main()
