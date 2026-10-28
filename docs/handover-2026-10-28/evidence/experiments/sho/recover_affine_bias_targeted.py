#!/usr/bin/env python3
"""Targeted recovery after peeling only SHO homogeneity.

The supplied hypothesis remains autonomous, time-invariant, first-order,
two-state, affine, and fully connected.  Only the fixed-zero bias is removed.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import time
import types

import numpy as np
import torch

from experiments import validate_d5_production_pipeline as production
from experiments.expand_chronological_archive_handoff_dataset import ArchiveN1Runner
from experiments.geometry.affine_sho_update_assembly import AffineSHOUpdateAssembly
from experiments.geometry.initial_design_remainder_assembly import (
    initial_design_with_remainder_assembly,
)
from experiments.geometry.n1_boundary_support_assembly import (
    N1BoundarySupportAssembly,
    scheduled_pool_with_boundary_assembly,
)
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3CurvatureRouteBinding,
    HybridN3DampedMatrixRHSBinding,
    HybridN3DampingBinding,
    Log,
    normalized_to_physical_assembly,
    run_n2_with_best_ever_assembly,
)
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho import run_full_k_transfer as sho
from experiments.sho import run_stage1b_n3_gradient_polishing as n3_production
from experiments.sho.assemble_n3_learned_polisher import ControlAssembly
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.kepler import run_four_parameter_peelback as staged


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/post_baseline_physical_peel_affine_bias")
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OLD = Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases")
CASES = ("companion_6", "companion_29", "companion_14", "interior_49")
LOWER = np.full(6, -1.99, np.float64)
UPPER = np.full(6, 1.99, np.float64)
FINAL_GATE = 1.0e-20


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class AffineObjective:
    def __init__(self, trajectory):
        self.trajectory = np.asarray(trajectory, np.float64)
        self.assembly = AffineSHOUpdateAssembly()
        self.invocations = 0

    def __call__(self, theta) -> float:
        self.invocations += 1
        return float(self.assembly.forward_objective(self.trajectory, theta).loss)


def post_baseline_composition(dimension: int, log: Log):
    """Production cell logic with only its recovered pool path substituted."""
    namespace = dict(edirect.teacher.__dict__)
    boundary = N1BoundarySupportAssembly()
    namespace["_candidate_pool"] = lambda m, case, iteration: (
        scheduled_pool_with_boundary_assembly(
            dimension, m, case, iteration, log, boundary,
        )[0]
    )
    namespace["_local_greedy_candidates"] = edirect.local_greedy
    for name in ("_cell_data", "_cell_proposals", "_greedy_proposal"):
        source = getattr(edirect.teacher, name)
        namespace[name] = types.FunctionType(
            source.__code__, namespace, name, source.__defaults__, source.__closure__,
        )
    return namespace


class AffineArchiveN1(ArchiveN1Runner):
    def __init__(self, objective, case_id, log):
        super().__init__(objective, LOWER, UPPER, case_id)
        self.log = log
        self._design, self.initial_design_trace = (
            initial_design_with_remainder_assembly(6, return_trace=True)
        )
        self.api = post_baseline_composition(6, log)
        self.api["_greedy_proposal"] = (
            lambda p, f, c, it: continuous.proposal(p, f, c, it, self.traces)
        )

    def evaluate(self, unit, source):
        unit = np.asarray(unit, np.float64)
        theta, _ = normalized_to_physical_assembly(
            unit[None, :], self.lower, self.upper, self.log,
        )
        value = float(self.objective(theta[0]))
        if not math.isfinite(value):
            raise RuntimeError("nonfinite affine objective")
        self.points.append(unit.copy())
        self.losses.append(value)
        best = int(np.argmin(self.losses))
        self.events.append({
            "evaluation": len(self.losses), "iteration": self.iteration,
            "source": source, "point": unit.tolist(), "theta": theta[0].tolist(),
            "loss": value, "best_index": best,
            "best_loss": float(self.losses[best]),
        })


class AffineN3:
    def __init__(self, trajectory, log):
        self.trajectory = np.asarray(trajectory, np.float64)
        self.assembly = AffineSHOUpdateAssembly()
        self.lower = torch.from_numpy(LOWER.copy())
        self.upper = torch.from_numpy(UPPER.copy())
        self.curvature = HybridN3CurvatureRouteBinding()
        self.damping = HybridN3DampingBinding(self.curvature)
        self.matrix_rhs = HybridN3DampedMatrixRHSBinding()
        self.controls = ControlAssembly()
        self.log = log
        _, self.direction_oracle, _ = staged._make_n3(
            lambda value: value.sum() * 0.0, LOWER, UPPER,
        )
        self.objective_evaluations = 0

    def scalar_loss(self, theta) -> float:
        self.objective_evaluations += 1
        return float(self.assembly.forward_objective(
            self.trajectory, np.asarray(theta, np.float64),
        ).loss)

    def run(self, start) -> dict:
        theta = torch.from_numpy(np.asarray(start, np.float64).copy())
        current_loss = self.scalar_loss(theta.numpy())
        accepted_steps = 0
        stagnation_count = 0
        history = [{"theta": theta.numpy().tolist(), "loss": current_loss}]
        route_counts = Counter()
        stop_reason = "not_started"
        backtracking_rejections = 0
        while True:
            result = self.assembly.evaluate(self.trajectory, theta.numpy())
            derivative_loss = float(result.loss)
            if np.float64(derivative_loss).view(np.uint64) != np.float64(current_loss).view(np.uint64):
                raise RuntimeError("affine forward and derivative objectives diverged")
            gradient = result.gradient
            hessian = result.symmetric_hessian
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            pre = self.controls.pre_step_route(0, gradient_norm)
            if pre != "CONTINUE":
                stop_reason = pre
                break
            spectrum, route = self.curvature.evaluate(gradient, hessian, self.log)
            route_counts[route["route"]] += 1
            initial = self.damping.initial(
                spectrum.scale, spectrum.minimum_eigenvalue, route["direct"], self.log,
            )
            for attempt in range(n3_production.MAX_DAMPING_TRIALS):
                self.damping.value(
                    spectrum.scale, initial, route["direct"], attempt, self.log,
                )
            direction, qp = dimension_general_box_qp_direction(
                theta, gradient, hessian, self.lower, self.upper,
                self.direction_oracle,
            )
            if direction is None:
                stop_reason = qp["route_status"]
                break
            self.matrix_rhs.construct(
                hessian, gradient, float(qp["damping"]), self.log,
            )
            descent, dot, _ = self.controls.descent.check(gradient, direction)
            if not descent:
                stop_reason = "no_descent_direction"
                break
            full_predicted = -float(qp["quadratic_value"])
            accepted = False
            unchanged = False
            for backtrack in range(n3_production.MAX_BACKTRACKS + 1):
                candidate = self.controls.candidate(theta, direction, backtrack)
                if bool(torch.equal(candidate, theta)):
                    unchanged = True
                    break
                if not self.controls.in_bounds(candidate, LOWER, UPPER):
                    backtracking_rejections += 1
                    continue
                candidate_loss = self.scalar_loss(candidate.numpy())
                if self.controls.accepts(candidate_loss, current_loss):
                    stagnation_count = n3_production._updated_float64_stagnation_count(
                        stagnation_count, current_loss, candidate_loss, full_predicted,
                    )
                    theta = candidate.detach()
                    current_loss = candidate_loss
                    accepted_steps += 1
                    accepted = True
                    history.append({
                        "theta": theta.numpy().tolist(), "loss": current_loss,
                        "backtrack": backtrack, "route": route["route"],
                    })
                    break
                backtracking_rejections += 1
            if accepted:
                if stagnation_count >= n3_production.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
                    stop_reason = n3_production.FLOAT64_STAGNATION_STOP_REASON
                    break
                continue
            numerical_stop, _, _ = self.controls.no_further_improvement(
                unchanged, dot, current_loss,
            )
            stop_reason = (
                "floating_point_no_further_improvement"
                if numerical_stop else "backtracking_no_improvement"
            )
            break
        return {
            "theta": theta.numpy().copy(), "loss": current_loss,
            "accepted_steps": accepted_steps, "stop_reason": stop_reason,
            "objective_evaluations": self.objective_evaluations,
            "backtracking_rejections": backtracking_rejections,
            "route_counts": dict(route_counts), "history": history,
        }


def rollout(prefix, teacher, recovered):
    truth = np.empty((sho.ROLLOUT_STEPS + 1, 2), np.float64)
    predicted = np.empty_like(truth)
    truth[0] = predicted[0] = prefix[-1]
    matrix_true = np.asarray(teacher, np.float64).reshape(2, 2)
    matrix = np.asarray(recovered[:4], np.float64).reshape(2, 2)
    bias = np.asarray(recovered[4:], np.float64)
    for index in range(sho.ROLLOUT_STEPS):
        truth[index + 1] = matrix_true @ truth[index]
        predicted[index + 1] = matrix @ predicted[index] + bias
    error = predicted - truth
    return {
        "steps": sho.ROLLOUT_STEPS,
        "continuation_max_error": float(np.max(np.abs(error))),
        "continuation_mse": float(np.mean(error * error)),
        "rollout_finite": bool(np.isfinite(predicted).all()),
        "bit_exact": bool(np.array_equal(
            predicted.view(np.uint64), truth.view(np.uint64),
        )),
    }


def branch(trajectory, objective, archive_theta, archive_loss, n, log):
    radius = (UPPER - LOWER) / float(n) ** (1.0 / 6.0)
    box_lower = np.maximum(LOWER, archive_theta - radius)
    box_upper = np.minimum(UPPER, archive_theta + radius)
    n2 = run_n2_with_best_ever_assembly(
        objective, archive_theta, archive_loss, box_lower, box_upper, log,
    )
    n3 = AffineN3(trajectory, log).run(n2["theta"])
    success = bool(n3["loss"] < FINAL_GATE)
    return n2, n3, success, box_lower, box_upper


def run_case(name, corpus, log):
    old = json.loads((OLD / name / "result.json").read_text())
    case_id = int(old["case_id"])
    trajectory = corpus[f"prefix_{case_id}"].copy()
    teacher = corpus["teachers"][case_id].reshape(-1).copy()
    objective = AffineObjective(trajectory)
    runner = AffineArchiveN1(objective, case_id, log)
    tested = 0
    successful = None
    # The archive is tested at every true evaluation count, newest first.
    while successful is None:
        runner.advance_one()
        n = len(runner.losses)
        order = [n - 1] + list(range(n - 1))
        for archive_index in order:
            tested += 1
            theta = LOWER + (UPPER - LOWER) * runner.points[archive_index]
            n2, n3, success, box_lower, box_upper = branch(
                trajectory, objective, theta, runner.losses[archive_index], n, log,
            )
            if success:
                successful = (archive_index, theta, n2, n3, box_lower, box_upper)
                break
        if n >= 5000:
            raise RuntimeError("targeted affine N1 reached the existing resource cap")
    index, archive_theta, n2, n3, box_lower, box_upper = successful
    final = np.asarray(n3["theta"], np.float64)
    error = final - np.concatenate((teacher, np.zeros(2, np.float64)))
    rollout_result = rollout(trajectory, teacher, final)
    full_success = bool(n3["loss"] < FINAL_GATE and rollout_result["rollout_finite"])
    return {
        "case": name, "case_id": case_id, "prefix_length": int(len(trajectory)),
        "transition_count": int(len(trajectory) - 1),
        "ground_truth_K": teacher.tolist(), "ground_truth_b": [0.0, 0.0],
        "recovered_K": final[:4].tolist(), "recovered_b": final[4:].tolist(),
        "final_parameters": final.tolist(), "final_loss": float(n3["loss"]),
        "existing_recovery_status": full_success,
        "parameter_error": error.tolist(),
        "parameter_l2_error": float(np.linalg.norm(error)),
        "parameter_max_error": float(np.max(np.abs(error))),
        "original_four_parameter_solution": old["final_theta"],
        "recovered_K_minus_original_four_parameter_solution": (
            final[:4] - np.asarray(old["final_theta"], np.float64)
        ).tolist(),
        "N1": {
            "stopping_evaluation": len(runner.losses),
            "stopping_phase": runner.phase,
            "evaluated_sources": [event["source"] for event in runner.events],
            "archive_branches_tested": tested,
            "handoff_archive_index": index,
            "handoff_insertion_evaluation": index + 1,
            "handoff_is_newest": index == len(runner.losses) - 1,
            "handoff_is_incumbent": index == int(np.argmin(runner.losses)),
            "handoff_theta": archive_theta.tolist(),
            "handoff_loss": float(runner.losses[index]),
            "initial_design_multipliers": runner.initial_design_trace["multipliers"],
        },
        "N2": {
            "box_lower": box_lower.tolist(), "box_upper": box_upper.tolist(),
            "endpoint": np.asarray(n2["theta"]).tolist(),
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
        "rollout": rollout_result,
        "objective_invocations": objective.invocations,
    }


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    # Explicitly record the newly locked reusable rule outside frozen assets.
    rule = {
        "status": "LOCKED_POST_BASELINE_PHYSICAL_PEELBACK_RULE",
        "rule": "new free update-law coefficient coordinates use [-1.99,1.99]",
        "feature_scale_rule": (
            "normalize/rescale a materially differently scaled feature; do not "
            "automatically widen the global coefficient box"
        ),
        "current_feature": "phi(z)=1", "current_feature_scaling": "none",
        "parameter_order": list(AffineSHOUpdateAssembly.parameter_order),
        "current_domain_lower": LOWER.tolist(),
        "current_domain_upper": UPPER.tolist(),
    }
    atomic_json(OUT / "coefficient_bound_rule.json", rule)

    log = Log()
    design, design_trace = initial_design_with_remainder_assembly(6, return_trace=True)
    boundary_points, boundary_labels, boundary_trace = N1BoundarySupportAssembly()(
        6, 50, edirect.teacher.SEARCH_SEED, log=log,
    )
    composition_audit = {
        "initial_design_shape": list(design.shape),
        "initial_design_multipliers": design_trace["multipliers"],
        "initial_design_exact_fallback_calls": (
            design_trace["production_gcd_fallback_calls"]
            + design_trace["production_remainder_fallback_calls"]
        ),
        "boundary_support_shape": list(boundary_points.shape),
        "boundary_identity_shape": list(boundary_labels.shape),
        "boundary_k": boundary_trace["k"],
        "boundary_old_oracle_calls": boundary_trace["old_boundary_oracle_calls"],
        "boundary_forbidden_fallback_calls": boundary_trace["forbidden_fallback_calls"],
    }

    corpus = np.load(DATA)
    rows = []
    first_failure = None
    for name in CASES:
        row = run_case(name, corpus, log)
        rows.append(row)
        atomic_json(OUT / "cases" / f"{name}.json", row)
        print(name, row["existing_recovery_status"], row["final_loss"],
              row["N1"]["stopping_evaluation"], row["N2"]["evaluations"],
              row["N3"]["accepted_steps"], flush=True)
        if not row["existing_recovery_status"]:
            first_failure = {
                "case": name, "classification": "N3 behavior",
                "detail": "completed affine route did not meet existing recovery semantics",
            }
            break
    after = hashes()
    passed = first_failure is None and len(rows) == len(CASES)
    report = {
        "status": (
            "HOMOGENEITY_ASSUMPTION_REMOVED_TARGETED" if passed
            else "TARGETED_AFFINE_RECOVERY_FAILED_CLOSED"
        ),
        "scope": "only z_next=K z+b; all other physical assumptions supplied",
        "bound_rule": rule,
        "live_binding": {
            "order": [
                "locked Kz matmul", "locked coordinatewise bias addition",
                "locked residual", "locked square", "locked production-order MSE",
            ],
            "parameter_routing": {
                "theta[0:4]": "row-major K", "theta[4]": "output-0 bias",
                "theta[5]": "output-1 bias",
            },
            "bias_fused_into_dot": False, "bias_fixed_to_zero": False,
            "bias_initialized_specially_at_zero": False,
            "bias_regularizer": None, "ground_truth_available_to_search": False,
            "free_coordinates": 6,
        },
        "search_composition": composition_audit,
        "targeted_subset_source": (
            "experiments.validate_n3_accepted_step_cap.SHO_CASES"
        ),
        "cases": rows, "first_failure": first_failure,
        "all_targeted_pass": passed,
        "deferred_D_oracles_unchanged": [
            "cKDTree exact-tie ownership", "np.log10",
            "ordered top-k retrieval", "np.exp",
        ],
        "binding_invocations": dict(log.counts),
        "checkpoint_hashes_unchanged": before == after,
        "frozen_teacher_modified": False,
        "canonical_corpus_modified": False,
        "hybrid_student_baseline_modified": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(OUT / "targeted_recovery_report.json", report)
    (OUT / "STATUS.txt").write_text(report["status"] + "\n")
    print(json.dumps({
        "status": report["status"], "cases": len(rows),
        "checkpoint_hashes_unchanged": before == after,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
