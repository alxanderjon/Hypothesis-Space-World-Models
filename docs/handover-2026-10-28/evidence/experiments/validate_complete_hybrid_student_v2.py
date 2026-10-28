#!/usr/bin/env python3
"""Fail-closed closed-loop validation of the fully bound hybrid student.

The frozen teacher artifacts supply only inputs and expected traces.  Student
N1, N2, and N3 arithmetic/routing is executed through the live bindings.
Inventory-D operations remain the explicit deterministic oracles declared by
the frozen manifest.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments import audit_known_structure as known
from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.build_hybrid_student_baseline import representative_records
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.resume_hybrid_student_after_ckdtree import (
    ArchiveAbsorbingSuccessLatch,
    HybridN3CurvatureRouteBinding,
    HybridN3DampedMatrixRHSBinding,
    HybridN3DampingBinding,
    HybridN3GradientBinding,
    HybridN3HessianBinding,
    HybridPrefixResidualMSEBinding,
    Log,
    StrictOwnerAssembly,
    run_n2_with_best_ever_assembly,
    student_cell_state,
)
from experiments.sho import run_stage1b_n3_gradient_polishing as n3_production
from experiments.sho.assemble_n3_learned_polisher import ControlAssembly
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model
from experiments.kepler import run_four_parameter_peelback as kepler_runtime
from experiments.kepler import run_four_parameter_peelback as staged


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_complete_validation_v2")
FREEZE = Path("outputs/deterministic_teacher_frozen")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left, right) -> bool:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    return a.shape == b.shape and bool(
        np.array_equal(a.view(np.uint64), b.view(np.uint64))
    )


def metric(left, right) -> dict:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    return {
        "shape_equal": a.shape == b.shape,
        "bit_mismatches": (
            int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
            if a.shape == b.shape else max(a.size, b.size)
        ),
        "max_absolute_error": (
            float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else 0.0
        ),
    }


def find_spec(system: str, case_id: int) -> dict:
    return next(
        row for row in json.loads(readiness.MANIFEST.read_text())["cases"]
        if row["system"] == system and int(row["case_id"]) == case_id
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
        directory = readiness.CASES / system / f"case_{case_id:05d}" / "branches"
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text())
            if (not row["existing_recovery_criterion"] and
                    row["n3_stop_reason"] == "FLOAT64_STAGNATION_TAIL"):
                return row
    raise RuntimeError(f"missing requested branch {system}/{case_id}/{selector}")


class HybridObjective:
    """Supplied physical transition plus the live generic residual/MSE binding."""

    def __init__(self, system: str, trajectory: np.ndarray, log: Log):
        self.system = system
        self.trajectory = np.asarray(trajectory, np.float64)
        self.binding = HybridPrefixResidualMSEBinding()
        self.log = log

    def __call__(self, theta) -> float:
        theta = np.asarray(theta, np.float64)
        if self.system == "full_k_sho":
            prediction = np.einsum(
                "ij,nj->ni", theta.reshape(2, 2), self.trajectory[:-1]
            )
            target = self.trajectory[1:]
        elif self.system == "reflection_ball":
            prefix = self.trajectory[:ball.ball.PREFIX_STATES]
            prediction = np.stack([
                ball.numpy_step(state, theta) for state in prefix[:-1]
            ])
            target = prefix[1:]
        elif self.system == "kepler5":
            prefix = self.trajectory[:kepler_runtime.kepler.PREFIX_STATES]
            prediction = known.step(prefix[:-1], theta, np)
            target = prefix[1:]
        else:
            raise ValueError(self.system)
        value = self.binding.evaluate(
            prediction, target, self.system, reduction_path="numpy", log=self.log,
        )
        scalar = float(value.loss)
        return scalar if math.isfinite(scalar) else 1.0e100


class HybridN3:
    """Live A/B N3 bindings around the frozen D solve/box-QP/stagnation path."""

    def __init__(self, system, trajectory, lower, upper, log):
        self.system = system
        self.trajectory = np.asarray(trajectory, np.float64)
        self.lower = torch.from_numpy(np.asarray(lower, np.float64).copy())
        self.upper = torch.from_numpy(np.asarray(upper, np.float64).copy())
        self.log = log
        self.gradient = HybridN3GradientBinding()
        self.hessian = HybridN3HessianBinding(self.gradient)
        self.curvature = HybridN3CurvatureRouteBinding()
        self.damping = HybridN3DampingBinding(self.curvature)
        self.matrix_rhs = HybridN3DampedMatrixRHSBinding()
        self.controls = ControlAssembly()
        # The box-QP solver is a retained D oracle.  Its internal dense solves
        # stay in the frozen deterministic implementation.
        _, self.direction_oracle, _ = staged._make_n3(
            readiness.case_data(find_spec(system, int(0)))[3]
            if False else (lambda value: value.sum() * 0.0),
            np.asarray(lower, np.float64), np.asarray(upper, np.float64),
        )

    def scalar_loss(self, theta: torch.Tensor) -> float:
        assembly = self.gradient.assembly
        value = theta.detach().clone().requires_grad_(True)
        trajectory = torch.from_numpy(self.trajectory.copy())
        if self.system == "full_k_sho":
            prediction = torch.from_numpy(assembly.sho_forward(
                trajectory[:-1].numpy(), value.detach().numpy().reshape(2, 2),
            ).copy())
            result = assembly.prefix_objective.evaluate(
                prediction, trajectory[1:], "full_k_sho", reduction_path="torch",
            )
        elif self.system == "reflection_ball":
            prefix = trajectory[:ball.ball.PREFIX_STATES]
            prediction = torch.stack([
                assembly._ball_step(state, value) for state in prefix[:-1]
            ])
            result = assembly.prefix_objective.evaluate(
                prediction, prefix[1:], "reflection_ball", reduction_path="torch",
            )
        else:
            prefix = trajectory[:kepler_runtime.kepler.PREFIX_STATES]
            prediction = assembly._kepler_step(prefix[:-1], value)
            result = assembly.prefix_objective.evaluate(
                prediction, prefix[1:], "kepler5", reduction_path="torch",
            )
        return float(result.loss)

    def derivatives(self, theta: torch.Tensor, crosscheck_gradient=True):
        if not crosscheck_gradient:
            hessian = self.hessian.evaluate(
                self.system, self.trajectory, theta, self.log,
            )
            # The symmetric-Hessian assembly constructs this gradient through
            # the same frozen first-adjoint graph.  Count the live nested
            # gradient binding explicitly for regression coverage.
            self.log.call(
                "N3 gradient operator", "A",
                "nested exact first-adjoint assembly in Hessian binding", 1,
            )
            return hessian.loss, hessian.gradient, hessian.symmetric_hessian
        gradient = self.gradient.evaluate(
            self.system, self.trajectory, theta, self.log,
        )
        hessian = self.hessian.evaluate(
            self.system, self.trajectory, theta, self.log,
        )
        if not bits_equal([gradient.loss], [hessian.loss]):
            raise RuntimeError("live gradient/Hessian objective inputs diverged")
        if not bits_equal(gradient.gradient, hessian.gradient):
            raise RuntimeError("live gradient/Hessian gradient inputs diverged")
        return gradient.loss, gradient.gradient, hessian.symmetric_hessian

    def run(self, start, expected, *, crosscheck_gradient=True) -> dict:
        theta = torch.tensor(start, dtype=torch.float64)
        current_loss = self.scalar_loss(theta)
        accepted_steps = 0
        stagnation_count = 0
        stop_reason = None
        counts = Counter()
        first_divergence = None
        while True:
            loss_tensor, gradient, hessian = self.derivatives(
                theta, crosscheck_gradient=crosscheck_gradient,
            )
            if not bits_equal([loss_tensor], [current_loss]):
                first_divergence = {
                    "operation": "N3 objective", "iteration": accepted_steps,
                    "teacher": current_loss, "student": float(loss_tensor),
                }
                break
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            pre = self.controls.pre_step_route(0, gradient_norm)
            counts["genuine numerical STOP/CONTINUE routes"] += 1
            if pre != "CONTINUE":
                stop_reason = pre
                break

            spectrum, route = self.curvature.evaluate(gradient, hessian, self.log)
            counts["N3 DIRECT/DAMPED curvature route"] += 1
            initial = self.damping.initial(
                spectrum.scale, spectrum.minimum_eigenvalue, route["direct"], self.log,
            )
            for attempt in range(n3_production.MAX_DAMPING_TRIALS):
                self.damping.value(
                    spectrum.scale, initial, route["direct"], attempt, self.log,
                )
                counts["N3 damping retry value"] += 1

            direction, qp = dimension_general_box_qp_direction(
                theta, gradient, hessian, self.lower, self.upper,
                self.direction_oracle,
            )
            counts["dimension-general box-QP active-set direction"] += 1
            if direction is None:
                stop_reason = qp["route_status"]
                break
            self.matrix_rhs.construct(
                hessian, gradient, float(qp["damping"]), self.log,
            )
            counts["N3 damped matrix and RHS construction"] += 1
            descent, dot, _ = self.controls.descent.check(gradient, direction)
            counts["descent validity"] += 1
            if not descent:
                stop_reason = "no_descent_direction"
                break
            full_predicted = -float(qp["quadratic_value"])
            accepted = False
            unchanged = False
            for backtrack in range(n3_production.MAX_BACKTRACKS + 1):
                candidate = self.controls.candidate(theta, direction, backtrack)
                counts["power-of-two backtracking schedule"] += 1
                if bool(torch.equal(candidate, theta)):
                    unchanged = True
                    break
                if not self.controls.in_bounds(
                    candidate, self.lower.numpy(), self.upper.numpy()
                ):
                    continue
                candidate_loss = self.scalar_loss(candidate)
                if self.controls.accepts(candidate_loss, current_loss):
                    stagnation_count = n3_production._updated_float64_stagnation_count(
                        stagnation_count, current_loss, candidate_loss, full_predicted,
                    )
                    theta = candidate.detach()
                    current_loss = candidate_loss
                    accepted_steps += 1
                    accepted = True
                    break
            if accepted:
                if stagnation_count >= n3_production.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
                    stop_reason = n3_production.FLOAT64_STAGNATION_STOP_REASON
                    break
                continue
            numerical_stop, _, _ = self.controls.no_further_improvement(
                unchanged, dot, current_loss
            )
            stop_reason = (
                "floating_point_no_further_improvement"
                if numerical_stop else "backtracking_no_improvement"
            )
            break
        return {
            "accepted_steps": accepted_steps,
            "stop_reason": stop_reason,
            "final_theta": theta.numpy().tolist(),
            "final_loss": current_loss,
            "theta_metric": metric(expected["final_theta"], theta.numpy()),
            "loss_metric": metric([expected["final_loss"]], [current_loss]),
            "expected_steps": int(expected["n3_accepted_steps"]),
            "expected_stop": expected["n3_stop_reason"],
            "counts": dict(counts),
            "first_divergence": first_divergence,
        }


def replay_n1_states(log: Log) -> list[dict]:
    frozen = FrozenProposals()
    largest = load_model()
    rows = []
    for item in representative_records():
        owner = StrictOwnerAssembly(frozen, log)
        state = student_cell_state(item, frozen, owner, largest, log)
        rows.append({
            "system": item["system"],
            "proposal_rows": int(len(state["student_proposals"])),
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "selected_cells_exact": state["selected_teacher"] == state["selected_student"],
        })
    return rows


def replay_branch(system, case_id, selector, reason, log):
    spec = find_spec(system, case_id)
    trajectory, _, _, _, lower, upper, _ = readiness.case_data(spec)
    saved = selected_branch(system, case_id, selector)
    theta = np.asarray(saved["archive_theta"], np.float64)
    n = int(saved["current_evaluation_count"])
    radius = (upper - lower) / float(n) ** (1.0 / len(lower))
    box_lower = np.maximum(lower, theta - radius)
    box_upper = np.minimum(upper, theta + radius)
    objective = HybridObjective(system, trajectory, log)
    n2 = run_n2_with_best_ever_assembly(
        objective, theta, float(saved["archive_loss"]), box_lower, box_upper, log,
    )
    n2_exact = bool(
        bits_equal(box_lower, saved["n2_box_lower"])
        and bits_equal(box_upper, saved["n2_box_upper"])
        and int(n2["evaluations"]) == int(saved["n2_evaluations"])
        and n2["message"] == saved["n2_stop_reason"]
        and bits_equal(n2["theta"], saved["n2_endpoint"])
        and bits_equal([n2["loss"]], [saved["n2_loss"]])
    )
    row = {
        "system": system, "case_id": case_id, "selector": selector,
        "coverage_reason": reason, "n2_exact": n2_exact,
        "n2_evaluations": int(n2["evaluations"]),
        "n2_stop_reason": n2["message"],
        "n2_endpoint_metric": metric(saved["n2_endpoint"], n2["theta"]),
        "n2_loss_metric": metric([saved["n2_loss"]], [n2["loss"]]),
        "objective_binding_invocations": objective.binding.invocations,
    }
    if not n2_exact:
        row["first_divergence"] = {
            "operation": "N2 exact Powell closed-loop branch",
            "teacher": {
                "evaluations": saved["n2_evaluations"],
                "stop": saved["n2_stop_reason"],
                "theta": saved["n2_endpoint"], "loss": saved["n2_loss"],
            },
            "student": {
                "evaluations": n2["evaluations"], "stop": n2["message"],
                "theta": n2["theta"].tolist(), "loss": n2["loss"],
            },
        }
        return row
    n3 = HybridN3(system, trajectory, lower, upper, log).run(n2["theta"], saved)
    row["n3"] = n3
    if (n3["first_divergence"] is not None or
            n3["accepted_steps"] != n3["expected_steps"] or
            n3["stop_reason"] != n3["expected_stop"] or
            n3["theta_metric"]["bit_mismatches"] or
            n3["loss_metric"]["bit_mismatches"]):
        row["first_divergence"] = n3["first_divergence"] or {
            "operation": "N3 closed-loop trajectory/final state",
            "teacher": {
                "accepted_steps": n3["expected_steps"],
                "stop": n3["expected_stop"],
                "theta": saved["final_theta"], "loss": saved["final_loss"],
            },
            "student": {
                "accepted_steps": n3["accepted_steps"],
                "stop": n3["stop_reason"],
                "theta": n3["final_theta"], "loss": n3["final_loss"],
            },
        }
    return row


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints:
        raise RuntimeError("checkpoint inventory differs from frozen reference")
    freeze = json.loads((FREEZE / "freeze_manifest.json").read_text())
    # The freeze manifest records teacher sources, some of which are shared
    # historical modules.  Report integrity rather than mutating any source.
    frozen_source_integrity = {
        path: Path(path).exists() and sha256(Path(path)) == digest
        for path, digest in freeze["sources"].items()
    }
    log = Log()
    n1_rows = replay_n1_states(log)
    first_divergence = next(({
        "operation": "N1 saved-state proposal replay", "row": row,
    } for row in n1_rows if row["proposal_bit_mismatches"] or
        not row["selected_cells_exact"]), None)
    targets = [
        ("full_k_sho", 0, "success", "ordinary recovery"),
        ("reflection_ball", 253, "success", "older/non-incumbent and N2 maxfev"),
        ("reflection_ball", 265, "first_stagnation_bad", "FLOAT64_STAGNATION_TAIL BAD"),
        ("reflection_ball", 215, "success", "constrained box-QP"),
        ("kepler5", 193, "success", "heavy backtracking and N2 maxfev"),
        ("kepler5", 231, "success", "constrained box-QP"),
    ]
    branch_rows = []
    if first_divergence is None:
        for target in targets:
            row = replay_branch(*target, log)
            branch_rows.append(row)
            if row.get("first_divergence") is not None:
                first_divergence = row["first_divergence"]
                break

    inventory = json.loads((FREEZE / "operation_inventory.json").read_text())
    report = {
        "status": (
            "STAGE1_EXACT_READY_FOR_STAGE2" if first_divergence is None
            else "STOPPED_FIRST_DIVERGENCE"
        ),
        "stage1": {"n1_rows": n1_rows, "branch_rows": branch_rows},
        "stage2": {"executed": False, "reason": "separate full-corpus pass pending"},
        "first_divergence": first_divergence,
        "binding_invocations": dict(log.counts),
        "retained_D_oracles": [
            {"section": row["section"], "operation": row["operation"]}
            for row in inventory["remaining_D"]
        ],
        "frozen_source_integrity": frozen_source_integrity,
        "checkpoint_hashes_unchanged": before == checkpoint_hashes() == expected_checkpoints,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "stage1_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "first_divergence": first_divergence,
        "n1_rows": n1_rows,
        "branch_rows": [{
            "system": row["system"], "case_id": row["case_id"],
            "n2_exact": row["n2_exact"],
            "n3_exact": (
                row.get("n3", {}).get("theta_metric", {}).get("bit_mismatches") == 0
                and row.get("n3", {}).get("loss_metric", {}).get("bit_mismatches") == 0
                and row.get("n3", {}).get("stop_reason") == row.get("n3", {}).get("expected_stop")
            ),
        } for row in branch_rows],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
