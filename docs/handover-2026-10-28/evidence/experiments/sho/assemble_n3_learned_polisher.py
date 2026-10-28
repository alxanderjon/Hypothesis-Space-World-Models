#!/usr/bin/env python3
"""Assemble and validate the remaining N3 controls with a temporary library solve."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# The saved broad-domain transfer used these settings before importing its runner.
os.environ.setdefault("BALL_TRANSFER_DOMAIN_LOWER", "0.5")
os.environ.setdefault("BALL_TRANSFER_DOMAIN_UPPER", "1.5")
os.environ.setdefault("BALL_TRANSFER_SEED", "20261001")
os.environ.setdefault("BALL_TRANSFER_TRAJECTORIES", "8")

import numpy as np
import torch

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import assemble_n3_hessian_operator as hessian
from experiments.sho import assemble_n3_newton_step_operator as prior_step
from experiments.sho import run_stage1b_n3_gradient_polishing as sho
from experiments.sho import train_n3_curvature_router as curvature
from experiments.sho.train_n3_derivative_significance import State

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/stage1B/n3_learned_polisher/control_and_closed_loop")
ROOT = Path("outputs/sho/stage1B/n3_learned_polisher")
SHO_REFERENCE = Path("outputs/sho/stage1B/n3_gradient_polishing/n3_per_trajectory.csv")
BALL_REFERENCE = Path("outputs/bouncing_ball/second_broad_n1_n2_n3_transfer/per_trajectory.csv")
DEPENDENCIES = (
    ROOT / "gradient_operator/n3_gradient_operator.pt",
    ROOT / "hessian_operator/n3_hessian_operator.pt",
    ROOT / "curvature_router/n3_curvature_router.pt",
    ROOT / "newton_step_operator/n3_damping_construction_locked.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, indent=2, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + "\n"


def bits(value: float) -> int:
    return int(np.float64(value).view(np.uint64))


def ordered_index(value: float) -> int:
    raw = bits(value)
    return (0xFFFF_FFFF_FFFF_FFFF - raw) if raw >> 63 else (raw + 0x8000_0000_0000_0000)


def ulp_distance(left: float, right: float) -> int:
    return abs(ordered_index(left) - ordered_index(right))


class ControlAssembly:
    """Learned primitive compositions; scheduling and state mutation stay structural."""

    def __init__(self) -> None:
        self.arithmetic = prior_step.ScalarArithmetic()
        self.curvature = curvature.CurvatureRouter()
        self.damping = prior_step.DampingAssembly(self.arithmetic)
        self.descent = prior_step.DescentAssembly(self.arithmetic, self.curvature)
        self.comparison = self.curvature.compare

    def strict_less(self, left: float, right: float) -> bool:
        # StrictLossSelector decides whether challenger=left replaces best=right.
        answer, _, _ = self.comparison.decide(
            np.asarray([right], dtype=np.float64), np.asarray([left], dtype=np.float64))
        return bool(answer[0])

    def at_most(self, value: float, limit: float) -> bool:
        return not self.strict_less(limit, value)

    def candidate(self, theta: torch.Tensor, direction: torch.Tensor, k: int) -> torch.Tensor:
        alpha = 2.0 ** (-k)
        return self.arithmetic.add(theta, self.arithmetic.multiply(alpha, direction))

    def in_bounds(self, candidate: torch.Tensor, lower: np.ndarray, upper: np.ndarray) -> bool:
        lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), (len(candidate),))
        upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), (len(candidate),))
        valid = True
        for value, low, high in zip(candidate.tolist(), lower.tolist(), upper.tolist()):
            below = self.strict_less(float(value), float(low))
            above = self.strict_less(float(high), float(value))
            valid = self.curvature.conjunction(valid, not (below or above))
        return valid

    def accepts(self, candidate_loss: float, current_loss: float) -> bool:
        return self.strict_less(candidate_loss, current_loss)

    def no_further_improvement(self, unchanged: bool, directional_derivative: float,
                               current_loss: float) -> tuple[bool, float, float]:
        predicted = max(0.0, float(self.arithmetic.subtract(0.0, directional_derivative)))
        spacing = abs(float(np.spacing(np.float64(current_loss))))
        threshold = float(self.arithmetic.multiply(32.0, spacing))
        too_small = self.at_most(predicted, threshold)
        return bool(unchanged or too_small), predicted, threshold

    def pre_step_route(self, accepted_steps: int, gradient_norm: float,
                       direction_status: str | None = None) -> str:
        if accepted_steps >= sho.MAX_ACCEPTED_STEPS:
            return "max_accepted_steps"
        if not math.isfinite(gradient_norm):
            return "nonfinite_gradient"
        if gradient_norm == 0.0:
            return "floating_point_zero_gradient"
        if direction_status is not None and direction_status != "ok":
            return direction_status
        return "CONTINUE"

    def exhausted_route(self, unchanged: bool, directional_derivative: float,
                        current_loss: float) -> str:
        no_further, _, _ = self.no_further_improvement(
            unchanged, directional_derivative, current_loss)
        return ("floating_point_no_further_improvement" if no_further
                else "backtracking_no_improvement")

    def newton_direction(self, g: torch.Tensor, H: torch.Tensor) -> dict:
        analysis = self.curvature.analyze(g, H)
        if analysis["status"] == "NONFINITE":
            return dict(direction=None, condition=math.inf, damping=math.nan,
                        status="nonfinite_derivatives", attempts=0)
        initial = self.damping.initial(float(analysis["scale"]),
                                       float(analysis["minimum_eigenvalue"]),
                                       bool(analysis["direct"]))
        identity = torch.eye(len(g), dtype=torch.float64)
        for attempt in range(sho.MAX_DAMPING_TRIALS):
            damping = self.damping.value(float(analysis["scale"]), initial,
                                         bool(analysis["direct"]), attempt)
            matrix = self.damping.matrix(H, damping)
            rhs = self.damping.rhs(g)
            try:
                # Explicitly temporary unresolved deterministic scaffold.
                direction = torch.linalg.solve(matrix, rhs)
            except torch.linalg.LinAlgError:
                continue
            descent, dot, _ = self.descent.check(g, direction)
            teacher = bool(torch.isfinite(direction).all()) and float(torch.dot(g, direction)) < 0.0
            if descent != teacher:
                raise AssertionError("learned descent routing diverged")
            if descent:
                return dict(direction=direction, condition=float(analysis["condition"]),
                            damping=damping, status="ok", attempts=attempt + 1,
                            directional_derivative=dot)
        return dict(direction=None, condition=float(analysis["condition"]), damping=math.nan,
                    status="no_descent_direction", attempts=sho.MAX_DAMPING_TRIALS)


@dataclass
class ReplayResult:
    theta: np.ndarray
    loss: float
    accepted_steps: int
    objective_evaluations: int
    backtracking_rejections: int
    accepted_with_backtracking: int
    damping_steps: int
    stop_reason: str
    routing_errors: dict[str, int]
    evaluated_thetas: list[np.ndarray]


def learned_derivatives(state: State, theta: torch.Tensor):
    return hessian.derivatives(lambda value: gradient.learned_objective(state, value),
                               theta, learned=True)


def polish(state: State, start: np.ndarray, lower: np.ndarray, upper: np.ndarray,
           controls: ControlAssembly) -> ReplayResult:
    theta = torch.as_tensor(start, dtype=torch.float64).clone()
    with torch.no_grad():
        current_loss = float(state.objective(theta))
    accepted_steps = objective_evaluations = backtracking_rejections = 0
    accepted_with_backtracking = damping_steps = 0
    routing = Counter()
    evaluated = [theta.numpy().copy()]
    stop_reason = "max_accepted_steps"
    while accepted_steps < sho.MAX_ACCEPTED_STEPS:
        _, g, _, H = learned_derivatives(state, theta)
        gradient_norm = float(torch.linalg.vector_norm(g))
        initial_route = controls.pre_step_route(accepted_steps, gradient_norm)
        if initial_route != "CONTINUE":
            stop_reason = initial_route; break
        step = controls.newton_direction(g, H)
        if step["direction"] is None:
            stop_reason = controls.pre_step_route(accepted_steps, gradient_norm, step["status"]); break
        direction = step["direction"]
        damping_steps += int(step["damping"] > 0.0 and math.isfinite(step["damping"]))
        teacher_dot = float(torch.dot(g, direction))
        routing["descent"] += int(not (torch.isfinite(direction).all() and teacher_dot < 0.0))
        accepted = unchanged = False
        for k in range(sho.MAX_BACKTRACKS + 1):
            candidate = controls.candidate(theta, direction, k)
            native_candidate = theta + (2.0 ** (-k)) * direction
            routing["candidate_bits"] += int(not torch.equal(candidate, native_candidate))
            if bool(torch.equal(candidate, theta)):
                unchanged = True; break
            learned_bounds = controls.in_bounds(candidate, lower, upper)
            teacher_bounds = bool(torch.all(candidate >= torch.from_numpy(lower)) and
                                  torch.all(candidate <= torch.from_numpy(upper)))
            routing["bounds"] += int(learned_bounds != teacher_bounds)
            if not learned_bounds:
                backtracking_rejections += 1; continue
            with torch.no_grad():
                candidate_loss = float(state.objective(candidate))
            objective_evaluations += 1
            evaluated.append(candidate.numpy().copy())
            learned_accept = controls.accepts(candidate_loss, current_loss)
            teacher_accept = candidate_loss < current_loss
            routing["strict_acceptance"] += int(learned_accept != teacher_accept)
            if learned_accept:
                theta = candidate.detach(); current_loss = candidate_loss
                accepted_steps += 1; accepted_with_backtracking += int(k > 0)
                accepted = True; break
            backtracking_rejections += 1
        if not accepted:
            learned_stop, _, _ = controls.no_further_improvement(unchanged, teacher_dot, current_loss)
            teacher_stop = unchanged or max(0.0, -teacher_dot) <= 32.0 * abs(float(np.spacing(np.float64(current_loss))))
            routing["stop_comparison"] += int(learned_stop != teacher_stop)
            stop_reason = controls.exhausted_route(unchanged, teacher_dot, current_loss)
            break
    return ReplayResult(theta.numpy().copy(), current_loss, accepted_steps,
                        objective_evaluations, backtracking_rejections,
                        accepted_with_backtracking, damping_steps, stop_reason,
                        dict(routing), evaluated)


def control_screens(controls: ControlAssembly) -> dict:
    tiny = np.nextafter(np.float64(0.0), np.float64(math.inf))
    descent_rows = []
    for name, g, d in (
        ("negative", [1., 0.], [-1., 0.]), ("zero", [1., 1.], [1., -1.]),
        ("positive", [1., 0.], [1., 0.]), ("nextafter_negative", [1., 0.], [-tiny, 0.]),
        ("nextafter_positive", [1., 0.], [tiny, 0.]),
        ("nonfinite_inf", [1., 0.], [math.inf, 0.]),
        ("nonfinite_nan", [1., 0.], [math.nan, 0.])):
        gt, dt = torch.tensor(g), torch.tensor(d)
        got, dot, _ = controls.descent.check(gt, dt)
        expected = bool(torch.isfinite(dt).all()) and float(torch.dot(gt, dt)) < 0.0
        descent_rows.append(dict(name=name, expected=expected, actual=got, error=int(expected != got)))
    acceptance_rows = []
    for scale in (0.0, 1.0, -1.0, 1e-300, 1e300):
        for candidate in (np.nextafter(np.float64(scale), -math.inf), np.float64(scale),
                          np.nextafter(np.float64(scale), math.inf)):
            expected = float(candidate) < scale
            got = controls.accepts(float(candidate), scale)
            acceptance_rows.append(dict(current=scale, candidate=float(candidate),
                                        expected=expected, actual=got, error=int(expected != got)))
    bound_rows = []
    for edge in (-1.0, 1.0):
        low, high = np.asarray([-1.0]), np.asarray([1.0])
        for value in (np.nextafter(edge, -math.inf), edge, np.nextafter(edge, math.inf)):
            expected = -1.0 <= value <= 1.0
            got = controls.in_bounds(torch.tensor([value]), low, high)
            bound_rows.append(dict(value=float(value), expected=expected, actual=got,
                                   error=int(expected != got)))
    stop_rows = []
    for loss in (0.0, 1e-300, 1e-28, 1.0, 1e300):
        threshold = 32.0 * abs(float(np.spacing(np.float64(loss))))
        for predicted in (np.nextafter(threshold, 0.0), threshold,
                          np.nextafter(threshold, math.inf)):
            derivative = -float(predicted)
            got, got_predicted, got_threshold = controls.no_further_improvement(False, derivative, loss)
            expected = max(0.0, -derivative) <= threshold
            stop_rows.append(dict(loss=loss, predicted=float(predicted), expected=expected,
                                  actual=got, predicted_exact=bits(got_predicted)==bits(max(0.0,-derivative)),
                                  threshold_exact=bits(got_threshold)==bits(threshold),
                                  error=int(expected != got)))
    state_controls = (
        ("continue", 0, 1.0, None, "CONTINUE"),
        ("nonfinite_nan", 0, math.nan, None, "nonfinite_gradient"),
        ("nonfinite_inf", 0, math.inf, None, "nonfinite_gradient"),
        ("exact_zero", 0, 0.0, None, "floating_point_zero_gradient"),
        ("nextafter_positive", 0, tiny, None, "CONTINUE"),
        ("no_descent", 0, 1.0, "no_descent_direction", "no_descent_direction"),
        ("nonfinite_derivatives", 0, 1.0, "nonfinite_derivatives", "nonfinite_derivatives"),
        ("accepted_cap", sho.MAX_ACCEPTED_STEPS, 1.0, None, "max_accepted_steps"),
    )
    state_rows = [dict(name=name, expected=expected,
                       actual=controls.pre_step_route(count, norm, direction),
                       error=int(controls.pre_step_route(count, norm, direction) != expected))
                  for name, count, norm, direction, expected in state_controls]
    post_controls = (
        ("unchanged", True, -1.0, 1.0, "floating_point_no_further_improvement"),
        ("exhausted_small", False, -(32.0*abs(float(np.spacing(1.0)))), 1.0,
         "floating_point_no_further_improvement"),
        ("exhausted_large", False, -1.0, 1.0, "backtracking_no_improvement"),
    )
    for name, unchanged, derivative, loss, expected in post_controls:
        actual = controls.exhausted_route(unchanged, derivative, loss)
        state_rows.append(dict(name=name, expected=expected, actual=actual,
                               error=int(actual != expected)))
    return dict(
        descent=dict(cases=len(descent_rows), errors=sum(x["error"] for x in descent_rows), rows=descent_rows),
        strict_acceptance=dict(cases=len(acceptance_rows), errors=sum(x["error"] for x in acceptance_rows), rows=acceptance_rows),
        bounds=dict(cases=len(bound_rows), errors=sum(x["error"] for x in bound_rows), rows=bound_rows),
        stop_32ulp=dict(cases=len(stop_rows), errors=sum(x["error"] for x in stop_rows),
                        arithmetic_errors=sum(not x["predicted_exact"] or not x["threshold_exact"] for x in stop_rows),
                        rows=stop_rows),
        stop_state=dict(cases=len(state_rows), errors=sum(x["error"] for x in state_rows), rows=state_rows),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def replay_sho(controls: ControlAssembly) -> tuple[dict, list[dict]]:
    references = {int(r["trajectory_id"]): r for r in read_csv(SHO_REFERENCE)
                  if int(r["requested_accepted_n3_steps"]) == 20}
    endpoints = sho._load_endpoints()
    positions, _, _ = sho._make_exact_labeled_split(sho.TRAJECTORY_COUNT, sho.FRESH_TEST_SEED)
    rows = []
    for i, endpoint in enumerate(endpoints):
        position = positions[i]
        state = State("sho", "full", str(i), torch.tensor(endpoint.theta),
                      lambda value, p=position: sho._scalar_loss(value, p), None)
        result = polish(state, endpoint.theta, np.asarray(sho.LOWER), np.asarray(sho.UPPER), controls)
        teacher = references[i]
        expected_theta = np.asarray([float(teacher["best_a"]), float(teacher["best_b"])])
        theta_ulps = [ulp_distance(float(a), float(b)) for a, b in zip(result.theta, expected_theta)]
        row = dict(trajectory_id=i, split="train" if i < 154 else "heldout" if i < 205 else "fresh",
                   previous_stagnation=int(endpoint.previous_plateau_case),
                   start_loss=float(state.objective(torch.tensor(endpoint.theta))),
                   final_loss=result.loss, teacher_final_loss=float(teacher["best_prefix_loss"]),
                   accepted_steps=result.accepted_steps,
                   teacher_accepted_steps=int(teacher["total_accepted_steps"]),
                   stop_reason=result.stop_reason, teacher_stop_reason=teacher["stop_reason"],
                   theta_ulp_max=max(theta_ulps), theta_ulps=theta_ulps,
                   routing_errors=sum(result.routing_errors.values()),
                   reached_1e_28=int(result.loss < 1e-28), improved=int(result.loss < float(state.objective(torch.tensor(endpoint.theta)))))
        rows.append(row)
        if (i + 1) % 32 == 0:
            print(f"SHO closed loop {i+1}/{len(endpoints)}", flush=True)
    groups = {}
    for split in ("train", "heldout", "fresh"):
        selected = [r for r in rows if r["split"] == split]
        groups[split] = dict(cases=len(selected), routing_errors=sum(r["routing_errors"] for r in selected),
                             exact_final_theta=sum(r["theta_ulp_max"] == 0 for r in selected),
                             stop_reason_errors=sum(r["stop_reason"] != r["teacher_stop_reason"] for r in selected))
    summary = dict(cases=len(rows), polished_successfully=sum(r["improved"] for r in rows),
                   routing_errors=sum(r["routing_errors"] for r in rows),
                   exact_final_theta=sum(r["theta_ulp_max"] == 0 for r in rows),
                   max_parameter_ulp_error=max(r["theta_ulp_max"] for r in rows),
                   stop_reason_errors=sum(r["stop_reason"] != r["teacher_stop_reason"] for r in rows),
                   accepted_step_count=sum(r["accepted_steps"] for r in rows),
                   teacher_accepted_step_count=sum(r["teacher_accepted_steps"] for r in rows),
                   reached_1e_28=sum(r["reached_1e_28"] for r in rows),
                   median_final_loss=float(np.median([r["final_loss"] for r in rows])),
                   worst_final_loss=max(r["final_loss"] for r in rows),
                   six_stagnation=[r for r in rows if r["previous_stagnation"]], groups=groups)
    return summary, rows


def replay_ball(controls: ControlAssembly) -> tuple[dict, list[dict]]:
    reference_rows = read_csv(BALL_REFERENCE)
    trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=8, n_timesteps=ball.TOTAL_STEPS, dt=ball.DT, seed=20_261_001)
    # Confirm the saved broad-run seed before accepting the reconstruction.
    if any(float(row["initial_x"]) != float(trajectory[0, 0])
           for row, trajectory in zip(reference_rows, trajectories)):
        trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
            n_trajectories=8, n_timesteps=ball.TOTAL_STEPS, dt=ball.DT, seed=ball.TRANSFER_SEED)
    rows = []
    for row, trajectory in zip(reference_rows, trajectories):
        i = int(row["trajectory_id"]); observed = torch.from_numpy(trajectory)
        start = np.asarray([float(row["n2_a"]), float(row["n2_b"])])
        state = State("ball", "fresh", str(i), torch.tensor(start),
                      lambda value, p=observed: ball.prefix_loss_torch(value, p), None)
        result = polish(state, start, np.asarray(ball.LOWER), np.asarray(ball.UPPER), controls)
        expected_theta = np.asarray([float(row["final_a"]), float(row["final_b"])])
        continuation_steps = ball.TOTAL_STEPS - (ball.PREFIX_STATES - 1)
        prediction = ball.numpy_rollout(trajectory[ball.PREFIX_STATES-1], result.theta, continuation_steps)
        truth = trajectory[ball.PREFIX_STATES-1:]
        error = prediction - truth
        topology = ball.topology_summary(result.evaluated_thetas, trajectory)
        rows.append(dict(trajectory_id=i, final_loss=result.loss,
                         teacher_final_loss=float(row["final_loss"]),
                         accepted_steps=result.accepted_steps,
                         teacher_accepted_steps=int(row["n3_steps"]),
                         stop_reason=result.stop_reason, teacher_stop_reason=row["n3_stop_reason"],
                         theta_ulp_max=max(ulp_distance(float(a), float(b)) for a,b in zip(result.theta,expected_theta)),
                         parameter_max_error=float(np.max(np.abs(result.theta - np.ones(2)))),
                         continuation_mse=float(np.mean(error*error)),
                         continuation_max_abs=float(np.max(np.abs(error))),
                         topology_changed_transitions=topology["topology_changed_transitions"],
                         routing_errors=sum(result.routing_errors.values())))
    summary = dict(cases=len(rows), polished_successfully=sum(r["final_loss"] < float(reference_rows[i]["n2_loss"]) for i,r in enumerate(rows)),
                   routing_errors=sum(r["routing_errors"] for r in rows),
                   exact_final_theta=sum(r["theta_ulp_max"] == 0 for r in rows),
                   max_parameter_ulp_error=max(r["theta_ulp_max"] for r in rows),
                   accepted_step_count=sum(r["accepted_steps"] for r in rows),
                   teacher_accepted_step_count=sum(r["teacher_accepted_steps"] for r in rows),
                   stop_reason_errors=sum(r["stop_reason"] != r["teacher_stop_reason"] for r in rows),
                   median_final_loss=float(np.median([r["final_loss"] for r in rows])),
                   worst_final_loss=max(r["final_loss"] for r in rows),
                   worst_parameter_max_error=max(r["parameter_max_error"] for r in rows),
                   worst_continuation_mse=max(r["continuation_mse"] for r in rows),
                   worst_continuation_max_abs=max(r["continuation_max_abs"] for r in rows),
                   topology_regressions=sum(r["topology_changed_transitions"] != 0 for r in rows))
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    before = {str(path): sha256(path) for path in DEPENDENCIES}
    gradient.OPS = gradient.LearnedArithmetic()
    hessian.activate_second_order_gradient()
    controls = ControlAssembly()
    screens = control_screens(controls)
    screen_pass = all(section["errors"] == 0 for section in screens.values()) and screens["stop_32ulp"]["arithmetic_errors"] == 0
    if not screen_pass:
        report = dict(status="STOPPED_CONTROL_SCREEN_FAILURE", screens=screens,
                      dependencies_unchanged=before == {str(path):sha256(path) for path in DEPENDENCIES})
        (OUT/"report.json").write_text(json_text(report)); print(json_text(report)); return

    component_files = {}
    for name, status, detail in (
        ("n3_descent_check", "LOCKED_ASSEMBLY", "finite(direction) AND dot(g,d)<0"),
        ("n3_backtracking_control", "VALIDATED_LEARNED_ASSEMBLY", "deterministic 2^-k schedule plus learned arithmetic/bounds/acceptance"),
        ("n3_strict_step_acceptance", "LOCKED_ASSEMBLY", "candidate_loss < current_loss; equality rejects"),
        ("n3_stop_continue", "LOCKED_ASSEMBLY", "production guard priority plus learned 32-ULP comparison")):
        path = OUT/f"{name}_locked.json"
        path.write_text(json_text(dict(status=status, parameters=0, detail=detail,
                                        reused_parameters=dict(strict_comparator=0, boolean_and=6),
                                        screens=screens)))
        component_files[name] = dict(path=str(path), sha256=sha256(path))

    sho_summary, sho_rows = replay_sho(controls)
    write_csv(OUT/"sho_closed_loop.csv", sho_rows)
    ball_summary, ball_rows = replay_ball(controls)
    write_csv(OUT/"ball_closed_loop.csv", ball_rows)
    after = {str(path): sha256(path) for path in DEPENDENCIES}
    passed = (sho_summary["cases"] == 256 and sho_summary["polished_successfully"] == 256 and
              sho_summary["routing_errors"] == 0 and sho_summary["stop_reason_errors"] == 0 and
              len(sho_summary["six_stagnation"]) == 6 and
              all(r["improved"] for r in sho_summary["six_stagnation"]) and
              ball_summary["cases"] == 8 and ball_summary["polished_successfully"] == 8 and
              ball_summary["routing_errors"] == 0 and ball_summary["stop_reason_errors"] == 0 and
              ball_summary["topology_regressions"] == 0 and before == after)
    checkpoint = None
    if passed:
        checkpoint = OUT/"n3_learned_polisher.pt"
        torch.save(dict(status="VALIDATED_LEARNED_ASSEMBLY", parameters=0,
                        components=component_files, dependencies=before,
                        temporary_remaining_scaffold=("torch.linalg.solve(H + lambda*I, -g)",
                            "structural indexing/loop bookkeeping", "physical/event branch scaffolding"),
                        sho=sho_summary, ball=ball_summary), checkpoint)
    report = dict(status="PASSED" if passed else "STOPPED_CLOSED_LOOP_GATE_FAILURE",
                  elapsed_seconds=time.time()-started, screens=screens,
                  components=component_files, sho=sho_summary, bouncing_ball=ball_summary,
                  checkpoint=None if checkpoint is None else dict(path=str(checkpoint),sha256=sha256(checkpoint)),
                  dependencies_before=before, dependencies_after=after, dependencies_unchanged=before==after,
                  remaining_deterministic_debt=["generic torch.linalg.solve",
                    "power-of-two/damping/backtracking loops and tensor indexing",
                    "declared physical collision/event branch routing"],
                  claim="N3 control peelback validated; full peelback is not complete")
    (OUT/"report.json").write_text(json_text(report))
    print(json_text(report))


if __name__ == "__main__":
    main()
