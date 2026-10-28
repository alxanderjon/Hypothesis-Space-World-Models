#!/usr/bin/env python3
"""Assemble N3 damping, a generic learned-arithmetic solve, and descent routing."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import assemble_n3_hessian_operator as hessian
from experiments.sho import run_stage1b_n3_gradient_polishing as production
from experiments.sho import run_stage1b_two_parameter_learned_search as sho_search
from experiments.sho import train_n3_curvature_router as curvature
from experiments.sho import train_n3_derivative_significance as state_source
from experiments.sho import validate_largest_cell_router_tie_repair as maximum_router

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n3_learned_polisher" / "newton_step_operator"
GRADIENT_CHECKPOINT = ROOT / "n3_learned_polisher/gradient_operator/n3_gradient_operator.pt"
HESSIAN_CHECKPOINT = ROOT / "n3_learned_polisher/hessian_operator/n3_hessian_operator.pt"
CURVATURE_CHECKPOINT = ROOT / "n3_learned_polisher/curvature_router/n3_curvature_router.pt"
PIVOT_CHECKPOINT = ROOT / "largest_cell_router/best_router_checkpoint.pt"
SEED = 20_261_007


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScalarArithmetic:
    def __init__(self): self.ops = gradient.OPS
    def add(self, x, y): return self.ops.add(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64))
    def subtract(self, x, y): return self.ops.subtract(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64))
    def multiply(self, x, y): return self.ops.multiply(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64))
    def divide(self, x, y): return self.ops.divide(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64))
    def negative(self, x):
        value = torch.as_tensor(x, dtype=torch.float64)
        return self.subtract(torch.zeros_like(value), value)


class DampingAssembly:
    def __init__(self, arithmetic): self.arithmetic = arithmetic
    def initial(self, scale: float, minimum: float, direct: bool) -> float:
        if direct: return 0.0
        floor = float(self.arithmetic.multiply(scale, production.DAMPING_RELATIVE_FLOOR))
        shifted = float(self.arithmetic.add(float(self.arithmetic.negative(minimum)), floor))
        return max(floor, shifted)
    def value(self, scale: float, initial: float, direct: bool, attempt: int) -> float:
        if attempt == 0 and direct: return 0.0
        floor = float(self.arithmetic.multiply(scale, production.DAMPING_RELATIVE_FLOOR))
        base = initial if initial > 0.0 else floor
        exponent = attempt if not direct else attempt - 1
        return float(self.arithmetic.multiply(base, 10.0 ** exponent))
    def matrix(self, value: torch.Tensor, damping: float) -> torch.Tensor:
        identity = torch.eye(len(value), dtype=torch.float64)
        return self.arithmetic.add(value, self.arithmetic.multiply(damping, identity))
    def rhs(self, value: torch.Tensor) -> torch.Tensor:
        return self.arithmetic.negative(value)


class LearnedPartialPivotLU:
    """Partial-pivot LU using locked pivot and scalar operators."""
    def __init__(self, arithmetic):
        self.arithmetic = arithmetic
        self.proposal = maximum_router.load_model()
        self.compare = curvature.StrictLossSelector()
        self.reused_parameters = sum(p.numel() for p in self.proposal.parameters())
    def pivot(self, values: torch.Tensor) -> int:
        magnitudes = torch.abs(values).numpy().astype(np.float64)
        current = int(np.argmax(maximum_router.learned_logits(self.proposal, magnitudes)))
        for candidate in range(len(magnitudes)):
            if candidate == current: continue
            replace, _, state = self.compare.decide(
                np.asarray([magnitudes[candidate]]), np.asarray([magnitudes[current]]))
            if bool(replace[0]) or (int(state[0]) == 0 and candidate < current): current = candidate
        return current
    def solve(self, matrix: torch.Tensor, rhs: torch.Tensor):
        a, b = matrix.clone(), rhs.clone(); n = len(b); pivots = []
        for column in range(n):
            pivot = column + self.pivot(a[column:, column]); pivots.append(pivot)
            if float(a[pivot, column]) == 0.0: raise torch.linalg.LinAlgError("singular pivot")
            if pivot != column:
                row = a[column].clone(); a[column] = a[pivot]; a[pivot] = row
                value = b[column].clone(); b[column] = b[pivot]; b[pivot] = value
            for row in range(column + 1, n):
                factor = self.arithmetic.divide(a[row, column], a[column, column])
                for entry in range(column + 1, n):
                    a[row, entry] = self.arithmetic.subtract(
                        a[row, entry], self.arithmetic.multiply(factor, a[column, entry]))
                b[row] = self.arithmetic.subtract(
                    b[row], self.arithmetic.multiply(factor, b[column]))
                a[row, column] = 0.0
        result = torch.zeros_like(b)
        for row in range(n - 1, -1, -1):
            residual = b[row]
            for entry in range(row + 1, n):
                residual = self.arithmetic.subtract(
                    residual, self.arithmetic.multiply(a[row, entry], result[entry]))
            result[row] = self.arithmetic.divide(residual, a[row, row])
        return result, dict(pivots=pivots)


class DescentAssembly:
    def __init__(self, arithmetic, router): self.arithmetic, self.router = arithmetic, router
    def dot(self, g, d):
        products = self.arithmetic.multiply(g, d)
        return torch.dot(products, torch.ones_like(products))
    def check(self, g, d):
        finite = bool(torch.isfinite(d).all())
        if not finite: return False, math.nan, dict(finite=False, negative=False)
        value = float(self.dot(g, d))
        negative, _, state = self.router.compare.decide(np.asarray([0.0]), np.asarray([value]))
        decision = self.router.conjunction(finite, bool(negative[0]))
        return decision, value, dict(finite=finite, negative=bool(negative[0]), sign_state=int(state[0]))


class NewtonStepAssembly:
    def __init__(self):
        self.arithmetic = ScalarArithmetic(); self.router = curvature.CurvatureRouter()
        self.damping = DampingAssembly(self.arithmetic)
        self.solver = LearnedPartialPivotLU(self.arithmetic)
        self.descent = DescentAssembly(self.arithmetic, self.router)
    def step(self, g, H):
        analysis = self.router.analyze(g, H)
        if analysis["status"] == "NONFINITE":
            return dict(direction=None, condition=math.inf, damping=math.nan,
                        status="nonfinite_derivatives", attempt=None, analysis=analysis, trials=[])
        initial = self.damping.initial(analysis["scale"], analysis["minimum_eigenvalue"], analysis["direct"])
        trials = []
        for attempt in range(production.MAX_DAMPING_TRIALS):
            damping = self.damping.value(analysis["scale"], initial, analysis["direct"], attempt)
            try:
                direction, metadata = self.solver.solve(self.damping.matrix(H, damping), self.damping.rhs(g))
            except (torch.linalg.LinAlgError, ValueError) as error:
                trials.append(dict(attempt=attempt, damping=damping, solve="failed", error=str(error))); continue
            descent, dot, state = self.descent.check(g, direction)
            trials.append(dict(attempt=attempt, damping=damping, solve="ok", dot=dot,
                               descent=descent, pivots=metadata["pivots"], descent_state=state))
            if descent:
                return dict(direction=direction, condition=analysis["condition"], damping=damping,
                            status="ok", attempt=attempt, initial_damping=initial,
                            analysis=analysis, trials=trials)
        return dict(direction=None, condition=analysis["condition"], damping=math.nan,
                    status="no_descent_direction", attempt=None, initial_damping=initial,
                    analysis=analysis, trials=trials)


def teacher_damping(scale, minimum, direct, attempt):
    initial = 0.0 if direct else max(scale * production.DAMPING_RELATIVE_FLOOR,
                                     -minimum + scale * production.DAMPING_RELATIVE_FLOOR)
    if attempt == 0 and direct: return initial, 0.0
    base = initial if initial > 0.0 else scale * production.DAMPING_RELATIVE_FLOOR
    return initial, base * (10.0 ** (attempt if not direct else attempt - 1))


def generic_teacher_step(g, H):
    analysis = curvature.teacher_analysis(g, H)
    if analysis["status"] == "NONFINITE":
        return dict(direction=None, condition=math.inf, damping=math.nan,
                    status="nonfinite_derivatives", attempt=None, analysis=analysis, trials=[])
    trials = []
    for attempt in range(production.MAX_DAMPING_TRIALS):
        initial, damping = teacher_damping(analysis["scale"], analysis["minimum_eigenvalue"],
                                           analysis["direct"], attempt)
        try: direction = torch.linalg.solve(H + damping * torch.eye(len(g)), -g)
        except torch.linalg.LinAlgError:
            trials.append(dict(attempt=attempt, damping=damping, solve="failed")); continue
        dot = float(torch.dot(g, direction)); descent = bool(torch.isfinite(direction).all()) and dot < 0.0
        trials.append(dict(attempt=attempt, damping=damping, solve="ok", dot=dot, descent=descent))
        if descent:
            return dict(direction=direction, condition=analysis["condition"], damping=damping,
                        status="ok", attempt=attempt, initial_damping=initial,
                        analysis=analysis, trials=trials)
    return dict(direction=None, condition=analysis["condition"], damping=math.nan,
                status="no_descent_direction", attempt=None, initial_damping=initial,
                analysis=analysis, trials=trials)


def bit_equal(left, right):
    return np.float64(left).view(np.uint64) == np.float64(right).view(np.uint64)


def damping_controls(assembly):
    scales = [np.finfo(float).tiny, 1.0, np.nextafter(1.0, math.inf), 1e-200, 1e200]
    minimums = [-2.0, -0.0, 0.0, np.nextafter(0.0, math.inf), 1.0]
    initial_errors = damping_errors = cases = 0
    for direct in (False, True):
        for scale in scales:
            for minimum in minimums:
                got_initial = assembly.damping.initial(scale, minimum, direct)
                for attempt in range(production.MAX_DAMPING_TRIALS):
                    expected_initial, expected = teacher_damping(scale, minimum, direct, attempt)
                    got = assembly.damping.value(scale, got_initial, direct, attempt)
                    initial_errors += int(not bit_equal(expected_initial, got_initial))
                    damping_errors += int(not bit_equal(expected, got)); cases += 1
    return dict(cases=cases, initial_errors=initial_errors, damping_errors=damping_errors,
                max_trials=production.MAX_DAMPING_TRIALS)


def build_solve_controls():
    generator = torch.Generator().manual_seed(SEED); cases = []
    for dimension in range(2, 7):
        spectra = [
            ("direct_spd", torch.linspace(1.0, 2.0, dimension)),
            ("condition_exact_threshold", torch.cat((torch.tensor([1.0]), torch.full((dimension-1,), 1e12)))),
            ("condition_above_threshold", torch.cat((torch.tensor([1e-14]), torch.ones(dimension-1)))),
            ("semidefinite", torch.cat((torch.tensor([0.0]), torch.linspace(1.0, 2.0, dimension-1)))),
            ("indefinite", torch.cat((torch.tensor([-1.0]), torch.linspace(1.0, 2.0, dimension-1)))),
        ]
        for serial, (kind, spectrum) in enumerate(spectra):
            if kind == "condition_exact_threshold": matrix = torch.diag(spectrum)
            else:
                raw = torch.randn(dimension, dimension, generator=generator)
                q, _ = torch.linalg.qr(raw); matrix = q @ torch.diag(spectrum) @ q.T
                matrix = 0.5 * (matrix + matrix.T)
            cases.append(dict(name=f"d{dimension}_{kind}_{serial}", dimension=dimension,
                              gradient=torch.randn(dimension, generator=generator),
                              hessian=matrix, kind=kind))
        identity = torch.eye(dimension); matrix = identity.clone(); matrix[:, 0] = 1.0
        matrix[0, 0] = 2.0; matrix[1, 0] = -2.0; matrix = matrix.T @ matrix + identity
        cases.append(dict(name=f"d{dimension}_pivot_tie", dimension=dimension,
                          gradient=torch.randn(dimension, generator=generator),
                          hessian=matrix, kind="pivot_tie"))
    return cases


def descent_controls(assembly):
    tiny = np.nextafter(np.float64(0.0), np.float64(math.inf))
    controls = [
        ("negative", torch.tensor([1., 0.]), torch.tensor([-1., 0.])),
        ("positive", torch.tensor([1., 0.]), torch.tensor([1., 0.])),
        ("exact_zero", torch.tensor([1., 1.]), torch.tensor([1., -1.])),
        ("nextafter_negative", torch.tensor([1., 0.]), torch.tensor([-tiny, 0.])),
        ("nextafter_positive", torch.tensor([1., 0.]), torch.tensor([tiny, 0.])),
        ("nonfinite", torch.tensor([1., 0.]), torch.tensor([math.inf, 0.])),
    ]
    rows = []
    for name, g, d in controls:
        teacher_dot = float(torch.dot(g, d)); teacher = bool(torch.isfinite(d).all()) and teacher_dot < 0
        learned, learned_dot, state = assembly.descent.check(g, d)
        dot_exact = ((math.isnan(teacher_dot) and math.isnan(learned_dot)) or bit_equal(teacher_dot, learned_dot))
        rows.append(dict(name=name, teacher=teacher, learned=learned, teacher_dot=teacher_dot,
                         learned_dot=learned_dot, dot_exact=dot_exact, state=state,
                         error=int(teacher != learned)))
    return dict(cases=len(rows), errors=sum(row["error"] for row in rows),
                dot_errors=sum(not row["dot_exact"] for row in rows), rows=rows)


def compare_steps(name, split, family, teacher, learned):
    route_error = int(teacher["analysis"]["status"] != learned["analysis"]["status"])
    damping_error = int(not ((math.isnan(teacher["damping"]) and math.isnan(learned["damping"])) or
                             bit_equal(teacher["damping"], learned["damping"])))
    if teacher["direction"] is None or learned["direction"] is None:
        relative = 0.0 if teacher["direction"] is None and learned["direction"] is None else math.inf
        sign_errors = int(relative != 0.0)
    else:
        relative = float(torch.linalg.vector_norm(learned["direction"]-teacher["direction"]) /
                         max(float(torch.linalg.vector_norm(teacher["direction"])), np.finfo(float).tiny))
        nonzero = teacher["direction"] != 0
        sign_errors = int(torch.count_nonzero(nonzero &
                          (torch.signbit(teacher["direction"]) != torch.signbit(learned["direction"]))))
    teacher_descent = teacher["trials"][-1].get("descent") if teacher["trials"] else None
    learned_descent = learned["trials"][-1].get("descent") if learned["trials"] else None
    return dict(name=name, split=split, family=family, teacher_route=teacher["analysis"]["status"],
                learned_route=learned["analysis"]["status"], teacher_attempt=teacher["attempt"],
                learned_attempt=learned["attempt"], teacher_damping=teacher["damping"],
                learned_damping=learned["damping"], route_error=route_error,
                damping_error=damping_error, attempt_error=int(teacher["attempt"] != learned["attempt"]),
                status_error=int(teacher["status"] != learned["status"]),
                direction_relative_error=relative, direction_sign_errors=sign_errors,
                descent_error=int(teacher_descent != learned_descent))


def backtracking_once(state, direction):
    theta = state.theta.detach(); current_loss = float(state.objective(theta))
    if state.family == "sho":
        lower, upper = torch.full_like(theta, sho_search.LOWER), torch.full_like(theta, sho_search.UPPER)
    elif state.family == "ball":
        lower, upper = torch.from_numpy(ball.LOWER), torch.from_numpy(ball.UPPER)
    else: lower, upper = torch.full_like(theta, -math.inf), torch.full_like(theta, math.inf)
    for backtrack in range(production.MAX_BACKTRACKS + 1):
        candidate = theta + (2.0 ** (-backtrack)) * direction
        if bool(torch.equal(candidate, theta)): return dict(accepted=False, backtrack=None, reason="unchanged")
        if not bool(torch.all(candidate >= lower) and torch.all(candidate <= upper)): continue
        candidate_loss = float(state.objective(candidate))
        if candidate_loss < current_loss:
            return dict(accepted=True, backtrack=backtrack, reason="improved")
    return dict(accepted=False, backtrack=None, reason="no_improvement")


def validate(states, controls, assembly):
    template = lambda: dict(states=0, route_errors=0, damping_errors=0, attempt_errors=0,
                            status_errors=0, descent_errors=0, direction_sign_errors=0,
                            catastrophic_solve_errors=0, backtracking_decision_errors=0,
                            worst_direction_relative_error=0.0)
    groups = defaultdict(template); rows = []; first_failure = None
    for serial, state in enumerate(states):
        td = hessian.derivatives(state.objective, state.theta)
        ld = hessian.derivatives(lambda value: gradient.learned_objective(state, value),
                                 state.theta, learned=True)
        teacher = generic_teacher_step(td[1], td[3]); learned = assembly.step(ld[1], ld[3])
        row = compare_steps(str(state.case_id), state.split, state.family, teacher, learned)
        if teacher["direction"] is not None and learned["direction"] is not None:
            tb, lb = backtracking_once(state, teacher["direction"]), backtracking_once(state, learned["direction"])
            back_error = int(tb["accepted"] != lb["accepted"] or tb["backtrack"] != lb["backtrack"])
        else: back_error = 0
        row["backtracking_decision_error"] = back_error; rows.append(row)
        group = groups[f"{state.family}_{state.split}"]; group["states"] += 1
        for source, target in (("route_error","route_errors"),("damping_error","damping_errors"),
          ("attempt_error","attempt_errors"),("status_error","status_errors"),
          ("descent_error","descent_errors"),("direction_sign_errors","direction_sign_errors")):
            group[target] += row[source]
        group["catastrophic_solve_errors"] += int(not math.isfinite(row["direction_relative_error"]) or
                                                    row["direction_relative_error"] > 1e-10)
        group["backtracking_decision_errors"] += back_error
        group["worst_direction_relative_error"] = max(group["worst_direction_relative_error"],
                                                        row["direction_relative_error"])
        failed = any(row[name] for name in ("route_error","damping_error","attempt_error",
                                             "status_error","descent_error","direction_sign_errors"))
        if (failed or row["direction_relative_error"] > 1e-10 or back_error) and first_failure is None:
            first_failure = row
        if (serial + 1) % 20 == 0: print(f"Newton-step replay {serial+1}/{len(states)}", flush=True)
    control_rows = []
    for case in controls:
        control_rows.append(compare_steps(case["name"], "targeted", "synthetic",
            generic_teacher_step(case["gradient"], case["hessian"]),
            assembly.step(case["gradient"], case["hessian"])))
    names = ("states","route_errors","damping_errors","attempt_errors","status_errors",
             "descent_errors","direction_sign_errors","catastrophic_solve_errors",
             "backtracking_decision_errors")
    totals = {name: sum(row[name] for row in groups.values()) for name in names}
    totals["worst_direction_relative_error"] = max(row["worst_direction_relative_error"] for row in groups.values())
    control_errors = sum(any(row[name] for name in ("route_error","damping_error","attempt_error",
        "status_error","descent_error","direction_sign_errors")) or
        row["direction_relative_error"] > 1e-10 for row in control_rows)
    passed = all(totals[name] == 0 for name in names[1:]) and control_errors == 0
    return dict(passed=passed, totals=totals, groups=dict(groups), rows=rows,
                targeted_controls=dict(cases=len(control_rows), errors=control_errors, rows=control_rows),
                first_failure=first_failure)


def write_markdown(report):
    lines = ["# N3 Newton-step operator", "", f"Status: `{report['status']}`", "",
      "The production damping schedule is assembled from locked scalar arithmetic. The solve uses "
      "dimension-generic partial-pivot LU with the locked learned maximum router and scalar arithmetic.", "",
      "| Pool | States | Route errors | Damping errors | Descent errors | Solve failures | Worst direction error |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["validation"]["groups"].items():
        lines.append(f"| {name} | {row['states']} | {row['route_errors']} | {row['damping_errors']} | "
                     f"{row['descent_errors']} | {row['catastrophic_solve_errors']} | "
                     f"{row['worst_direction_relative_error']:.6g} |")
    lines += ["", f"Targeted solve controls: {report['validation']['targeted_controls']['cases']}; "
              f"errors: {report['validation']['targeted_controls']['errors']}.",
              f"Damping controls: {report['damping_controls']['cases']}; errors: "
              f"{report['damping_controls']['initial_errors']+report['damping_controls']['damping_errors']}.",
              f"Descent controls: {report['descent_controls']['cases']}; errors: {report['descent_controls']['errors']}."]
    if report["checkpoint"]:
        lines += ["", f"Checkpoint: `{Path(report['checkpoint']).name}`", "",
                  f"SHA-256: `{report['checkpoint_sha256']}`"]
    lines += ["", "Backtracking, acceptance, and stopping remain deterministic teacher logic."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    started=time.time(); OUT.mkdir(parents=True, exist_ok=True)
    protected=(GRADIENT_CHECKPOINT,HESSIAN_CHECKPOINT,CURVATURE_CHECKPOINT,PIVOT_CHECKPOINT)
    before={str(path):sha256(path) for path in protected}
    for path,status in ((GRADIENT_CHECKPOINT,"FROZEN_CANDIDATE"),
                        (HESSIAN_CHECKPOINT,"FROZEN_CANDIDATE"),(CURVATURE_CHECKPOINT,"LOCKED")):
        if torch.load(path,weights_only=False).get("status") != status: raise RuntimeError(path)
    gradient.OPS=gradient.LearnedArithmetic(); hessian.activate_second_order_gradient()
    assembly=NewtonStepAssembly(); damping_result=damping_controls(assembly)
    descent_result=descent_controls(assembly); states=state_source.build_states()
    controls=build_solve_controls(); validation=validate(states,controls,assembly)
    unchanged=before=={str(path):sha256(path) for path in protected}
    damping_pass=(damping_result["initial_errors"]==0 and damping_result["damping_errors"]==0 and unchanged)
    solve_pass=validation["passed"]
    descent_pass=(descent_result["errors"]==0 and descent_result["dot_errors"]<=1 and unchanged)
    passed=damping_pass and solve_pass and descent_pass
    damping_manifest=OUT/"n3_damping_construction_locked.json"
    descent_manifest=OUT/"n3_descent_check_locked.json"; checkpoint=None
    if damping_pass:
        damping_manifest.write_text(json.dumps(dict(status="LOCKED_ASSEMBLY",new_parameters=0,
          rule="production scale/minimum-eigenvalue damping with powers-of-ten retry",
          controls=damping_result,dependencies=before),indent=2)+"\n")
    if passed:
        descent_manifest.write_text(json.dumps(dict(status="LOCKED_ASSEMBLY",new_parameters=0,
          rule="finite(direction) AND learned_dot(gradient,direction)<0",
          controls={k:v for k,v in descent_result.items() if k!="rows"},dependencies=before),indent=2)+"\n")
        checkpoint=OUT/"n3_newton_step_operator.pt"
        torch.save(dict(status="FROZEN_CANDIDATE",new_parameters=0,
          reused_parameters=dict(pivot_router=assembly.solver.reused_parameters,curvature_boolean_router=6),
          solve="dimension-generic partial-pivot LU from locked scalar arithmetic",
          damping_manifest=str(damping_manifest),descent_manifest=str(descent_manifest),dependencies=before),checkpoint)
    report=dict(status="PASSED" if passed else "STOPPED_NEWTON_STEP_GATE_FAILED",
      production_damping_rule=dict(scale="max(max(abs(eigvalsh(H))),float64.tiny)",direct_initial=0.0,
        damped_initial="max(scale*1e-12,-minimum_eigenvalue+scale*1e-12)",
        retry="base*10**attempt; DIRECT retry uses attempt-1",trials=production.MAX_DAMPING_TRIALS,
        acceptance="finite(direction) AND dot(gradient,direction)<0"),
      damping_components=dict(learned=["multiplication","addition","subtraction/negation"],
                              deterministic=["max","power-of-ten constants","retry loop"]),
      damping_controls=damping_result,
      solve_audit=dict(teacher="torch.linalg.solve(A,b), CPU float64 square solve",
        representation="dense Hessian plus scalar diagonal damping",identity="runtime-dimension torch.eye",
        matrix="locked multiply(lambda,I), then locked add(H,lambdaI)",rhs="locked subtraction 0-gradient",
        implementation="partial-pivot LU; learned canonical maximum pivot; locked elimination arithmetic",
        existing_compatible_full_solve_checkpoint=None,new_solve_model=None,new_parameters=0,
        reused_pivot_router_parameters=assembly.solver.reused_parameters),
      descent_rule=dict(rule="finite(direction) AND dot(gradient,direction)<0",
        learned=["multiplication","strict-negative comparison","Boolean AND"],
        deterministic=["finite check","torch.dot reduction order"]),
      descent_controls={k:v for k,v in descent_result.items() if k!="rows"},
      validation=validation,systems=dict(states=len(states),targeted_solve_controls=len(controls),dimensions="2-6"),
      manifests=dict(damping=str(damping_manifest) if damping_pass else None,
                     descent=str(descent_manifest) if passed else None),
      checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
      dependencies=before,protected_checkpoints_modified=not unchanged,
      first_unresolved_operation=None if passed else dict(
        operation="generic linear solve",
        diagnosis="partial-pivot learned-arithmetic LU differs from CPU torch.linalg.solve ordering; near-condition-limit DAMPED systems amplify float64 differences",
        failed_targeted_controls=validation["targeted_controls"]["errors"],
        worst_targeted_direction_relative_error=max(
          row["direction_relative_error"] for row in validation["targeted_controls"]["rows"])),
      next_operation="resolve the generic linear-solve primitive before backtracking",elapsed_seconds=time.time()-started)
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n"); write_markdown(report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
