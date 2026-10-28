#!/usr/bin/env python3
"""Peel back only N3 curvature classification and direct/damped routing."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import assemble_n3_hessian_operator as hessian
from experiments.sho import run_stage1b_n3_gradient_polishing as production
from experiments.sho import train_n2_direction_controller as n2_router
from experiments.sho import train_n3_derivative_significance as state_source
from experiments.sho.validate_representative_tie_selector import StrictLossSelector


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n3_learned_polisher" / "curvature_router"
GRADIENT_CHECKPOINT = ROOT / "n3_learned_polisher/gradient_operator/n3_gradient_operator.pt"
HESSIAN_CHECKPOINT = ROOT / "n3_learned_polisher/hessian_operator/n3_hessian_operator.pt"
AND_CHECKPOINT = ROOT / "n2_direction_controller/local_direction_routers.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CurvatureRouter:
    """Finite guard plus learned conjunction of exact categorical comparisons."""

    def __init__(self) -> None:
        saved = torch.load(AND_CHECKPOINT, map_location="cpu", weights_only=False)
        if saved.get("status") != "LOCAL_OPERATORS_LOCKED":
            raise RuntimeError("locked Boolean router is unavailable")
        self.and_router = n2_router.TinyRouter(2, 2)
        self.and_router.load_state_dict(saved["state_dicts"]["boolean_and"])
        self.and_router.eval().requires_grad_(False)
        self.compare = StrictLossSelector()

    def conjunction(self, left: bool, right: bool) -> bool:
        values = torch.tensor([[float(left), float(right)]], dtype=torch.float64)
        with torch.no_grad():
            return bool(self.and_router(values).argmax(1)[0])

    def positive(self, value: float) -> bool:
        # 0 < value, expressed through the locked strict comparison assembly.
        answer, _, _ = self.compare.decide(
            np.asarray([value], dtype=np.float64), np.asarray([0.0], dtype=np.float64))
        return bool(answer[0])

    def at_most(self, value: float, limit: float) -> bool:
        # Equality must pass: reject only when limit < value.
        above, _, _ = self.compare.decide(
            np.asarray([value], dtype=np.float64), np.asarray([limit], dtype=np.float64))
        return not bool(above[0])

    def analyze(self, gradient_value: torch.Tensor,
                hessian_value: torch.Tensor) -> dict:
        finite = bool(torch.isfinite(gradient_value).all() and
                      torch.isfinite(hessian_value).all())
        if not finite:
            return dict(status="NONFINITE", finite=False, direct=False,
                        singular_values=None, smallest_singular=None,
                        largest_singular=None, condition=math.inf,
                        eigenvalues=None, minimum_eigenvalue=None, scale=None,
                        positive=False, condition_ok=False)
        singular_values = torch.linalg.svdvals(hessian_value)
        smallest = float(torch.min(singular_values))
        largest = float(torch.max(singular_values))
        condition = largest / smallest if smallest > 0.0 else math.inf
        eigenvalues = torch.linalg.eigvalsh(hessian_value)
        minimum = float(torch.min(eigenvalues))
        scale = max(float(torch.max(torch.abs(eigenvalues))),
                    torch.finfo(torch.float64).tiny)
        positive = self.positive(minimum)
        condition_ok = self.at_most(condition, production.CONDITION_LIMIT)
        direct = self.conjunction(positive, condition_ok)
        return dict(
            status="DIRECT" if direct else "DAMPED", finite=True, direct=direct,
            singular_values=singular_values.detach().tolist(),
            smallest_singular=smallest, largest_singular=largest,
            condition=condition, eigenvalues=eigenvalues.detach().tolist(),
            minimum_eigenvalue=minimum, scale=scale,
            positive=positive, condition_ok=condition_ok)


def teacher_analysis(gradient_value: torch.Tensor,
                     hessian_value: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(gradient_value).all() and
                  torch.isfinite(hessian_value).all())
    if not finite:
        return dict(status="NONFINITE", finite=False, direct=False,
                    condition=math.inf, minimum_eigenvalue=None, scale=None,
                    positive=False, condition_ok=False)
    singular_values = torch.linalg.svdvals(hessian_value)
    smallest = float(torch.min(singular_values))
    largest = float(torch.max(singular_values))
    condition = largest / smallest if smallest > 0.0 else math.inf
    eigenvalues = torch.linalg.eigvalsh(hessian_value)
    minimum = float(torch.min(eigenvalues))
    scale = max(float(torch.max(torch.abs(eigenvalues))),
                torch.finfo(torch.float64).tiny)
    positive = minimum > 0.0
    condition_ok = condition <= production.CONDITION_LIMIT
    direct = positive and condition_ok
    return dict(status="DIRECT" if direct else "DAMPED", finite=True,
                direct=direct, condition=condition, minimum_eigenvalue=minimum,
                scale=scale, positive=positive, condition_ok=condition_ok)


def direction_with_route(gradient_value: torch.Tensor, hessian_value: torch.Tensor,
                         analysis: dict) -> tuple[torch.Tensor | None, float, float, str]:
    """Unchanged production solve/damping schedule with route supplied by router."""
    if analysis["status"] == "NONFINITE":
        return None, math.inf, math.nan, "nonfinite_derivatives"
    condition = float(analysis["condition"])
    minimum = float(analysis["minimum_eigenvalue"])
    scale = float(analysis["scale"])
    direct_allowed = bool(analysis["direct"])
    initial_damping = 0.0
    if not direct_allowed:
        initial_damping = max(
            scale * production.DAMPING_RELATIVE_FLOOR,
            -minimum + scale * production.DAMPING_RELATIVE_FLOOR,
        )
    identity = torch.eye(len(gradient_value), dtype=torch.float64)
    for attempt in range(production.MAX_DAMPING_TRIALS):
        if attempt == 0 and direct_allowed:
            damping = 0.0
        else:
            base = (initial_damping if initial_damping > 0.0 else
                    scale * production.DAMPING_RELATIVE_FLOOR)
            damping = base * (10.0 ** (attempt if not direct_allowed else attempt - 1))
        try:
            direction = torch.linalg.solve(hessian_value + damping * identity,
                                           -gradient_value)
        except torch.linalg.LinAlgError:
            continue
        if (bool(torch.isfinite(direction).all()) and
                float(torch.dot(gradient_value, direction)) < 0.0):
            return direction, condition, damping, "ok"
    return None, condition, math.nan, "no_descent_direction"


def boundary_controls(router: CurvatureRouter) -> list[dict]:
    tiny = np.nextafter(np.float64(0.0), np.float64(math.inf))
    limit = np.float64(production.CONDITION_LIMIT)
    below = np.nextafter(limit, np.float64(0.0))
    above = np.nextafter(limit, np.float64(math.inf))
    controls = [
        ("positive_well_conditioned", torch.tensor([1.0, -1.0]), torch.diag(torch.tensor([1.0, 2.0]))),
        ("minimum_eigenvalue_negative", torch.ones(2), torch.diag(torch.tensor([-1.0, 1.0]))),
        ("minimum_eigenvalue_zero", torch.ones(2), torch.diag(torch.tensor([0.0, 1.0]))),
        ("minimum_eigenvalue_nextafter_above_zero", torch.ones(2), torch.diag(torch.tensor([tiny, tiny]))),
        ("condition_nextafter_below", torch.ones(2), torch.diag(torch.tensor([1.0, below]))),
        ("condition_exact_threshold", torch.ones(2), torch.diag(torch.tensor([1.0, limit]))),
        ("condition_nextafter_above", torch.ones(2), torch.diag(torch.tensor([1.0, above]))),
        ("nonfinite_gradient_nan", torch.tensor([math.nan, 1.0]), torch.eye(2)),
        ("nonfinite_hessian_inf", torch.ones(2), torch.tensor([[math.inf, 0.0], [0.0, 1.0]])),
        ("nonfinite_hessian_nan", torch.ones(2), torch.tensor([[math.nan, 0.0], [0.0, 1.0]])),
    ]
    result = []
    for name, gradient_value, hessian_value in controls:
        teacher = teacher_analysis(gradient_value, hessian_value)
        learned = router.analyze(gradient_value, hessian_value)
        result.append(dict(name=name, teacher_status=teacher["status"],
                           learned_status=learned["status"],
                           teacher_positive=teacher["positive"],
                           learned_positive=learned["positive"],
                           teacher_condition_ok=teacher["condition_ok"],
                           learned_condition_ok=learned["condition_ok"],
                           condition=teacher["condition"],
                           minimum_eigenvalue=teacher["minimum_eigenvalue"],
                           error=int(teacher["status"] != learned["status"])))
    return result


def truth_table(router: CurvatureRouter) -> list[dict]:
    rows = []
    for positive in (False, True):
        for condition_ok in (False, True):
            prediction = router.conjunction(positive, condition_ok)
            teacher = positive and condition_ok
            rows.append(dict(positive=positive, condition_ok=condition_ok,
                             teacher="DIRECT" if teacher else "DAMPED",
                             prediction="DIRECT" if prediction else "DAMPED",
                             error=int(prediction != teacher)))
    return rows


def replay(states, router: CurvatureRouter) -> dict:
    groups = defaultdict(lambda: dict(states=0, direct=0, damped=0,
                                      nonfinite=0, route_errors=0,
                                      same_input_direction_errors=0,
                                      router_attributable_direction_differences=0,
                                      worst_teacher_direction_relative_error=0.0))
    first_failure = None
    for serial, state in enumerate(states):
        teacher = hessian.derivatives(state.objective, state.theta)
        learned = hessian.derivatives(
            lambda value: gradient.learned_objective(state, value),
            state.theta, learned=True)
        _, teacher_gradient, _, teacher_hessian = teacher
        _, learned_gradient, _, learned_hessian = learned
        teacher_state = teacher_analysis(teacher_gradient, teacher_hessian)
        learned_state = router.analyze(learned_gradient, learned_hessian)
        deterministic_learned_state = teacher_analysis(learned_gradient, learned_hessian)
        key = f"{state.family}_{state.split}"
        group = groups[key]
        group["states"] += 1
        group[teacher_state["status"].lower()] += 1
        route_error = int(learned_state["status"] != teacher_state["status"])
        same_input_error = int(learned_state["status"] != deterministic_learned_state["status"])
        group["route_errors"] += route_error
        group["same_input_direction_errors"] += same_input_error

        if len(teacher_gradient) == 2:
            teacher_direction, _, _, teacher_status = production._newton_direction(
                teacher_gradient, teacher_hessian)
        else:
            # Production N3 constructs eye(2).  For the requested d=3-6 generic
            # controls, retain identical route/damping/solve semantics with the
            # already dimension-generic diagnostic identity construction.
            teacher_direction, _, _, teacher_status = direction_with_route(
                teacher_gradient, teacher_hessian, teacher_state)
        learned_direction, _, _, learned_status = direction_with_route(
            learned_gradient, learned_hessian, learned_state)
        deterministic_direction, _, _, deterministic_status = direction_with_route(
            learned_gradient, learned_hessian, deterministic_learned_state)
        if learned_direction is None or deterministic_direction is None:
            attributable = int(not (learned_direction is None and
                                     deterministic_direction is None and
                                     learned_status == deterministic_status))
        else:
            attributable = int(not torch.equal(learned_direction, deterministic_direction))
        group["router_attributable_direction_differences"] += attributable
        if teacher_direction is None or learned_direction is None:
            relative = (0.0 if teacher_direction is None and learned_direction is None
                        and teacher_status == learned_status else math.inf)
        else:
            relative = float(
                torch.linalg.vector_norm(learned_direction - teacher_direction) /
                max(float(torch.linalg.vector_norm(teacher_direction)),
                    np.finfo(np.float64).tiny))
        group["worst_teacher_direction_relative_error"] = max(
            group["worst_teacher_direction_relative_error"], relative)
        if ((route_error or same_input_error or relative > 1.0e-10) and
                first_failure is None):
            first_failure = dict(family=state.family, split=state.split,
                                 case_id=state.case_id,
                                 teacher_route=teacher_state["status"],
                                 learned_route=learned_state["status"],
                                 teacher_direction_status=teacher_status,
                                 learned_direction_status=learned_status,
                                 direction_relative_error=relative)
        if (serial + 1) % 20 == 0:
            print(f"curvature replay {serial + 1}/{len(states)}", flush=True)
    totals = dict(
        states=sum(row["states"] for row in groups.values()),
        direct=sum(row["direct"] for row in groups.values()),
        damped=sum(row["damped"] for row in groups.values()),
        nonfinite=sum(row["nonfinite"] for row in groups.values()),
        route_errors=sum(row["route_errors"] for row in groups.values()),
        same_input_direction_errors=sum(row["same_input_direction_errors"] for row in groups.values()),
        router_attributable_direction_differences=sum(
            row["router_attributable_direction_differences"] for row in groups.values()),
        worst_teacher_direction_relative_error=max(
            row["worst_teacher_direction_relative_error"] for row in groups.values()),
    )
    passed = (totals["route_errors"] == 0 and
              totals["same_input_direction_errors"] == 0 and
              totals["router_attributable_direction_differences"] == 0 and
              totals["worst_teacher_direction_relative_error"] <= 1.0e-10)
    return dict(passed=passed, totals=totals, groups=dict(groups),
                first_failure=first_failure)


def write_markdown(report: dict) -> None:
    replay_result = report["replay"]
    lines = [
        "# N3 curvature router", "", f"Status: `{report['status']}`", "",
        "## Production rule", "",
        "```text", "finite = all_finite(gradient) AND all_finite(H)",
        "s = svdvals(H)", "condition = max(s)/min(s), or +inf when min(s) <= 0",
        "e = eigvalsh(H)", "positive = min(e) > 0",
        f"condition_ok = condition <= {production.CONDITION_LIMIT:.17g}",
        "DIRECT = learned_AND(positive, condition_ok)",
        "```", "",
        "The finite guard and eig/SVD feature extraction remain deterministic. The locked "
        "six-parameter Boolean router is reused without retraining.", "",
        "## Replay", "",
        "| Pool | States | Direct | Damped | Nonfinite | Route errors | Worst direction relative error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in replay_result["groups"].items():
        lines.append(f"| {name} | {row['states']} | {row['direct']} | {row['damped']} | "
                     f"{row['nonfinite']} | {row['route_errors']} | "
                     f"{row['worst_teacher_direction_relative_error']:.6g} |")
    lines.extend(["", f"Boundary controls: `{report['boundary_controls']['cases']}`; "
                  f"errors: `{report['boundary_controls']['errors']}`.", "",
                  f"Checkpoint: `{Path(report['checkpoint']).name}`", "",
                  f"SHA-256: `{report['checkpoint_sha256']}`", "",
                  "Linear solve, damping arithmetic, backtracking, and stopping were not learned or modified."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dependencies = {str(path): sha256(path) for path in
                    (GRADIENT_CHECKPOINT, HESSIAN_CHECKPOINT, AND_CHECKPOINT)}
    for path, status in ((GRADIENT_CHECKPOINT, "FROZEN_CANDIDATE"),
                         (HESSIAN_CHECKPOINT, "FROZEN_CANDIDATE")):
        if torch.load(path, weights_only=False).get("status") != status:
            raise RuntimeError(f"protected dependency not {status}: {path}")
    gradient.OPS = gradient.LearnedArithmetic()
    hessian.activate_second_order_gradient()
    router = CurvatureRouter()
    dependencies[str(router.compare.sub_path)] = sha256(router.compare.sub_path)
    dependencies[str(router.compare.zero_path)] = sha256(router.compare.zero_path)
    table = truth_table(router)
    controls = boundary_controls(router)
    states = state_source.build_states()
    replay_result = replay(states, router)
    table_errors = sum(row["error"] for row in table)
    control_errors = sum(row["error"] for row in controls)
    passed = table_errors == 0 and control_errors == 0 and replay_result["passed"]
    checkpoint = None
    if passed:
        checkpoint = OUT / "n3_curvature_router.pt"
        torch.save(dict(
            status="LOCKED", architecture="2-input/2-class linear Boolean router",
            new_parameters=0, reused_parameters=6,
            state_dict=router.and_router.state_dict(),
            finite_guard="deterministic production early return",
            inputs=["minimum_eigenvalue_strictly_positive",
                    "condition_at_most_1e12"],
            dependencies=dependencies,
        ), checkpoint)
    report = dict(
        status="PASSED" if passed else "STOPPED_ROUTER_GATE_FAILED",
        exact_production_rule=dict(
            finite_guard="all gradient and Hessian values finite; otherwise NONFINITE",
            evaluation_order=["svdvals(H)", "smallest=min(s)", "largest=max(s)",
                              "condition=largest/smallest if smallest>0 else +inf",
                              "eigvalsh(H)", "minimum_eigenvalue=min(e)",
                              "scale=max(max(abs(e)), float64.tiny)",
                              "positive=(minimum_eigenvalue>0)",
                              "condition_ok=(condition<=1e12)",
                              "direct=positive AND condition_ok"],
            equality_semantics="zero eigenvalue is DAMPED; condition exactly 1e12 may be DIRECT",
        ),
        classification=dict(
            deterministic_linear_algebra=["isfinite", "svdvals", "min/max", "eigvalsh", "abs", "scale"],
            existing_learned_primitives=["locked strict float64 comparison/sign assembly",
                                         "locked six-parameter Boolean AND router"],
            newly_trained_primitives=[],
            genuinely_missing_after_reuse=[]),
        router=dict(architecture="reused float64 Linear(2,2) categorical AND",
                    new_parameters=0, reused_parameters=6,
                    inputs=["minimum_eigenvalue > 0", "condition <= 1e12"],
                    outputs=["DAMPED", "DIRECT"]),
        truth_table=dict(cases=len(table), errors=table_errors, rows=table),
        boundary_controls=dict(cases=len(controls), errors=control_errors, rows=controls),
        replay=replay_result,
        systems=dict(states=len(states), dimensions="SHO/ball d=2; synthetic d=3-6"),
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        dependencies=dependencies,
        protected_checkpoints_modified=False,
        next_operation="Newton damping arithmetic and linear solve",
        elapsed_seconds=time.time() - started,
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
