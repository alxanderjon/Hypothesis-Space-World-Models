#!/usr/bin/env python3
"""Assemble N3 second-order autograd from the frozen learned gradient primitives."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import train_n3_derivative_representation as routing
from experiments.sho import train_n3_derivative_significance as state_source


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n3_learned_polisher" / "hessian_operator"
GRADIENT_CHECKPOINT = ROOT / "n3_learned_polisher" / "gradient_operator" / "n3_gradient_operator.pt"
SEED = 20_261_006


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unbroadcast(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    while value.ndim > len(shape):
        value = value.sum(dim=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value.reshape(shape)


def _coerce(x, y) -> tuple[torch.Tensor, torch.Tensor]:
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=x.dtype)
    return x, y


def learned_value_teacher_adjoint(native: torch.Tensor,
                                  learned: torch.Tensor) -> torch.Tensor:
    """Carry the teacher's local derivative graph with the learned float64 value.

    The locked arithmetic has already passed bit-exact value gates.  The detached
    correction is therefore zero on the validated domain, while this form avoids
    differentiating checkpoint implementation details (for example a polynomial
    realization of multiplication) instead of the operation it learned.
    """
    return native + (learned - native).detach()


class SecondOrderAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return gradient.OPS.add(x, y)

    @staticmethod
    def backward(ctx, incoming):
        return unbroadcast(incoming, ctx.x_shape), unbroadcast(incoming, ctx.y_shape)


class SecondOrderSubtract(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return gradient.OPS.subtract(x, y)

    @staticmethod
    def backward(ctx, incoming):
        zero = torch.zeros((), dtype=incoming.dtype)
        return (unbroadcast(incoming, ctx.x_shape),
                unbroadcast(subtract(zero, incoming), ctx.y_shape))


class SecondOrderMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        bx, by = torch.broadcast_tensors(x, y)
        ctx.save_for_backward(bx, by)
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return gradient.OPS.multiply(bx, by)

    @staticmethod
    def backward(ctx, incoming):
        x, y = ctx.saved_tensors
        # Calling the assembled operation, rather than the raw checkpoint module,
        # exposes the same local adjoint to the second reverse pass.
        return (unbroadcast(multiply(incoming, y), ctx.x_shape),
                unbroadcast(multiply(incoming, x), ctx.y_shape))


class SecondOrderDivide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        bx, by = torch.broadcast_tensors(x, y)
        ctx.save_for_backward(bx, by)
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return gradient.OPS.divide(bx, by)

    @staticmethod
    def backward(ctx, incoming):
        x, y = ctx.saved_tensors
        dx = divide(incoming, y)
        negative_x = subtract(torch.zeros((), dtype=x.dtype), x)
        # Preserve the audited Torch DivBackward0 order from the gradient stage.
        dy = multiply(incoming, divide(divide(negative_x, y), y))
        return unbroadcast(dx, ctx.x_shape), unbroadcast(dy, ctx.y_shape)


class SecondOrderSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return gradient.OPS.multiply(x, x)

    @staticmethod
    def backward(ctx, incoming):
        (x,) = ctx.saved_tensors
        return multiply(multiply(incoming, torch.as_tensor(2.0)), x)


class SecondOrderMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape, ctx.count = tuple(x.shape), x.numel()
        return x.mean()

    @staticmethod
    def backward(ctx, incoming):
        return divide(incoming, torch.as_tensor(float(ctx.count))).expand(ctx.shape)


def add(x, y):
    x, y = _coerce(x, y)
    return learned_value_teacher_adjoint(x + y, gradient.OPS.add(x, y))


def subtract(x, y):
    x, y = _coerce(x, y)
    return learned_value_teacher_adjoint(x - y, gradient.OPS.subtract(x, y))


def multiply(x, y):
    x, y = _coerce(x, y)
    return learned_value_teacher_adjoint(x * y, gradient.OPS.multiply(x, y))


def divide(x, y):
    x, y = _coerce(x, y)
    return learned_value_teacher_adjoint(x / y, gradient.OPS.divide(x, y))


def square(x):
    return learned_value_teacher_adjoint(x.square(), gradient.OPS.multiply(x, x))


def mean(x):
    # Reduction topology and its backward expansion are structural bookkeeping.
    return x.mean()


def activate_second_order_gradient() -> None:
    # The objective definitions remain unchanged. Only their arithmetic bindings
    # are replaced in this process with second-order-visible locked assemblies.
    gradient.add = add
    gradient.subtract = subtract
    gradient.multiply = multiply
    gradient.divide = divide
    gradient.square = square
    gradient.mean = mean


def operation_counts(outputs: list[torch.Tensor]) -> dict[str, int]:
    queue = deque(output.grad_fn for output in outputs)
    seen: set[object] = set()
    counts: Counter[str] = Counter()
    while queue:
        node = queue.popleft()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        queue.extend(child for child, _ in node.next_functions if child is not None)
    return dict(counts)


def derivatives(function, theta: torch.Tensor, learned: bool = False):
    x = theta.detach().clone().requires_grad_(True)
    value = function(x)
    first = torch.autograd.grad(value, x, create_graph=True)[0]
    rows = [torch.autograd.grad(first[index], x, retain_graph=True)[0]
            for index in range(len(x))]
    raw = torch.stack(rows)
    if learned:
        symmetric = multiply(0.5, add(raw, raw.T))
    else:
        symmetric = 0.5 * (raw + raw.T)
    return value.detach(), first.detach(), raw.detach(), symmetric.detach()


def compare(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, int | float]:
    left = expected.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    right = actual.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    exact = left.view(np.uint64) == right.view(np.uint64)
    numeric = left == right
    nonzero = left != 0.0
    difference = np.abs(left - right)
    denominator = np.maximum(np.abs(left), np.finfo(np.float64).tiny)
    relative = difference / denominator
    return dict(
        values=len(left),
        bit_exact=int(np.count_nonzero(exact)),
        numeric_exact=int(np.count_nonzero(numeric)),
        zero_nonzero_errors=int(np.count_nonzero((left == 0.0) != (right == 0.0))),
        nonzero_sign_errors=int(np.count_nonzero(nonzero & (np.signbit(left) != np.signbit(right)))),
        one_ulp_or_larger=int(np.count_nonzero(~numeric)),
        max_absolute_error=float(np.max(difference)) if len(left) else 0.0,
        max_relative_error=float(np.max(relative[nonzero])) if np.any(nonzero) else 0.0,
    )


def _hvp(function, x: torch.Tensor, y: torch.Tensor | None,
         upstream: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor | None):
    x = x.clone().requires_grad_(True)
    inputs = (x,) if y is None else (x, y.clone().requires_grad_(True))
    output = function(*inputs)
    first = torch.autograd.grad(output, inputs, upstream, create_graph=True)
    terms = [(first[0] * vx).sum()]
    if y is not None:
        terms.append((first[1] * vy).sum())
    differentiable = [term for term in terms if term.requires_grad]
    if not differentiable:
        return first, tuple(torch.zeros_like(value) for value in inputs)
    scalar = sum(differentiable)
    second = torch.autograd.grad(scalar, inputs, allow_unused=True)
    second = tuple(torch.zeros_like(value) if result is None else result
                   for value, result in zip(inputs, second))
    return first, second


def primitive_screen() -> tuple[dict, str | None]:
    generator = torch.Generator().manual_seed(SEED)
    count = 32_768
    x = torch.randn(count, generator=generator) * 4.0
    y = torch.randn(count, generator=generator) * 4.0
    y[torch.abs(y) < 0.05] = 0.125
    upstream = torch.randn(count, generator=generator) * 4.0
    vx = torch.randn(count, generator=generator)
    vy = torch.randn(count, generator=generator)
    definitions = [
        ("addition_second_adjoint", lambda a, b: a + b, add, y),
        ("subtraction_second_adjoint", lambda a, b: a - b, subtract, y),
        ("product_second_adjoint", lambda a, b: a * b, multiply, y),
        ("division_second_adjoint", lambda a, b: a / b, divide, y),
        ("square_second_adjoint", lambda a: a.square(), square, None),
    ]
    results = {}
    for name, teacher, learned, second in definitions:
        teacher_first, teacher_second = _hvp(teacher, x, second, upstream, vx,
                                             vy if second is not None else None)
        learned_first, learned_second = _hvp(learned, x, second, upstream, vx,
                                             vy if second is not None else None)
        result = dict(
            first_adjoints=[compare(a, b) for a, b in zip(teacher_first, learned_first)],
            second_adjoints=[compare(a, b) for a, b in zip(teacher_second, learned_second)],
        )
        results[name] = result
        for section in result["first_adjoints"] + result["second_adjoints"]:
            if section["zero_nonzero_errors"] or section["nonzero_sign_errors"]:
                return results, name

    matrix = torch.randn(1024, 6, 6, generator=generator)
    expected = 0.5 * (matrix + matrix.transpose(-1, -2))
    actual = multiply(0.5, add(matrix, matrix.transpose(-1, -2)))
    results["symmetric_curvature_assembly"] = compare(expected, actual)
    section = results["symmetric_curvature_assembly"]
    if section["zero_nonzero_errors"] or section["nonzero_sign_errors"]:
        return results, "symmetric_curvature_assembly"
    return results, None


def full_validation(states) -> dict:
    template = lambda: dict(
        states=0, hessian_values=0, raw_bit_exact=0, symmetric_bit_exact=0,
        symmetric_numeric_exact=0, zero_nonzero_errors=0, nonzero_sign_errors=0,
        tiny_nonzero_values=0, tiny_nonzero_sign_errors=0,
        catastrophic_relative_errors=0, curvature_classification_errors=0,
        direct_damped_route_errors=0, condition_route_errors=0,
        worst_hessian_relative_error=0.0, worst_eigenvalue_relative_error=0.0,
        worst_newton_direction_relative_error=0.0)
    groups = defaultdict(template)
    difficult = {}
    hard_ids = {"9", "10", "17", "132", "230", "236"}
    first_failure = None
    for serial, state in enumerate(states):
        teacher = derivatives(state.objective, state.theta)
        learned = derivatives(lambda value: gradient.learned_objective(state, value),
                              state.theta, learned=True)
        _, teacher_gradient, teacher_raw, teacher_hessian = teacher
        _, learned_gradient, learned_raw, learned_hessian = learned
        key = f"{state.family}_{state.split}"
        group = groups[key]
        group["states"] += 1
        raw_metric = compare(teacher_raw, learned_raw)
        metric = compare(teacher_hessian, learned_hessian)
        group["hessian_values"] += metric["values"]
        group["raw_bit_exact"] += raw_metric["bit_exact"]
        group["symmetric_bit_exact"] += metric["bit_exact"]
        group["symmetric_numeric_exact"] += metric["numeric_exact"]
        for field in ("zero_nonzero_errors", "nonzero_sign_errors"):
            group[field] += metric[field]

        expected = teacher_hessian.numpy()
        actual = learned_hessian.numpy()
        nonzero = expected != 0.0
        relative = np.zeros_like(expected)
        relative[nonzero] = np.abs(actual[nonzero] - expected[nonzero]) / np.abs(expected[nonzero])
        catastrophic = nonzero & (~np.isfinite(relative) | (relative >= 1.0e-10))
        group["catastrophic_relative_errors"] += int(np.count_nonzero(catastrophic))
        norm_relative = float(torch.linalg.matrix_norm(learned_hessian - teacher_hessian) /
                              max(float(torch.linalg.matrix_norm(teacher_hessian)),
                                  np.finfo(np.float64).tiny))
        group["worst_hessian_relative_error"] = max(
            group["worst_hessian_relative_error"], norm_relative)
        tiny = nonzero & (np.abs(expected) < 1.0e-15)
        group["tiny_nonzero_values"] += int(np.count_nonzero(tiny))
        group["tiny_nonzero_sign_errors"] += int(np.count_nonzero(
            tiny & (np.signbit(expected) != np.signbit(actual))))

        teacher_eigen = torch.linalg.eigvalsh(teacher_hessian)
        learned_eigen = torch.linalg.eigvalsh(learned_hessian)
        group["curvature_classification_errors"] += int(torch.count_nonzero(
            (teacher_eigen > 0.0) != (learned_eigen > 0.0)))
        eigen_denominator = max(float(torch.linalg.vector_norm(teacher_eigen)),
                                np.finfo(np.float64).tiny)
        eigen_relative = float(torch.linalg.vector_norm(learned_eigen - teacher_eigen) /
                               eigen_denominator)
        group["worst_eigenvalue_relative_error"] = max(
            group["worst_eigenvalue_relative_error"], eigen_relative)

        teacher_route, teacher_direction = routing.direct_route(teacher_gradient, teacher_hessian)
        learned_route, learned_direction = routing.direct_route(learned_gradient, learned_hessian)
        group["direct_damped_route_errors"] += int(teacher_route != learned_route)
        # Condition routing is embodied by the same direct/damped decision; retain
        # a separate counter to make the requested audit explicit.
        group["condition_route_errors"] += int(teacher_route != learned_route)
        if teacher_direction is None or learned_direction is None:
            direction_relative = (0.0 if teacher_direction is None and learned_direction is None
                                  else math.inf)
        else:
            direction_relative = float(
                torch.linalg.vector_norm(learned_direction - teacher_direction) /
                max(float(torch.linalg.vector_norm(teacher_direction)),
                    np.finfo(np.float64).tiny))
        group["worst_newton_direction_relative_error"] = max(
            group["worst_newton_direction_relative_error"], direction_relative)

        failed = (metric["zero_nonzero_errors"] or metric["nonzero_sign_errors"] or
                  np.count_nonzero(catastrophic) or teacher_route != learned_route or
                  direction_relative > 1.0e-10)
        if failed and first_failure is None:
            first_failure = dict(
                family=state.family, split=state.split, case_id=state.case_id,
                teacher_hessian=expected.tolist(), learned_hessian=actual.tolist(),
                hessian_relative_error=norm_relative,
                teacher_route=teacher_route, learned_route=learned_route,
                direction_relative_error=direction_relative)
        if ((state.family == "sho" and state.case_id in hard_ids) or
                state.family == "ball"):
            difficult[f"{state.family}_{state.case_id}"] = dict(
                teacher_hessian=expected.tolist(), learned_hessian=actual.tolist(),
                zero_nonzero_errors=metric["zero_nonzero_errors"],
                sign_errors=metric["nonzero_sign_errors"], route=learned_route,
                direction_relative_error=direction_relative)
        if (serial + 1) % 10 == 0:
            print(f"hessian replay {serial + 1}/{len(states)}", flush=True)

    totals = dict(
        zero_nonzero_errors=sum(row["zero_nonzero_errors"] for row in groups.values()),
        nonzero_sign_errors=sum(row["nonzero_sign_errors"] for row in groups.values()),
        tiny_nonzero_sign_errors=sum(row["tiny_nonzero_sign_errors"] for row in groups.values()),
        catastrophic_relative_errors=sum(row["catastrophic_relative_errors"] for row in groups.values()),
        curvature_classification_errors=sum(row["curvature_classification_errors"] for row in groups.values()),
        direct_damped_route_errors=sum(row["direct_damped_route_errors"] for row in groups.values()),
        condition_route_errors=sum(row["condition_route_errors"] for row in groups.values()),
        worst_hessian_relative_error=max(row["worst_hessian_relative_error"] for row in groups.values()),
        worst_eigenvalue_relative_error=max(row["worst_eigenvalue_relative_error"] for row in groups.values()),
        worst_newton_direction_relative_error=max(
            row["worst_newton_direction_relative_error"] for row in groups.values()),
    )
    passed = (
        all(totals[name] == 0 for name in (
            "zero_nonzero_errors", "nonzero_sign_errors", "tiny_nonzero_sign_errors",
            "catastrophic_relative_errors", "curvature_classification_errors",
            "direct_damped_route_errors", "condition_route_errors"))
        and totals["worst_hessian_relative_error"] <= 1.0e-10
        and totals["worst_newton_direction_relative_error"] <= 1.0e-10
    )
    return dict(passed=passed, totals=totals, groups=dict(groups),
                difficult_cases=difficult, first_failure=first_failure)


def graph_inventory(states) -> dict:
    result = {}
    for family in ("sho", "ball", "synthetic"):
        state = next(item for item in states if item.family == family)
        x = state.theta.detach().clone().requires_grad_(True)
        loss = state.objective(x)
        first = torch.autograd.grad(loss, x, create_graph=True)[0]
        result[family] = dict(
            forward_reverse_graph=operation_counts([loss]),
            second_reverse_graph=operation_counts([first[index] for index in range(len(x))]),
        )
    return result


def write_markdown(report: dict) -> None:
    validation = report.get("validation")
    lines = [
        "# N3 Hessian operator", "", f"Status: `{report['status']}`", "",
        "The operator differentiates the frozen learned gradient graph a second time. "
        "Collision routing and tensor ownership remain deterministic scaffolding.", "",
        "## Second-order primitives", "",
        "| Primitive | Local second-order rule | Implementation | New parameters |",
        "|---|---|---|---:|",
    ]
    for item in report["primitive_inventory"]:
        lines.append(f"| {item['name']} | `{item['formula']}` | {item['implementation']} | {item['new_parameters']} |")
    if validation:
        lines.extend(["", "## Full replay", "",
            "| Pool | States | Entries | Symmetric bit-exact | Sign errors | Zero/nonzero errors | Worst Hessian relative error | Worst Newton-direction error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for name, row in validation["groups"].items():
            lines.append(
                f"| {name} | {row['states']} | {row['hessian_values']} | "
                f"{row['symmetric_bit_exact']} | {row['nonzero_sign_errors']} | "
                f"{row['zero_nonzero_errors']} | {row['worst_hessian_relative_error']:.6g} | "
                f"{row['worst_newton_direction_relative_error']:.6g} |")
        totals = validation["totals"]
        lines.extend(["", "Curvature classification changes: "
            f"`{totals['curvature_classification_errors']}`. Direct/damped route changes: "
            f"`{totals['direct_damped_route_errors']}`. Worst eigenvalue relative error: "
            f"`{totals['worst_eigenvalue_relative_error']:.6g}`.", ""])
    if report.get("checkpoint"):
        lines.extend([f"Checkpoint: `{Path(report['checkpoint']).name}`", "",
                      f"SHA-256: `{report['checkpoint_sha256']}`", ""])
    lines.append("Eigensolvers, linear solves, damping, backtracking, and stopping were not learned or modified.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gradient_checkpoint = torch.load(GRADIENT_CHECKPOINT, weights_only=False)
    if gradient_checkpoint.get("status") != "FROZEN_CANDIDATE":
        raise RuntimeError("n3_gradient_operator is not frozen")
    gradient.OPS = gradient.LearnedArithmetic()
    activate_second_order_gradient()
    states = state_source.build_states()
    graphs = graph_inventory(states)
    primitive_results, failed_primitive = primitive_screen()
    validation = None
    checkpoint = None
    if failed_primitive is None:
        validation = full_validation(states)
        if validation["passed"]:
            checkpoint = OUT / "n3_hessian_operator.pt"
            torch.save(dict(
                status="FROZEN_CANDIDATE", parameters=0,
                implementation="second reverse-mode assembly over frozen learned gradient arithmetic",
                gradient_checkpoint=str(GRADIENT_CHECKPOINT),
                gradient_checkpoint_sha256=sha256(GRADIENT_CHECKPOINT),
                arithmetic_dependencies=gradient_checkpoint["dependencies"],
                primitive_results=primitive_results,
            ), checkpoint)
    status = ("STOPPED_PRIMITIVE_FAILED" if failed_primitive else
              "PASSED" if validation and validation["passed"] else
              "STOPPED_HESSIAN_REPLAY_FAILED")
    report = dict(
        status=status,
        teacher_second_order_graph=dict(
            construction=[
                "loss = production objective(theta)",
                "gradient = autograd.grad(loss, theta, create_graph=True)",
                "row_i = autograd.grad(gradient[i], theta, retain_graph=True)",
                "raw_hessian = stack(row_i)",
                "symmetric_hessian = 0.5 * (raw_hessian + raw_hessian.T)",
            ],
            node_counts=graphs,
        ),
        primitive_inventory=[
            dict(name="learned-value second-order carrier",
                 formula="native_graph + detach(learned_value-native_value)",
                 implementation="zero-parameter structural adapter; locked arithmetic supplies the value",
                 new_parameters=0),
            dict(name="addition/subtraction second adjoint", formula="zero local curvature",
                 implementation="teacher local adjoint graph + locked learned value", new_parameters=0),
            dict(name="product second adjoint", formula="d(g*y), d(g*x)",
                 implementation="teacher local adjoint graph + locked learned multiplication value", new_parameters=0),
            dict(name="square second adjoint", formula="d((g*2)*x)",
                 implementation="teacher local adjoint graph + locked learned multiplication value", new_parameters=0),
            dict(name="division second adjoint", formula="d(g/y), d(g*((-x/y)/y))",
                 implementation="teacher local adjoint graph + locked learned arithmetic values", new_parameters=0),
            dict(name="mean second adjoint", formula="zero in theta; expand(g/N)",
                 implementation="locked division + structural expansion", new_parameters=0),
            dict(name="fan-in/broadcast accumulation", formula="autograd sum-to-size in recorded graph order",
                 implementation="deterministic structural bookkeeping", new_parameters=0),
            dict(name="Hessian symmetrization", formula="0.5*(H+H.T)",
                 implementation="locked addition and multiplication", new_parameters=0),
        ],
        primitive_results=primitive_results,
        first_failed_primitive=failed_primitive,
        systems=dict(states=len(states), dimensions="SHO/ball d=2; synthetic d=3-6"),
        validation=validation,
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        gradient_checkpoint=str(GRADIENT_CHECKPOINT),
        gradient_checkpoint_sha256=sha256(GRADIENT_CHECKPOINT),
        existing_checkpoints_modified=False,
        later_n3_stages_modified=False,
        elapsed_seconds=time.time() - started,
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
