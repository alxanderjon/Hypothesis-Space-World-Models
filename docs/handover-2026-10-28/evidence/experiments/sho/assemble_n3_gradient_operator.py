#!/usr/bin/env python3
"""Assemble the exact active-branch N3 gradient from locked arithmetic."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_division_rounding_router import DivisionAssembly, RoundingRouter
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_subtraction_operator import Subtraction
from experiments.sho import run_stage1b_n3_gradient_polishing as n3
from experiments.sho import run_stage1b_two_parameter_learned_search as sho_search
from experiments.sho import train_n3_derivative_representation as prior
from experiments.sho import train_n3_derivative_significance as state_source


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n3_learned_polisher" / "gradient_operator"
ADD_PATH = ROOT / "addition_operator_smoke" / "addition_before_evaluation.pt"
MUL_PATH = ROOT / "multiplication_operator_smoke" / "multiplication_before_evaluation.pt"
SUB_PATH = ROOT / "subtraction_operator_training" / "subtraction_before_evaluation.pt"
DIV_ROUTER_PATH = ROOT / "division_rounding_router_training" / "rounding_router_before_evaluation.pt"
SEED = 20_261_005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LearnedArithmetic:
    def __init__(self) -> None:
        self.add_model = Addition()
        self.add_model.load_state_dict(torch.load(ADD_PATH, weights_only=False)["state_dict"])
        self.mul_model = QuadraticArithmetic()
        self.mul_model.load_state_dict(torch.load(MUL_PATH, weights_only=False)["state_dict"])
        self.sub_model = Subtraction()
        self.sub_model.load_state_dict(torch.load(SUB_PATH, weights_only=False)["state_dict"])
        router = RoundingRouter()
        router.load_state_dict(torch.load(DIV_ROUTER_PATH, weights_only=False)["state_dict"])
        self.division = DivisionAssembly(router)
        for model in (self.add_model, self.mul_model, self.sub_model):
            model.eval().requires_grad_(False)

    @staticmethod
    def _pair(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        x, y = torch.broadcast_tensors(x, y)
        shape = tuple(x.shape)
        return torch.stack((x.reshape(-1), y.reshape(-1)), dim=1), shape

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pair, shape = self._pair(x, y)
        return self.add_model(pair).reshape(shape)

    def subtract(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pair, shape = self._pair(x, y)
        return self.sub_model(pair).reshape(shape)

    def multiply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pair, shape = self._pair(x, y)
        return self.mul_model(pair).reshape(shape)

    def divide(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pair, shape = self._pair(x, y)
        array = pair.detach().cpu().numpy()
        values, _, _ = self.division.predict(array)
        return torch.from_numpy(values.copy()).reshape(shape)


OPS: LearnedArithmetic | None = None


def tensor(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=like.dtype)


def unbroadcast(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # This matches PyTorch's structural sum-to-size ownership bookkeeping.
    while value.ndim > len(shape):
        value = value.sum(dim=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value.reshape(shape)


class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return OPS.add(x, y)

    @staticmethod
    def backward(ctx, gradient):
        return unbroadcast(gradient, ctx.x_shape), unbroadcast(gradient, ctx.y_shape)


class SubtractFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return OPS.subtract(x, y)

    @staticmethod
    def backward(ctx, gradient):
        zero = torch.zeros((), dtype=gradient.dtype)
        return unbroadcast(gradient, ctx.x_shape), unbroadcast(OPS.subtract(zero, gradient), ctx.y_shape)


class MultiplyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        bx, by = torch.broadcast_tensors(x, y)
        ctx.save_for_backward(bx, by)
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return OPS.multiply(bx, by)

    @staticmethod
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        return (unbroadcast(OPS.multiply(gradient, y), ctx.x_shape),
                unbroadcast(OPS.multiply(gradient, x), ctx.y_shape))


class DivideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        bx, by = torch.broadcast_tensors(x, y)
        ctx.save_for_backward(bx, by)
        ctx.x_shape, ctx.y_shape = tuple(x.shape), tuple(y.shape)
        return OPS.divide(bx, by)

    @staticmethod
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        gx = OPS.divide(gradient, y)
        # Exact audited DivBackward0 order: grad * ((-x / y) / y).
        negative_x = OPS.subtract(torch.zeros((), dtype=x.dtype), x)
        gy = OPS.multiply(gradient, OPS.divide(OPS.divide(negative_x, y), y))
        return unbroadcast(gx, ctx.x_shape), unbroadcast(gy, ctx.y_shape)


class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return OPS.multiply(x, x)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return OPS.multiply(OPS.multiply(gradient, torch.as_tensor(2.0)), x)


class MeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape, ctx.count = tuple(x.shape), x.numel()
        # Production mean reduction stays deterministic scaffolding.
        return x.mean()

    @staticmethod
    def backward(ctx, gradient):
        scaled = OPS.divide(gradient, torch.as_tensor(float(ctx.count)))
        return scaled.expand(ctx.shape)


def add(x, y):
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    y = tensor(y, x)
    return AddFunction.apply(x, y)


def subtract(x, y):
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    y = tensor(y, x)
    return SubtractFunction.apply(x, y)


def multiply(x, y):
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    y = tensor(y, x)
    return MultiplyFunction.apply(x, y)


def divide(x, y):
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    y = tensor(y, x)
    return DivideFunction.apply(x, y)


def square(x):
    return SquareFunction.apply(x)


def mean(x):
    return MeanFunction.apply(x)


def sho_objective(theta: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    prefix = position[:sho_search.P]
    parameter_a, parameter_b = theta.unbind()
    loss = torch.zeros((), dtype=torch.float64)
    for horizon in sho_search.J_ALL:
        starts = torch.arange(1, sho_search.P-horizon)
        previous, current = prefix[starts-1], prefix[starts]
        predictions = []
        for _ in range(horizon):
            following = add(multiply(parameter_a, previous), multiply(parameter_b, current))
            predictions.append(following)
            previous, current = current, following
        prediction = torch.stack(predictions, dim=-1)
        truth = torch.stack([prefix[starts+step] for step in range(1,horizon+1)], dim=-1)
        term = multiply(float(sho_search.DRIFT_HEAVY[horizon]), mean(square(subtract(prediction, truth))))
        loss = add(loss, term)
    return loss


def ball_step(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    a, b = theta.unbind(); x, y, vx, vy = state.unbind()
    ux, uy = multiply(a, vx), multiply(a, vy)
    t_remain: float | torch.Tensor = ball.DT
    for _ in range(ball.MAX_COLLISIONS):
        if float(t_remain) <= ball.EPS: break
        tx: float | torch.Tensor = math.inf; ty: float | torch.Tensor = math.inf
        if float(ux.detach()) > ball.EPS: tx = divide(subtract(ball.XMAX, x), ux)
        elif float(ux.detach()) < -ball.EPS: tx = divide(subtract(ball.XMIN, x), ux)
        if float(uy.detach()) > ball.EPS: ty = divide(subtract(ball.YMAX, y), uy)
        elif float(uy.detach()) < -ball.EPS: ty = divide(subtract(ball.YMIN, y), uy)
        tx_value, ty_value = float(tx), float(ty)
        t_hit: float | torch.Tensor = tx if tx_value <= ty_value else ty
        if float(t_hit) > float(t_remain) or math.isinf(float(t_hit)):
            x = add(x, multiply(ux, t_remain)); y = add(y, multiply(uy, t_remain))
            t_remain = 0.0; break
        x = add(x, multiply(ux, t_hit)); y = add(y, multiply(uy, t_hit))
        t_remain = subtract(t_remain, t_hit)
        hit_x = abs(tx_value-float(t_hit)) < ball.EPS
        hit_y = abs(ty_value-float(t_hit)) < ball.EPS
        if hit_x:
            x = torch.as_tensor(ball.XMAX if float(ux.detach()) > 0.0 else ball.XMIN)
            ux = subtract(0.0, ux); vx = subtract(0.0, vx)
        if hit_y:
            y = torch.as_tensor(ball.YMAX if float(uy.detach()) > 0.0 else ball.YMIN)
            uy = subtract(0.0, uy); vy = subtract(0.0, vy)
    return torch.stack((x, y, multiply(b, vx), multiply(b, vy)))


def ball_objective(theta: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
    prefix = trajectory[:ball.PREFIX_STATES]
    errors = [subtract(ball_step(prefix[index], theta), prefix[index+1])
              for index in range(ball.PREFIX_STATES-1)]
    return mean(square(torch.stack(errors)))


def synthetic_objective(theta: torch.Tensor, matrix: torch.Tensor,
                        center: torch.Tensor) -> torch.Tensor:
    delta = subtract(theta, center)
    quadratic = torch.zeros((), dtype=torch.float64)
    quartic = torch.zeros((), dtype=torch.float64)
    for left in range(len(theta)):
        for right in range(len(theta)):
            quadratic = add(quadratic, multiply(multiply(delta[left], matrix[left,right]), delta[right]))
        squared = multiply(delta[left], delta[left])
        quartic = add(quartic, multiply(squared, squared))
    return add(multiply(0.5, quadratic), multiply(0.01, quartic))


def learned_objective(state, theta: torch.Tensor) -> torch.Tensor:
    defaults = state.objective.__defaults__
    if state.family == "sho": return sho_objective(theta, defaults[0])
    if state.family == "ball": return ball_objective(theta, defaults[0])
    if state.family == "synthetic": return synthetic_objective(theta, defaults[0], defaults[1])
    raise ValueError(state.family)


def operation_counts(loss: torch.Tensor) -> dict[str, int]:
    queue = deque([loss.grad_fn]); seen = set(); counts = Counter()
    while queue:
        node = queue.popleft()
        if node is None or node in seen: continue
        seen.add(node); counts[type(node).__name__] += 1
        queue.extend(child for child,_ in node.next_functions if child is not None)
    return dict(counts)


def compare(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, int | float]:
    e = expected.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    a = actual.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    exact = e.view(np.uint64) == a.view(np.uint64)
    numerical = e == a
    nonzero = e != 0.0
    return dict(values=len(e), bit_exact=int(np.count_nonzero(exact)),
        numeric_exact=int(np.count_nonzero(numerical)),
        zero_nonzero_errors=int(np.count_nonzero((e==0.0)!=(a==0.0))),
        nonzero_sign_errors=int(np.count_nonzero(nonzero & (np.signbit(e)!=np.signbit(a)))),
        one_ulp_or_larger=int(np.count_nonzero(~numerical)),
        max_absolute_error=float(np.max(np.abs(e-a))) if len(e) else 0.0)


def primitive_screen() -> tuple[dict, str | None]:
    generator = torch.Generator().manual_seed(SEED)
    count = 32768
    x = torch.randn(count, generator=generator)*4.0
    y = torch.randn(count, generator=generator)*4.0
    y[torch.abs(y)<0.05] = 0.125
    upstream = torch.randn(count, generator=generator)*4.0
    screens = {}

    def run(name, teacher_function, learned_function, inputs):
        teacher_inputs = [value.clone().requires_grad_(True) for value in inputs]
        learned_inputs = [value.clone().requires_grad_(True) for value in inputs]
        teacher_output = teacher_function(*teacher_inputs)
        learned_output = learned_function(*learned_inputs)
        teacher_gradient = torch.autograd.grad(teacher_output, teacher_inputs, upstream)[0]
        learned_gradient = torch.autograd.grad(learned_output, learned_inputs, upstream)[0]
        result = dict(forward=compare(teacher_output, learned_output),
                      first_input_adjoint=compare(teacher_gradient, learned_gradient))
        screens[name] = result
        return result

    ordered = [
        ("product_adjoint", lambda a,b:a*b, lambda a,b:multiply(a,b), [x,y]),
        ("addition_adjoint", lambda a,b:a+b, lambda a,b:add(a,b), [x,y]),
        ("subtraction_adjoint", lambda a,b:a-b, lambda a,b:subtract(a,b), [x,y]),
        ("square_adjoint", lambda a:a.square(), lambda a:square(a), [x]),
        ("division_adjoint", lambda a,b:a/b, lambda a,b:divide(a,b), [x,y]),
    ]
    for name, teacher_function, learned_function, inputs in ordered:
        result = run(name, teacher_function, learned_function, inputs)
        for section in result.values():
            if section["zero_nonzero_errors"] or section["nonzero_sign_errors"]:
                return screens, name

    tx = x.clone().requires_grad_(True); lx = x.clone().requires_grad_(True)
    teacher_value, learned_value = tx.mean(), mean(lx)
    tg = torch.autograd.grad(teacher_value,tx)[0]; lg = torch.autograd.grad(learned_value,lx)[0]
    screens["mean_adjoint"] = dict(forward=compare(teacher_value,learned_value),
                                    first_input_adjoint=compare(tg,lg))
    result = screens["mean_adjoint"]
    if any(section["zero_nonzero_errors"] or section["nonzero_sign_errors"] for section in result.values()):
        return screens, "mean_adjoint"
    return screens, None


def full_validation(states) -> dict:
    groups = defaultdict(lambda: dict(states=0,gradient_values=0,bit_exact=0,
        numeric_exact=0,zero_nonzero_errors=0,nonzero_sign_errors=0,
        catastrophic_relative_errors=0,tiny_nonzero_values=0,tiny_nonzero_sign_errors=0,
        curvature_route_errors=0,worst_gradient_relative_error=0.0,
        worst_direction_relative_error=0.0,objective_numeric_mismatches=0))
    difficult = {}; first_failure = None
    hard_ids = {"9","10","17","132","230","236"}
    for index,state in enumerate(states):
        teacher_theta = state.theta.detach().clone().requires_grad_(True)
        learned_theta = state.theta.detach().clone().requires_grad_(True)
        teacher_loss = state.objective(teacher_theta)
        learned_loss = learned_objective(state, learned_theta)
        teacher_gradient = torch.autograd.grad(teacher_loss,teacher_theta)[0].detach()
        learned_gradient = torch.autograd.grad(learned_loss,learned_theta)[0].detach()
        _, teacher_hessian, _ = state_source.exact(state)
        true_route,true_direction = prior.direct_route(teacher_gradient,torch.from_numpy(teacher_hessian))
        route,direction = prior.direct_route(learned_gradient,torch.from_numpy(teacher_hessian))
        key=f"{state.family}_{state.split}"; group=groups[key]; group["states"]+=1
        metric=compare(teacher_gradient,learned_gradient)
        for field in ("values","bit_exact","numeric_exact","zero_nonzero_errors","nonzero_sign_errors"):
            target="gradient_values" if field=="values" else field; group[target]+=metric[field]
        group["objective_numeric_mismatches"] += int(float(teacher_loss)!=float(learned_loss))
        teacher_np=teacher_gradient.numpy();learned_np=learned_gradient.numpy();nonzero=teacher_np!=0
        relative=np.zeros_like(teacher_np)
        relative[nonzero]=np.abs(learned_np[nonzero]-teacher_np[nonzero])/np.abs(teacher_np[nonzero])
        group["catastrophic_relative_errors"] += int(np.count_nonzero(nonzero & (~np.isfinite(relative)|(relative>=1.0))))
        group["worst_gradient_relative_error"]=max(group["worst_gradient_relative_error"],
            float(np.linalg.norm(learned_np-teacher_np)/max(np.linalg.norm(teacher_np),np.finfo(float).tiny)))
        tiny=nonzero&(np.abs(teacher_np)<1e-15);group["tiny_nonzero_values"]+=int(np.count_nonzero(tiny))
        group["tiny_nonzero_sign_errors"]+=int(np.count_nonzero(tiny&(np.signbit(teacher_np)!=np.signbit(learned_np))))
        group["curvature_route_errors"]+=int(route!=true_route)
        direction_relative=math.inf if direction is None or true_direction is None else float(
            torch.linalg.vector_norm(direction-true_direction)/max(float(torch.linalg.vector_norm(true_direction)),np.finfo(float).tiny))
        group["worst_direction_relative_error"]=max(group["worst_direction_relative_error"],direction_relative)
        failed=(metric["zero_nonzero_errors"] or metric["nonzero_sign_errors"] or
                np.count_nonzero(nonzero & (~np.isfinite(relative)|(relative>=1.0))) or route!=true_route)
        if failed and first_failure is None:
            first_failure=dict(family=state.family,split=state.split,case_id=state.case_id,
                teacher_gradient=teacher_np.tolist(),learned_gradient=learned_np.tolist(),
                teacher_loss=float(teacher_loss),learned_loss=float(learned_loss),
                direction_relative_error=direction_relative)
        if (state.family=="sho" and state.case_id in hard_ids) or state.family=="ball":
            difficult[f"{state.family}_{state.case_id}"]=dict(
                teacher_gradient=teacher_np.tolist(),learned_gradient=learned_np.tolist(),
                sign_errors=metric["nonzero_sign_errors"],zero_nonzero_errors=metric["zero_nonzero_errors"],
                direction_relative_error=direction_relative)
        if (index+1)%20==0: print(f"gradient replay {index+1}/{len(states)}",flush=True)
    totals=dict(
        zero_nonzero_errors=sum(v["zero_nonzero_errors"] for v in groups.values()),
        nonzero_sign_errors=sum(v["nonzero_sign_errors"] for v in groups.values()),
        catastrophic_relative_errors=sum(v["catastrophic_relative_errors"] for v in groups.values()),
        tiny_nonzero_sign_errors=sum(v["tiny_nonzero_sign_errors"] for v in groups.values()),
        curvature_route_errors=sum(v["curvature_route_errors"] for v in groups.values()),
        worst_direction_relative_error=max(v["worst_direction_relative_error"] for v in groups.values()),
    )
    passed=(all(totals[k]==0 for k in ("zero_nonzero_errors","nonzero_sign_errors",
        "catastrophic_relative_errors","tiny_nonzero_sign_errors","curvature_route_errors")) and
        totals["worst_direction_relative_error"] <= 1e-10)
    return dict(passed=passed,totals=totals,groups=dict(groups),difficult_cases=difficult,
                first_failure=first_failure)


def main() -> None:
    global OPS
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    dependencies={str(path):sha256(path) for path in (ADD_PATH,MUL_PATH,SUB_PATH,DIV_ROUTER_PATH)}
    OPS=LearnedArithmetic()
    primitive_results,failed_primitive=primitive_screen()
    states=state_source.build_states()
    graph_inventory={}
    for family in ("sho","ball","synthetic"):
        state=next(item for item in states if item.family==family)
        x=state.theta.detach().clone().requires_grad_(True)
        graph_inventory[family]=operation_counts(state.objective(x))
    validation=None;checkpoint=None
    if failed_primitive is None:
        validation=full_validation(states)
        if validation["passed"]:
            checkpoint=OUT/"n3_gradient_operator.pt"
            torch.save(dict(status="FROZEN_CANDIDATE",parameters=0,
                implementation="custom reverse-mode assembly from locked arithmetic",
                dependencies=dependencies,primitive_results=primitive_results),checkpoint)
    status=("STOPPED_PRIMITIVE_FAILED" if failed_primitive else
            "PASSED" if validation and validation["passed"] else "STOPPED_GRADIENT_REPLAY_FAILED")
    report=dict(status=status,
        teacher_graph=dict(
            SHO="multihorizon recurrence p'=a*previous+b*current; weighted mean squared reconstruction errors",
            bouncing_ball="one-step active-event-branch K_ball; mean squared state-transition errors",
            execution_order=["mean adjoint seed","square adjoint","subtract/stack routing",
                "product/add/divide active-branch adjoints","broadcast sum-to-size","parameter gradient"],
            node_counts=graph_inventory),
        primitive_inventory=[
            dict(name="product adjoint",formula="dx=g*y; dy=g*x",implementation="locked multiplication",new_parameters=0),
            dict(name="addition adjoint",formula="dx=g; dy=g",implementation="structural routing",new_parameters=0),
            dict(name="subtraction adjoint",formula="dx=g; dy=0-g",implementation="locked subtraction",new_parameters=0),
            dict(name="square adjoint",formula="dx=(g*2)*x",implementation="locked multiplication",new_parameters=0),
            dict(name="division adjoint",formula="dx=g/y; dy=g*((-x/y)/y)",implementation="locked division/multiply/subtract",new_parameters=0),
            dict(name="mean adjoint",formula="expand(g/N)",implementation="locked division plus structural expand",new_parameters=0),
            dict(name="broadcast accumulation",formula="sum_to_size",implementation="deterministic structural bookkeeping",new_parameters=0),
            dict(name="event branch routing",formula="unchanged active teacher branch",implementation="deterministic scaffold",new_parameters=0)],
        primitive_results=primitive_results,first_failed_primitive=failed_primitive,
        systems=dict(states=len(states),dimensions="2 and synthetic 3-6"),
        validation=validation,
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        dependencies=dependencies,existing_checkpoints_modified=False,
        elapsed_seconds=time.time()-started)
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
