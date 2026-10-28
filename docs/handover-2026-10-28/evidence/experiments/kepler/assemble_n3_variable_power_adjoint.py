#!/usr/bin/env python3
"""Peel back the positive-base variable-power adjoint used by Kepler N3."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from experiments.kepler import deterministic_teacher as kepler
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import assemble_n3_hessian_operator as hessian
from experiments.sho import train_n3_curvature_router as curvature
from experiments.sho import train_n3_derivative_significance as prior_states

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/kepler/n3_variable_power_adjoint")
ROOT = Path("outputs/sho/stage1B/n3_learned_polisher")
DEPENDENCIES = (
    ROOT / "gradient_operator/n3_gradient_operator.pt",
    ROOT / "hessian_operator/n3_hessian_operator.pt",
    ROOT / "curvature_router/n3_curvature_router.pt",
    Path("outputs/sho/stage1B/sqrt_derivative_branch_assembly/sqrt_derivative_branch_validated.json"),
    Path("outputs/sho/stage1B/log_derivative_branch_assembly/log_derivative_branch_validated.json"),
)
SEED = 20_261_009


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LearnedLogAdjoint(torch.autograd.Function):
    """Exact Torch log value with its positive-domain adjoint assembled as g/x."""

    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return torch.log(value)

    @staticmethod
    def backward(ctx, incoming):
        (value,) = ctx.saved_tensors
        return hessian.divide(incoming, value)


def learned_log(value: torch.Tensor) -> torch.Tensor:
    return LearnedLogAdjoint.apply(value)


class LearnedVariablePowerAdjoint(torch.autograd.Function):
    """Torch pow forward with PowBackward1 reproduced from locked arithmetic."""

    @staticmethod
    def forward(ctx, base, exponent):
        result = torch.pow(base, exponent)
        # PowBackward1 saves and reuses the forward result on the exponent
        # path.  Recomputing base**exponent is mathematically equivalent but
        # changes the second-order reduction by one bit in ill-conditioned
        # Kepler states.
        ctx.save_for_backward(base, exponent, result)
        return result

    @staticmethod
    def backward(ctx, incoming):
        base, exponent, result = ctx.saved_tensors
        exponent_minus_one = hessian.subtract(exponent, 1.0)
        base_local = hessian.multiply(exponent, variable_power(base, exponent_minus_one))
        exponent_local = hessian.multiply(result, learned_log(base))
        # This order matches PowBackward1: incoming multiplies each local adjoint last.
        return (hessian.multiply(incoming, base_local),
                hessian.multiply(incoming, exponent_local))


def variable_power(base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    return LearnedVariablePowerAdjoint.apply(base, exponent)


def learned_acceleration(state: torch.Tensor, theta: torch.Tensor,
                         system: kepler.System = kepler.DEFAULT_SYSTEM) -> torch.Tensor:
    strength, exponent = theta.unbind()
    dx = hessian.subtract(state[4], state[0]); dy = hessian.subtract(state[5], state[1])
    radius_squared = hessian.add(hessian.multiply(dx, dx), hessian.multiply(dy, dy))
    # Fixed sqrt is an already validated path and is outside this peelback target.
    radius = torch.sqrt(radius_squared)
    denominator = variable_power(radius, exponent)
    common = hessian.divide(hessian.multiply(system.gravitational_constant, strength), denominator)
    scale_1 = hessian.multiply(common, system.mass_2)
    scale_2 = hessian.multiply(common, system.mass_1)
    # Preserve the teacher's unary-negation node; replacing it by 0-scale changes
    # the second-order reduction graph by one last bit on ill-conditioned states.
    negative_scale_2 = -scale_2
    return torch.stack((hessian.multiply(scale_1, dx), hessian.multiply(scale_1, dy),
                        hessian.multiply(negative_scale_2, dx),
                        hessian.multiply(negative_scale_2, dy)))


def learned_step(state: torch.Tensor, theta: torch.Tensor,
                 system: kepler.System = kepler.DEFAULT_SYSTEM) -> torch.Tensor:
    dt = torch.as_tensor(system.dt); half_dt = hessian.multiply(0.5, dt)
    acceleration = learned_acceleration(state, theta, system)
    v1hx = hessian.add(state[2], hessian.multiply(half_dt, acceleration[0]))
    v1hy = hessian.add(state[3], hessian.multiply(half_dt, acceleration[1]))
    v2hx = hessian.add(state[6], hessian.multiply(half_dt, acceleration[2]))
    v2hy = hessian.add(state[7], hessian.multiply(half_dt, acceleration[3]))
    x1 = hessian.add(state[0], hessian.multiply(dt, v1hx))
    y1 = hessian.add(state[1], hessian.multiply(dt, v1hy))
    x2 = hessian.add(state[4], hessian.multiply(dt, v2hx))
    y2 = hessian.add(state[5], hessian.multiply(dt, v2hy))
    drifted = torch.stack((x1, y1, v1hx, v1hy, x2, y2, v2hx, v2hy))
    next_acceleration = learned_acceleration(drifted, theta, system)
    return torch.stack((x1, y1,
        hessian.add(v1hx, hessian.multiply(half_dt, next_acceleration[0])),
        hessian.add(v1hy, hessian.multiply(half_dt, next_acceleration[1])),
        x2, y2,
        hessian.add(v2hx, hessian.multiply(half_dt, next_acceleration[2])),
        hessian.add(v2hy, hessian.multiply(half_dt, next_acceleration[3]))))


def power_adjoint_acceleration(state: torch.Tensor, theta: torch.Tensor,
                               system: kepler.System = kepler.DEFAULT_SYSTEM) -> torch.Tensor:
    """Exact teacher graph with only PowBackward1 replaced by the new adjoint."""
    strength, exponent = theta.unbind()
    dx = state[4] - state[0]; dy = state[5] - state[1]
    radius_squared = dx * dx + dy * dy
    radius = torch.sqrt(radius_squared)
    denominator = variable_power(radius, exponent)
    common = torch.as_tensor(system.gravitational_constant) * strength / denominator
    scale_1 = common * torch.as_tensor(system.mass_2)
    scale_2 = common * torch.as_tensor(system.mass_1)
    return torch.stack((scale_1*dx, scale_1*dy, -scale_2*dx, -scale_2*dy))


def power_adjoint_step(state: torch.Tensor, theta: torch.Tensor,
                       system: kepler.System = kepler.DEFAULT_SYSTEM) -> torch.Tensor:
    dt=torch.as_tensor(system.dt);half_dt=0.5*dt
    acceleration=power_adjoint_acceleration(state,theta,system)
    v1hx=state[2]+half_dt*acceleration[0];v1hy=state[3]+half_dt*acceleration[1]
    v2hx=state[6]+half_dt*acceleration[2];v2hy=state[7]+half_dt*acceleration[3]
    x1=state[0]+dt*v1hx;y1=state[1]+dt*v1hy;x2=state[4]+dt*v2hx;y2=state[5]+dt*v2hy
    drifted=torch.stack((x1,y1,v1hx,v1hy,x2,y2,v2hx,v2hy))
    next_acceleration=power_adjoint_acceleration(drifted,theta,system)
    return torch.stack((x1,y1,v1hx+half_dt*next_acceleration[0],v1hy+half_dt*next_acceleration[1],
                        x2,y2,v2hx+half_dt*next_acceleration[2],v2hy+half_dt*next_acceleration[3]))


def learned_objective(theta: torch.Tensor, trajectory: torch.Tensor,
                      prefix_states: int = kepler.PREFIX_STATES) -> torch.Tensor:
    prefix = trajectory[:prefix_states]
    errors=[]
    for index in range(prefix_states-1):
        adjoint_value=power_adjoint_step(prefix[index],theta)
        assembled_value=learned_step(prefix[index],theta)
        prediction=adjoint_value+(assembled_value-adjoint_value).detach()
        # Preserve the production objective graph exactly outside the isolated
        # variable-power path.  The straight-through value above still keeps
        # the locked arithmetic forward value, while native subtraction and
        # square retain Torch's exact first/second-order evaluation graph.
        errors.append(prediction-prefix[index+1])
    stacked = torch.stack(errors)
    # Preserve Tensor.square's second-order node rather than substituting x*x.
    return torch.mean(stacked.square())


def scalar_derivatives(function, base: torch.Tensor, exponent: torch.Tensor,
                       incoming: torch.Tensor) -> tuple[torch.Tensor, ...]:
    x = base.detach().clone().requires_grad_(True)
    p = exponent.detach().clone().requires_grad_(True)
    value = function(x, p)
    scalar = torch.sum(incoming * value)
    dx, dp = torch.autograd.grad(scalar, (x, p), create_graph=True)
    dxx, dxp = torch.autograd.grad(dx.sum(), (x, p), retain_graph=True)
    dpx, dpp = torch.autograd.grad(dp.sum(), (x, p))
    return value.detach(), dx.detach(), dp.detach(), dxx.detach(), dxp.detach(), dpx.detach(), dpp.detach()


def metric(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    left = expected.numpy().reshape(-1); right = actual.numpy().reshape(-1)
    exact = left.view(np.uint64) == right.view(np.uint64)
    nonzero = left != 0.0
    relative = np.abs(right-left) / np.maximum(np.abs(left), np.finfo(np.float64).tiny)
    return dict(values=len(left), bit_exact=int(exact.sum()), numeric_exact=int((left==right).sum()),
                signed_zero_mismatches=int(np.sum((left==0.0)&(right==0.0)&(np.signbit(left)!=np.signbit(right)))),
                zero_nonzero_errors=int(np.sum((left==0.0)!=(right==0.0))),
                nonzero_sign_errors=int(np.sum(nonzero & (np.signbit(left)!=np.signbit(right)))),
                max_absolute_error=float(np.max(np.abs(right-left))),
                max_relative_error=float(np.max(relative)))


def primitive_screen() -> dict:
    rng = np.random.default_rng(SEED)
    generic_x = np.exp2(rng.uniform(-100.0, 100.0, 8192))
    generic_p = rng.uniform(2.0, 4.0, len(generic_x))
    incoming = rng.normal(size=len(generic_x))
    incoming[::97] = 0.0
    special = []
    for center in (0.25, 0.5, 1.0, 2.0, 4.0):
        for x in (np.nextafter(center, 0.0), center, np.nextafter(center, math.inf)):
            for p in (2.0, np.nextafter(2.0, math.inf), 3.0,
                      np.nextafter(4.0, 0.0), 4.0):
                for upstream in (-1.0, -0.0, 0.0, 1.0):
                    special.append((x,p,upstream))
    special = np.asarray(special, dtype=np.float64)
    x = torch.from_numpy(np.concatenate((generic_x, special[:,0])))
    p = torch.from_numpy(np.concatenate((generic_p, special[:,1])))
    upstream = torch.from_numpy(np.concatenate((incoming, special[:,2])))
    teacher = scalar_derivatives(torch.pow, x, p, upstream)
    learned = scalar_derivatives(variable_power, x, p, upstream)
    names = ("value","d_base","d_exponent","d_base_base","d_base_exponent",
             "d_exponent_base","d_exponent_exponent")
    results = {name:metric(expected,actual) for name,expected,actual in zip(names,teacher,learned)}
    results["cases"] = len(x)
    results["all_bit_exact"] = all(results[name]["bit_exact"] == results[name]["values"] for name in names)
    results["gate_passed"] = all(
        results[name]["numeric_exact"] == results[name]["values"] and
        results[name]["zero_nonzero_errors"] == 0 and
        results[name]["nonzero_sign_errors"] == 0 for name in names)
    results["formula"] = dict(
        d_base="incoming * (p * x**(p-1))",
        d_exponent="incoming * (x**p * log(x))",
        d_base_base="incoming * p*(p-1)*x**(p-2)",
        d_base_exponent="incoming * x**(p-1)*(1+p*log(x))",
        d_exponent_exponent="incoming * x**p*log(x)**2")
    return results


def graph_counts(outputs: list[torch.Tensor]) -> dict[str,int]:
    queue=deque(output.grad_fn for output in outputs); seen=set(); counts=Counter()
    while queue:
        node=queue.popleft()
        if node is None or node in seen: continue
        seen.add(node); counts[type(node).__name__]+=1
        queue.extend(child for child,_ in node.next_functions if child is not None)
    return dict(counts)


def objective_derivatives(function, theta: torch.Tensor):
    x=theta.detach().clone().requires_grad_(True); value=function(x)
    first=torch.autograd.grad(value,x,create_graph=True)[0]
    rows=[torch.autograd.grad(first[i],x,retain_graph=True)[0] for i in range(len(x))]
    raw=torch.stack(rows); symmetric=0.5*(raw+raw.T)
    return value.detach(),first.detach(),raw.detach(),symmetric.detach(),graph_counts([value,first[0],first[1]])


def state_pool() -> list[dict]:
    rng=np.random.default_rng(SEED+1)
    geometries=((0.10,0.8,0.0,"train"),(0.35,1.0,0.3,"train"),
                (0.60,1.2,1.1,"heldout"),(0.25,1.4,2.4,"fresh"))
    fixed=(kepler.TEACHER_THETA,
           np.asarray((np.nextafter(1.0,math.inf),np.nextafter(3.0,0.0))),
           np.asarray((0.25,2.0)),np.asarray((2.0,4.0)),
           np.asarray((0.5,3.8)),np.asarray((1.75,2.2)))
    states=[]
    for geometry_id,(e,rp,angle,split) in enumerate(geometries):
        trajectory=kepler.numpy_rollout(kepler.initial_state(e,rp,angle))
        points=list(fixed)+[rng.uniform(kepler.LOWER,kepler.UPPER) for _ in range(2)]
        for point_id,point in enumerate(points):
            states.append(dict(case_id=f"g{geometry_id}_p{point_id}",split=split,
                trajectory=torch.from_numpy(trajectory),theta=torch.from_numpy(np.asarray(point,dtype=np.float64))))
    return states


def compare_kepler(states: list[dict]) -> dict:
    router=curvature.CurvatureRouter(); groups=defaultdict(lambda:Counter(states=0,gradient_values=0,
        hessian_values=0,gradient_bit_exact=0,hessian_bit_exact=0,gradient_sign_errors=0,
        hessian_sign_errors=0,zero_nonzero_errors=0,curvature_errors=0,route_errors=0,
        directions=0,direction_failures=0))
    worst_gradient=worst_hessian=worst_direction=0.0; rows=[]
    for index,state in enumerate(states):
        trajectory=state["trajectory"];theta=state["theta"]
        teacher=objective_derivatives(lambda value,p=trajectory:kepler.prefix_loss_torch(value,p),theta)
        learned=objective_derivatives(lambda value,p=trajectory:learned_objective(value,p),theta)
        tv,tg,_,th,tgraph=teacher;lv,lg,_,lh,lgraph=learned
        gm=metric(tg,lg);hm=metric(th,lh);group=groups[state["split"]]
        group.update(states=1,gradient_values=gm["values"],hessian_values=hm["values"],
                     gradient_bit_exact=gm["bit_exact"],hessian_bit_exact=hm["bit_exact"],
                     gradient_sign_errors=gm["nonzero_sign_errors"],hessian_sign_errors=hm["nonzero_sign_errors"],
                     zero_nonzero_errors=gm["zero_nonzero_errors"]+hm["zero_nonzero_errors"])
        ta=curvature.teacher_analysis(tg,th);la=router.analyze(lg,lh)
        min_teacher=float(torch.linalg.eigvalsh(th).min());min_learned=float(torch.linalg.eigvalsh(lh).min())
        curvature_error=int((min_teacher>0)!=(min_learned>0));route_error=int(ta["status"]!=la["status"])
        group.update(curvature_errors=curvature_error,route_errors=route_error)
        td=curvature.direction_with_route(tg,th,ta)[0];ld=curvature.direction_with_route(lg,lh,la)[0]
        direction_relative=0.0
        if td is not None or ld is not None:
            group.update(directions=1)
            if td is None or ld is None: direction_relative=math.inf;group.update(direction_failures=1)
            else:
                direction_relative=float(torch.linalg.vector_norm(ld-td)/max(float(torch.linalg.vector_norm(td)),np.finfo(float).tiny))
                group.update(direction_failures=int(not math.isfinite(direction_relative) or direction_relative>1e-10))
        grad_relative=float(torch.linalg.vector_norm(lg-tg)/max(float(torch.linalg.vector_norm(tg)),np.finfo(float).tiny))
        hess_relative=float(torch.linalg.vector_norm(lh-th)/max(float(torch.linalg.vector_norm(th)),np.finfo(float).tiny))
        worst_gradient=max(worst_gradient,grad_relative);worst_hessian=max(worst_hessian,hess_relative);worst_direction=max(worst_direction,direction_relative)
        rows.append(dict(case_id=state["case_id"],split=state["split"],theta=theta.tolist(),
            teacher_loss=float(tv),learned_loss=float(lv),gradient=gm,hessian=hm,
            curvature_error=curvature_error,route_error=route_error,direction_relative_error=direction_relative,
            teacher_graph=tgraph,learned_graph=lgraph))
        print(f"Kepler derivative replay {index+1}/{len(states)}",flush=True)
    totals=dict(states=len(states),gradient_sign_errors=sum(v["gradient_sign_errors"] for v in groups.values()),
        hessian_sign_errors=sum(v["hessian_sign_errors"] for v in groups.values()),
        zero_nonzero_errors=sum(v["zero_nonzero_errors"] for v in groups.values()),
        curvature_errors=sum(v["curvature_errors"] for v in groups.values()),
        route_errors=sum(v["route_errors"] for v in groups.values()),
        direction_failures=sum(v["direction_failures"] for v in groups.values()),
        worst_gradient_relative_error=worst_gradient,worst_hessian_relative_error=worst_hessian,
        worst_newton_direction_relative_error=worst_direction)
    passed=(all(totals[k]==0 for k in ("gradient_sign_errors","hessian_sign_errors","zero_nonzero_errors",
        "curvature_errors","route_errors","direction_failures")) and worst_direction<=1e-10)
    return dict(passed=passed,totals=totals,groups={k:dict(v) for k,v in groups.items()},rows=rows)


def regression_replay() -> dict:
    states=[state for state in prior_states.build_states() if state.family in ("sho","ball")]
    result=hessian.full_validation(states)
    return dict(passed=result["passed"],states=len(states),totals=result["totals"],groups=result["groups"])


def main() -> None:
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    before={str(path):sha256(path) for path in DEPENDENCIES}
    gradient.OPS=gradient.LearnedArithmetic();hessian.activate_second_order_gradient()
    primitive=primitive_screen()
    if not primitive["gate_passed"]:
        report=dict(status="STOPPED_PRIMITIVE_FAILURE",primitive=primitive,
                    dependencies_unchanged=before=={str(p):sha256(p) for p in DEPENDENCIES})
        (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2));return
    kepler_validation=compare_kepler(state_pool())
    regression=regression_replay() if kepler_validation["passed"] else None
    after={str(path):sha256(path) for path in DEPENDENCIES}
    passed=kepler_validation["passed"] and regression is not None and regression["passed"] and before==after
    checkpoint=None
    inventory=dict(
        torch_graph=("PowBackward1 saves base, exponent, result; base path uses WhereBackward0->incoming*(p*pow(base,p-1)); exponent path uses incoming*(result*log(base))",),
        positive_domain="Kepler radius is strictly positive; PowBackward1 zero-base guards never activate.",
        first_order=("d/dx x^p = p*x^(p-1)","d/dp x^p = x^p*log(x)"),
        second_order=("dxx=p*(p-1)*x^(p-2)","dxp=x^(p-1)*(1+p*log(x))","dpp=x^p*log(x)^2"),
        reused=("locked add/subtract/multiply/divide assemblies","validated reciprocal/log derivative semantics","validated fixed-sqrt derivative path","structural indexing/broadcast/reduction"),
        new=("positive-base variable-power custom adjoint assembly",),new_parameters=0,
        deterministic_forward_values=("torch.pow", "torch.log", "torch.sqrt"))
    if passed:
        checkpoint=OUT/"n3_variable_power_adjoint.pt"
        torch.save(dict(status="FROZEN_CANDIDATE",parameters=0,module=__name__,
                        entrypoints=("variable_power","learned_objective"),inventory=inventory,
                        dependencies=before,primitive=primitive,
                        kepler_totals=kepler_validation["totals"],regression_totals=regression["totals"]),checkpoint)
    report=dict(status="PASSED" if passed else "STOPPED_VALIDATION_FAILURE",elapsed_seconds=time.time()-started,
        inventory=inventory,primitive=primitive,kepler=kepler_validation,regression=regression,
        checkpoint=None if checkpoint is None else dict(path=str(checkpoint),sha256=sha256(checkpoint)),
        dependencies_before=before,dependencies_after=after,dependencies_unchanged=before==after,
        existing_n3_checkpoints_modified=False,full_kepler_transfer_run=False)
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="kepler"},indent=2))
    print(json.dumps(dict(kepler_passed=kepler_validation["passed"],kepler_totals=kepler_validation["totals"],
                          kepler_groups=kepler_validation["groups"]),indent=2))


if __name__=="__main__":
    main()
