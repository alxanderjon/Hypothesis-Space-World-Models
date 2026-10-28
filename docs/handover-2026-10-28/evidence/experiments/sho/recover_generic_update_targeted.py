"""Established four-case screen, generic physical graph, unchanged recovery rules."""
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.sho import recover_affine_bias_targeted as parent
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.validate_generic_update_network_adapter import protected
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.expand_chronological_archive_handoff_dataset import ArchiveN1Runner
from experiments.geometry.initial_design_remainder_assembly import initial_design_with_remainder_assembly
from experiments.resume_hybrid_student_after_ckdtree import (
    Log, normalized_to_physical_assembly, run_n2_with_best_ever_assembly,
    HybridN3CurvatureRouteBinding, HybridN3DampingBinding, HybridN3DampedMatrixRHSBinding)
from experiments.sho.assemble_n3_learned_polisher import ControlAssembly
from experiments.n3_box_qp import dimension_general_box_qp_direction

n3_production=parent.n3_production
OUT=Path('outputs/sho/generic_update_network_targeted_recovery')
CASES=parent.CASES
FINAL_GATE=parent.FINAL_GATE


def clean(x):
    if is_dataclass(x): return clean(asdict(x))
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(float,np.floating)): return float(x) if math.isfinite(x) else str(x)
    if isinstance(x,(int,np.integer)): return int(x)
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    return x


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n'); tmp.replace(path)


class Objective:
    def __init__(self,adapter,trajectory):
        self.adapter=adapter; self.trajectory=trajectory; self.invocations=0
        self.capture=None
    def __call__(self,phi):
        self.invocations+=1
        with torch.no_grad(): value=float(self.adapter.objective(phi,self.trajectory))
        if self.capture is not None: self.capture.append({'phi':np.asarray(phi).copy(),'loss':value})
        return value


class GenericArchiveN1(ArchiveN1Runner):
    def __init__(self,objective,adapter,case_id,log):
        super().__init__(objective,adapter.lower,adapter.upper,case_id)
        self.log=log
        self._design,self.initial_design_trace=initial_design_with_remainder_assembly(self.dimension,return_trace=True)
        self.api=parent.post_baseline_composition(self.dimension,log)
        self.api['_greedy_proposal']=lambda p,f,c,it: parent.continuous.proposal(p,f,c,it,self.traces)
    def evaluate(self,unit,source):
        unit=np.asarray(unit,np.float64)
        phi,_=normalized_to_physical_assembly(unit[None,:],self.lower,self.upper,self.log)
        value=float(self.objective(phi[0]))
        if not math.isfinite(value): raise RuntimeError('nonfinite generic objective')
        self.points.append(unit.copy()); self.losses.append(value)
        best=int(np.argmin(self.losses))
        self.events.append(dict(evaluation=len(self.losses),iteration=self.iteration,source=source,
                               point=unit.tolist(),theta=phi[0].tolist(),loss=value,
                               best_index=best,best_loss=float(self.losses[best])))


class GenericN3:
    def __init__(self,adapter,trajectory,log,trace_path):
        self.adapter=adapter; self.trajectory=trajectory; self.log=log
        self.lower=torch.from_numpy(adapter.lower.copy()); self.upper=torch.from_numpy(adapter.upper.copy())
        self.curvature=HybridN3CurvatureRouteBinding()
        self.damping=HybridN3DampingBinding(self.curvature)
        self.matrix_rhs=HybridN3DampedMatrixRHSBinding()
        self.controls=ControlAssembly()
        _,self.direction_oracle,_=parent.staged._make_n3(lambda x:x.sum()*0.,adapter.lower,adapter.upper)
        self.objective_evaluations=0
        self.trace_path=trace_path; self.diagnostics=[]; self.last_printed=-1
        reference=OUT/'instrumentation_replay_reference'/trace_path.relative_to(OUT)
        self.reference=json.loads(reference.read_text()) if reference.exists() else None
        self.replay_compared=0
        self.stream=trace_path.with_suffix('.jsonl')
        self.stream.write_text('')
    def scalar_loss(self,theta):
        self.objective_evaluations+=1
        with torch.no_grad(): return float(self.adapter.objective(theta,self.trajectory))
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
            result = self.adapter.derivatives(theta.numpy(), self.trajectory)
            derivative_loss = float(result.loss)
            if np.float64(derivative_loss).view(np.uint64) != np.float64(current_loss).view(np.uint64):
                raise RuntimeError("generic forward and derivative objectives diverged")
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
            self.diagnostics.append({
                'accepted_before': accepted_steps, 'loss': current_loss,
                'gradient': gradient.numpy().tolist(), 'hessian': hessian.numpy().tolist(),
                'gradient_norm': gradient_norm, 'curvature_route': route['route'],
                'qp': qp, 'phi': theta.numpy().tolist(),
                'direction': None if direction is None else direction.numpy().tolist(),
            })
            if self.reference is not None:
                old_history=self.reference.get('history', self.reference.get('result',{}).get('history',[]))
                if accepted_steps < len(old_history):
                    expected=old_history[accepted_steps]
                    for key in ('theta','loss'):
                        before=np.asarray(expected[key],np.float64)
                        after=np.asarray(history[-1][key],np.float64)
                        if not np.array_equal(before.view(np.uint64),after.view(np.uint64)):
                            raise RuntimeError(f'instrumentation replay mismatch at step {accepted_steps}: {key}')
                    self.replay_compared+=1
            with self.stream.open('a') as handle:
                handle.write(json.dumps(clean({'history':history[-1], 'diagnostics':self.diagnostics[-1]}),allow_nan=False)+'\n')
            if accepted_steps % 50 == 0:
                atomic(self.trace_path, {'stage':'N3_RUNNING','history':history,
                                        'diagnostics':self.diagnostics,
                                        'instrumentation_replay_steps_exact':self.replay_compared})
            if accepted_steps % 25 == 0 and accepted_steps != self.last_printed:
                print(f'N3 {self.trace_path.name} steps={accepted_steps} loss={current_loss:.17g}', flush=True)
                self.last_printed = accepted_steps
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
                if not self.controls.in_bounds(candidate, self.lower.numpy(), self.upper.numpy()):
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


def rollout_diagnostic(adapter,trajectory,teacher,phi):
    predicted=adapter.rollout(phi,trajectory[-1],4096).numpy()
    # Teacher used only here, after optimization; never supplied to the hypothesis.
    truth=np.empty_like(predicted); truth[0]=trajectory[-1]
    matrix=np.asarray(teacher).reshape(2,2)
    for i in range(4096): truth[i+1]=matrix@truth[i]
    with np.errstate(over='ignore',invalid='ignore'):
        error=predicted-truth
        return dict(steps=4096,continuation_max_error=float(np.max(np.abs(error))),
                    continuation_mse=float(np.mean(error*error)),rollout_finite=bool(np.isfinite(predicted).all()),
                    bit_exact=bool(np.array_equal(predicted.view(np.uint64),truth.view(np.uint64))))


def run_case(name,corpus,log):
    old=json.loads((parent.OLD/name/'result.json').read_text())
    case_id=int(old['case_id']); trajectory=corpus[f'prefix_{case_id}'].copy()
    adapter=GenericSHOUpdateNetworkAdapter(); objective=Objective(adapter,trajectory)
    runner=GenericArchiveN1(objective,adapter,case_id,log)
    root=OUT/name; root.mkdir(parents=True,exist_ok=True)
    branches=[]; success=False; n2=n3=None
    atomic(root/'initial_design.json',dict(dimension=adapter.dimension,unit_points=runner._design,
         trace=runner.initial_design_trace,lower=adapter.lower,upper=adapter.upper))
    while not success:
        runner.advance_one(); n=len(runner.losses)
        atomic(root/'n1_history.json',runner.events)
        for i in [n-1]+list(range(n-1)):
            phi=adapter.lower+(adapter.upper-adapter.lower)*runner.points[i]
            lower,upper=adapter.archive_box(phi,n)
            stem=f'n{n:05d}_i{i:05d}'
            print(f'{name} {stem} start loss={runner.losses[i]:.17g}',flush=True)
            atomic(root/'progress.json',dict(stage='N2',evaluation_count=n,archive_index=i))
            objective.capture=[]
            n2=run_n2_with_best_ever_assembly(objective,phi,runner.losses[i],lower,upper,log)
            evaluations=objective.capture; objective.capture=None
            n2_record={k:v for k,v in n2.items() if k not in ('tracker','powell')}
            atomic(root/(stem+'_n2.json'),dict(box_lower=lower,box_upper=upper,result=n2_record,history=evaluations))
            print(f'{name} {stem} N2 nfev={n2["evaluations"]} loss={n2["loss"]:.17g}',flush=True)
            solver=GenericN3(adapter,trajectory,log,root/(stem+'_n3.json'))
            n3=solver.run(n2['theta'])
            success=bool(n3['loss']<FINAL_GATE)
            atomic(root/(stem+'_n3.json'),dict(stage='N3_COMPLETE',result=n3,diagnostics=solver.diagnostics))
            branches.append(dict(evaluation_count=n,archive_index=i,N2=n2_record,N3={k:v for k,v in n3.items() if k!='history'},valid_handoff=success))
            atomic(root/'branches.json',branches)
            print(f'{name} {stem} N3 {n3["stop_reason"]} steps={n3["accepted_steps"]} loss={n3["loss"]:.17g} handoff={success}',flush=True)
            if success: break
        if n>=5000 and not success: break
    rollout=rollout_diagnostic(adapter,trajectory,corpus['teachers'][case_id],n3['theta'])
    final_success=bool(success and rollout['rollout_finite'])
    result=dict(case=name,case_id=case_id,prefix_length=len(trajectory),dimension=adapter.dimension,
                first_evaluated_site=runner.events[0],N1_evaluations=len(runner.losses),
                branches_tested=len(branches),archive_handoff_index=i if success else None,
                handoff_is_newest=(i==n-1) if success else None,
                handoff_is_incumbent=(i==int(np.argmin(runner.losses))) if success else None,
                N2=n2_record,N3={k:v for k,v in n3.items() if k!='history'},recovered_phi=n3['theta'],
                final_transition_mse=n3['loss'],rollout=rollout,existing_recovery_status=final_success,
                failure_stage=None if final_success else ('rollout' if success else 'N1_RESOURCE_INCOMPLETE'),
                adapter_invocations=adapter.calls)
    atomic(root/'result.json',result)
    return result


def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    before=protected(); ckpt=hashes(); log=Log(); rows=[]; forbidden=Counter()
    # Audit completed before any objective/search call.
    atomic(OUT/'success_contract_audit.json',dict(
        required_parameter_specific_conditions=[],
        valid_handoff={'class':'BEHAVIORAL','condition':'completed N3 loss < 1e-20'},
        absorbing_success={'class':'BEHAVIORAL','condition':'break archive and N1 at first valid handoff'},
        N2_to_N3={'class':'NUMERICAL_CONTROL','condition':'always pass best-ever N2 endpoint, including maxfev exits'},
        final_success={'class':'BEHAVIORAL','condition':'valid handoff AND finite 4096-step rollout'},
        rollout_error={'class':'BEHAVIORAL','role':'offline raw diagnostic; no additional numerical cutoff'},
        parameter_error={'class':'PARAMETER_SPECIFIC','role':'offline affine-only diagnostic; not a required gate; omitted'},
        N3_stops={'class':'NUMERICAL_CONTROL','meaning':'termination alone is not recovered-law detection'},
        budgets={'N1_cap':5000,'N2_maxiter':200,'N2_maxfev':1500,'N2_xtol':1e-12,'N2_ftol':1e-15,'N3_accepted_step_cap':None},
        parent_source='experiments/sho/recover_affine_bias_targeted.py'))
    def profile(frame,event,arg):
        if event!='call': return
        obj=frame.f_locals.get('self')
        if obj is not None and type(obj).__name__ in ('AffineSHOUpdateAssembly','AffineN3','SHOExactSolutionDecoder','FullKSHOHessianAssembly'):
            forbidden[type(obj).__name__]+=1
            raise RuntimeError('forbidden affine physical path')
    report={}
    try:
        sys.setprofile(profile)
        corpus=np.load(parent.DATA)
        for name in CASES:
            row=run_case(name,corpus,log); rows.append(row)
            atomic(OUT/'partial_report.json',dict(cases=rows))
            if not row['existing_recovery_status']: break
        report=dict(status='GENERIC_UPDATE_NETWORK_RECOVERY_TARGETED' if len(rows)==4 and all(x['existing_recovery_status'] for x in rows) else 'FIRST_TARGETED_RECOVERY_FAILURE',cases=rows)
    except Exception as exc:
        report=dict(status='FAIL_CLOSED',cases=rows,exception_type=type(exc).__name__,exception=str(exc))
        raise
    finally:
        sys.setprofile(None)
        report.update(forbidden_affine_calls=dict(forbidden),forbidden_affine_call_total=sum(forbidden.values()),
                      operation_counts=dict(log.counts),operation_implementations=log.implementation,
                      protected_unchanged=before==protected(),checkpoint_count=len(ckpt),checkpoint_hashes_unchanged=ckpt==hashes(),
                      full_corpus_run=False,new_baseline_frozen=False,
                      adapter_sha256=hashlib.sha256(Path('experiments/sho/generic_update_network_adapter.py').read_bytes()).hexdigest(),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic(OUT/'report.json',report)
    print(json.dumps(clean(report),indent=2),flush=True)


if __name__=='__main__': main()
