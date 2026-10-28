"""Train only a positive-input power primitive; no quotient assembly."""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn

from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

REFINE = '--refine-training-residuals' in sys.argv
OUT = ROOT/('reciprocal_operator_training_refine' if REFINE else 'reciprocal_operator_smoke')
SEED = 27462026
EPS = np.finfo(np.float64).eps
HOLDOUT = [(-290.,-250.),(-180.,-160.),(-30.,-10.),(10.,30.),(160.,180.),(250.,290.)]


class PositivePower(nn.Module):
    """Raw x -> exp(log_scale)*pow(x,p). No reciprocal or division in inference."""
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(.3*torch.randn(2,dtype=torch.float64))

    def forward(self,x):
        p,log_scale = self.weights
        return torch.exp(log_scale)*torch.pow(x,p)

    def training_log_output(self,x):
        p,log_scale = self.weights
        return p*torch.log(x)+log_scale


def predict(model,x):
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(x,dtype=np.float64))).numpy()


def metrics(y,p):
    relative=(p-y)/y
    return dict(examples=len(y),relative_rmse=float(np.sqrt(np.mean(relative*relative))),
                relative_mae=float(np.mean(abs(relative))),max_relative_error=float(abs(relative).max()),
                exact_agreement=int(np.sum(y==p)),nonfinite=int(np.sum(~np.isfinite(p))),nonpositive=int(np.sum(p<=0)))


def run():
    started=time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    exp=rng.uniform(-300.,300.,10000); keep=np.ones(len(exp),bool)
    for lo,hi in HOLDOUT: keep &= ~((exp>=lo)&(exp<=hi))
    x=np.r_[np.power(10.,exp[keep][:4096]),np.exp(rng.uniform(-.01,.01,256)),np.ones(128)]
    for lo,hi in HOLDOUT: assert not np.any((np.log10(x)>=lo)&(np.log10(x)<=hi))
    target=np.divide(1.,x)  # Teacher labeling only.
    tx=torch.from_numpy(x); log_target=np.log(target)
    model=PositivePower(); initial=model.weights.detach().numpy().copy()
    assert initial[0]!=-1.
    history=[]; best=[np.inf,None]

    def set_weights(w):
        with torch.no_grad(): model.weights.copy_(torch.from_numpy(w))

    def log_residual(w):
        set_weights(w)
        with torch.no_grad(): error=model.training_log_output(tx).numpy()-log_target
        loss=float(np.mean(error*error))
        history.append(dict(phase='log_regression',evaluation=len(history)+1,p=float(w[0]),log_scale=float(w[1]),rmse=float(np.sqrt(loss))))
        if loss<best[0]: best[:]=[loss,w.copy()]
        return error

    with torch.no_grad(): logx=torch.log(tx).numpy()
    jac=np.column_stack((logx,np.ones(len(x))))
    resume_path=ROOT/'reciprocal_operator_smoke/reciprocal_before_evaluation.pt'
    if REFINE:
        saved=torch.load(resume_path,weights_only=False)
        archived=np.load(resume_path.parent/'training_data.npz')
        assert np.array_equal(x,archived['x']) and np.array_equal(target,archived['teacher'])
        model.load_state_dict(saved['state_dict'])
        log_residual(model.weights.detach().numpy().copy())
        fit=SimpleNamespace(message='Resumed original trained diagnostic; identical training data; no held-out supervision')
    else:
        fit=least_squares(log_residual,initial,jac=lambda _:jac,method='trf',max_nfev=40,ftol=1e-14,xtol=1e-14,gtol=1e-14)
    # Refine only against the same training labels using actual power inference.
    # No exponent/scale bounds, target assignments, or coefficient snapping.
    best_relative=[np.inf,None]

    def relative_residual(w):
        set_weights(w); pred=predict(model,x); error=(pred-target)/target
        loss=float(np.mean(error*error))
        history.append(dict(phase='relative_regression',evaluation=len(history)+1,p=float(w[0]),log_scale=float(w[1]),rmse=float(np.sqrt(loss))))
        if loss<best_relative[0]: best_relative[:]=[loss,w.copy()]
        return error

    def relative_jacobian(w):
        set_weights(w); ratio=predict(model,x)/target
        return ratio[:,None]*jac

    fit_relative=least_squares(relative_residual,best[1],jac=relative_jacobian,method='trf',max_nfev=30,
                               ftol=1e-14,xtol=1e-14,gtol=1e-14)
    if REFINE:
        # Generic Gauss-Newton training-residual correction below SciPy's
        # gradient stopping tolerance. Parameters remain wholly unconstrained.
        for _ in range(16):
            w=best_relative[1].copy(); loss=best_relative[0]
            residual=relative_residual(w); derivative=relative_jacobian(w)
            step=np.linalg.lstsq(derivative,-residual,rcond=None)[0]
            if np.array_equal(w,w+step): break
            relative_residual(w+step)
            if best_relative[0]>=loss: break
    set_weights(best_relative[1])
    p,b=map(float,best_relative[1]); scale=float(torch.exp(model.weights[1]).detach())
    config=dict(seed=SEED,dtype='float64',model='exp(log_scale)*pow(x,p)',inputs='raw positive finite x only',
                initial_p=float(initial[0]),initial_log_scale=float(initial[1]),initial_scale=float(np.exp(initial[1])),
                learned_parameters=['p','log_scale'],exponent_constraints=None,
                inference_operations=['exp(log_scale)','pow(x,p)','multiply'],
                training_examples=len(x),training_log10_range=[-300,300],heldout_log10_bands=HOLDOUT,
                teacher='float64 1/x, labeling and validation only',no_epsilon=True,
                optimizer='unconstrained TRF least squares: log-output regression, then relative training-residual refinement',
                relative_gate=16*EPS,identity_relative_gate=16*EPS,no_assembly=True)
    if REFINE:
        config.update(resumed_checkpoint=str(resume_path),resumed_sha256=sha(resume_path),
                      refinement='At most 16 unconstrained Gauss-Newton steps from original training residuals; stop at no progress')
    checkpoint=OUT/'reciprocal_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,status='SAVED_BEFORE_EVALUATION',
                                          training_history=history,optimizer_messages=[fit.message,fit_relative.message]))
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(OUT/'training_data.npz',x=x,teacher=target)
    write_csv(OUT/'training_history.csv',history)
    print(json.dumps(dict(event='saved_before_evaluation',p=p,scale=scale,log_scale=b,initial=initial.tolist(),sha256=digest)),flush=True)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']); model.eval().requires_grad_(False)

    families=[('ordinary_log_uniform',np.power(10.,rng.uniform(-300,300,8192))),
              ('near_one',np.exp(rng.uniform(-.01,.01,4096)))]
    for lo,hi in HOLDOUT: families.append((f'heldout_{lo:g}_{hi:g}',np.power(10.,rng.uniform(lo,hi,2048))))
    families += [('tiny_extrapolation',np.power(10.,rng.uniform(-308,-300,2048))),
                 ('large_extrapolation',np.power(10.,rng.uniform(300,308,2048)))]
    min_finite_input=np.nextafter(np.ldexp(np.float64(1),-1024),np.inf)
    assert np.isfinite(np.divide(1.,min_finite_input))
    # Covers extreme subnormal inputs with finite reciprocals, all binary joins,
    # near-one adjacent floats and the largest finite inputs.
    bases=np.ldexp(np.ones(2048),np.arange(-1024,1024))
    bits=bases.view(np.int64)[:,None]+np.arange(-16,17,dtype=np.int64)
    min_bits=int(min_finite_input.view(np.int64)); max_bits=int(np.float64(np.finfo(float).max).view(np.int64))
    boundaries=bits[(bits>=min_bits)&(bits<=max_bits)].view(np.float64)
    extreme_bits=np.r_[np.arange(min_bits,min_bits+65,dtype=np.int64),np.arange(max_bits-64,max_bits+1,dtype=np.int64)]
    families += [('binary_boundaries_adjacent',boundaries),('extreme_finite_cases',extreme_bits.view(np.float64))]
    raw=rng.integers(min_bits,max_bits+1,32768,dtype=np.int64).view(np.float64)
    families.append(('random_float64_bits',raw))
    vx=np.concatenate([v for _,v in families]); tags=np.concatenate([np.repeat(k,len(v)) for k,v in families])
    truth=np.divide(1.,vx); pred=predict(model,vx)
    assert np.all(np.isfinite(truth)&(truth>0))
    rows=[dict(grouping='all',group='all',**metrics(truth,pred))]
    for group,_ in families:
        mask=tags==group; rows.append(dict(grouping='family',group=group,**metrics(truth[mask],pred[mask])))
    logs=np.log10(vx)
    for low in range(-320,301,20):
        mask=(logs>=low)&(logs<low+20)
        if mask.any(): rows.append(dict(grouping='log10_input_band',group=f'[{low},{low+20})',**metrics(truth[mask],pred[mask])))
    write_csv(OUT/'reciprocal_validation.csv',rows)
    np.savez_compressed(OUT/'validation_data.npz',x=vx,teacher=truth,prediction=pred,family=tags)

    ordered=np.unique(np.r_[vx,np.logspace(np.log10(min_finite_input),np.log10(np.finfo(float).max)-.001,32768)])
    ordered=ordered[(ordered>=min_finite_input)&np.isfinite(ordered)]
    outputs=predict(model,ordered); changes=np.diff(outputs)
    reversal_ids=np.flatnonzero(changes>0)
    product=vx*pred; teacher_product=vx*truth
    with np.errstate(over='ignore'):
        teacher_double=np.divide(1.,truth)
    # Eligibility comes from the teacher. Never mask an unexpected learned failure.
    eligible=np.isfinite(teacher_double)&(teacher_double>0)
    double=predict(model,pred[eligible]); double_relative=(double-vx[eligible])/vx[eligible]
    teacher_double_relative=(teacher_double[eligible]-vx[eligible])/vx[eligible]
    properties=dict(monotonic_inputs=len(ordered),monotonicity_reversals=len(reversal_ids),
                    adjacent_float_pairs=int(np.sum(np.diff(ordered.view('u8'))==1)),
                    product_identity_relative_rmse=float(np.sqrt(np.mean((product-1)**2))),
                    product_identity_max_error=float(abs(product-1).max()),
                    teacher_product_identity_max_error=float(abs(teacher_product-1).max()),
                    double_reciprocal_examples=int(eligible.sum()),double_reciprocal_teacher_nonfinite_exclusions=int((~eligible).sum()),
                    double_reciprocal_relative_rmse=float(np.sqrt(np.mean(double_relative**2))),
                    double_reciprocal_max_relative_error=float(abs(double_relative).max()),
                    double_reciprocal_nonfinite=int(np.sum(~np.isfinite(double))),
                    teacher_double_reciprocal_max_relative_error=float(abs(teacher_double_relative).max()),
                    min_tested_input=float(vx.min()),max_tested_input=float(vx.max()))
    (OUT/'reciprocal_properties.json').write_text(json.dumps(properties,indent=2))
    np.savez_compressed(OUT/'identity_checks.npz',ordered_x=ordered,ordered_prediction=outputs,
                        x=vx,product=product,double_eligible=eligible,double_reciprocal=double)
    if len(reversal_ids):
        write_csv(OUT/'monotonicity_failures.csv',[dict(left=float(ordered[i]),right=float(ordered[i+1]),left_output=float(outputs[i]),right_output=float(outputs[i+1])) for i in reversal_ids])
    overall=rows[0]
    passed=bool(overall['nonfinite']==0 and overall['nonpositive']==0 and overall['max_relative_error']<=config['relative_gate']
                and len(reversal_ids)==0 and properties['product_identity_max_error']<=config['identity_relative_gate']
                and properties['double_reciprocal_nonfinite']==0 and properties['double_reciprocal_max_relative_error']<=config['identity_relative_gate'])
    unchanged=all(sha(type(OUT)(path))==h for path,h in protected.items()); assert unchanged
    result=dict(operator='reciprocal_operator',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',gate_passed=passed,
                learned_p=p,learned_s=scale,learned_log_scale=b,initial_weights=initial.tolist(),objective_evaluations=len(history),
                overall=overall,properties=properties,checkpoint=str(checkpoint),sha256=digest,
                protected_checkpoints=len(protected),all_protected_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-started)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:
        (OUT/'candidate_manifest.json').write_text(json.dumps(dict(operator='reciprocal_operator',status='FROZEN_CANDIDATE',
             checkpoint=str(checkpoint),sha256=digest,module='experiments.geometry.train_reciprocal_operator',model='PositivePower',
             learned_p=p,learned_s=scale,input_domain='positive finite float64 x for which float64 1/x remains finite',
             evaluation=str(OUT/'results.json')),indent=2))
    text=[f"reciprocal_operator = {result['status']}",f'Learned p={p:.17g}, s={scale:.17g}, log_scale={b:.17g}.',
          f'Random initial p={initial[0]:.17g}, log_scale={initial[1]:.17g}. No parameter constraints or snapping.',
          f'Overall: {json.dumps(overall)}',f'Properties: {json.dumps(properties)}',
          f'Checkpoint: {checkpoint}; SHA256: {digest}. Persisted before evaluation.',
          f'Protected checkpoints unchanged: {len(protected)}. Elapsed: {result["elapsed_seconds"]:.3f} seconds.',
          'Inference explicitly uses exp, pow, and multiplication only; no reciprocal or division call. Backend implementation of pow remains a library primitive.',
          'Double-reciprocal cases whose teacher result overflows are reported as excluded; all other cases enter the gate.',
          'No derivative quotient, root finding, minimization, recursion or search assembled or executed.']
    (OUT/'reciprocal_summary.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
