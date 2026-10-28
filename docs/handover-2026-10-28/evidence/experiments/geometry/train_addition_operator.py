"""Synthetic addition regression followed by one read-only full-log check."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn

from experiments.geometry.train_exponent_log2_operator import ExponentLog2, sha, write_csv
from experiments.geometry.train_mantissa_log import MonotoneMantissaLog
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT/'addition_operator_smoke'
SEED = 27452026
EPS = np.finfo(np.float64).eps
EDGES = np.array([-1074,-900,-600,-300,-32,0,32,300,600,900,1024])


class Addition(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(.3*torch.randn(2,dtype=torch.float64))

    def forward(self, xy):
        return self.weights[0]*xy[...,0]+self.weights[1]*xy[...,1]


def predict(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(x,dtype=np.float64))).numpy()


def reserved(x):
    _, exp = np.frexp(abs(x))
    bands = np.searchsorted(EDGES[1:-1], exp-1, side='right')
    return ((bands[:,0]+2*bands[:,1])%4==1) & np.all(x!=0,axis=1)


def pairs(n, seed, heldout):
    rng = np.random.default_rng(seed); chunks=[]; count=0
    while count<n:
        x = np.ldexp(rng.uniform(1,2,(n,2)),rng.integers(-1000,1001,(n,2)))
        x *= rng.choice([-1.,1.],x.shape)
        x = x[reserved(x)==heldout]
        chunks.append(x); count += len(x)
    return np.concatenate(chunks)[:n]


def cancellations(n, seed):
    rng=np.random.default_rng(seed)
    x=np.ldexp(rng.uniform(1,2,n),rng.integers(-1074,1023,n))*rng.choice([-1.,1.],n)
    y=-x.copy()
    y[1::4]=-np.nextafter(x[1::4],np.inf)
    y[2::4]=-np.nextafter(x[2::4],-np.inf)
    y[3::4]=-x[3::4]*(1-np.ldexp(np.ones(len(x[3::4])),rng.integers(-52,-10,len(x[3::4]))))
    return np.column_stack((x,y))


def metric(y,p):
    err=p-y; maximum=float(np.max(abs(err))); nz=y!=0
    rel=err[nz]/y[nz]
    return dict(examples=len(y),rmse=float(maximum*np.sqrt(np.mean((err/maximum)**2))) if maximum else 0.,
                mae=float(np.mean(abs(err))),max_error=maximum,
                max_relative_error=float(np.max(abs(rel))) if len(rel) else 0.,
                nonfinite=int(np.sum(~np.isfinite(p))),exact_agreement=int(np.sum(y==p)),
                bit_exact_agreement=int(np.sum(y.view('u8')==p.view('u8'))),
                zero_failures=int(np.sum((y==0)&(p!=0))),
                sign_failures=int(np.sum((y!=0)&(np.sign(y)!=np.sign(p)))))


def assembled_check(adder):
    models=[]; manifests=[]
    for folder, cls in [('mantissa_log_operator_support_refine',MonotoneMantissaLog),('exponent_log2_operator_smoke',ExponentLog2)]:
        manifest=json.loads((ROOT/folder/'candidate_manifest.json').read_text())
        path=Path(manifest['checkpoint']); assert sha(path)==manifest['sha256']
        model=cls(); model.load_state_dict(torch.load(path,weights_only=False)['state_dict'])
        model.eval().requires_grad_(False); models.append(model); manifests.append(manifest)
    mantissa,exponent=models

    def loghat(v):
        v=np.asarray(v,dtype=np.float64); assert np.all(np.isfinite(v)&(v>0))
        m,e=np.frexp(v); m=2*m; e=e-1
        assert np.array_equal(np.ldexp(m,e),v)
        return predict(adder,np.column_stack((predict(mantissa,m),predict(exponent,e.astype(np.float64)))))

    rng=np.random.default_rng(SEED+100)
    raw=rng.integers(np.uint64(1),np.uint64(0x7ff0000000000000),size=32768,dtype=np.uint64).view(np.float64)
    powers=np.ldexp(np.ones(2098),np.arange(-1074,1024))
    offsets=np.arange(-16,17,dtype=np.int64)
    bits=powers.view(np.int64)[:,None]+offsets
    bits=bits[(bits>0)&(bits<0x7ff0000000000000)]
    boundary=bits.view(np.float64)
    old=np.load(ROOT/'log_operator_smoke/identity_inputs.npz')['monotonic_inputs']
    mantissa_ordered=np.load(ROOT/'mantissa_log_operator_support_refine/ordered_validation.npz')['m']
    groups=[('random_float64',raw),('all_binary_boundaries_adjacent_floats',boundary),
            ('original_monotonicity_pool',old),('compact_mantissa_ordered_pool',mantissa_ordered),
            ('magnitude_grid',np.logspace(-323,308,16384)),
            ('float64_extremes',np.array([np.nextafter(0.,1.),np.finfo(float).tiny,np.nextafter(1.,0.),1.,np.nextafter(1.,2.),np.finfo(float).max]))]
    rows=[]
    for name,v in groups:
        rows.append(dict(group=name,**metric(np.log(v),loghat(v))))
    ordered=np.unique(np.concatenate([v for _,v in groups])); teacher=np.log(ordered); output=loghat(ordered)
    rows.append(dict(group='all_unique_ordered',**metric(teacher,output)))
    dy=np.diff(output); dt=np.diff(teacher); adjacent=np.diff(ordered.view('u8'))==1
    _,exp=np.frexp(ordered); boundary_pair=np.diff(exp)!=0
    monotonic=dict(inputs=len(ordered),pairs=len(ordered)-1,reversals=int(np.sum(dy<0)),
                   teacher_strict_collapses=int(np.sum((dt>0)&(dy==0))),
                   adjacent_float_pairs=int(adjacent.sum()),adjacent_reversals=int(np.sum(adjacent&(dy<0))),
                   adjacent_teacher_strict_collapses=int(np.sum(adjacent&(dt>0)&(dy==0))),
                   binary_boundary_reversals=int(np.sum(boundary_pair&(dy<0))),
                   max_downward_step=float(-dy[dy<0].min()) if np.any(dy<0) else 0.)
    bad=[]
    for i in np.flatnonzero((dy<0)|((dt>0)&(dy==0))):
        bad.append(dict(kind='reversal' if dy[i]<0 else 'teacher_strict_collapse',
                        left=float(ordered[i]),right=float(ordered[i+1]),teacher_left=float(teacher[i]),teacher_right=float(teacher[i+1]),
                        learned_left=float(output[i]),learned_right=float(output[i+1]),adjacent_float=bool(adjacent[i]),binary_boundary=bool(boundary_pair[i])))
    xy=np.ldexp(rng.uniform(1,2,(16384,2)),rng.integers(-500,500,(16384,2)))
    lx,ly=loghat(xy[:,0]),loghat(xy[:,1]); product=xy[:,0]*xy[:,1]
    rp=loghat(product)-predict(adder,np.column_stack((lx,ly)))
    tp=np.log(product)-(np.log(xy[:,0])+np.log(xy[:,1]))
    rx=np.ldexp(rng.uniform(1,2,16384),rng.integers(-1000,1000,16384))
    rr=loghat(1/rx)+loghat(rx); tr=np.log(1/rx)+np.log(rx)
    identities=[]
    for name,r,t in [('product',rp,tp),('reciprocal',rr,tr)]:
        identities.append(dict(identity=name,examples=len(r),rmse=float(np.sqrt(np.mean(r*r))),max_error=float(abs(r).max()),
                               teacher_max_error=float(abs(t).max()),max_residual_difference_from_teacher=float(abs(r-t).max())))
    write_csv(OUT/'assembled_log_validation.csv',rows)
    write_csv(OUT/'assembled_log_ordering_cases.csv',bad or [dict(kind='no_cases')])
    write_csv(OUT/'assembled_log_identities.csv',identities)
    np.savez_compressed(OUT/'assembled_log_check_inputs.npz',ordered=ordered,teacher=teacher,predictions=output,product_pairs=xy,reciprocal_inputs=rx)
    result=dict(overall=rows[-1],monotonicity=monotonic,identities=identities,
                log_one=float(loghat(np.array([1.]))[0]),frozen_inputs=[dict(checkpoint=m['checkpoint'],sha256=m['sha256']) for m in manifests],
                decomposition='Exact deterministic np.frexp, shift to 1<=m<2; not learned or replaced',
                status='DIAGNOSTIC_ASSEMBLY_ONLY',no_production_replacement=True)
    (OUT/'assembled_log_results.json').write_text(json.dumps(result,indent=2))
    return result


def run():
    started=time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    canc=cancellations(2048,SEED+1); canc=canc[~reserved(canc)]
    rng=np.random.default_rng(SEED+2)
    vals=np.ldexp(rng.uniform(-1,1,128),rng.integers(-1000,1000,128))
    zeros=np.r_[np.column_stack((vals,np.zeros(len(vals)))),np.column_stack((np.zeros(len(vals)),vals))]
    x=np.concatenate((pairs(4096,SEED,False),canc,zeros))
    assert not reserved(x).any()
    y=x[:,0]+x[:,1]; scale=np.max(abs(x),axis=1); scale[scale==0]=1.
    model=Addition(); initial=model.weights.detach().numpy().copy(); assert np.all(initial!=1)
    history=[]; best=[np.inf,None]; tx=torch.from_numpy(x)

    def residual(w):
        with torch.no_grad(): model.weights.copy_(torch.from_numpy(w))
        error=(predict(model,x)-y)/scale
        loss=float(np.mean(error*error))
        history.append(dict(evaluation=len(history)+1,w1=float(w[0]),w2=float(w[1]),normalized_rmse=float(np.sqrt(loss))))
        if loss<best[0]: best[:]=[loss,w.copy()]
        return error

    jac=x/scale[:,None]
    opt=least_squares(residual,initial,jac=lambda _:jac,method='trf',max_nfev=30,ftol=1e-14,xtol=1e-14,gtol=1e-14)
    # Training-residual least-squares refinement; never assign known coefficients.
    for _ in range(8):
        if best[0]==0: break
        w=best[1].copy(); delta=np.linalg.lstsq(jac,-residual(w),rcond=None)[0]
        if np.array_equal(w,w+delta): break
        residual(w+delta)
    with torch.no_grad(): model.weights.copy_(torch.from_numpy(best[1]))
    config=dict(seed=SEED,dtype='float64',model='w1*x+w2*y, separately evaluated products then sum',bias=False,
                initial_weights=initial.tolist(),training_examples=len(x),magnitude_band_edges_base2=EDGES.tolist(),
                heldout_combinations='(x_band+2*y_band)%4==1, nonzero operands; excluded from every training family',
                optimizer='TRF continuous least squares, input-scale-normalized residuals and analytic Jacobian; at most 8 training-residual correction steps',
                target='float64 x+y',relative_precision_gate=32*EPS,no_coefficient_snapping=True)
    checkpoint=OUT/'addition_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,status='SAVED_BEFORE_EVALUATION',
                                          objective_calls=len(history),optimizer_message=opt.message))
    print(json.dumps(dict(event='trained_saved_before_evaluation',weights=best[1].tolist(),initial=initial.tolist(),sha256=digest)),flush=True)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(OUT/'training_pairs.npz',inputs=x,targets=y)
    write_csv(OUT/'training_history.csv',history)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
    extremes=np.array([-np.finfo(float).max,-np.finfo(float).tiny,-np.nextafter(0.,1.),-1.,-0.,0.,np.nextafter(0.,1.),np.finfo(float).tiny,1.,np.finfo(float).max])
    edge=np.array(np.meshgrid(extremes,extremes)).reshape(2,-1).T
    with np.errstate(over='ignore'): edge=edge[np.isfinite(edge[:,0]+edge[:,1])]
    zval=np.r_[vals,extremes]
    groups=[('seen_magnitude_combinations',pairs(8192,SEED+10,False)),('heldout_magnitude_combinations',pairs(16384,SEED+11,True)),
            ('near_and_exact_cancellation',cancellations(8192,SEED+12)),('extreme_finite_pairs',edge),
            ('zeros',np.r_[np.column_stack((zval,np.zeros(len(zval)))),np.column_stack((np.zeros(len(zval)),zval))])]
    rows=[]
    for name,xx in groups: rows.append(dict(grouping='family',group=name,**metric(xx[:,0]+xx[:,1],predict(model,xx))))
    vx=np.concatenate([xx for _,xx in groups]); vy=vx[:,0]+vx[:,1]; vp=predict(model,vx)
    rows.append(dict(grouping='all',group='all',**metric(vy,vp)))
    for s1 in (-1,0,1):
        for s2 in (-1,0,1):
            keep=(np.sign(vx[:,0])==s1)&(np.sign(vx[:,1])==s2)
            if keep.any(): rows.append(dict(grouping='sign',group=f'{s1},{s2}',**metric(vy[keep],vp[keep])))
    _,exps=np.frexp(np.max(abs(vx),axis=1)); exps=exps-1
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        keep=(exps>=lo)&(exps<hi)&np.any(vx!=0,axis=1)
        if keep.any(): rows.append(dict(grouping='max_operand_magnitude_base2',group=f'[{lo},{hi})',**metric(vy[keep],vp[keep])))
    comm=predict(model,vx[:,::-1].copy())
    anti=predict(model,np.column_stack((zval,-zval)))
    left=predict(model,np.column_stack((zval,np.zeros(len(zval)))))
    right=predict(model,np.column_stack((np.zeros(len(zval)),zval)))
    properties=dict(commutativity_failures=int(np.sum(comm!=vp)),left_zero_failures=int(np.sum(left!=zval)),
                    right_zero_failures=int(np.sum(right!=zval)),exact_cancellation_failures=int(np.sum(anti!=0)))
    total=metric(vy,vp)
    passed=bool(total['nonfinite']==0 and total['max_relative_error']<=32*EPS and total['zero_failures']==0
                and total['sign_failures']==0 and not any(properties.values()))
    write_csv(OUT/'addition_validation.csv',rows)
    (OUT/'addition_properties.json').write_text(json.dumps(properties,indent=2))
    np.savez_compressed(OUT/'addition_validation_pairs.npz',inputs=vx,targets=vy,predictions=vp)
    if passed:
        (OUT/'candidate_manifest.json').write_text(json.dumps(dict(operator='addition_operator',status='FROZEN_CANDIDATE',
            checkpoint=str(checkpoint),sha256=digest,module='experiments.geometry.train_addition_operator',model='Addition'),indent=2))
    assembly=assembled_check(model) if passed else None
    unchanged=all(sha(Path(p))==h for p,h in protected.items()); assert unchanged
    result=dict(operator='addition_operator',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',gate_passed=passed,
                learned_weights=best[1].tolist(),bias=0,initial_weights=initial.tolist(),objective_calls=len(history),validation=total,
                properties=properties,checkpoint=str(checkpoint),sha256=digest,assembly=assembly,
                protected_checkpoints=len(protected),all_protected_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-started)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    lines=[f"addition_operator = {result['status']}",f"Weights: {best[1].tolist()}; bias absent; random initial weights: {initial.tolist()}.",
           f'Validation: {json.dumps(total)}',f'Identities: {json.dumps(properties)}',f'Checkpoint: {checkpoint}; SHA256: {digest}.',
           f'Existing checkpoint hashes unchanged: {len(protected)}. Elapsed {result["elapsed_seconds"]:.3f} seconds.']
    if assembly:
        lines += ['One diagnostic log assembly only, with exact deterministic decomposition and unchanged frozen branches.',
                  f'Log errors: {json.dumps(assembly["overall"])}',f'Ordering: {json.dumps(assembly["monotonicity"])}',
                  f'Product/reciprocal identities: {json.dumps(assembly["identities"])}',
                  'No production full-log replacement. Any assembly ordering defects are reported without additional training.']
    (OUT/'addition_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
