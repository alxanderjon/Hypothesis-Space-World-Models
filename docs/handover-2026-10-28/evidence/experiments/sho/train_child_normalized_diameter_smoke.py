"""Supervised child q=diagonal/n^(1/d); all model powers learned.

Only synthetic geometry. No frozen operator is used as a label or retrained.
"""
import copy
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import (
    generate, inputs, persist_data, metrics, write_csv, sha,
    TRAIN_DIMS, TRAIN_BUDGETS, INTERPOLATED_BUDGETS, EXTRAPOLATED_BUDGETS, ASPECT_HOLDOUT)

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'child_normalized_diameter_smoke'
SEED=27202026
EPS=np.finfo(np.float64).eps
GATES=dict(relative_error=256*EPS,d2_reference_relative_error=256*EPS,
           scale_equivariance_relative_error=512*EPS,permutation_bit_mismatches=0)


class ChildNormalizedDiameter(nn.Module):
    """Learned positive power-sum and count/dimension scaling.

    alpha=exp(raw_alpha)
    log_q = beta * log(sum_j exp(alpha*log(side_j)))
            + gamma*exp(delta*log(d))*log(n) + offset
    q=exp(log_q)

    No norm, count-root, or final teacher coefficient is an input/constant.
    Sorting is deterministic coordinate bookkeeping before summation.
    """
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(.2*torch.randn(5,dtype=torch.float64))

    def log_forward(self,log_sides,mask,log_n,d,weights=None):
        raw_alpha,beta,gamma,delta,offset=self.weights if weights is None else weights
        terms=(torch.exp(raw_alpha)*log_sides).masked_fill(~mask,-torch.inf)
        ordered=torch.sort(terms,dim=1).values
        log_power_sum=torch.logsumexp(ordered,dim=1)
        return beta*log_power_sum+gamma*torch.exp(delta*torch.log(d))*log_n+offset

    def forward(self,log_sides,mask,log_n,d):
        return torch.exp(self.log_forward(log_sides,mask,log_n,d))


def target(sides,n,d):
    assert n>=1 and np.all(np.isfinite(sides)) and np.all(sides>0)
    diagonal=np.sqrt(np.add.reduce(sides*sides))
    count_root=np.power(np.float64(n),np.float64(1)/np.float64(d))
    q=diagonal/count_root
    assert np.isfinite(q) and q>0
    return float(q)


def boxes(d,count,seed,group,force_count=None):
    # Reuse only the synthetic box sampler, never the spacing network/weights.
    records=generate(d,count,seed,group)
    for r in records:
        if force_count is not None:r['n']=int(force_count)
        r['target']=target(r['sides'],r['n'],r['d'])
        r.pop('volume',None)
        if force_count is not None:r['group']='small_count_grid'
    return records


def count_bucket(n):
    return '1' if n==1 else '2-5' if n<=5 else '6-20' if n<=20 else '21-1000' if n<=1000 else '1001-32768' if n<=32768 else '>32768'


def aspect_bucket(a):
    return '[1,2)' if a<2 else '[2,10)' if a<10 else '[10,100)' if a<100 else '[100,10000)' if a<10000 else '>=10000'


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for folder in ('characteristic_spacing_smoke','geometry_allocation_curriculum',
        'recursive_region_splitter_poc','split_quality_smoke') for p in (ROOT/folder).glob('*.pt')}
    config=dict(seed=SEED,dtype='float64',train_dimensions=TRAIN_DIMS,heldout_dimensions=[2,4],
        train_counts=TRAIN_BUDGETS,unseen_interpolation_counts=INTERPOLATED_BUDGETS,
        unseen_extrapolation_counts=EXTRAPOLATED_BUDGETS,aspect_holdout_bands=ASPECT_HOLDOUT,
        geometric_mean_scale_train=[1e-3,1e3],scale_extrapolation=[[1e-12,1e-8],[1e8,1e12]],
        aspect_extrapolation=[1e3,1e6],gates=GATES,protected_hashes=protected,
        model='exp(beta*logsumexp(exp(raw_alpha)*individual_log_sides) + gamma*exp(delta*log(d))*log(n) + offset)',
        inputs='individual log side lengths, mask, log integer count, dimension; no diagonal/count-root/q',
        learned_parameters=['raw_alpha','beta','gamma','delta','offset'],
        initialization='all five independent normal(0,0.2), no teacher coefficients',
        coordinate_permutation='sort per-coordinate powered log terms before stable logsumexp',
        target='sqrt(add.reduce(sides*sides)) / np.power(float64(n),float64(1)/float64(d))',
        objective='continuous squared log-output residual to log(exact teacher q), equal example weighting',
        optimizer='TRF least_squares with exact forward-mode autodiff Jacobian; max_nfev=160; ftol/xtol/gtol=1e-14',
        checkpoint_selection='best training objective; persist before held-out evaluation',
        structure='symmetric learned power-sum, learned outer power and count/dimension exponents; positive exponential output',
        no_epsilon=True,recursion=False,search=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    train=[]
    for d in TRAIN_DIMS:train+=boxes(d,1536,SEED+d,'train')
    persist_data(OUT/'training_boxes.npz',train)
    tx=inputs(train);log_targets=np.log(np.array([r['target'] for r in train]))
    model=ChildNormalizedDiameter();initial=model.weights.detach().numpy().copy()
    history=[];best=[np.inf,None]
    def residual(w):
        with torch.no_grad():z=model.log_forward(*tx,weights=torch.from_numpy(w)).numpy()
        e=z-log_targets;mse=float(np.mean(e*e))
        history.append(dict(evaluation=len(history)+1,training_log_rmse=math.sqrt(mse),weights=json.dumps(w.tolist())))
        if mse<best[0]:best[:]=[mse,w.copy()]
        return e
    def jacobian(w):
        return torch.func.jacfwd(lambda v:model.log_forward(*tx,weights=v))(torch.from_numpy(w)).detach().numpy()
    opt=least_squares(residual,initial,jac=jacobian,method='trf',max_nfev=160,ftol=1e-14,xtol=1e-14,gtol=1e-14)
    with torch.no_grad():model.weights.copy_(torch.from_numpy(best[1]))
    checkpoint=OUT/'child_normalized_diameter_before_evaluation.pt'
    checkpoint_hash=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,
        initial_weights=initial.tolist(),training_log_rmse=math.sqrt(best[0]),objective_calls=len(history),
        optimizer_status=int(opt.status),optimizer_message=opt.message,status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'training_history.csv',history)
    print(json.dumps(dict(event='trained_and_saved',initial=initial.tolist(),learned=model.weights.detach().tolist(),
        learned_side_power=float(model.weights[0].detach().exp()),training_log_rmse=math.sqrt(best[0]),calls=len(history))),flush=True)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval()
    validation=[]
    families=('combinations_and_count_interpolation','aspect_interpolation','aspect_extrapolation','budget_extrapolation','scale_extrapolation')
    for d in range(1,6):
        for g,group in enumerate(families):validation+=boxes(d,256,SEED+100+d*10+g,group)
        for n in range(1,21):validation+=boxes(d,16,SEED+1000+100*d+n,'combinations',force_count=n)
    persist_data(OUT/'heldout_boxes.npz',validation)
    vx=inputs(validation)
    with torch.no_grad():pred=model(*vx).numpy()
    truth=np.array([r['target'] for r in validation]);dims=np.array([r['d'] for r in validation])
    rows=[]
    for d in ('all',1,2,3,4,5):
        for group in ('all',)+families+('small_count_grid',):
            mask=np.ones(len(validation),dtype=bool)
            if d!='all':mask &=dims==d
            if group!='all':mask &=np.array([r['group']==group for r in validation])
            rows.append(dict(record_type='teacher_metrics',dimension=d,group=group,
                             dimension_heldout=d in (2,4),**metrics(truth[mask],pred[mask])))
    for name,fn in [('count_range',lambda r:count_bucket(r['n'])),('aspect_ratio',lambda r:aspect_bucket(r['aspect']))]:
        labels=np.array([fn(r) for r in validation])
        for value in sorted(set(labels)):
            for d in ('all',1,2,3,4,5):
                mask=labels==value
                if d!='all':mask &=dims==d
                if mask.any():rows.append(dict(record_type=name,group=value,dimension=d,**metrics(truth[mask],pred[mask])))
    d2=np.flatnonzero(dims==2)
    audited=np.array([np.linalg.norm(validation[i]['sides'][None,:],axis=1)[0]/np.sqrt(np.int64(validation[i]['n'])) for i in d2])
    d2_metric=metrics(audited,pred[d2]);d2_teacher_diff=metrics(audited,truth[d2])
    rows.append(dict(record_type='d2_audited_norm_over_sqrt',dimension=2,group='all',**d2_metric))
    rows.append(dict(record_type='d2_general_teacher_vs_audited',dimension=2,group='all',**d2_teacher_diff))
    # Exact same model and count; changed side scale only, no teacher label involved.
    base=validation[::13]
    with torch.no_grad():base_pred=model(*inputs(base)).numpy()
    equiv_max=0.
    for c in (1e-6,.137,3.,1e6):
        scaled=[dict(r,sides=r['sides']*c) for r in base]
        with torch.no_grad():scaled_pred=model(*inputs(scaled)).numpy()
        expected=c*base_pred
        for d in range(1,6):
            mask=np.array([r['d']==d for r in base]);m=metrics(expected[mask],scaled_pred[mask])
            equiv_max=max(equiv_max,m['max_relative_error'])
            rows.append(dict(record_type='scale_equivariance',dimension=d,c=c,**m))
    rng=np.random.default_rng(SEED+9000)
    permuted=[dict(r,sides=rng.permutation(r['sides'])) for r in validation]
    with torch.no_grad():perm_pred=model(*inputs(permuted)).numpy()
    perm_mismatches=int(np.sum(perm_pred.view(np.uint64)!=pred.view(np.uint64)))
    perm_max=float(np.max(abs((perm_pred-pred)/pred)))
    for d in range(1,6):
        mask=dims==d
        rows.append(dict(record_type='random_coordinate_permutation',dimension=d,
            bit_mismatches=int(np.sum(perm_pred[mask].view(np.uint64)!=pred[mask].view(np.uint64))),**metrics(pred[mask],perm_pred[mask])))
    # Exhaust all d! coordinate orders for 16 held-out geometries in each d.
    exhaustive_records=[];exhaustive_original=[]
    for d in range(1,6):
        for index in np.flatnonzero(dims==d)[:16]:
            for order in itertools.permutations(range(d)):
                exhaustive_records.append(dict(validation[index],sides=validation[index]['sides'][list(order)]))
                exhaustive_original.append(pred[index])
    with torch.no_grad():ep=model(*inputs(exhaustive_records)).numpy()
    eo=np.array(exhaustive_original)
    exhaustive_mismatches=int(np.sum(ep.view(np.uint64)!=eo.view(np.uint64)))
    rows.append(dict(record_type='exhaustive_coordinate_permutation',dimension='all',bit_mismatches=exhaustive_mismatches,**metrics(eo,ep)))
    cases=[]
    sqrt_lookup={i:audited[j] for j,i in enumerate(d2)}
    for i,r in enumerate(validation):
        row=dict(case=i,dimension=r['d'],group=r['group'],n=r['n'],aspect=r['aspect'],scale=r['scale'],
            teacher=r['target'],learned=float(pred[i]),absolute_error=float(pred[i]-truth[i]),
            relative_error=float((pred[i]-truth[i])/truth[i]),permutation_bit_exact=bool(pred[i].view(np.uint64)==perm_pred[i].view(np.uint64)))
        if i in sqrt_lookup:row.update(audited_d2_teacher=float(sqrt_lookup[i]),relative_error_vs_audited_d2=float((pred[i]-sqrt_lookup[i])/sqrt_lookup[i]))
        cases.append(row)
    overall=metrics(truth,pred)
    passed=(np.isfinite(pred).all() and np.all(pred>0) and overall['max_relative_error']<=GATES['relative_error']
        and d2_metric['max_relative_error']<=GATES['d2_reference_relative_error']
        and equiv_max<=GATES['scale_equivariance_relative_error'] and perm_mismatches==0 and exhaustive_mismatches==0)
    write_csv(OUT/'child_normalized_diameter_smoke.csv',rows)
    write_csv(OUT/'child_normalized_diameter_cases.csv',cases)
    assert all(sha(p)==h for p,h in protected.items())
    w=model.weights.detach().tolist()
    result=dict(status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',passed=bool(passed),
        checkpoint=str(checkpoint),sha256=checkpoint_hash,initial_weights=initial.tolist(),learned_raw_weights=w,
        learned_effective_coefficients=dict(side_power=math.exp(w[0]),outer_power=w[1],count_coefficient=w[2],dimension_power=w[3],log_offset=w[4]),
        training_examples=len(train),heldout_examples=len(validation),training_objective_calls=len(history),
        overall=overall,d2_vs_audited=d2_metric,d2_teacher_rounding_difference=d2_teacher_diff,
        scale_equivariance_max_relative_error=equiv_max,scale_equivariance_cases=len(base)*4,
        permutation_max_relative_error=perm_max,random_permutation_cases=len(validation),random_permutation_bit_mismatches=perm_mismatches,
        exhaustive_permutation_cases=len(eo),exhaustive_permutation_bit_mismatches=exhaustive_mismatches,
        frozen_components_unchanged=True,elapsed_seconds=time.monotonic()-started,recursion_run=False,search_run=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps(dict(child_normalized_diameter_operator='FROZEN_CANDIDATE',**result),indent=2))
    print(json.dumps(dict(event='finished',**result)),flush=True)


if __name__=='__main__':run()
