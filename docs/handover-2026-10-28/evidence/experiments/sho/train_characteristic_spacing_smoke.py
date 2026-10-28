"""Learn characteristic spacing from per-side logs, log n, and dimension.

Synthetic geometry only. Dimension power and every affine coefficient are
learned. No V, log(V), or h is supplied to the model as an input feature.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares

from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/characteristic_spacing_smoke')
ROOT=Path('outputs/sho/stage1B')
SEED=27192026
TRAIN_DIMS=(1,3,5)
TRAIN_BUDGETS=(2,8,32,128,512,2048,8192)
INTERPOLATED_BUDGETS=(3,13,37,137,997,4093)
EXTRAPOLATED_BUDGETS=(1,32768,1000000,1000000000)
ASPECT_HOLDOUT=((3.,4.),(12.,18.))
EPS=np.finfo(np.float64).eps
GATES=dict(max_relative_error=256*EPS,d2_sqrt_max_relative_error=256*EPS,
           scale_equivariance_max_relative_error=512*EPS)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(p,rows):
    import csv
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


class CharacteristicSpacing(nn.Module):
    """Shared affine side encoder + learned budget term and dimension power.

    z = sum_j (a*log_side_j+c) + b*log_n + e
    log_prediction = exp(p*log(d)) * z
    prediction = exp(log_prediction)

    All five parameters a,b,c,e,p start random. There is no fixed mean/1/d,
    learned-dimension lookup, or fixed teacher coefficient in the forward path.
    """
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(.2*torch.randn(5,dtype=torch.float64))

    def log_forward(self,log_sides,mask,log_n,d,weights=None):
        a,b,c,e,p=self.weights if weights is None else weights
        side_terms=torch.where(mask,a*log_sides+c,torch.zeros_like(log_sides))
        z=side_terms.sum(1)+b*log_n+e
        return torch.exp(p*torch.log(d))*z

    def forward(self,log_sides,mask,log_n,d):
        return torch.exp(self.log_forward(log_sides,mask,log_n,d))


def generate(dim,count,seed,group):
    rng=np.random.default_rng(seed);records=[]
    while len(records)<count:
        if dim==1:ratio=1.
        elif group=='aspect_extrapolation':ratio=10**rng.uniform(3,6)
        else:
            ratio=np.exp(rng.uniform(0,np.log(64)))
            reserved=any(a<=ratio<=b for a,b in ASPECT_HOLDOUT)
            if group=='train' and reserved:continue
            if group=='aspect_interpolation' and not reserved:continue
        if dim==1:offset=np.zeros(1)
        else:
            u=rng.random(dim);u[0]=0;u[1]=1;rng.shuffle(u)
            offset=(u-u.mean())*np.log(ratio)
        log_scale=rng.uniform(-3,3)
        if group=='scale_extrapolation':log_scale=rng.uniform(-12,-8) if rng.random()<.5 else rng.uniform(8,12)
        sides=10**log_scale*np.exp(offset)
        if group=='budget_extrapolation':n=int(rng.choice(EXTRAPOLATED_BUDGETS))
        elif group=='train':n=int(rng.choice(TRAIN_BUDGETS))
        else:n=int(rng.choice(INTERPOLATED_BUDGETS))
        assert np.all(sides>0) and n>0
        volume=np.prod(sides)
        assert np.isfinite(volume) and volume>0
        target=np.exp((np.log(volume)-np.log(n))/dim)
        assert target>0 and np.isfinite(target)
        records.append(dict(d=dim,n=n,sides=sides,target=float(target),volume=float(volume),
                            group=group,aspect=float(sides.max()/sides.min()),scale=float(10**log_scale)))
    return records


def inputs(records):
    maximum=max(r['d'] for r in records)
    log_sides=np.zeros((len(records),maximum));mask=np.zeros_like(log_sides,dtype=bool)
    for i,r in enumerate(records):
        log_sides[i,:r['d']]=np.log(r['sides']);mask[i,:r['d']]=True
    return (torch.from_numpy(log_sides),torch.from_numpy(mask),
            torch.from_numpy(np.log(np.array([r['n'] for r in records]))),
            torch.tensor([r['d'] for r in records],dtype=torch.float64))


def persist_data(path,records):
    maximum=max(r['d'] for r in records);sides=np.ones((len(records),maximum))
    for i,r in enumerate(records):sides[i,:r['d']]=r['sides']
    np.savez_compressed(path,sides=sides,d=[r['d'] for r in records],n=[r['n'] for r in records],
                        target=[r['target'] for r in records],group=[r['group'] for r in records],
                        aspect=[r['aspect'] for r in records],scale=[r['scale'] for r in records])


def metrics(y,p):
    e=p-y;rel=e/y
    return dict(examples=len(y),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),
        max_error=float(abs(e).max()),relative_rmse=float(np.sqrt(np.mean(rel*rel))),
        relative_mae=float(np.mean(abs(rel))),max_relative_error=float(abs(rel).max()),
        target_min=float(y.min()),target_max=float(y.max()),bit_exact=int(np.sum(y.view(np.uint64)==p.view(np.uint64))))


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for folder in ('geometry_allocation_curriculum','recursive_region_splitter_poc','split_quality_smoke')
               for p in (ROOT/folder).glob('*.pt')}
    config=dict(seed=SEED,dtype='float64',train_dimensions=TRAIN_DIMS,heldout_dimensions=[2,4],
        train_budgets=TRAIN_BUDGETS,interpolation_budgets=INTERPOLATED_BUDGETS,extrapolation_budgets=EXTRAPOLATED_BUDGETS,
        aspect_holdout_bands=ASPECT_HOLDOUT,train_scale_range=[1e-3,1e3],heldout_scale_ranges=[[1e-12,1e-8],[1e8,1e12]],
        gates=GATES,model='exp(exp(p*log(d)) * (sum_j(a*log(side_j)+c) + b*log(n) + e))',
        model_inputs='individual log side lengths + validity mask, log integer budget, dimension',
        learned_parameters=['a','b','c','e','p'],initialization='independent random normal * 0.2, float64',
        target='np.exp((np.log(np.prod(sides))-np.log(n))/d)',
        objective='continuous squared log-output residual to log(exact teacher h), equal weight per example',
        optimizer='scipy least_squares TRF; exact torch forward-mode Jacobian; max_nfev=120; ftol/xtol/gtol=1e-14',
        geometric_prior='shared additive encoder on per-side logs, learned power-law dependence on d, exponential positive output',
        no_volume_input=True,no_hardcoded_dimension_reciprocal=True,protected_hashes=protected)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    train=[]
    for d in TRAIN_DIMS:train+=generate(d,1536,SEED+d,'train')
    persist_data(OUT/'training_boxes.npz',train)
    tx=inputs(train);log_target=np.log(np.array([r['target'] for r in train]))
    model=CharacteristicSpacing();initial=model.weights.detach().numpy().copy();history=[];best=[np.inf,None]
    def residual(w):
        with torch.no_grad():z=model.log_forward(*tx,weights=torch.from_numpy(w)).numpy()
        residuals=z-log_target;loss=float(np.mean(residuals**2))
        history.append(dict(evaluation=len(history)+1,log_rmse=math.sqrt(loss),weights=json.dumps(w.tolist())))
        if loss<best[0]:best[:]=[loss,w.copy()]
        return residuals
    def jacobian(w):
        fn=lambda v:model.log_forward(*tx,weights=v)
        return torch.func.jacfwd(fn)(torch.from_numpy(w)).detach().numpy()
    optimization=least_squares(residual,initial,jac=jacobian,method='trf',max_nfev=120,
                               ftol=1e-14,xtol=1e-14,gtol=1e-14)
    with torch.no_grad():model.weights.copy_(torch.from_numpy(best[1]))
    checkpoint=OUT/'characteristic_spacing_before_evaluation.pt'
    checkpoint_hash=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,
        initial_weights=initial.tolist(),optimizer_status=int(optimization.status),optimizer_message=optimization.message,
        objective_calls=len(history),training_log_rmse=math.sqrt(best[0]),status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'training_history.csv',history)
    print(json.dumps(dict(event='trained_and_saved',initial=initial.tolist(),learned=model.weights.detach().tolist(),
        calls=len(history),training_log_rmse=math.sqrt(best[0]),sha256=checkpoint_hash)),flush=True)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval()
    validation=[]
    groups=('combinations_and_budget_interpolation','aspect_interpolation','aspect_extrapolation','budget_extrapolation','scale_extrapolation')
    for d in range(1,6):
        for g,group in enumerate(groups):validation+=generate(d,256,SEED+100+d*10+g,group)
    persist_data(OUT/'heldout_boxes.npz',validation)
    with torch.no_grad():pred=model(*inputs(validation)).numpy()
    truth=np.array([r['target'] for r in validation]);dimensions=np.array([r['d'] for r in validation])
    rows=[]
    for d in ('all',1,2,3,4,5):
        for group in ('all',)+groups:
            mask=np.ones(len(validation),dtype=bool)
            if d!='all':mask &= dimensions==d
            if group!='all':mask &=np.array([r['group']==group for r in validation])
            rows.append(dict(record_type='teacher_label_metrics',dimension=d,group=group,
                dimension_heldout=d in (2,4),**metrics(truth[mask],pred[mask])))
    cases=[];sqrt_true=[];sqrt_pred=[];exp_true=[]
    for i,r in enumerate(validation):
        case=dict(case=i,dimension=r['d'],group=r['group'],n=r['n'],aspect=r['aspect'],scale=r['scale'],
                  teacher=r['target'],learned=float(pred[i]),relative_error=float((pred[i]-truth[i])/truth[i]))
        if r['d']==2:
            exact_sqrt=math.sqrt(float(np.prod(r['sides']))/r['n'])
            case.update(sqrt_teacher=exact_sqrt,relative_error_vs_sqrt=float((pred[i]-exact_sqrt)/exact_sqrt),
                        exp_teacher_relative_to_sqrt=float((truth[i]-exact_sqrt)/exact_sqrt))
            sqrt_true.append(exact_sqrt);sqrt_pred.append(pred[i]);exp_true.append(truth[i])
        cases.append(case)
    sqrt_true=np.array(sqrt_true);sqrt_pred=np.array(sqrt_pred);exp_true=np.array(exp_true)
    sqrt_metrics=metrics(sqrt_true,sqrt_pred)
    label_difference=metrics(sqrt_true,exp_true)
    rows.append(dict(record_type='d2_vs_audited_sqrt',dimension=2,group='all',**sqrt_metrics))
    rows.append(dict(record_type='d2_exp_teacher_vs_sqrt_teacher',dimension=2,group='all',**label_difference))
    equiv=[];max_equiv=0.
    base=validation[::13]
    with torch.no_grad():base_pred=model(*inputs(base)).numpy()
    for c in (1e-6,.137,3.,1e6):
        scaled=[dict(r,sides=r['sides']*c) for r in base]
        with torch.no_grad():scaled_pred=model(*inputs(scaled)).numpy()
        expected=c*base_pred
        for d in range(1,6):
            mask=np.array([r['d']==d for r in base]);m=metrics(expected[mask],scaled_pred[mask])
            max_equiv=max(max_equiv,m['max_relative_error'])
            equiv.append(dict(record_type='scale_equivariance',dimension=d,c=c,**m))
    rows+=equiv
    overall=metrics(truth,pred)
    passed=(np.isfinite(pred).all() and np.all(pred>0) and overall['max_relative_error']<=GATES['max_relative_error']
            and sqrt_metrics['max_relative_error']<=GATES['d2_sqrt_max_relative_error']
            and max_equiv<=GATES['scale_equivariance_max_relative_error'])
    write_csv(OUT/'characteristic_spacing_smoke.csv',rows)
    write_csv(OUT/'characteristic_spacing_cases.csv',cases)
    assert all(sha(p)==h for p,h in protected.items())
    result=dict(status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',passed=bool(passed),
        checkpoint=str(checkpoint),sha256=checkpoint_hash,learned_weights=model.weights.detach().tolist(),
        initial_weights=initial.tolist(),training_examples=len(train),heldout_examples=len(validation),
        overall=overall,d2_vs_sqrt=sqrt_metrics,d2_teacher_formula_difference=label_difference,
        max_scale_equivariance_relative_error=max_equiv,protected_checkpoints_unchanged=True,
        elapsed_seconds=time.monotonic()-start,training_objective_calls=len(history))
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps(dict(characteristic_spacing_operator='FROZEN_CANDIDATE',**result),indent=2))
    print(json.dumps(dict(event='finished',**result)),flush=True)


if __name__=='__main__':run()
