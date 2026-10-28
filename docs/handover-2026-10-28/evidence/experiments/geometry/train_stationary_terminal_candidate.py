"""Small terminal-location screen; frozen locator and high-precision root labels."""
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_stationary_three_state_router import load_router,DEFER
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT=ref.ROOT
OUT=ROOT/'stationary_terminal_candidate_training'
SEED=28312026


class TerminalCandidate(nn.Module):
    """Six learned parameters; sigmoid bounds the predicted bracket fraction."""
    def __init__(self):
        super().__init__();self.linear=nn.Linear(5,1).double()
    def forward(self,features):return torch.sigmoid(self.linear(features)).squeeze(1)
    def predict_fraction(self,features):
        with torch.no_grad():return self(torch.from_numpy(np.asarray(features,dtype=np.float64))).numpy()
    def locate(self,L,R,features):return L+self.predict_fraction(features)*(R-L)


def features(router,coeff,L,R,adjacent):
    gaps=[]
    for c in (L,R):
        with torch.no_grad():
            states,gap,opposite,terms=router(torch.from_numpy(np.column_stack((coeff[:,:4],c,coeff[:,4:6]))),details=True)
        assert np.all(states.numpy()!=DEFER) and np.all(opposite.numpy())
        gaps.append(gap.numpy())
    # All quantities are available at termination. No root or minimizer feature.
    eps=np.finfo(float).eps
    return np.column_stack((np.log(gaps[0]/eps),np.log(gaps[1]/eps),
        np.log((R-L)/(np.maximum(abs(L),abs(R))*eps)),adjacent.astype(float),(L+R)/2))


def quality(L,R,roots,lam,target):
    pred=L+lam*(R-L)
    distance=np.array([float(abs(mp.mpf(float(c))-root)) for c,root in zip(pred,roots)])
    rel=np.abs(lam-target)
    return dict(cases=len(L),root_error_median=float(np.median(distance)),root_error_p99=float(np.quantile(distance,.99)),
        root_error_max=float(distance.max()),fraction_MAE=float(rel.mean()),fraction_RMSE=float(np.sqrt(np.mean(rel**2))),
        fraction_max_error=float(rel.max()),outside_bracket=int(np.sum((pred<L)|(pred>R))))


def quick_screen():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=90
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('locked','validated','manifest'))})
    locator=ROOT/'stationary_stopping_operator_training'
    lock=json.loads((locator/'results.json').read_text());assert lock['status']=='LOCKED' and lock['teacher_roots_contained']==7093
    brackets=list(csv.DictReader((locator/'stopping_bracket_replay.csv').open()))
    teacher=list(csv.DictReader((ROOT/'defer_aware_stationary_locator_audit/defer_aware_root_recovery.csv').open()))
    pool=ROOT/'stationary_float64_minimizer_full_audit';archive=np.load(pool/'all_float64_candidates.npz')
    rr=list(csv.DictReader((pool/'stationary_float64_minimizers.csv').open()))
    cmap={(r['scope'],r['case']):archive['inputs'][archive['offsets'][i],[0,1,2,3,5,6,7]] for i,r in enumerate(rr)}
    coeff=np.array([cmap[(r['scope'],r['case'])] for r in brackets])
    L=np.array([float(r['L']) for r in brackets]);R=np.array([float(r['R']) for r in brackets])
    roots=[mp.mpf(r['c_teacher_decimal']) for r in teacher]
    target=np.array([float((root-mp.mpf(float(l)))/(mp.mpf(float(r))-mp.mpf(float(l)))) for root,l,r in zip(roots,L,R)])
    assert np.all((target>=0)&(target<=1))
    adjacent=np.array([r['termination']=='adjacent_float' for r in brackets])
    router,refs=load_router();xx=features(router,coeff,L,R,adjacent)
    # A small stratified screen, split by saved case ID, before any full-pool fit.
    rng=np.random.default_rng(SEED+1);selected=[]
    scope=np.array([r['scope'] for r in brackets])
    for group in np.unique(scope):
        for stop in (False,True):
            ids=np.flatnonzero((scope==group)&(adjacent==stop));rng.shuffle(ids)
            selected.extend(ids[:min(192,len(ids))])
    selected=np.array(selected);rng.shuffle(selected);cut=int(.7*len(selected));train=selected[:cut];valid=selected[cut:]
    # Fixed screen criterion, registered before fitting. Location quality must
    # match midpoint to within 2% relative RMSE, without worsening physical p99
    # by more than 2%; out-of-bracket predictions forbidden. No tuning on gate.
    plan=dict(model='5 local features -> affine -> sigmoid; 6 float64 parameters',seed=SEED,
        feature_names=['log(left_reliability_gap/eps)','log(right_reliability_gap/eps)','log(relative_bracket_width/eps)','adjacent_stop','bracket_center'],
        target='high-precision (root-L)/(R-L), converted to float64 only after division',
        train_indices=train.tolist(),validation_indices=valid.tolist(),max_updates=600,
        gate='validation fraction RMSE <= 1.02 * midpoint RMSE; physical root-error p99 <= 1.02 * midpoint p99; zero containment failures',
        no_argmin_supervision=True,all_roots_contained=True)
    (OUT/'quick_screen_plan.json').write_text(json.dumps(plan,indent=2))
    np.savez_compressed(OUT/'archive_terminal_features.npz',features=xx,target_fraction=target,L=L,R=R,coeff=coeff,adjacent=adjacent,
                        root_decimal=np.array([str(r) for r in roots]),train=train,validation=valid)
    model=TerminalCandidate();initial={k:v.detach().tolist() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=.025)
    x=torch.from_numpy(xx[train]);y=torch.from_numpy(target[train]);history=[]
    for step in range(1,601):
        opt.zero_grad();loss=((model(x)-y)**2).mean();loss.backward();opt.step()
        if step%20==0:history.append(dict(update=step,training_MSE=float(loss.detach())))
    model.eval().requires_grad_(False)
    cp=OUT/'quick_screen_before_evaluation.pt'
    digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),optimizer_state=opt.state_dict(),initial_parameters=initial,
        seed=SEED,updates=step,architecture=plan['model'],features=plan['feature_names'],target=plan['target']))
    ref.write_csv(OUT/'quick_training_history.csv',history)
    pred=model.predict_fraction(xx)
    report=[]
    for name,ids in [('quick_training',train),('quick_validation',valid)]:
        for stop in ('all','ADJACENT','NO_PROGRESS'):
            keep=ids if stop=='all' else ids[adjacent[ids] == (stop=='ADJACENT')]
            if not len(keep):continue
            rr0=[roots[i] for i in keep]
            for arm,lam in [('midpoint',np.full(len(keep),.5)),('learned',pred[keep])]:
                report.append(dict(group=name,stopping_type=stop,arm=arm,**quality(L[keep],R[keep],rr0,lam,target[keep])))
    ref.write_csv(OUT/'terminal_candidate_quick_screen.csv',report)
    base=next(r for r in report if r['group']=='quick_validation' and r['stopping_type']=='all' and r['arm']=='midpoint')
    learned=next(r for r in report if r['group']=='quick_validation' and r['stopping_type']=='all' and r['arm']=='learned')
    passed=(learned['fraction_RMSE']<=1.02*base['fraction_RMSE'] and learned['root_error_p99']<=1.02*base['root_error_p99'] and learned['outside_bracket']==0)
    result=dict(status='QUICK_SCREEN_PROMISING' if passed else 'DIAGNOSTIC_ONLY_STOP',screen_passed=passed,
        learned_parameters={k:v.tolist() for k,v in model.state_dict().items()},baseline=base,learned=learned,
        fraction_RMSE_ratio=learned['fraction_RMSE']/base['fraction_RMSE'],physical_p99_ratio=learned['root_error_p99']/base['root_error_p99'],
        checkpoint=str(cp),checkpoint_sha256=digest,train_cases=len(train),validation_cases=len(valid),updates=step,
        frozen_refs=refs,all_existing_protected_hashes_unchanged=all(ref.sha(Path(p))==h for p,h in protected.items()),
        elapsed_seconds=time.monotonic()-start)
    assert result['all_existing_protected_hashes_unchanged']
    (OUT/'quick_screen_results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':quick_screen()
