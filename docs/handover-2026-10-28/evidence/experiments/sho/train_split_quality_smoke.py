"""Continuous geometry-score regression; frozen axis and allocation networks."""
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sho.train_recursive_region_splitter_poc import (
    Splitter, LockedAllocation, children, quality, digest, csv_write, ALLOC, ALLOC_SHA)
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/split_quality_smoke')
AXIS=Path('outputs/sho/stage1B/recursive_region_splitter_poc/splitter_best_before_evaluation.pt')
AXIS_SHA='a8ef44e81d877b96773e4ca04887f504d8d5875f42b34b8f3fec2e19e13b62fa'
SEED=27182026
TRAIN_CUTS=np.array([.15,.2,.3,.4,.5,.6,.7,.8,.85])
TEST_CUTS=np.array([.137,.237,.337,.437,.563,.663,.763,.863])
RESERVED_ASPECTS=((1.3,1.5),(2.7,3.3),(6.,8.))
TEST_BUDGETS=(3,7,13,19,23,37,73,149,251)
GATES=dict(score_rmse=.005,score_max_error=.03,small_budget_winner_agreement=.995,
           all_winner_agreement=.99,small_budget_max_relative_regret=.001,previous_n4_correct=True)


class QualityNet(nn.Module):
    """Generic symmetric learned scorer; no diameter/aspect-score inputs."""
    def __init__(self):
        super().__init__()
        self.child=nn.Sequential(nn.Linear(7,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,8))
        self.head=nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,1))
        self.double()

    def forward(self,x):
        h=self.child(x)
        return self.head(torch.cat((h.amax(1),h.mean(1)),dim=1)).squeeze(1)


def bucket(n):return '2-5' if n<=5 else '6-10' if n<=10 else '11-20' if n<=20 else '>20'


def parents(count,seed,kind):
    rng=np.random.default_rng(seed);lo=[];hi=[];budgets=[]
    groups=([2,4,5],[6,8,9,10],[11,12,14,15,16,17,18,20],[24,40,56,96,160,224])
    heldgroups=([3],[7],[13,19],[23,37,73,149,251])
    while len(budgets)<count:
        ratio=np.exp(rng.uniform(0,np.log(16)))
        held=any(a<=ratio<=b for a,b in RESERVED_ASPECTS)
        if kind=='train' and held:continue
        if kind in ('aspect','joint') and not held:continue
        # 90% small budgets, balanced across the three small-budget buckets.
        which=int(rng.choice(4,p=[.3,.3,.3,.1]))
        source=heldgroups if kind in ('budget','joint') else groups
        n=int(rng.choice(source[which]))
        scale=10**rng.uniform(-2,2)
        if kind=='joint':scale=10**rng.uniform(-6,-3) if rng.random()<.5 else 10**rng.uniform(3,6)
        length=scale*np.array([np.sqrt(ratio),1/np.sqrt(ratio)])
        if rng.random()<.5:length=length[::-1]
        lower=rng.uniform(-2,2,2)*scale
        lo.append(lower);hi.append(lower+length);budgets.append(n)
    return np.array(lo),np.array(hi),np.array(budgets,dtype=int)


def make_pool(lo,hi,n,cuts,axis_model,allocator):
    with torch.no_grad():axis_scores,_=axis_model(torch.from_numpy(hi-lo),torch.from_numpy(n.astype(float)))
    axes=axis_scores.argmax(1).numpy()
    relative=[];lengths=[];parent=[];candidate_budgets=[];parent_index=[];cut_values=[]
    for i in range(len(n)):
        for q in cuts:
            cl,ch=children(lo[i],hi[i],int(axes[i]),q)
            lengths.append(ch-cl);relative.append((ch-cl)/(hi[i]-lo[i]));parent.append(hi[i]-lo[i])
            candidate_budgets.append(n[i]);parent_index.append(i);cut_values.append(q)
    relative=np.array(relative);lengths=np.array(lengths);parent=np.array(parent);bn=np.array(candidate_budgets)
    counts,prob,exact=allocator.allocate(relative,bn)
    valid=counts.min(1)>0
    y=np.array([quality(l,c,p,int(b)) for l,c,p,b in zip(lengths,counts,parent,bn)])
    # Only elementary geometry/count features, never norms or teacher-score terms.
    normalized=lengths/np.sqrt(np.prod(parent,axis=1))[:,None,None]
    normalized=np.sort(normalized,axis=2) # coordinate-axis naming is irrelevant
    fraction=counts/bn[:,None]
    safe_fraction=np.maximum(fraction,np.finfo(float).tiny) # invalids are excluded/masked
    volume=np.prod(relative,axis=2)
    x=np.concatenate((normalized,np.log(normalized),fraction[:,:,None],
                      np.log(safe_fraction)[:,:,None],volume[:,:,None]),axis=2)
    return dict(x=x,y=y,valid=valid,counts=counts,allocation_fraction=prob,
                volume_count_mismatch=np.any(counts!=exact,axis=1),lower=lo,upper=hi,budget=n,
                candidate_budget=bn,parent_index=np.array(parent_index),cut=np.array(cut_values),axis=axes)


def metric(y,p):
    e=p-y
    return dict(samples=len(y),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),
                max_error=float(abs(e).max()),correlation=float(np.corrcoef(y,p)[0,1]) if len(y)>1 else 1.)


def evaluate(model,pool,group,aggregate,candidate_rows,parent_rows):
    prediction=np.full(len(pool['y']),np.inf)
    valid=pool['valid']
    with torch.no_grad():prediction[valid]=model(torch.from_numpy(pool['x'][valid])).numpy()
    for key in ('all','2-5','6-10','11-20','>20'):
        mask=valid.copy()
        if key!='all':mask &=np.array([bucket(int(n))==key for n in pool['candidate_budget']])
        if mask.any():aggregate.append(dict(record_type='scores',group=group,budget_bucket=key,**metric(pool['y'][mask],prediction[mask])))
    for i in range(len(prediction)):
        candidate_rows.append(dict(record_type='candidate',group=group,parent=int(pool['parent_index'][i]),
            n=int(pool['candidate_budget'][i]),cut=float(pool['cut'][i]),valid=bool(valid[i]),
            counts=json.dumps(pool['counts'][i].tolist()),teacher_score=float(pool['y'][i]),
            learned_score=float(prediction[i]),score_error=float(prediction[i]-pool['y'][i]) if valid[i] else '',
            volume_count_mismatch=bool(pool['volume_count_mismatch'][i])))
    for parent,n in enumerate(pool['budget']):
        indices=np.flatnonzero((pool['parent_index']==parent)&valid)
        y=pool['y'][indices];p=prediction[indices];q=pool['cut'][indices]
        teacher_order=np.argsort(y,kind='stable');learned_order=np.argsort(p,kind='stable')
        best=float(y.min());equivalent=y<=best+1e-12*max(1,best)
        canonical=int(np.flatnonzero(equivalent)[0]);winner=int(learned_order[0])
        top2=learned_order[:2];teacher_top2=teacher_order[:2]
        tol=1e-12*max(1,best)
        top2_quality_threshold=y[teacher_top2[-1]]+tol
        parent_rows.append(dict(record_type='ranking',group=group,parent=parent,n=int(n),budget_bucket=bucket(int(n)),
            axis=int(pool['axis'][parent]),teacher_cut=float(q[canonical]),learned_cut=float(q[winner]),
            teacher_best=float(best),learned_winner_teacher_score=float(y[winner]),
            teacher_best_agreement=bool(equivalent[winner]),canonical_agreement=bool(winner==canonical),
            teacher_best_in_learned_top2=bool(equivalent[top2].any()),
            top2_set_agreement=bool(set(top2)==set(teacher_top2)),
            top2_quality_agreement=bool(np.all(y[top2]<=top2_quality_threshold)),
            absolute_penalty=float(y[winner]-best),relative_penalty=float(y[winner]/best-1),
            best_to_next_strict_margin=float(np.min(y[~equivalent])-best) if (~equivalent).any() else '',
            chosen_counts=json.dumps(pool['counts'][indices[winner]].tolist()),
            count_conservation=bool(pool['counts'][indices[winner]].sum()==n)))
    return prediction


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    assert digest(AXIS)==AXIS_SHA
    axis_model=Splitter();axis_model.load_state_dict(torch.load(AXIS,weights_only=False)['state_dict'])
    axis_model.eval().requires_grad_(False)
    allocator=LockedAllocation()
    config=dict(seed=SEED,gates=GATES,axis_checkpoint=str(AXIS),axis_sha256=AXIS_SHA,
        allocator_checkpoint=str(ALLOC),allocator_sha256=ALLOC_SHA,dimension=2,
        train_cuts=TRAIN_CUTS.tolist(),heldout_cuts=TEST_CUTS.tolist(),heldout_budgets=TEST_BUDGETS,
        reserved_aspect_bands=RESERVED_ASPECTS,features=['child sorted side lengths / sqrt(parent area)',
            'log of those normalized lengths','integer child count / n','log count fraction','child volume / parent volume'],
        model='shared child MLP 7->48 SiLU->48 SiLU->8, max/mean pooling, 16->32 SiLU->1',
        target='unchanged quality() from previous splitter: normalized worst child diameter + .05 worst log aspect',
        loss='MSE on finite continuous teacher scores ONLY; no winning-cut supervision',
        invalid='zero-count splits have infinite quality, excluded from regression and deterministically masked in ranking',
        training='Adam 2000 updates lr .002 batch1024; then at most 120 LBFGS iterations on fixed training subset',
        checkpoint_selection='lowest training MSE, no heldout selection',fresh_small_budget_sampling_fraction=.9)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    train=make_pool(*parents(2048,SEED,'train'),TRAIN_CUTS,axis_model,allocator)
    np.savez_compressed(OUT/'training_pool.npz',**train)
    x=torch.from_numpy(train['x'][train['valid']]);y=torch.from_numpy(train['y'][train['valid']])
    model=QualityNet();initial=copy.deepcopy(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    rng=np.random.default_rng(SEED+1);best=float('inf');best_state=None;history=[]
    for step in range(2000):
        idx=torch.from_numpy(rng.integers(len(x),size=1024))
        pred=model(x[idx]);loss=(pred-y[idx]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%200==0:
            with torch.no_grad():mse=float((model(x)-y).square().mean())
            if mse<best:best=mse;best_state=copy.deepcopy(model.state_dict());best_at=f'Adam {step+1}'
            history.append(dict(phase='Adam',update=step+1,training_rmse=math.sqrt(mse)))
            print(json.dumps(dict(event='training',**history[-1])),flush=True)
    model.load_state_dict(best_state)
    # Fixed train-only polish; no validation feedback or automatic extension.
    idx=torch.from_numpy(rng.choice(len(x),size=min(4096,len(x)),replace=False));fx=x[idx];fy=y[idx]
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=20,max_eval=25,history_size=30,
                         tolerance_grad=1e-10,tolerance_change=1e-13,line_search_fn='strong_wolfe')
    closures=0
    for chunk in range(6):
        def closure():
            nonlocal closures
            opt.zero_grad();loss=(model(fx)-fy).square().mean();loss.backward();closures+=1;return loss
        opt.step(closure)
        with torch.no_grad():mse=float((model(x)-y).square().mean())
        if mse<best:best=mse;best_state=copy.deepcopy(model.state_dict());best_at=f'LBFGS chunk {chunk+1}'
        history.append(dict(phase='LBFGS',update=chunk+1,training_rmse=math.sqrt(mse),closure_calls=closures))
        print(json.dumps(dict(event='training',**history[-1])),flush=True)
    model.load_state_dict(best_state)
    checkpoint=OUT/'split_quality_before_evaluation.pt'
    sha=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),initial_state_dict=initial,config=config,
        training_mse=best,best_at=best_at,status='SAVED_BEFORE_HELDOUT_EVALUATION'))
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval()
    csv_write(OUT/'training_history.csv',history)
    aggregate=[];candidate_rows=[];parent_rows=[];all_y=[];all_p=[];all_n=[]
    for i,group in enumerate(('aspect','budget','cuts','joint')):
        cuts=TEST_CUTS if group in ('cuts','joint') else TRAIN_CUTS
        pool=make_pool(*parents(384,SEED+10+i,group),cuts,axis_model,allocator)
        np.savez_compressed(OUT/f'heldout_{group}.npz',**pool)
        pred=evaluate(model,pool,group,aggregate,candidate_rows,parent_rows)
        all_y.append(pool['y'][pool['valid']]);all_p.append(pred[pool['valid']]);all_n.append(pool['candidate_budget'][pool['valid']])
        print(json.dumps(dict(event='heldout',group=group,**metric(all_y[-1],all_p[-1]))),flush=True)
    # Exact archived n=4 failure: original box, original cuts, frozen axis.
    old=np.load('outputs/sho/stage1B/recursive_region_splitter_poc/heldout_scale.npz')
    pool=make_pool(old['lower'][3:4],old['upper'][3:4],old['budget'][3:4],
                   np.array([.2,.3,.4,.5,.6,.7,.8]),axis_model,allocator)
    assert pool['budget'][0]==4
    np.savez_compressed(OUT/'previous_n4_failure.npz',**pool)
    evaluate(model,pool,'previous_n4_failure',aggregate,candidate_rows,parent_rows)
    yy=np.concatenate(all_y);pp=np.concatenate(all_p);nn=np.concatenate(all_n)
    overall=metric(yy,pp)
    for key in ('all','2-5','6-10','11-20','>20'):
        mask=np.ones(len(nn),dtype=bool) if key=='all' else np.array([bucket(int(n))==key for n in nn])
        aggregate.append(dict(record_type='scores',group='combined',budget_bucket=key,**metric(yy[mask],pp[mask])))
    rankstats=[]
    for group in ('aspect','budget','cuts','joint','combined','previous_n4_failure'):
        for key in ('all','2-5','6-10','11-20','>20','<=20'):
            selected=[r for r in parent_rows if (r['group']==group or (group=='combined' and r['group']!='previous_n4_failure'))
                      and (key=='all' or r['budget_bucket']==key or (key=='<=20' and r['n']<=20))]
            if not selected:continue
            diff=[r for r in selected if not r['teacher_best_agreement']]
            row=dict(record_type='ranking_summary',group=group,budget_bucket=key,parents=len(selected),
                teacher_best_agreement=float(np.mean([r['teacher_best_agreement'] for r in selected])),
                canonical_agreement=float(np.mean([r['canonical_agreement'] for r in selected])),
                teacher_best_in_top2=float(np.mean([r['teacher_best_in_learned_top2'] for r in selected])),
                top2_set_agreement=float(np.mean([r['top2_set_agreement'] for r in selected])),
                top2_quality_agreement=float(np.mean([r['top2_quality_agreement'] for r in selected])),
                wrong_winners=len(diff),max_relative_penalty=max(r['relative_penalty'] for r in selected),
                mean_relative_penalty=float(np.mean([r['relative_penalty'] for r in selected])),
                mean_relative_penalty_when_wrong=float(np.mean([r['relative_penalty'] for r in diff])) if diff else 0.)
            aggregate.append(row);rankstats.append(row)
    small=next(r for r in rankstats if r['group']=='combined' and r['budget_bucket']=='<=20')
    ranking=next(r for r in rankstats if r['group']=='combined' and r['budget_bucket']=='all')
    prior=next(r for r in parent_rows if r['group']=='previous_n4_failure')
    score_gates=all(r['rmse']<=GATES['score_rmse'] and r['max_error']<=GATES['score_max_error']
                    for r in aggregate if r['record_type']=='scores' and r['group']!='previous_n4_failure')
    passed=(score_gates and small['teacher_best_agreement']>=GATES['small_budget_winner_agreement']
            and ranking['teacher_best_agreement']>=GATES['all_winner_agreement']
            and small['max_relative_penalty']<=GATES['small_budget_max_relative_regret']
            and prior['teacher_best_agreement'])
    csv_write(OUT/'split_quality_smoke.csv',aggregate)
    csv_write(OUT/'split_quality_small_budget.csv',[r for r in parent_rows+candidate_rows if r['n']<=20])
    csv_write(OUT/'all_parent_rankings.csv',parent_rows)
    assert digest(AXIS)==AXIS_SHA and digest(ALLOC)==ALLOC_SHA
    result=dict(status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',passed=passed,score_gate=score_gates,
        overall=overall,small_budget_ranking=small,all_ranking=ranking,previous_n4=prior,
        checkpoint=str(checkpoint),sha256=sha,axis_sha256=AXIS_SHA,allocator_sha256=ALLOC_SHA,
        protected_checkpoints_unchanged=True,allocator=allocator.stats(),best_at=best_at,
        training_rmse=math.sqrt(best),elapsed_seconds=time.monotonic()-start,recursion_run=False,search_run=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps(dict(split_quality_score='FROZEN_CANDIDATE',**result),indent=2))
    print(json.dumps(dict(event='finished',**result)),flush=True)


if __name__=='__main__':run()
