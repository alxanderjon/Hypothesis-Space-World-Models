"""Train only equality-based Brent formula selection, preserving all formulas."""
import csv
import inspect
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq

from experiments.geometry import audit_exact_branch_brent as audit
from experiments.geometry import train_branch_location_residual as fresh_source
from experiments.geometry.train_brent_interpolation_eligibility import BrentEligibility
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'brent_interpolation_branch_selector_training'
SOURCE=ROOT/'exact_branch_switch_brent_audit'
ELIG=ROOT/'brent_interpolation_eligibility_training'
SEED=928202619
LINEAR,INVERSE_QUADRATIC=0,1


class BrentBranchSelector(nn.Module):
    def __init__(self,threshold):
        super().__init__()
        self.weights=nn.Parameter(.3*torch.randn(2,dtype=torch.float64))
        self.head=nn.Linear(3,2,bias=False,dtype=torch.float64)
        self.threshold=threshold.eval().requires_grad_(False)

    def sign_features(self,x):
        margin=self.weights[0]*x[:,0]+self.weights[1]*x[:,1]
        signs=self.threshold.decision(margin)
        return nn.functional.one_hot((signs+1).long(),3).double()

    def forward(self,x):return self.head(self.sign_features(x))

    def predict(self,x):
        with torch.no_grad():
            return self(torch.from_numpy(np.asarray(x,dtype=np.float64).reshape(-1,2))).argmax(1).numpy()


def truth(x):return (x[:,0]!=x[:,1]).astype(np.int64)


def metrics(model,x,name):
    y=truth(x);p=model.predict(x);bad=p!=y
    return dict(group=name,cases=len(x),errors=int(bad.sum()),LINEAR_errors=int((bad&(y==LINEAR)).sum()),
                INVERSE_QUADRATIC_errors=int((bad&(y==INVERSE_QUADRATIC)).sum()),
                exact_equality_cases=int((y==LINEAR).sum()),exact_equality_errors=int((bad&(y==LINEAR)).sum()))


def boundaries(seed,n):
    rng=np.random.default_rng(seed)
    base=np.r_[np.ldexp(rng.uniform(1,2,n),rng.integers(-1000,900,n)),
               0.,np.nextafter(0.,1.),np.finfo(float).tiny,.25,.5,1.,2.]
    base=np.r_[base,-base];pairs=[];tags=[]
    for b in base:
        pairs.append((b,b));tags.append('equality')
        for direction in (-np.inf,np.inf):
            c=np.nextafter(b,direction)
            pairs.extend(((b,c),(c,b)));tags.extend(('one_ULP','one_ULP'))
    pairs.extend(((0.,-0.),(-0.,0.)));tags.extend(('signed_zero','signed_zero'))
    return np.array(pairs),np.array(tags)


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('manifest','locked','validated'))})
    for folder in (SOURCE,ELIG):protected.update({str(p):sha(p) for p in folder.iterdir() if p.is_file()})
    protected[str(Path(audit.__file__))]=sha(Path(audit.__file__))
    em=json.loads((ELIG/'brent_interpolation_eligibility_locked.json').read_text())
    assert em['status']=='LOCKED' and sha(Path(em['checkpoint']))==em['sha256']
    eligibility=BrentEligibility(MagnitudeThreshold())
    eligibility.load_state_dict(torch.load(em['checkpoint'],weights_only=False)['state_dict']);eligibility.eval().requires_grad_(False)
    threshold=MagnitudeThreshold();threshold.load_state_dict(eligibility.threshold.state_dict())
    model=BrentBranchSelector(threshold);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    cases=json.loads((SOURCE/'solve_inputs.json').read_text())
    splits={int(r['case_id']):r['group'] for r in audit.read(ELIG/'solve_splits.csv')}
    data=[];ids=[];saved_queries={};saved_methods=[]
    with (SOURCE/'brent_iterations.csv').open() as f:
        for row in csv.DictReader(f):
            i=int(row['case_id']);saved_queries.setdefault(i,[])
            if row['query']:saved_queries[i].append((float(row['query']),float(row['new_function'])))
            if row['stop']=='True' or row['eligible']!='True':continue
            s=json.loads(row['normalized_state']);data.append([s['xpre'],s['xblk']]);ids.append(i)
            saved_methods.append(row['attempt_method'])
    x=np.array(data);ids=np.array(ids);assert len(x)==25814
    assert np.array_equal(truth(x),np.array([0 if m=='LINEAR_INTERPOLATION' else 1 for m in saved_methods]))
    fit=np.array([splits[i]=='fit' for i in ids]);held=np.array([splits[i]=='heldout_archive' for i in ids]);fresh=ids>=774
    assert not set(ids[fit])&set(ids[~fit])
    train_boundary,_=boundaries(SEED+1,96);test_boundary,tags=boundaries(SEED+2,256)
    rng=np.random.default_rng(SEED+3);ordinary=rng.uniform(-1,1,(4096,2))
    train=np.concatenate((x[fit],train_boundary,ordinary))
    scale=abs(train).max(1);design=np.divide(train,scale[:,None],out=np.zeros_like(train),where=scale[:,None]>0)
    target=design[:,0]-design[:,1];history=[]
    for step in range(9):
        w=model.weights.detach().numpy().copy();pred=w[0]*design[:,0]+w[1]*design[:,1];res=target-pred
        history.append(dict(part='margin',step=step,weights=w.tolist(),max_error=float(abs(res).max())))
        if np.all(res==0):break
        proposal=w+np.linalg.lstsq(design,res,rcond=None)[0]
        if np.array_equal(proposal,w):break
        with torch.no_grad():model.weights.copy_(torch.from_numpy(proposal))
    table=torch.eye(3,dtype=torch.float64);table_y=torch.tensor([1,0,1])
    with torch.no_grad():train_features=model.sign_features(torch.from_numpy(train))
    y=torch.from_numpy(truth(train));opt=torch.optim.Adam(model.head.parameters(),lr=.05)
    for step in range(1500):
        opt.zero_grad();loss=nn.functional.cross_entropy(model.head(train_features),y)+nn.functional.cross_entropy(model.head(table),table_y)
        loss.backward();opt.step()
        with torch.no_grad():
            scores=model.head(table);confidence=(scores.gather(1,table_y[:,None])-scores.gather(1,(1-table_y)[:,None])).min()
        if confidence>4:break
    history.append(dict(part='head',step=step+1,loss=float(loss.detach()),min_logit_margin=float(confidence)))
    config=dict(seed=SEED,trainable_parameters=8,dtype='float64',inputs=['xpre','xblk'],
        architecture='Learned bias-free 2-weight margin -> existing frozen three-state zero threshold -> generic sign one-hot -> learned bias-free Linear(3,2)',
        teacher='LINEAR iff xpre == xblk, otherwise INVERSE_QUADRATIC',
        inference='Raw coordinates; no raw coordinate equality flag or teacher branch label supplied. Argmax of learned head decides formula.',
        training='Supervised margin residual least squares from random weights, then cross entropy routing; no snapping.',
        normalization='Training-only common max-absolute scale; none at inference.',
        labels=['LINEAR','INVERSE_QUADRATIC'],split='Reuse whole-parent/whole-solve split; all saved fresh solves excluded; no validation tuning',
        fit_attempts=int(fit.sum()),heldout_attempts=int(held.sum()),saved_fresh_attempts=int(fresh.sum()),
        frozen_eligibility=em,threshold_reused=True,no_other_training=True,no_recursion=True)
    checkpoint=OUT/'branch_selector_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),initial_state=initial,config=config,history=history))
    json_write(OUT/'config.json',config);json_write(OUT/'training_history.json',history)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    rows=[metrics(model,x,'all_saved_25814_attempts'),metrics(model,x[fit],'fit_solves'),metrics(model,x[held],'heldout_solves'),
          metrics(model,x[fresh],'saved_fresh_44_solves'),metrics(model,test_boundary,'heldout_boundary_all')]
    for tag in sorted(set(tags)):rows.append(metrics(model,test_boundary[tags==tag],tag))
    conflict=audit.read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv')
    for g in range(21):
        ii={int(r['unique_id']) for r in conflict if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==g}
        rows.append(metrics(model,x[np.isin(ids,list(ii))],f'conflicting_group_{g}'))
    rows.append(metrics(model,x[ids==765],'recursive_fixture_no_recursion'))
    pred=model.predict(x);write_csv(OUT/'attempt_predictions.csv',[dict(case_id=int(i),xpre=a,xblk=b,teacher=int(t),learned=int(p)) for i,(a,b),t,p in zip(ids,x,truth(x),pred)])
    write_csv(OUT/'validation.csv',rows)
    print(json.dumps(dict(event='saved_and_screened',weights=model.weights.tolist(),errors=sum(r['errors'] for r in rows),sha256=digest)),flush=True)
    if any(r['errors'] for r in rows):
        json_write(OUT/'results.json',dict(status='DIAGNOSTIC_ONLY_STOP',metrics=rows,checkpoint=str(checkpoint),sha256=digest));return

    # Both compared interpreters share the LOCKED eligibility operator. Exactly
    # one further expression differs: formula-branch equality -> learned router.
    original=inspect.getsource(audit.traced_brent)
    eold='eligible=previous_step_large and residual_decreasing';enew='eligible=eligibility_hook(sp,delta,fc,fp)'
    assert original.count(eold)==1
    baseline=original.replace(eold,enew);assert baseline.count('if xp==xb:')==1
    changed=baseline.replace('if xp==xb:','if branch_hook(xp,xb):')
    (OUT/'brent_with_learned_branch.py').write_text(changed)
    branch_calls=[]
    def eligibility_hook(sp,d,fc,fp):return bool(eligibility.predict([[sp,d,fc,fp]])[1][0])
    def branch_hook(xp,xb):
        p=int(model.predict([[xp,xb]])[0]);branch_calls.append((xp,xb,p));return p==LINEAR
    env=dict(audit.__dict__,eligibility_hook=eligibility_hook,branch_hook=branch_hook)
    exec(compile(baseline,'brent_locked_eligibility.py','exec'),env);base=env['traced_brent']
    exec(compile(changed,'brent_with_learned_branch.py','exec'),env);learned=env['traced_brent']
    a=audit.Audit.__new__(audit.Audit);replays=[];fresh_inputs=[]
    def replay(case,i,group):
        a.setup(case['parent'],case['axis']);count=case['count'];calls=[]
        def callback(c):
            f=a.formula(float(c),count);ia=int(f[4][0])
            return float(f[1][0]),dict(q_active=int(f[3][0]),aspect_active=ia,aspect_power=int(f[5][0,ia]),competing_q=f[6][0].tolist(),competing_log_aspect=f[7][0].tolist())
        root,tr,n,reason=base(callback,case['left'],case['right'],i)
        def installed(c):
            v,_=callback(c);calls.append((float(c),v));return v
        target,info=brentq(installed,case['left'],case['right'],xtol=audit.XTOL,rtol=audit.RTOL,maxiter=audit.MAXITER,full_output=True)
        assert root.hex()==target.hex()==float(case['teacher_cut']).hex()
        old=len(branch_calls);got,lr,ln,lreason=learned(callback,case['left'],case['right'],i)
        tq=[r for r in tr if not r['stop']];lq=[r for r in lr if not r['stop']]
        assert n==info.function_calls==len(calls)
        assert [(r['query'],r['new_function']) for r in tq]==calls[2:]
        if i<818:assert [(r['query'],r['new_function']) for r in tq]==saved_queries[i]
        eq=sum(u['query'].hex()==v['query'].hex() for u,v in zip(tq,lq))
        first=next((j for j,(u,v) in enumerate(zip(tq,lq)) if u['query'].hex()!=v['query'].hex()),None)
        if first is None and len(tq)!=len(lq):first=min(len(tq),len(lq))
        errors=sum(p!=int(xp!=xb) for xp,xb,p in branch_calls[old:])
        replays.append(dict(case_id=i,group=group,label=case['label'],interpolation_attempts=len(branch_calls)-old,branch_errors=errors,
            queries=len(lq),exact_queries=eq,first_query_divergence=first,exact_all_states=tr==lr,exact_return=root.hex()==got.hex(),
            teacher_return=root,learned_return=got))
        if i>=818:fresh_inputs.extend([[r['normalized_state']['xpre'],r['normalized_state']['xblk']] for r in tr if r['eligible']])
    for i,r in enumerate(cases):replay(r,i,splits[i])
    print(json.dumps(dict(event='saved_replayed',solves=818,exact=sum(r['exact_all_states'] for r in replays))),flush=True)
    # The eight independently generated eligibility-stage solves stay held out.
    oldfresh=json.loads((ELIG/'new_fresh_candidates.json').read_text())
    old_seed=fresh_source.SEED
    try:
        fresh_source.SEED=SEED+41
        newfresh,parents,fa=fresh_source.fresh_data()
    finally:fresh_source.SEED=old_seed
    assert newfresh
    knownparents={json.dumps(c['parent'],sort_keys=True) for c in cases+oldfresh}
    assert not knownparents&{json.dumps(p,sort_keys=True) for p in parents}
    json_write(OUT/'new_fresh_candidates.json',newfresh);json_write(OUT/'fresh_generation.json',dict(parents=parents,audit=fa))
    for j,r in enumerate(oldfresh+newfresh):
        replay(dict(parent=r['parent'],axis=r['axis'],count=r['k'],left=r['c_left'],right=r['c_right'],teacher_cut=r['teacher_cut'],label=r['case']),
               818+j,'eligibility_stage_fresh' if j<len(oldfresh) else 'new_fresh')
    rows.append(metrics(model,np.array(fresh_inputs),'additional_fresh_attempts'))
    write_csv(OUT/'validation.csv',rows);write_csv(OUT/'solve_replay.csv',replays)
    unchanged=all(sha(Path(p))==h for p,h in protected.items());assert unchanged
    passed=all(r['errors']==0 for r in rows) and all(r['exact_all_states'] and r['branch_errors']==0 and r['exact_return'] for r in replays)
    result=dict(operator='brent_interpolation_branch_selector',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        module='experiments.geometry.train_brent_interpolation_branch_selector',model='BrentBranchSelector',implementation_sha256=sha(Path(__file__)),
        checkpoint=str(checkpoint),sha256=digest,margin_weights=model.weights.tolist(),head_weights=model.head.weight.tolist(),
        frozen_eligibility_checkpoint=em['checkpoint'],frozen_eligibility_sha256=em['sha256'],metrics=rows,
        saved_solves=818,saved_fresh_solves=44,eligibility_stage_fresh_solves=len(oldfresh),new_fresh_solves=len(newfresh),
        replay_solves=len(replays),replayed_interpolation_attempts=sum(r['interpolation_attempts'] for r in replays),
        branch_errors=sum(r['branch_errors'] for r in replays),total_queries=sum(r['queries'] for r in replays),
        exact_queries=sum(r['exact_queries'] for r in replays),exact_full_state_replays=sum(r['exact_all_states'] for r in replays),
        first_query_divergence=next((r for r in replays if r['first_query_divergence'] is not None),None),
        protected_files=len(protected),protected_hashes_unchanged=unchanged,no_other_training=True,no_recursion=True,elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    if passed:json_write(OUT/'brent_interpolation_branch_selector_locked.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
