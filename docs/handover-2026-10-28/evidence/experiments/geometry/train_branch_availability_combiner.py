"""Discrete availability combiner only; preserve frozen upstream operators."""
import csv
import itertools
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from torch import nn

from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.geometry.train_branch_switch_sign_transition import SignTransition
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.validate_stationary_three_state_router import load_router
from experiments.geometry.audit_stationary_candidate_availability import scan_roots, math_roots
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint

OUT = ROOT/'branch_switch_availability_combiner_training'
SOURCE = ROOT/'refined_branch_switch_availability_training'
SEED = 28412026
FEATURES = ['sign_transition','branch_signature_changed','valid_pair','correct_owner']


class AvailabilityCombiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2, dtype=torch.float64)

    def forward(self, state):
        if state.ndim != 2 or state.shape[1] != 4 or not ((state==0)|(state==1)).all():
            raise ValueError('Expected four discrete binary state inputs.')
        return self.head(state)


def generate_fresh():
    """Independent parents; analytic teacher labels only, no learned cut prediction."""
    rng=np.random.default_rng(SEED+1); a=Audit(); parents=[]; rows=[]; inputs=[]; flags=[]
    for d in (2,3,4):
        for n in (6,13,21):
            scale=10**rng.uniform(-2,2); lengths=scale*np.exp(rng.uniform(-1.2,1.2,d)); lo=scale*rng.uniform(-.3,.3,d)
            p=dict(d=d,n=n,lower=lo.tolist(),upper=(lo+lengths).tolist());pi=len(parents);parents.append(p)
            for axis in range(d):
                a.setup(p,axis)
                edge=max(1e-15,4*np.spacing(abs(a.lo[axis])+abs(a.hi[axis]))/a.a)
                xx=np.linspace(edge,1-edge,max(2049,n*32+1)); counts=a.counts(xx)[:,0]
                lb=np.array([xx[np.flatnonzero(counts<k)[-1]] for k in range(1,n+1)])
                ub=np.array([xx[np.flatnonzero(counts>=k)[0]] for k in range(1,n+1)])
                for _ in range(60):
                    mid=(lb+ub)/2; less=a.counts(mid)[:,0]<np.arange(1,n+1)
                    lb=np.where(less,mid,lb);ub=np.where(less,ub,mid)
                for k in range(1,n):
                    breaks=a.breaks(ub[k-1],lb[k],k)
                    for j,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
                        f=a.formula(np.array([(l+r)/2]),k);iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
                        coeff=(1/(a.h*np.power(float(k if iq==0 else n-k),1/d)),a.B,a.a,.05*power,iq,ia)
                        roots=scan_roots(lambda c:a.formula(c,k)[1],l,r)
                        smooth=[z for z in math_roots(coeff) if mp.mpf(l)<z<mp.mpf(r)] if roots else []
                        branchroots=[] if smooth else roots
                        scan=np.linspace(np.nextafter(l,r),np.nextafter(r,l),129)
                        for side in (0,1):
                            idx=[0,1] if side==0 else [127,128];c=scan[idx];f=a.formula(c,k)
                            qi=f[3]; ai=f[4]; powers=f[5][np.arange(2),ai]
                            qchange=int(qi[0]!=qi[1]);achange=int(ai[0]!=ai[1]);pchange=int(powers[0]!=powers[1])
                            valid=bool(l<c[0]<c[1]<r)
                            A=1/(a.h*np.power(np.where(qi==0,k,n-k).astype(float),1/d))
                            inputs.extend(np.column_stack((A,np.full(2,a.B),np.full(2,a.a),.05*powers,c,qi,ai)))
                            flags.append([int(qchange or achange or pchange),int(valid),1])
                            label=int(any((0 if abs(root-l)<=abs(r-root) else 1)==side for root in branchroots))
                            rows.append(dict(group='new_fresh_geometry',case=f'{pi}:{axis}:{k}:{j}',dimension=d,budget=n,
                                boundary_side=side,pair_index=idx[0],lower=l,upper=r,sample_1=c[0],sample_2=c[1],
                                q_switch=qchange,aspect_switch=achange,power_switch=pchange,smooth_present=bool(smooth),
                                label=label,teacher_sign_1=int(np.sign(f[1][0])),teacher_sign_2=int(np.sign(f[1][1]))))
    router,_=load_router()
    with torch.no_grad():terms=router.term_assembly.terms(torch.from_numpy(np.array(inputs))).numpy()
    den=abs(terms).sum(1)
    margins=np.divide(terms.sum(1),den,out=np.zeros(len(terms)),where=den>0).reshape(-1,2)
    return parents,rows,margins,np.array(flags,float)


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('manifest','locked','validated'))})
    audited_files=[SOURCE/'feature_dataset.npz',SOURCE/'availability_predictions.csv',
        Path('experiments/geometry/audit_branch_availability_false_positives.py'),
        Path('experiments/geometry/train_branch_switch_availability.py'),
        Path('experiments/geometry/audit_stationary_candidate_availability.py')]
    protected.update({str(p):sha(p) for p in audited_files})
    lockpath=ROOT/'branch_switch_sign_transition_training/branch_switch_sign_transition_locked.json'
    upstream=json.loads(lockpath.read_text()); assert upstream['status']=='LOCKED'
    cp=Path(upstream['checkpoint']);assert sha(cp)==upstream['sha256']
    transition=SignTransition(MagnitudeThreshold())
    transition.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);transition.eval().requires_grad_(False)
    data=np.load(SOURCE/'feature_dataset.npz');features=data['features']; labels=data['labels']
    meta=list(csv.DictReader((SOURCE/'availability_predictions.csv').open()))
    with torch.no_grad():st=transition(torch.from_numpy(features[:,:2].copy())).argmax(1).numpy()
    # Ownership is derived from scheduling, never from a saved root/winner.
    owners=[]
    for r in meta:
        l,u=float(r['lower']),float(r['upper']);side=int(r['boundary_side'])
        scan=np.linspace(np.nextafter(l,u),np.nextafter(u,l),129);ii=[0,1] if side==0 else [127,128]
        owners.append(int(float(r['sample_1'])==scan[ii[0]] and float(r['sample_2'])==scan[ii[1]]))
    x=np.column_stack((st,features[:,4],features[:,5],owners)).astype(float)
    parents,freshmeta,margins,freshflags=generate_fresh()
    with torch.no_grad():fresh_st=transition(torch.from_numpy(margins)).argmax(1).numpy()
    xf=np.column_stack((fresh_st,freshflags));yf=np.array([r['label'] for r in freshmeta])
    sign_errors=int(np.sum(np.sign(margins)!=np.array([[r['teacher_sign_1'],r['teacher_sign_2']] for r in freshmeta])))
    rule=lambda z:np.all(z==1,axis=1).astype(int)  # Teacher supervision/audit only.
    audit_errors=int(np.sum(rule(x)!=labels)); fresh_rule_errors=int(np.sum(rule(xf)!=yf))
    audit_text='''Exact teacher audit and domain contract
The analytic teacher forms 129 points with linspace(nextafter(l,r),nextafter(r,l),129),
reselects active branches at each sample, and detects strict derivative-product
sign changes. Offline brentq supplies labels. Existing branch labels exclude
fixed-branch smooth roots and assign remaining candidates to the nearest boundary
(left owns an exact distance tie). No score-equivalence suppression occurs.

For this endpoint-associated class, relevant pairs are [0,1] on the left and
[127,128] on the right: two interior samples, never straddling the boundary.
Combined signature change includes q-child, aspect-child, AND aspect-log-power.
q/aspect flags are audited separately but redundant once the combined flag is given.

On the saved pool and the independent geometry generated here, availability is
equivalent to sign_transition AND signature_changed AND valid_pair AND correct_owner.
This is an empirically verified local decomposition, not the literal full-scan
implementation or a proof for arbitrary multi-switch intervals.

Saved ownership is correct by construction. The input ownership bit means the
supplied scan pair belongs to the requested oriented boundary, not which side a
predicted root would occupy. It is checked from the existing pair schedule only.
Incorrect-ownership tests deliberately keep a pair but request the other boundary.
They are caller-contract negative controls, not new physical teacher candidates.
Invalid-pair controls are likewise finite-rule supervision; actual collapsed pairs
are retained in the geometry replay. Boundary side alone is not correctness.
The four-bit combiner never sees margins, coordinates, roots, labels or winner IDs.
'''
    audit_text+=f'\nSaved rule mismatches: {audit_errors}/{len(x)}. New geometry rule mismatches: {fresh_rule_errors}/{len(xf)}. New sampled-sign errors: {sign_errors}.\n'
    (OUT/'teacher_rule_audit.txt').write_text(audit_text)
    (OUT/'new_fresh_parents.json').write_text(json.dumps(parents,indent=2))
    write_csv(OUT/'new_fresh_teacher_states.csv',[dict(r,margin_1=margins[i,0],margin_2=margins[i,1],transition=int(fresh_st[i])) for i,r in enumerate(freshmeta)])
    if audit_errors or fresh_rule_errors or sign_errors:
        (OUT/'STOP.json').write_text(json.dumps(dict(status='AUDIT_FAILED_STOP',saved_errors=audit_errors,fresh_errors=fresh_rule_errors,sign_errors=sign_errors)))
        print((OUT/'STOP.json').read_text(),flush=True);return
    print(f'Audit passed on {len(x)} saved and {len(xf)} newly generated geometry states. Training four-bit combiner only.',flush=True)
    table=np.array(list(itertools.product([0.,1.],repeat=4)))
    trainmask=data['trainmask'];held=data['heldmask']
    train=np.vstack([x[trainmask],table]);target=np.r_[labels[trainmask],rule(table)]
    state_ids=(train*np.array([8,4,2,1])).sum(1).astype(int);counts=np.bincount(state_ids,minlength=16)
    weights=torch.from_numpy(1/(16*counts[state_ids]))
    model=AvailabilityCombiner();initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=.05);xt=torch.from_numpy(train);yt=torch.from_numpy(target);history=[]
    for step in range(1,2001):
        opt.zero_grad();loss=(nn.functional.cross_entropy(model(xt),yt,reduction='none')*weights).sum();loss.backward();opt.step()
        with torch.no_grad():
            logits=model(torch.from_numpy(table));tt=torch.from_numpy(rule(table));errors=int((logits.argmax(1)!=tt).sum())
            separation=float((logits.gather(1,tt[:,None])-logits.gather(1,(1-tt)[:,None])).min())
        history.append(dict(update=step,loss=float(loss.detach()),table_errors=errors,min_separation=separation))
        if errors==0 and separation>=2:break
    config=dict(seed=SEED,features=FEATURES,architecture='float64 Linear(4,2) with bias; 10 trainable parameters',
        classes=['NO_BRANCH_SWITCH_CANDIDATE','BRANCH_SWITCH_CANDIDATE_EXISTS'],
        training_cases=len(train),archived_training_cases=int(trainmask.sum()),finite_truth_table_states=16,
        supervision='Existing saved teacher availability labels for archived fit parents plus audited 16-state contract. No heldout or fresh rows in fitting.',
        optimizer='Adam lr=.05; stop at exact finite rule and minimum logit separation >=2',
        upstream=dict(lock=str(lockpath),lock_sha256=sha(lockpath),checkpoint=str(cp),sha256=sha(cp)),
        inference='Discrete state -> learned logits argmax; no AND override.',
        scope='Endpoint-associated branch-switch class under exact existing scan scheduling; not arbitrary generic multi-root subpieces.')
    saved=OUT/'combiner_before_evaluation.pt'
    digest=save_checkpoint(saved,dict(state_dict=model.state_dict(),initial_state=initial,optimizer_state=opt.state_dict(),config=config,updates=step))
    model.load_state_dict(torch.load(saved,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    metrics=[]
    def evaluate(z,y,name):
        with torch.no_grad():pred=model(torch.from_numpy(z)).argmax(1).numpy()
        metrics.append(dict(group=name,cases=len(y),positives=int(y.sum()),false_NO=int(np.sum((y==1)&(pred==0))),
            false_YES=int(np.sum((y==0)&(pred==1))),errors=int(np.sum(pred!=y))))
        return pred
    pred=evaluate(x,labels,'all_saved_states');fp=evaluate(xf,yf,'new_fresh_geometry')
    evaluate(x[trainmask],labels[trainmask],'archived_training');evaluate(x[held],labels[held],'heldout_archived')
    for group in sorted({r['group'] for r in meta}):
        mask=np.array([r['group']==group for r in meta]);evaluate(x[mask],labels[mask],group)
    for name,key in [('known_366','known_366'),('previous_120_false_positives','previous_false_positive'),
                     ('five_prior_failures','error'),('12_smooth_exceptions','smooth_control'),('all_smooth_controls','smooth_present')]:
        mask=np.array([r[key]=='True' for r in meta]);evaluate(x[mask],labels[mask],name)
    for key in ('switch_type','dimension','budget'):
        for value in sorted({r[key] for r in meta}):
            mask=np.array([r[key]==value for r in meta]);evaluate(x[mask],labels[mask],f'{key}:{value}')
    tp=evaluate(table,rule(table),'complete_16_state_rule')
    # Expanded q/aspect/power input patterns validate compression, including simultaneous changes.
    expanded=np.array(list(itertools.product([0.,1.],repeat=6)))  # transition,q,aspect,power,valid,owner
    ez=np.column_stack((expanded[:,0],np.any(expanded[:,1:4],axis=1),expanded[:,4:6])).astype(float)
    evaluate(ez,rule(ez),'expanded_64_branch_signature_states')
    controls=[]
    for valid,owner in ((0,1),(1,0),(0,0)):
        z=x.copy();z[:,2]=valid;z[:,3]=owner
        evaluate(z,np.zeros(len(z),dtype=int),f'contract_control_valid_{valid}_owner_{owner}');controls.append(z)
    for col,name in enumerate(FEATURES):
        for val in (0,1):
            mask=table[:,col]==val;evaluate(table[mask],rule(table[mask]),f'truth_table_{name}:{val}')
    rng=np.random.default_rng(SEED+2);heldstate=rng.integers(0,2,(10000,4)).astype(float)
    evaluate(heldstate,rule(heldstate),'fresh_discrete_state_sequences')
    write_csv(OUT/'truth_table.csv',[dict(zip(FEATURES,row.tolist()),teacher=int(rule(table)[i]),prediction=int(tp[i])) for i,row in enumerate(table)])
    rows=[dict(group=r['group'],case=r['case'],boundary_side_metadata=r['boundary_side'],switch_type=r['switch_type'],
        **dict(zip(FEATURES,x[i].astype(int).tolist())),teacher=int(labels[i]),prediction=int(pred[i]),error=bool(labels[i]!=pred[i]),
        known_366=r['known_366'],previous_false_positive=r['previous_false_positive'],prior_five_failure=r['error'],smooth_exception=r['smooth_control']) for i,r in enumerate(meta)]
    write_csv(OUT/'availability_replay.csv',rows)
    write_csv(OUT/'required_cases_replay.csv',[r for r in rows if any(r[k]=='True' for k in ('known_366','previous_false_positive','prior_five_failure','smooth_exception'))])
    write_csv(OUT/'new_fresh_replay.csv',[dict(r,**dict(zip(FEATURES,xf[i].astype(int).tolist())),prediction=int(fp[i]),error=bool(yf[i]!=fp[i])) for i,r in enumerate(freshmeta)])
    write_csv(OUT/'availability_validation.csv',metrics);write_csv(OUT/'training_history.csv',history)
    (OUT/'learned_parameters.json').write_text(json.dumps(dict(config=config,initial={k:v.tolist() for k,v in initial.items()},learned={k:v.tolist() for k,v in model.state_dict().items()}),indent=2))
    unchanged=all(sha(Path(p))==h for p,h in protected.items());assert unchanged
    passed=all(r['errors']==0 for r in metrics)
    result=dict(operator='branch_switch_candidate_availability',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        checkpoint=str(saved),sha256=digest,config=config,metrics=metrics,updates=step,
        saved_teacher_rule_errors=audit_errors,new_fresh_teacher_rule_errors=fresh_rule_errors,new_fresh_sign_errors=sign_errors,
        protected_files=len(protected),all_protected_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-start,
        implementation=str(Path(__file__)),implementation_sha256=sha(Path(__file__)),no_candidate_prediction=True,no_recursion=True,no_search=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:
        lock=OUT/'branch_switch_candidate_availability_locked.json';lock.write_text(json.dumps(result,indent=2))
        lock.with_suffix('.sha256').write_text(sha(lock)+'  '+lock.name+'\n')
    summary=[f"branch_switch_candidate_availability = {result['status']}",audit_text,
        f"Random float64 affine router, 10 parameters, {step} Adam updates. Checkpoint saved/reloaded before evaluation.",
        'Weights and biases in learned_parameters.json. Upstream sign transition reused unchanged.',
        'Results below overlap; invalid/incorrect-owner controls are separately labeled caller-contract tests.']
    summary += [f"{r['group']}: n={r['cases']}, positive={r['positives']}, false NO={r['false_NO']}, false YES={r['false_YES']}" for r in metrics]
    summary += [f"All {len(protected)} protected hashes unchanged. Runtime {result['elapsed_seconds']:.3f}s excluding implementation/report inspection.",
        f'Checkpoint SHA256: {digest}', 'No candidate-location model, smooth-root change, recursive partitioning or search.']
    (OUT/'availability_combiner_summary.txt').write_text('\n'.join(summary)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','metrics')},indent=2),flush=True)
    print(json.dumps(metrics[:20],indent=2),flush=True)


if __name__=='__main__':run()
