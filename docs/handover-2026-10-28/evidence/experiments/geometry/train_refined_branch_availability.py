"""Only refined scan-pair availability routing. Existing arithmetic is frozen."""
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_parent_split_selector import safe_json
from experiments.geometry.validate_terminal_candidate_full import original_parents
from experiments.geometry.validate_stationary_three_state_router import load_router
from experiments.geometry.audit_stationary_candidate_availability import scan_roots,math_roots
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT=ref.ROOT;OUT=ROOT/'refined_branch_switch_availability_training';SEED=28382026
NAMES=['margin_1','margin_2','q_change','aspect_change','signature_change','valid_pair','boundary_side']


class RefinedAvailability(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(7,12),nn.Tanh(),nn.Linear(12,1)).double()
    def forward(self,x):return self.net(x).squeeze(1)


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    old=ROOT/'branch_switch_candidate_availability_training';auditdir=ROOT/'branch_availability_false_positive_audit'
    original=original_parents();oldfresh=json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text())
    previous=json.loads((ROOT/'stationary_candidate_availability_audit/fresh_parents.json').read_text())
    independent=json.loads((old/'additional_fresh_parents.json').read_text())
    prior=list(csv.DictReader((old/'availability_predictions.csv').open()))
    scans=list(csv.DictReader((auditdir/'all_scan_states.csv').open()))
    states=[]
    def getparent(row):
        pi=int(row['case'].split(':')[0]);g=row['group']
        return original[pi] if g=='archived_geometry' else oldfresh[str(pi)] if g=='existing_fresh_geometry' else independent[pi] if g=='independent_fresh_geometry' else previous[pi]
    for i,r in enumerate(scans):
        states.append(dict(group=r['group'],case=r['case'],parent_key=prior[i]['parent_key'],parent=getparent(r),
            l=float(r['lower']),r=float(r['upper']),side=int(r['boundary_side']),label=int(r['teacher_label']),
            known=prior[i]['known_positive']=='True',previous_FP=prior[i]['error']=='True' and prior[i]['prediction']=='1',
            smooth_control=r['pair_strict_sign_transition']=='True' and r['teacher_label']=='0',
            smooth_present=r['smooth_fixed_root_present']=='True',distance=prior[i]['candidate_ULP_distance']))
    for r in csv.DictReader((auditdir/'fresh_boundary_scan_states.csv').open()):
        states.append(dict(group='previous_fresh_perturbations',case=r['case'],parent_key='perturbed:'+r['case'].split(':')[0],parent=getparent(r),
            l=float(r['lower']),r=float(r['upper']),side=int(r['boundary_side']),label=int(r['teacher_label']),
            known=False,previous_FP=False,smooth_control=False,smooth_present=r['smooth_fixed_root_present']=='True',distance=''))
    # New geometry and labels, never used in quick checkpoint selection.
    rng=np.random.default_rng(SEED);geometry=Audit();newparents=[]
    for d in (2,3,4):
        for n in (5,10,24):
            scale=10**rng.uniform(-2,2);s=scale*np.exp(rng.uniform(-1,1,d));lo=scale*rng.uniform(-.3,.3,d)
            parent=dict(d=d,n=n,lower=lo.tolist(),upper=(lo+s).tolist());pi=len(newparents);newparents.append(parent)
            for axis in range(d):
                a=geometry;a.setup(parent,axis);edge=max(1e-15,4*np.spacing(abs(a.lo[axis])+abs(a.hi[axis]))/a.a)
                xx=np.linspace(edge,1-edge,max(2049,n*32+1));cc=a.counts(xx)[:,0]
                lower=np.array([xx[np.flatnonzero(cc<k)[-1]] for k in range(1,n+1)])
                upper=np.array([xx[np.flatnonzero(cc>=k)[0]] for k in range(1,n+1)])
                for _ in range(60):
                    mid=(lower+upper)/2;mask=a.counts(mid)[:,0]<np.arange(1,n+1)
                    lower=np.where(mask,mid,lower);upper=np.where(mask,upper,mid)
                for k in range(1,n):
                    br=a.breaks(upper[k-1],lower[k],k)
                    for j,(l,r) in enumerate(zip(br[:-1],br[1:])):
                        f=a.formula(np.array([(l+r)/2]),k);iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
                        coeff=(1/(a.h*np.power(float(k if iq==0 else n-k),1/d)),a.B,a.a,.05*power,iq,ia)
                        roots=scan_roots(lambda c:a.formula(c,k)[1],l,r)
                        smooth=[z for z in math_roots(coeff) if mp.mpf(l)<z<mp.mpf(r)] if roots else []
                        roots=[] if smooth else roots
                        for side in (0,1):
                            candidates=[c for c in roots if (0 if abs(c-l)<=abs(r-c) else 1)==side]
                            distance=min((min(abs(int(np.float64(c).view(np.uint64))-int(np.float64(b).view(np.uint64))) for b in (l,r)) for c in candidates),default='')
                            states.append(dict(group='new_independent_geometry',case=f'{pi}:{axis}:{k}:{j}',parent_key='new:'+str(pi),parent=parent,
                                l=float(l),r=float(r),side=side,label=int(bool(candidates)),known=False,previous_FP=False,
                                smooth_control=False,smooth_present=bool(smooth),distance=distance))
    (OUT/'new_fresh_parents.json').write_text(safe_json(newparents))
    router,refs=load_router();inputs=[];other=[];teacher_derivatives=[];metadata=[]
    for p in states:
        a=geometry;axis,k=map(int,p['case'].split(':')[1:3]);a.setup(p['parent'],axis);l=p['l'];r=p['r']
        xx=np.linspace(np.nextafter(l,r),np.nextafter(r,l),129);ii=[0,1] if p['side']==0 else [127,128];c=xx[ii]
        f=a.formula(c,k);iq=f[3];ia=f[4];power=f[5][np.arange(2),ia];n=a.n;d=a.d
        A=1/(a.h*np.power(np.where(iq==0,k,n-k).astype(float),1/d))
        inputs.extend(np.column_stack((A,np.full(2,a.B),np.full(2,a.a),.05*power,c,iq,ia)))
        qc=int(iq[0]!=iq[1]);ac=int(ia[0]!=ia[1]);pc=int(power[0]!=power[1]);valid=l<c[0]<c[1]<r
        other.append([qc,ac,int(qc or ac or pc),int(valid),p['side']]);teacher_derivatives.append(f[1])
        typ='multiple' if qc+ac+pc>1 else 'q_max_only' if qc else 'aspect_only' if ac or pc else 'no_switch'
        metadata.append(dict(group=p['group'],case=p['case'],parent_key=p['parent_key'],dimension=d,budget=n,child_left=k,child_right=n-k,
            lower=l,upper=r,boundary_side=p['side'],sample_1=float(c[0]),sample_2=float(c[1]),label=p['label'],
            switch_type=typ,q_switch=qc,aspect_switch=ac,aspect_power_switch=pc,valid_pair=valid,
            known_366=p['known'],previous_false_positive=p['previous_FP'],smooth_control=p['smooth_control'],
            smooth_present=p['smooth_present'],candidate_endpoint_ULPs=p['distance']))
    inp=np.array(inputs)
    with torch.no_grad():terms=router.term_assembly.terms(torch.from_numpy(inp)).numpy()
    den=abs(terms[:,0])+abs(terms[:,1]);margins=np.divide(terms.sum(1),den,out=np.zeros(len(terms)),where=den>0).reshape(-1,2)
    assert np.isfinite(margins).all()
    x=np.column_stack((margins,np.array(other)));y=np.array([p['label'] for p in states]);td=np.array(teacher_derivatives)
    feature_errors=np.sign(margins)!=np.sign(td)
    # Pure audit only: inference never overrides the trained logits with this rule.
    geometric_rule=(margins[:,0]*margins[:,1]<0)&(x[:,4]>0)&(x[:,5]>0)
    feature_rule_disagreements=int(np.sum(geometric_rule!=y))
    archivekeys=sorted({p['parent_key'] for p in states if p['group']=='archived_geometry'});heldkeys=set(archivekeys[::5])
    trainmask=np.array([p['group']=='archived_geometry' and p['parent_key'] not in heldkeys for p in states])
    heldmask=np.array([p['group']=='archived_geometry' and p['parent_key'] in heldkeys for p in states])
    oldfreshmask=np.array([p['group'] in ('existing_fresh_geometry','new_fresh_geometry','saved_failure_geometry') for p in states])
    def sample(mask,positives,negatives):
        chosen=[]
        for label,cap in ((1,positives),(0,negatives)):
            ids=np.flatnonzero(mask&(y==label));rng.shuffle(ids);chosen.extend(ids[:cap])
        return np.array(chosen,dtype=int)
    quicktrain=sample(trainmask,300,1500)
    controls=np.array([i for i,p in enumerate(states) if trainmask[i] and (p['previous_FP'] or p['smooth_present'])],dtype=int)
    quicktrain=np.unique(np.r_[quicktrain,controls])
    selection=sample(heldmask|oldfreshmask,300,900)
    plan=dict(seed=SEED,features=NAMES,architecture='7 -> 12 tanh -> 1, float64, 109 parameters',
        feature_source='Frozen refined-sqrt term assembly with deterministic per-sample active branches. m=(T1+T2)/(|T1|+|T2|), zero when both terms zero. No teacher derivative substitution.',
        scaffold='Exact deterministic 129-point interior scan coordinates, active identities, sample validity and boundary ownership. No final candidate location input/output.',
        quick_train_indices=quicktrain.tolist(),quick_selection_indices=selection.tolist(),
        gates='Zero quick selection and fit errors before full continuation; zero held-out, independent fresh, known-366, prior-FP, smooth-control and complete-pool errors for lock.',
        quick_max_updates=4000,full_max_updates=4000,learning_rate=.01)
    (OUT/'training_plan.json').write_text(safe_json(plan))
    np.savez_compressed(OUT/'feature_dataset.npz',features=x,labels=y,teacher_derivatives=td,learned_terms=terms,
                        trainmask=trainmask,heldmask=heldmask,quicktrain=quicktrain,selection=selection)
    model=RefinedAvailability();initial=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.01);history=[]
    def stage(name,trainids,maxupdates):
        xt=torch.from_numpy(x[trainids]);yt=torch.from_numpy(y[trainids].astype(float));xv=torch.from_numpy(x[selection]);yv=torch.from_numpy(y[selection].astype(float))
        wt=torch.where(yt>0,.5/float(yt.sum()),.5/float((1-yt).sum()))
        best=None;bestkey=None;beststep=0
        for step in range(1,maxupdates+1):
            opt.zero_grad();loss=(nn.functional.binary_cross_entropy_with_logits(model(xt),yt,reduction='none')*wt).sum();loss.backward();opt.step()
            if step%25:continue
            with torch.no_grad():
                lp=model(xt);lv=model(xv);te=int(((lp>0)!=(yt>0)).sum());ve=int(((lv>0)!=(yv>0)).sum())
                vl=float(nn.functional.binary_cross_entropy_with_logits(lv,yv))
            history.append(dict(stage=name,update=step,train_errors=te,selection_errors=ve,selection_loss=vl))
            key=(ve,te,vl)
            if bestkey is None or key<bestkey:
                bestkey=key;best=copy.deepcopy(model.state_dict());bestopt=copy.deepcopy(opt.state_dict());beststep=step
            if te==0 and ve==0:break
            if step-beststep>=750:break
        model.load_state_dict(best);opt.load_state_dict(bestopt)
        cp=OUT/(name+'_before_evaluation.pt')
        h=save_checkpoint(cp,dict(state_dict=model.state_dict(),optimizer_state=opt.state_dict(),initial_state=initial,
            seed=SEED,stage=name,best_update=beststep,stop_update=step,features=NAMES,architecture=plan['architecture']))
        print(safe_json(dict(stage=name,best_update=beststep,stop_update=step,training_errors=bestkey[1],selection_errors=bestkey[0],sha256=h)),flush=True)
        return dict(checkpoint=str(cp),sha256=h,best_update=beststep,stop_update=step,training_errors=bestkey[1],selection_errors=bestkey[0])
    quick=stage('quick',quicktrain,4000)
    quickpass=quick['training_errors']==0 and quick['selection_errors']==0
    full=stage('full',np.flatnonzero(trainmask),4000) if quickpass else None
    current=full or quick;model.eval().requires_grad_(False)
    with torch.no_grad():logits=model(torch.from_numpy(x)).numpy()
    pred=logits>0
    def metric(ids):
        ids=np.asarray(ids,dtype=int);yy=y[ids];pp=pred[ids]
        return dict(examples=len(ids),positives=int(yy.sum()),negatives=int(len(ids)-yy.sum()),
            false_NO=int(np.sum((yy==1)&~pp)),false_YES=int(np.sum((yy==0)&pp)),errors=int(np.sum(yy!=pp)))
    reports=dict(all=metric(np.arange(len(y))),quick_train=metric(quicktrain),quick_selection=metric(selection),
        full_train=metric(np.flatnonzero(trainmask)),heldout_archived=metric(np.flatnonzero(heldmask)))
    for name,test in [('known_366',lambda p:p['known']),('previous_120_false_positives',lambda p:p['previous_FP']),
                      ('12_smooth_exceptions',lambda p:p['smooth_control']),('all_smooth_controls',lambda p:p['smooth_present']),
                      ('independent_previous_fresh',lambda p:p['group']=='independent_fresh_geometry'),
                      ('new_independent_fresh',lambda p:p['group']=='new_independent_geometry'),
                      ('previous_fresh_perturbations',lambda p:p['group']=='previous_fresh_perturbations')]:
        reports[name]=metric([i for i,p in enumerate(states) if test(p)])
    assert reports['known_366']['examples']==366 and reports['previous_120_false_positives']['examples']==120 and reports['12_smooth_exceptions']['examples']==12
    breakdown=[]
    for field in ('group','switch_type','dimension','budget'):
        for value in sorted({p[field] for p in metadata}):
            breakdown.append(dict(category=field,value=value,**metric([i for i,p in enumerate(metadata) if p[field]==value])))
    for name,test in [('1_ULP',lambda d:d==1),('2_to_4_ULP',lambda d:2<=d<=4),('5_to_16_ULP',lambda d:5<=d<=16),('above_16_ULP',lambda d:d>16)]:
        breakdown.append(dict(category='candidate_distance',value=name,**metric([i for i,p in enumerate(metadata) if p['candidate_endpoint_ULPs']!='' and test(int(p['candidate_endpoint_ULPs']))])))
    rows=[dict(p,prediction=int(pred[i]),logit=float(logits[i]),error=bool(pred[i]!=y[i]),margin_1=x[i,0],margin_2=x[i,1],feature_sign_errors=int(feature_errors[i].sum())) for i,p in enumerate(metadata)]
    ref.write_csv(OUT/'availability_predictions.csv',rows);ref.write_csv(OUT/'availability_breakdown.csv',breakdown)
    ref.write_csv(OUT/'replay_required_cases.csv',[r for r in rows if r['known_366'] or r['previous_false_positive'] or r['smooth_control']])
    ref.write_csv(OUT/'availability_errors.csv',[r for r in rows if r['error']] or [dict(error=False)])
    ref.write_csv(OUT/'training_history.csv',history)
    (OUT/'learned_parameters.json').write_text(safe_json({k:v.tolist() for k,v in model.state_dict().items()}))
    assert all(ref.sha(Path(k))==h for k,h in protected.items())
    passed=quickpass and reports['all']['errors']==0
    result=dict(operator='branch_switch_candidate_availability',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        quick_passed=quickpass,architecture=plan['architecture'],parameters=sum(p.numel() for p in model.parameters()),features=NAMES,
        quick=quick,full=full,metrics=reports,learned_vs_teacher_sample_sign_disagreements=int(feature_errors.sum()),
        feature_rule_label_disagreements=feature_rule_disagreements,feature_source=plan['feature_source'],deterministic_scaffold=plan['scaffold'],
        checkpoint=current['checkpoint'],checkpoint_sha256=current['sha256'],refs=refs,elapsed_seconds=time.monotonic()-started,
        protected_files=len(protected),all_protected_hashes_unchanged=True,no_architecture_increase=True,no_location_prediction=True,no_recursion=True,no_search=True,
        implementation_sha256=ref.sha(Path(__file__)))
    if passed:
        path=OUT/'branch_switch_candidate_availability_locked.json';path.write_text(safe_json(result));h=ref.sha(path)
        path.with_suffix('.sha256').write_text(h+'  '+path.name+'\n');result['lock_manifest_sha256']=h
    (OUT/'results.json').write_text(safe_json(result));(OUT/'protected_hashes.json').write_text(safe_json(protected))
    print(safe_json(result),flush=True)


if __name__=='__main__':run()
