"""Only a small branch-boundary availability classifier; no location output."""
import copy
import csv
import json
import time
from collections import defaultdict, Counter
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn

from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.audit_stationary_candidate_availability import scan_roots, math_roots
from experiments.geometry.validate_terminal_candidate_full import original_parents
from experiments.geometry.train_parent_split_selector import safe_json
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT=ref.ROOT;SOURCE=ROOT/'stationary_candidate_availability_audit'
OUT=ROOT/'branch_switch_candidate_availability_training';SEED=28372026


class BranchAvailability(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(features,24),nn.Tanh(),nn.Linear(24,1)).double()
    def forward(self,x):return self.net(x).squeeze(1)


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('manifest','locked','validated'))})
    audit=Audit();original=original_parents()
    oldfresh=json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text())
    previousfresh=json.loads((SOURCE/'fresh_parents.json').read_text())
    pieces=[]
    for r in csv.DictReader((SOURCE/'stationary_availability_all.csv').open()):
        if not r['group'].endswith('geometry'):continue
        pi,axis,k,j=map(int,r['case'].split(':'))
        parent=original[pi] if r['group']=='archived_geometry' else oldfresh[str(pi)] if r['group']=='existing_fresh_geometry' else previousfresh[pi]
        pieces.append(dict(group=r['group'],case=r['case'],parent_key=r['group']+':'+str(pi),parent=parent,axis=axis,k=k,l=float(r['lower']),r=float(r['upper'])))
    known={(r['group'],r['case']) for r in csv.DictReader((SOURCE/'stationary_availability_mismatches.csv').open())
           if r['group'].endswith('geometry') and r['teacher_available']=='True' and r['learned_available']=='False'}
    assert len(known)==366
    rng=np.random.default_rng(SEED)
    additional=[]
    for d in (2,3,4):
        for n in (4,8,16,29):
            scale=10**rng.uniform(-2,2);s=scale*np.exp(rng.uniform(-1.1,1.1,d));lo=scale*rng.uniform(-.3,.3,d)
            parent=dict(d=d,n=n,lower=lo.tolist(),upper=(lo+s).tolist());pi=len(additional);additional.append(parent)
            for axis in range(d):
                audit.setup(parent,axis);a=audit
                edge=max(1e-15,4*np.spacing(abs(a.lo[axis])+abs(a.hi[axis]))/a.a)
                x=np.linspace(edge,1-edge,max(2049,n*32+1));counts=a.counts(x)[:,0]
                lower=np.array([x[np.flatnonzero(counts<k)[-1]] for k in range(1,n+1)])
                upper=np.array([x[np.flatnonzero(counts>=k)[0]] for k in range(1,n+1)])
                for _ in range(60):
                    mid=(lower+upper)/2;mask=a.counts(mid)[:,0]<np.arange(1,n+1)
                    lower=np.where(mask,mid,lower);upper=np.where(mask,upper,mid)
                for k in range(1,n):
                    br=a.breaks(upper[k-1],lower[k],k)
                    for j,(l,r) in enumerate(zip(br[:-1],br[1:])):
                        pieces.append(dict(group='independent_fresh_geometry',case=f'{pi}:{axis}:{k}:{j}',parent_key=f'independent:{pi}',parent=parent,axis=axis,k=k,l=float(l),r=float(r)))
    (OUT/'additional_fresh_parents.json').write_text(safe_json(additional))
    rows=[];features=[];label_details=[];feature_names=[];offsets=(-512,-16,-1,0,1,16,512)
    for idx,p in enumerate(pieces):
        parent=p['parent'];audit.setup(parent,p['axis']);a=audit;k=p['k'];l=p['l'];r=p['r'];n=parent['n'];d=parent['d']
        f=a.formula(np.array([(l+r)/2]),k);iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
        coeff=(1/(a.h*np.power(float(k if iq==0 else n-k),1/d)),a.B,a.a,.05*power,iq,ia)
        roots=scan_roots(lambda c:a.formula(c,k)[1],l,r)
        # Smooth stationary roots are explicitly a different candidate class.
        smooth=[z for z in math_roots(coeff) if mp.mpf(l)<z<mp.mpf(r)] if roots else []
        branchroots=roots if roots and not smooth else []
        if (p['group'],p['case']) in known:assert branchroots
        # Labels are per oriented subpiece-boundary state. Shared boundaries may
        # have different availability in their left and right incident subpieces.
        targets=[[],[]]
        for c in branchroots:targets[0 if abs(c-l)<=abs(r-c) else 1].append(c)
        for side,b in enumerate((l,r)):
            bits=int(np.float64(b).view(np.uint64));xx=np.array([np.uint64(bits+off).view(np.float64) for off in offsets])
            assert np.all((xx>0)&(xx<1))
            val=a.formula(xx,k);q=val[6];aspect=val[7]
            qscale=np.maximum(q[:,0],q[:,1]);qm=(q[:,0]-q[:,1])/qscale
            am=(aspect[:,0]-aspect[:,1])/np.maximum(1.,np.maximum(abs(aspect[:,0]),abs(aspect[:,1])))
            shape=np.zeros(5);shape[:d]=a.s/a.s.max()
            base=[d/5,np.log1p(n)/6,k/n,(n-k)/n,p['axis']/max(1,d-1),b,side,r-l,*shape]
            names=['dimension','log_budget','left_count_fraction','right_count_fraction','axis','boundary','boundary_side','subpiece_width']+[f'side_{j}' for j in range(5)]
            for h,off in enumerate(offsets):
                # Preserve tiny comparison evidence using an invertible scaled
                # asinh; do not use root coordinates, teacher winner or labels.
                base.extend([float(val[3][h]),float(val[4][h]),float(np.arcsinh(qm[h]/np.finfo(float).eps)/40),
                             float(np.arcsinh(am[h]/np.finfo(float).eps)/40),float(np.log1p(q[h,0]/a.h)),float(np.log1p(q[h,1]/a.h)),
                             float(aspect[h,0]/20),float(aspect[h,1]/20)])
                names.extend([f'{name}_{off}' for name in ('q_active','aspect_active','q_margin','aspect_margin','q_left','q_right','aspect_left','aspect_right')])
            qswitch=len(set(val[3].tolist()))>1;aswitch=len(set(val[4].tolist()))>1
            typ='multiple' if qswitch and aswitch else 'q_max' if qswitch else 'aspect_max' if aswitch else 'no_observed_switch'
            yes=bool(targets[side]);nearest=min((min(abs(c-l),abs(r-c)) for c in targets[side]),default=None)
            ulp=min((min(abs(int(np.float64(c).view(np.uint64))-int(np.float64(z).view(np.uint64))) for z in (l,r)) for c in targets[side]),default=None)
            rows.append(dict(group=p['group'],case=p['case'],parent_key=p['parent_key'],dimension=d,budget=n,child_left=k,child_right=n-k,
                boundary_side=side,boundary=b,lower=l,upper=r,label=int(yes),known_miss=(p['group'],p['case']) in known,
                known_positive=(p['group'],p['case']) in known and yes,switch_type=typ,q_switch=qswitch,aspect_switch=aswitch,
                candidate_distance=nearest,candidate_ULP_distance=ulp,smooth_roots_present=bool(smooth)))
            features.append(base);feature_names=names
            label_details.append(dict(group=p['group'],case=p['case'],boundary_side=side,teacher_branch_candidates=targets[side],
                                      teacher_roots=roots,high_precision_smooth_roots=[mp.nstr(z,60) for z in smooth]))
        if (idx+1)%2000==0:print(safe_json(dict(stage='labels',pieces=idx+1,seconds=time.monotonic()-start)),flush=True)
    x=np.array(features);y=np.array([r['label'] for r in rows]);assert np.isfinite(x).all()
    assert sum(r['known_positive'] for r in rows)==366
    archkeys=sorted({r['parent_key'] for r in rows if r['group']=='archived_geometry'})
    heldkeys=set(archkeys[::5]);trainmask=np.array([r['group']=='archived_geometry' and r['parent_key'] not in heldkeys for r in rows])
    # Old fresh parents are selection/validation only; newly generated parents
    # remain untouched until the saved quick-screen model is evaluated.
    selmask=np.array([(r['group']=='archived_geometry' and r['parent_key'] in heldkeys) or r['group'] in ('existing_fresh_geometry','new_fresh_geometry','saved_failure_geometry') for r in rows])
    def subset(mask,positive,negative):
        out=[]
        for label,cap in [(1,positive),(0,negative)]:
            ii=np.flatnonzero(mask&(y==label));rng.shuffle(ii);out.extend(ii[:cap])
        out=np.array(out);rng.shuffle(out);return out
    train=subset(trainmask,256,768);selection=subset(selmask,192,384)
    plan=dict(seed=SEED,architecture=f'{x.shape[1]} local normalized geometry features -> Linear(24) -> tanh -> Linear(1)',
        maximum_updates=2500,plateau_patience=500,check_every=25,lr=.003,
        quick_gate='zero false NO and <=0.5% total selection error; if failed STOP without continuation or architecture changes',
        final_gate='zero false NO on known 366; <=0.5% held-out archived and fresh errors, with zero dimension/type subgroup above 2% error',
        train_indices=train.tolist(),selection_indices=selection.tolist(),feature_names=feature_names,
        feature_contract='Boundary location/oriented subpiece domain, normalized geometry/counts and competing branch values/identities at fixed -512,-16,-1,0,1,16,512 ULP offsets. No teacher candidate coordinate, derivative-scan outcome, root, label or winning identity inputs.',
        label_contract='Exact dynamic 129-sample teacher scan, excluding subpieces with high-precision fixed-branch smooth roots; associate residual branch-switch candidates with their nearest oriented subpiece endpoint. No endpoint score-equivalence suppression.',
        zero_switch_control_policy='Keep all endpoint states, including ordinary boundaries with no observed active switch, as negative controls.')
    (OUT/'training_plan.json').write_text(safe_json(plan))
    ref.write_csv(OUT/'boundary_dataset.csv',rows)
    (OUT/'teacher_label_details.json').write_text(safe_json(label_details))
    np.savez_compressed(OUT/'boundary_features.npz',features=x,labels=y,train=train,selection=selection,feature_names=np.array(feature_names))
    model=BranchAvailability(x.shape[1]);initial=copy.deepcopy(model.state_dict());optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    xt=torch.from_numpy(x[train]);yt=torch.from_numpy(y[train].astype(float));xv=torch.from_numpy(x[selection]);yv=torch.from_numpy(y[selection].astype(float))
    weights=torch.where(yt>0,.5/float(yt.sum()),.5/float((1-yt).sum()))
    best=None;best_key=None;best_step=0;history=[]
    for step in range(1,2501):
        optimizer.zero_grad();loss=(nn.functional.binary_cross_entropy_with_logits(model(xt),yt,reduction='none')*weights).sum();loss.backward();optimizer.step()
        if step%25:continue
        with torch.no_grad():
            logits=model(xv);pred=logits>0;fn=int(((yv>0)&~pred).sum());fp=int(((yv==0)&pred).sum());vl=float(nn.functional.binary_cross_entropy_with_logits(logits,yv))
        history.append(dict(update=step,training_loss=float(loss.detach()),selection_false_NO=fn,selection_false_YES=fp,selection_loss=vl))
        key=(fn+fp,vl)
        if best_key is None or key<best_key:
            best_key=key;best_step=step;best=copy.deepcopy(model.state_dict());best_opt=copy.deepcopy(optimizer.state_dict())
        if fn==0 and (fn+fp)/len(selection)<=.005:break
        if step-best_step>=500:break
    model.load_state_dict(best);model.eval().requires_grad_(False)
    cp=OUT/'availability_quick_before_evaluation.pt'
    digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),optimizer_state=best_opt,initial_state=initial,seed=SEED,best_update=best_step,stop_update=step,
                                   feature_names=feature_names,architecture=plan['architecture']))
    print(safe_json(dict(stage='saved_quick_checkpoint',best_update=best_step,stop_update=step,sha256=digest)),flush=True)
    ref.write_csv(OUT/'training_history.csv',history)
    with torch.no_grad():logits=model(torch.from_numpy(x)).numpy()
    pred=logits>0
    def metric(ids):
        ids=np.asarray(ids,dtype=int);t=y[ids];p=pred[ids]
        return dict(examples=len(ids),positive=int(t.sum()),errors=int(np.sum(t!=p)),false_NO=int(np.sum((t==1)&~p)),false_YES=int(np.sum((t==0)&p)),
                    error_rate=float(np.mean(t!=p)) if len(ids) else 0.)
    reports=dict(quick_training=metric(train),quick_selection=metric(selection),
        heldout_archived=metric([i for i,r in enumerate(rows) if r['parent_key'] in heldkeys]),
        independent_fresh=metric([i for i,r in enumerate(rows) if r['group']=='independent_fresh_geometry']),
        known_366=metric([i for i,r in enumerate(rows) if r['known_positive']]),all=metric(np.arange(len(rows))))
    breakdown=[]
    for field in ('group','dimension','budget','switch_type'):
        for value in sorted({r[field] for r in rows}):
            ids=[i for i,r in enumerate(rows) if r[field]==value]
            breakdown.append(dict(category=field,value=value,**metric(ids)))
    for name,test in [('1_ULP',lambda d:d==1),('2_to_4_ULP',lambda d:2<=d<=4),('5_to_16_ULP',lambda d:5<=d<=16),('above_16_ULP',lambda d:d>16)]:
        ids=[i for i,r in enumerate(rows) if r['candidate_ULP_distance'] is not None and test(r['candidate_ULP_distance'])]
        breakdown.append(dict(category='positive_candidate_endpoint_distance',value=name,**metric(ids)))
    predictions=[dict(r,prediction=int(pred[i]),logit=float(logits[i]),error=bool(pred[i]!=y[i]),training=bool(i in set(train)),selection=bool(i in set(selection))) for i,r in enumerate(rows)]
    ref.write_csv(OUT/'availability_predictions.csv',predictions)
    ref.write_csv(OUT/'availability_errors.csv',[r for r in predictions if r['error']] or [dict(error=False)])
    ref.write_csv(OUT/'known_366_replay.csv',[r for r in predictions if r['known_positive']])
    ref.write_csv(OUT/'availability_breakdown.csv',breakdown)
    # Inspect indistinguishable rounded feature vectors, without fitting a new model.
    ambiguous={}
    for decimals in (None,12,8):
        xx=x if decimals is None else np.round(x,decimals)
        bins=defaultdict(set)
        for feature,label in zip(xx,y):bins[feature.tobytes()].add(int(label))
        ambiguous[str(decimals)]=sum(len(v)>1 for v in bins.values())
    q=reports['quick_selection'];quick_pass=q['false_NO']==0 and q['error_rate']<=.005
    passed=quick_pass and reports['known_366']['false_NO']==0 and all(reports[k]['error_rate']<=.005 for k in ('heldout_archived','independent_fresh'))
    passed=passed and not any(r['error_rate']>.02 for r in breakdown if r['category'] in ('dimension','switch_type'))
    assert all(ref.sha(Path(k))==v for k,v in protected.items())
    result=dict(operator='branch_switch_candidate_availability',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        quick_screen_passed=quick_pass,architecture=plan['architecture'],parameters=sum(p.numel() for p in model.parameters()),
        training_examples=len(train),selection_examples=len(selection),best_update=best_step,stop_update=step,metrics=reports,
        conflicting_feature_vectors=ambiguous,checkpoint=str(cp),sha256=digest,
        elapsed_seconds=time.monotonic()-start,all_protected_hashes_unchanged=True,protected_files=len(protected),
        no_model_size_increase=True,no_candidate_location_prediction=True,no_recursion=True,no_global_search=True,
        stop_reason=None if passed else 'Availability smoke gate failed; no larger training run, architecture increase, or promotion.',
        implementation_sha256=ref.sha(Path(__file__)))
    if passed:
        (OUT/'candidate_manifest.json').write_text(safe_json(result))
    (OUT/'results.json').write_text(safe_json(result));(OUT/'protected_hashes.json').write_text(safe_json(protected))
    print(safe_json(result),flush=True)


if __name__=='__main__':run()
