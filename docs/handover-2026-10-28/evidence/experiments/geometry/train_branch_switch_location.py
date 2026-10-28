"""Audit and fit branch-switch locations only; all other modules are read-only."""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq

from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_sign_transition import SignTransition
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_branch_availability_combiner import AvailabilityCombiner
from experiments.geometry.validate_terminal_candidate_full import original_parents
from experiments.geometry.validate_stationary_three_state_router import load_router
from experiments.geometry.verify_learned_continuous_recursion import Selector
from experiments.geometry.train_subpiece_minimum_selector import canonical,teacher_select
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'branch_switch_candidate_location_training';SEED=28422026
SOURCE=ROOT/'refined_branch_switch_availability_training'
COMB=ROOT/'branch_switch_availability_combiner_training'


def readcsv(p):return list(csv.DictReader(p.open()))


def load_locked_models():
    sm=json.loads((ROOT/'branch_switch_sign_transition_training/branch_switch_sign_transition_locked.json').read_text())
    am=json.loads((COMB/'branch_switch_candidate_availability_locked.json').read_text())
    for m in (sm,am):assert m['status']=='LOCKED' and sha(Path(m['checkpoint']))==m['sha256']
    s=SignTransition(MagnitudeThreshold());s.load_state_dict(torch.load(sm['checkpoint'],weights_only=False)['state_dict']);s.eval().requires_grad_(False)
    a=AvailabilityCombiner();a.load_state_dict(torch.load(am['checkpoint'],weights_only=False)['state_dict']);a.eval().requires_grad_(False)
    return s,a,dict(sign_transition=sm['sha256'],availability=am['sha256'])


def collect():
    s,comb,refs=load_locked_models()
    original=original_parents()
    oldfresh=json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text())
    prev=json.loads((ROOT/'stationary_candidate_availability_audit/fresh_parents.json').read_text())
    independent=json.loads((ROOT/'branch_switch_candidate_availability_training/additional_fresh_parents.json').read_text())
    new=json.loads((SOURCE/'new_fresh_parents.json').read_text())
    newest=json.loads((COMB/'new_fresh_parents.json').read_text())
    perturb=iter(readcsv(ROOT/'branch_availability_false_positive_audit/fresh_boundary_scan_states.csv'))
    def parent(group,case):
        i=int(case.split(':')[0])
        return original[i] if group=='archived_geometry' else oldfresh[str(i)] if group=='existing_fresh_geometry' else independent[i] if group=='independent_fresh_geometry' else new[i] if group=='new_independent_geometry' else newest[i] if group=='new_combiner_geometry' else prev[i]
    states=readcsv(SOURCE/'availability_predictions.csv');d=np.load(SOURCE/'feature_dataset.npz');features=d['features']
    with torch.no_grad():
        transitions=s(torch.from_numpy(features[:,:2].copy())).argmax(1).numpy()
        yes=comb(torch.from_numpy(np.column_stack((transitions,features[:,4:6],np.ones(len(features)))))).argmax(1).numpy()
    assert np.array_equal(yes,d['labels'])
    rows=[];parents={}
    for i,r in enumerate(states):
        pr=next(perturb) if r['group']=='previous_fresh_perturbations' else r
        p=parent(pr['group'],pr['case'])
        if not yes[i]:continue
        key=r['group']+':'+r['case'].split(':')[0];parents[key]=p
        rows.append(dict(r,parent_key=key,parent=p,heldout=bool(d['heldmask'][i]),train=bool(d['trainmask'][i]),
                         margin_1=float(features[i,0]),margin_2=float(features[i,1])))
    latest=readcsv(COMB/'new_fresh_replay.csv')
    for r in latest:
        if r['label']!='1':continue
        r=dict(r,group='new_combiner_geometry');p=parent(r['group'],r['case']);key=r['group']+':'+r['case'].split(':')[0];parents[key]=p
        rows.append(dict(r,parent_key=key,parent=p,heldout=False,train=False,known_366=False,previous_false_positive=False))
    a=Audit();router,_=load_router();audited=[];failures=[]
    for idx,r in enumerate(rows):
        axis,k=map(int,r['case'].split(':')[1:3]);a.setup(r['parent'],axis)
        l=float(r['lower']);u=float(r['upper']);side=int(r['boundary_side']);xx=np.linspace(np.nextafter(l,u),np.nextafter(u,l),129)
        f=a.formula(xx,k);trans=np.flatnonzero(f[1][:-1]*f[1][1:]<0);roots=[]
        for j in trans:
            root=float(brentq(lambda c:float(a.formula(c,k)[1][0]),xx[j],xx[j+1],xtol=5e-324,rtol=1e-15))
            if (0 if abs(root-l)<=abs(u-root) else 1)==side:roots.append((int(j),root))
        expected=0 if side==0 else 127
        if len(roots)!=1 or roots[0][0]!=expected:
            failures.append(dict(index=idx,group=r['group'],case=r['case'],side=side,roots=roots,reason='not one root from expected interior pair'));continue
        j,cstar=roots[0];cs=xx[[j,j+1]];iq=f[3][[j,j+1]];ia=f[4][[j,j+1]];power=f[5][[j,j+1],ia]
        A=1/(a.h*np.power(np.where(iq==0,k,a.n-k).astype(float),1/a.d))
        inp=np.column_stack((A,np.full(2,a.B),np.full(2,a.a),.05*power,cs,iq,ia))
        with torch.no_grad():terms=router.term_assembly.terms(torch.from_numpy(inp)).numpy()
        margins=terms.sum(1)/abs(terms).sum(1)
        branch=int(iq[0]!=iq[1] or ia[0]!=ia[1] or power[0]!=power[1]);valid=int(l<cs[0]<cs[1]<u)
        with torch.no_grad():
            st=int(s(torch.from_numpy(margins[None,:])).argmax(1)[0]);available=int(comb(torch.tensor([[st,branch,valid,1]],dtype=torch.float64)).argmax(1)[0])
        if not available or not cs[0]<=cstar<=cs[1]:
            failures.append(dict(index=idx,reason='availability or interval mismatch'));continue
        boundary=l if side==0 else u;width=cs[1]-cs[0];lam=(cstar-cs[0])/width
        qchange=int(iq[0]!=iq[1]);achange=int(ia[0]!=ia[1]);pchange=int(power[0]!=power[1]);typ='simultaneous' if qchange+achange+pchange>1 else 'q_only' if qchange else 'aspect_only'
        audited.append(dict(index=idx,group=r['group'],case=r['case'],parent_key=r['parent_key'],parent=r['parent'],
            dimension=a.d,budget=a.n,axis=axis,k=k,side=side,lower=l,upper=u,c_left=cs[0],c_right=cs[1],boundary=boundary,
            margin_1=margins[0],margin_2=margins[1],teacher_derivative_1=f[1][j],teacher_derivative_2=f[1][j+1],
            q_identity_1=int(iq[0]),q_identity_2=int(iq[1]),aspect_identity_1=int(ia[0]),aspect_identity_2=int(ia[1]),
            aspect_power_1=int(power[0]),aspect_power_2=int(power[1]),switch_type=typ,
            teacher_cut=cstar,teacher_lambda=lam,boundary_fraction=(boundary-cs[0])/width,normalized_spacing=width/(u-l),
            train=r['train'] if 'train' in r else False,heldout=r['heldout'] if 'heldout' in r else False,
            known_366=str(r.get('known_366','False'))=='True',mechanism='dynamic_active_branch_brentq',
            at_sample_endpoint=bool(cstar in cs),teacher_sign_transition_count=len(trans)))
    return audited,failures,parents,refs


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x))


def audit_run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('manifest','locked','validated'))})
    json_write(OUT/'protected_hashes.json',protected)
    rows,failures,parents,refs=collect();json_write(OUT/'location_cases.json',rows);json_write(OUT/'parents.json',parents)
    write_csv(OUT/'teacher_location_audit.csv',[{k:v for k,v in r.items() if k!='parent'} for r in rows])
    json_write(OUT/'teacher_audit_failures.json',failures)
    lam=np.array([r['teacher_lambda'] for r in rows]);widths=np.array([r['c_right']-r['c_left'] for r in rows])
    text='''Teacher location rule, before training
The existing analytic continuous Selector.select scans linspace(nextafter(l,r),
nextafter(r,l),129). It recomputes active q/aspect branches at every sample and
every solver evaluation. A strict product of neighboring derivatives <0 invokes
scipy.optimize.brentq on those two samples with xtol=5e-324, rtol=1e-15, default
maxiter=100, float64 inputs/outputs. It sorts/deduplicates returned floats.
These branch-switch labels exclude ordinary fixed-branch smooth roots; the root
solver can localize a rounded active-branch discontinuity rather than a smooth
mathematical derivative zero. Do not replace its target by a mathematical root.
Boundary association uses nearest subpiece endpoint, with left owning equal
distance. The supplied local pair is [0,1] or [127,128], BOTH interior samples.
Candidate coordinates come from brentq, not by assigning a branch boundary or
sample location, even when its returned float happens to coincide with one.
The brentq bracket confines its return to the two scan samples. Full scans may
produce multiple candidates in general; this audit checks each available local
boundary request for exactly one candidate from its associated pair.
No candidate coordinates enter model inputs. Endpoints define reconstruction;
boundary coordinate is geometric metadata and may lie OUTSIDE the scan pair.
'''
    text+=f'\nAvailable states audited: {len(rows)}. Mechanism/availability/containment failures: {len(failures)}.\n'
    text+=f'Lambda min/median/max: {np.min(lam)}/{np.median(lam)}/{np.max(lam)}. Scan widths min/max: {widths.min()}/{widths.max()}.\n'
    (OUT/'teacher_location_rule.txt').write_text(text)
    assert all(sha(Path(p))==h for p,h in protected.items())
    json_write(OUT/'audit_result.json',dict(cases=len(rows),failures=len(failures),refs=refs,elapsed_seconds=time.monotonic()-start,
        lambda_quantiles=np.quantile(lam,[0,.25,.5,.75,1]).tolist(),sample_endpoint_returns=sum(r['at_sample_endpoint'] for r in rows)))
    print(text,flush=True);print(json.dumps(dict(groups={g:sum(r['group']==g for r in rows) for g in sorted({r['group'] for r in rows})},types={t:sum(r['switch_type']==t for r in rows) for t in sorted({r['switch_type'] for r in rows})})),flush=True)


class BranchLocation(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(3,1,dtype=torch.float64)
        nn.init.normal_(self.linear.weight,mean=0.,std=.1)
        nn.init.uniform_(self.linear.bias,.25,.75)

    def forward(self,x):return self.linear(x).squeeze(1).clamp(0.,1.)

    def locate(self,left,right,x):
        with torch.no_grad():lam=self(torch.from_numpy(x)).numpy()
        return left+lam*(right-left)


def location_metrics(rows,cuts,name):
    target=np.array([r['teacher_cut'] for r in rows]);left=np.array([r['c_left'] for r in rows]);right=np.array([r['c_right'] for r in rows])
    err=abs(cuts-target);ulp=np.array([abs(int(np.float64(c).view(np.uint64))-int(np.float64(t).view(np.uint64))) for c,t in zip(cuts,target)])
    return dict(group=name,cases=len(rows),exact=int((ulp==0).sum()),one_ULP=int((ulp==1).sum()),larger_ULP=int((ulp>1).sum()),
        absolute_error_median=float(np.median(err)),absolute_error_p99=float(np.quantile(err,.99)),absolute_error_max=float(err.max()),
        ULP_median=float(np.median(ulp)),ULP_p99=float(np.quantile(ulp,.99)),ULP_max=int(ulp.max()),
        bracket_relative_MAE=float(np.mean(err/(right-left))),outside_pair=int(np.sum((cuts<left)|(cuts>right))))


def quick_train():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    audit=json.loads((OUT/'audit_result.json').read_text());assert audit['failures']==0
    rows=json.loads((OUT/'location_cases.json').read_text())
    x=np.array([[r['margin_1'],r['margin_2'],r['side']] for r in rows]);y=np.array([r['teacher_lambda'] for r in rows])
    left=np.array([r['c_left'] for r in rows]);right=np.array([r['c_right'] for r in rows]);target=np.array([r['teacher_cut'] for r in rows])
    rng=np.random.default_rng(SEED);fit=[]
    for typ in sorted({r['switch_type'] for r in rows}):
        for side in (0,1):
            ids=np.array([i for i,r in enumerate(rows) if r['train'] and r['switch_type']==typ and r['side']==side],int)
            rng.shuffle(ids);fit.extend(ids[:32])
    fit=np.array(sorted(fit),int)
    model=BranchLocation();initial={k:v.detach().clone() for k,v in model.state_dict().items()};opt=torch.optim.Adam(model.parameters(),lr=.03)
    xt=torch.from_numpy(x[fit]);yt=torch.from_numpy(y[fit]);history=[]
    for step in range(1,2001):
        opt.zero_grad();loss=((model(xt)-yt)**2).mean();loss.backward();opt.step()
        history.append(dict(update=step,training_MSE=float(loss.detach())))
        if float(loss.detach())<1e-18:break
        if step>300 and min(r['training_MSE'] for r in history[-100:])>=min(r['training_MSE'] for r in history[:-100]):break
    cp=OUT/'location_quick_before_evaluation.pt'
    digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),optimizer_state=opt.state_dict(),initial_state=initial,seed=SEED,
        fit_indices=fit.tolist(),updates=step,features=['normalized_margin_1','normalized_margin_2','boundary_side'],
        architecture='float64 Linear(3,1) -> clamp[0,1], four trained parameters',
        target='teacher dynamic-branch brentq returned float, normalized within the audited interior pair',
        reconstruction='left + lambda*(right-left)',no_coefficient_snapping=True))
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    cuts=model.locate(left,right,x);mid=(left+right)/2;boundary=np.array([r['boundary'] for r in rows]);edge=np.clip(boundary,left,right)
    metrics=[];a=Audit();quality=[]
    for name,cc in [('learned',cuts),('scan_midpoint',mid),('raw_geometric_boundary',boundary),('boundary_projected_to_scan_pair',edge)]:
        metrics.append(location_metrics(rows,cc,name))
    for i,r in enumerate(rows):
        a.setup(r['parent'],r['axis']);cv=np.array([r['teacher_cut'],cuts[i],mid[i],boundary[i],edge[i]])
        ss=a.formula(cv,r['k'])[0];tol=1e-12*max(1.,ss[0])
        quality.append(dict(group=r['group'],case=r['case'],side=r['side'],switch_type=r['switch_type'],known_366=r['known_366'],
            teacher_cut=target[i],learned_cut=cuts[i],teacher_score=ss[0],learned_score=ss[1],midpoint_score=ss[2],boundary_score=ss[3],edge_score=ss[4],
            learned_score_difference=abs(ss[1]-ss[0]),learned_penalty=max(0.,ss[1]-ss[0]),outside_equivalence=bool(ss[1]>ss[0]+tol),
            learned_worse_than_boundary_outside_equivalence=bool(ss[1]>ss[3]+tol),learned_worse_than_edge_outside_equivalence=bool(ss[1]>ss[4]+tol)))
    for name,mask in [('heldout_archived',[r['heldout'] for r in rows]),('quick_fit',np.isin(np.arange(len(rows)),fit)),
                      ('independent_fresh',[r['group'] in ('independent_fresh_geometry','new_independent_geometry','new_combiner_geometry') for r in rows]),
                      ('known_366',[r['known_366'] for r in rows])]:
        ids=np.flatnonzero(mask)
        metrics.append(location_metrics([rows[i] for i in ids],cuts[ids],name))
    for typ in sorted({r['switch_type'] for r in rows}):
        ids=np.array([i for i,r in enumerate(rows) if r['switch_type']==typ]);metrics.append(location_metrics([rows[i] for i in ids],cuts[ids],typ))
    write_csv(OUT/'quick_location_metrics.csv',metrics);write_csv(OUT/'quick_candidate_quality.csv',quality);write_csv(OUT/'quick_training_history.csv',history)
    # Coordinate quality must also match the geometric boundary baseline; score
    # equivalence alone is not evidence of learning a better localizer.
    quick_pass=metrics[0]['outside_pair']==0 and metrics[0]['absolute_error_max']<=metrics[2]['absolute_error_max'] and metrics[0]['absolute_error_median']<=metrics[2]['absolute_error_median'] and not any(r['outside_equivalence'] or r['learned_worse_than_boundary_outside_equivalence'] or r['learned_worse_than_edge_outside_equivalence'] for r in quality)
    result=dict(status='QUICK_LOCATION_SCREEN_PASSED_DOWNSTREAM_PENDING' if quick_pass else 'DIAGNOSTIC_ONLY_STOP',quick_pass=quick_pass,
        checkpoint=str(cp),sha256=digest,train_cases=len(fit),updates=step,parameters={k:v.tolist() for k,v in model.state_dict().items()},
        metrics=metrics,outside_score_equivalence=sum(r['outside_equivalence'] for r in quality),elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'quick_result.json',result)
    protected=json.loads((OUT/'protected_hashes.json').read_text());assert all(sha(Path(p))==h for p,h in protected.items())
    print(json.dumps(result,indent=2),flush=True)


def diagnostic_replay():
    """Read-only replay of the saved quick candidate; no further fitting."""
    import ast
    start=time.monotonic();torch.set_num_threads(1)
    result=json.loads((OUT/'quick_result.json').read_text());rows=json.loads((OUT/'location_cases.json').read_text())
    model=BranchLocation();model.load_state_dict(torch.load(result['checkpoint'],weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    x=np.array([[r['margin_1'],r['margin_2'],r['side']] for r in rows]);L=np.array([r['c_left'] for r in rows]);R=np.array([r['c_right'] for r in rows]);cuts=model.locate(L,R,x)
    sel=Selector();subresults=[];subdata=[]
    for i,r in enumerate(rows):
        base=dict(axis=r['axis'],k=r['k']);tc=[dict(base,identity=r['case']+':'+kind,kind=kind,cut=c) for kind,c in [('left',r['lower']),('right',r['upper']),('stationary',r['teacher_cut'])]]
        lc=[dict(e) for e in tc];lc[2]['cut']=float(cuts[i]);sel.score(r['parent'],tc);sel.score(r['parent'],lc)
        subdata.append((tc,lc))
    ts=np.array([[e['teacher_score'] for e in tc] for tc,lc in subdata]);ls=np.array([[e['score'] for e in lc] for tc,lc in subdata]);valid=np.ones(ts.shape,bool)
    ti=teacher_select(ts,valid);li=sel.sub.select(sel.sub.normalize(ls,valid),valid)
    for i,(r,(tc,lc)) in enumerate(zip(rows,subdata)):
        t=tc[ti[i]];l=lc[li[i]];best=min(e['teacher_score'] for e in tc);tol=1e-12*max(1.,best)
        subresults.append(dict(group=r['group'],case=r['case'],known_366=r['known_366'],teacher_identity=t['identity'],learned_identity=l['identity'],
            winner_changed=t['identity']!=l['identity'],teacher_cut=t['cut'],learned_cut=l['cut'],
            outside_equivalence=l['teacher_score']>best+tol,teacher_score_penalty=max(0.,l['teacher_score']-best),
            branch_candidate_identity_changed=False,availability_mismatch=False))
    write_csv(OUT/'quick_subpiece_selector_replay.csv',subresults)
    # Existing cached learned smooth candidates and endpoints stay unchanged.
    # Only the corresponding saved branch-switch coordinates/scores are replaced.
    cached=readcsv(ROOT/'subpiece_minimum_selector_training/frozen_candidate_scores.csv');original=original_parents()
    freshparents=json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text())
    lookup={(('archive_geometry' if r['group']=='archived_geometry' else 'fresh_geometry'),tuple(map(int,r['case'].split(':')))):(r,float(cuts[i]))
            for i,r in enumerate(rows) if r['group'] in ('archived_geometry','existing_fresh_geometry')}
    poolmap=defaultdict(list)
    for e in cached:
        if e['group'] not in ('archive_geometry','fresh_geometry'):continue
        r=dict(e);r.update(parent=int(e['parent']),axis=int(e['axis']),k=int(e['k']),cut=float(e['cut']),teacher_score=float(e['teacher_score']),score=float(e['score']))
        case=ast.literal_eval(r['case']) if r['case'].startswith('(') else tuple(map(int,r['case'].split(':')))
        r['piece']=str(case);r['slot']='left' if r['kind'] in ('left','saved0') else 'right' if r['kind'] in ('right','saved1') else 'stationary'
        poolmap[(r['group'],case)].append(r)
    replaced=0;inserted=0
    allteacher=[];alllearned=[]
    for key,pool in poolmap.items():
        tc=[dict(e) for e in pool];lc=[dict(e) for e in pool]
        if key in lookup:
            r,c=lookup[key];record=r['parent'];current=next((j for j,e in enumerate(tc) if e['slot']=='stationary'),None)
            if current is None:
                e=dict(tc[0],identity=str(key[1])+':stationary',kind='stationary',slot='stationary',cut=r['teacher_cut'])
                tc.append(e);lc.append(dict(e));current=len(tc)-1;inserted+=1
            else:replaced+=1
            tc[current]['cut']=r['teacher_cut'];lc[current]['cut']=c
            # Exact cached score convention: normalized side-scaled children and
            # fixed interval counts, as in train_subpiece_minimum_selector.run.
            a=sel.audit;a.setup(record,r['axis'])
            for e in (tc[current],lc[current]):
                a.setup(record,e['axis']);e['teacher_score']=float(a.formula(e['cut'],e['k'])[0])
                from experiments.sho.train_finite_split_score_smoke import upstream,predict
                side=a.s;ell=np.tile(side,(1,2,1));ell[0,0,e['axis']]*=e['cut'];ell[0,1,e['axis']]*=1-e['cut']
                e['score']=float(predict(a.finite,upstream(a.models,a.execution,side[None,:],np.array([a.n]),ell,np.array([[e['k'],a.n-e['k']]])))[0])
        allteacher.extend(tc);alllearned.extend(lc)

    def reduce(pool,teacher_arm=False):
        by=defaultdict(list)
        for e in pool:by[(e.get('group','fixture'),e.get('parent',0),e['piece'])].append(e)
        slots=[]
        for pp in by.values():
            named={e.get('slot',e['kind']):e for e in pp};slots.append([named['left'],named['right'],named.get('stationary')])
        ss=np.array([[e['teacher_score' if teacher_arm else 'score'] if e else 0 for e in row] for row in slots]);vv=np.array([[e is not None for e in row] for row in slots])
        ids=teacher_select(ss,vv) if teacher_arm else sel.sub.select(sel.sub.normalize(ss,vv),vv)
        ip=defaultdict(list);sp={}
        for key,row,idx in zip(by,slots,ids):
            e=row[idx];sp[key]=e;ip[(e.get('group','fixture'),e.get('parent',0),e['axis'],e['k'])].append(e)
        order=lambda e:(e['axis'],e['cut'],e['identity'])
        keys=sorted(ip);ordered=[sorted(ip[k],key=order) for k in keys]
        if teacher_arm:
            iw=[canonical(p) for p in ordered]
        else:
            ii,_,_=sel.interval.reduce([np.array([e['score'] for e in p]) for p in ordered]);iw=[p[j] for p,j in zip(ordered,ii)]
        pp=defaultdict(list)
        for e in iw:pp[(e.get('group','fixture'),e.get('parent',0))].append(e)
        pkeys=sorted(pp);pl=[sorted(pp[k],key=order) for k in pkeys]
        if teacher_arm:pw=[canonical(p) for p in pl]
        else:
            pi,_,_=sel.parent.reduce([np.array([e['score'] for e in p]) for p in pl]);pw=[p[j] for p,j in zip(pl,pi)]
        return dict(subpiece=sp,interval=dict(zip(keys,iw)),parent=dict(zip(pkeys,pw)))
    tr=reduce(allteacher,True);lr=reduce(alllearned);changes=[]
    for level in tr:
        for key,t in tr[level].items():
            l=lr[level][key];tol=1e-12*max(1.,t['teacher_score'])
            changes.append(dict(scope='cached_archived_and_fresh',level=level,key=str(key),teacher_identity=t['identity'],learned_identity=l['identity'],
                identity_changed=t['identity']!=l['identity'],teacher_cut=t['cut'],learned_cut=l['cut'],coordinate_changed=t['cut']!=l['cut'],
                outside_equivalence=l['teacher_score']>t['teacher_score']+tol,score_difference=abs(l['teacher_score']-t['teacher_score'])))
    # Explicit original d=2,N=8 fixture: retain its saved learned smooth roots.
    fixture=json.loads((ROOT/'learned_continuous_recursive_verification/first_failure.json').read_text())['selector_context'];record=fixture['parent']
    ft=fixture['teacher_candidates'];fl=[dict(e) for e in fixture['learned_candidates']]
    for i,r in enumerate(rows):
        if r['group']!='saved_failure_geometry':continue
        piece=':'.join(r['case'].split(':')[1:]);e=dict(axis=r['axis'],k=r['k'],piece=piece,identity=piece+':stationary0',kind='stationary',cut=float(cuts[i]));fl.append(e)
    sel.score(record,ft);sel.score(record,fl);frt=reduce(ft,True);frl=reduce(fl)
    fixture_rows=[]
    for level in frt:
        for key,t in frt[level].items():
            l=frl[level][key];tol=1e-12*max(1.,t['teacher_score'])
            fixture_rows.append(dict(level=level,key=str(key),teacher_identity=t['identity'],learned_identity=l['identity'],identity_changed=t['identity']!=l['identity'],
                teacher_cut=t['cut'],learned_cut=l['cut'],coordinate_changed=t['cut']!=l['cut'],outside_equivalence=l['teacher_score']>t['teacher_score']+tol,
                teacher_score=t['teacher_score'],learned_teacher_score=l['teacher_score']))
    write_csv(OUT/'quick_cached_selector_replay.csv',changes);write_csv(OUT/'d2_n8_fixture_replay.csv',fixture_rows)
    json_write(OUT/'d2_n8_candidate_sets.json',dict(teacher=ft,learned=fl))
    metrics=result['metrics'];baseline_pass=metrics[0]['absolute_error_max']<=metrics[2]['absolute_error_max'] and metrics[0]['absolute_error_median']<=metrics[2]['absolute_error_median']
    stats={level:dict(cases=sum(r['level']==level for r in changes),identity_changes=sum(r['level']==level and r['identity_changed'] for r in changes),
        coordinate_changes=sum(r['level']==level and r['coordinate_changed'] for r in changes),outside_equivalence=sum(r['level']==level and r['outside_equivalence'] for r in changes)) for level in tr}
    protected=json.loads((OUT/'protected_hashes.json').read_text());assert all(sha(Path(p))==h for p,h in protected.items())
    final=dict(status='DIAGNOSTIC_ONLY_STOP',promoted=False,full_training_started=False,quick_baseline_gate_passed=baseline_pass,
        stop_reason='Quick learned location is less accurate than the geometric-boundary baseline; no larger training authorized after that screen fails.',
        checkpoint=result['checkpoint'],sha256=result['sha256'],quick_updates=result['updates'],quick_fit_cases=result['train_cases'],
        location_metrics=metrics,available_branch_cases=len(rows),availability_mismatches=0,branch_candidate_identity_changes=0,
        all_branch_subpiece_winner_changes=sum(r['winner_changed'] for r in subresults),all_branch_subpiece_outside_equivalence=sum(r['outside_equivalence'] for r in subresults),
        cached_branch_replacements=replaced,cached_branch_insertions=inserted,downstream=stats,fixture=fixture_rows,
        downstream_scope='All 522 positive subpieces; 106 archived + 8 cached fresh parents; explicit saved d2 N8 parent. Smooth candidates reused unchanged from existing caches. No newly regenerated full parent candidate trees.',
        all_protected_hashes_unchanged=True,protected_files=len(protected),replay_seconds=time.monotonic()-start,
        implementation_sha256=sha(Path(__file__)))
    json_write(OUT/'final_result.json',final)
    print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':
    import sys
    diagnostic_replay() if '--replay' in sys.argv else quick_train() if '--quick' in sys.argv else audit_run()
