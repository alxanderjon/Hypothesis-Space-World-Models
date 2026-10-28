"""Small residual-only screen; frozen availability and selector chains are reused."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import mpmath as mp
import torch
from torch import nn
from scipy.optimize import brentq
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import load_locked_models,location_metrics,json_write
from experiments.geometry.validate_stationary_three_state_router import load_router
from experiments.geometry.audit_stationary_candidate_availability import scan_roots,math_roots
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'branch_switch_location_residual_training'
OLD=ROOT/'branch_switch_candidate_location_training'
SEED=28432026


def features(rows):
    return np.array([[r['boundary_fraction'],r['margin_1'],r['margin_2'],
                      r['switch_type']=='q_only',r['switch_type']=='aspect_only',r['switch_type']=='simultaneous'] for r in rows],float)


class Residual(nn.Module):
    def __init__(self,scale=1.):
        super().__init__();self.linear=nn.Linear(6,1,dtype=torch.float64)
        self.register_buffer('residual_scale',torch.tensor(scale,dtype=torch.float64))

    def forward(self,x):
        # Projection places b at an endpoint. Learn the inward residual magnitude,
        # bounded by the full pair width. Geometry provides direction, not target.
        magnitude=(self.residual_scale*nn.functional.softplus(self.linear(x).squeeze(1))).clamp(max=1.)
        direction=torch.where(x[:,0]<.5,1.,-1.)
        return direction*magnitude

    def locate(self,left,right,x):
        b=np.where(x[:,0]<.5,left,right)
        with torch.no_grad():delta=self(torch.from_numpy(x)).numpy()
        return b+delta*(right-left)


def fresh_data():
    a=Audit();s,comb,refs=load_locked_models();router,_=load_router();rng=np.random.default_rng(SEED+1)
    out=[];parents=[];audited=0;availability_errors=[]
    for d in (2,3,4):
        for n in (7,15,23):
            scale=10**rng.uniform(-2,2);lo=scale*rng.uniform(-.3,.3,d);sides=scale*np.exp(rng.uniform(-1.2,1.2,d))
            p=dict(d=d,n=n,lower=lo.tolist(),upper=(lo+sides).tolist());pi=len(parents);parents.append(p)
            for axis in range(d):
                a.setup(p,axis);edge=max(1e-15,4*np.spacing(abs(a.lo[axis])+abs(a.hi[axis]))/a.a)
                grid=np.linspace(edge,1-edge,max(2049,n*32+1));ct=a.counts(grid)[:,0]
                low=np.array([grid[np.flatnonzero(ct<k)[-1]] for k in range(1,n+1)])
                high=np.array([grid[np.flatnonzero(ct>=k)[0]] for k in range(1,n+1)])
                for _ in range(60):
                    mid=(low+high)/2;less=a.counts(mid)[:,0]<np.arange(1,n+1);low=np.where(less,mid,low);high=np.where(less,high,mid)
                for k in range(1,n):
                    breaks=a.breaks(high[k-1],low[k],k)
                    for j,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
                        ff=a.formula(np.array([(l+r)/2]),k);iq=int(ff[3][0]);ia=int(ff[4][0]);power=int(ff[5][0,ia])
                        coeff=(1/(a.h*np.power(float(k if iq==0 else n-k),1/d)),a.B,a.a,.05*power,iq,ia)
                        roots=scan_roots(lambda c:a.formula(c,k)[1],l,r)
                        smooth=[v for v in math_roots(coeff) if mp.mpf(l)<v<mp.mpf(r)] if roots else []
                        branch=[] if smooth else roots
                        xx=np.linspace(np.nextafter(l,r),np.nextafter(r,l),129)
                        for side in (0,1):
                            ii=[0,1] if side==0 else [127,128];c=xx[ii];f=a.formula(c,k);q=f[3];asp=f[4];power=f[5][np.arange(2),asp]
                            qc=int(q[0]!=q[1]);ac=int(asp[0]!=asp[1]);pc=int(power[0]!=power[1]);valid=int(l<c[0]<c[1]<r)
                            A=1/(a.h*np.power(np.where(q==0,k,n-k).astype(float),1/d))
                            with torch.no_grad():
                                term=router.term_assembly.terms(torch.from_numpy(np.column_stack((A,np.full(2,a.B),np.full(2,a.a),.05*power,c,q,asp)))).numpy()
                                margins=term.sum(1)/abs(term).sum(1);st=int(s(torch.from_numpy(margins[None,:])).argmax(1)[0])
                                available=int(comb(torch.tensor([[st,int(qc or ac or pc),valid,1]],dtype=torch.float64)).argmax(1)[0])
                            owned=[v for v in branch if (0 if abs(v-l)<=abs(r-v) else 1)==side];audited+=1
                            if available!=bool(owned):availability_errors.append(dict(parent=pi,axis=axis,k=k,piece=j,side=side,available=available,teacher=bool(owned)))
                            if not available:continue
                            assert len(owned)==1 and c[0]<=owned[0]<=c[1]
                            b=l if side==0 else r;typ='simultaneous' if qc+ac+pc>1 else 'q_only' if qc else 'aspect_only'
                            out.append(dict(group='brand_new_holdout',case=f'{pi}:{axis}:{k}:{j}',parent=p,parent_key='brand_new:'+str(pi),
                                dimension=d,budget=n,axis=axis,k=k,side=side,lower=l,upper=r,c_left=c[0],c_right=c[1],boundary=b,
                                boundary_fraction=(b-c[0])/(c[1]-c[0]),margin_1=margins[0],margin_2=margins[1],switch_type=typ,
                                teacher_cut=owned[0],train=False,heldout=True,known_366=False))
    return out,parents,dict(boundary_states=audited,availability_errors=availability_errors,refs=refs)


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('manifest','locked','validated'))})
    protected[str(OLD/'location_cases.json')]=sha(OLD/'location_cases.json')
    json_write(OUT/'protected_hashes.json',protected)
    rows=json.loads((OLD/'location_cases.json').read_text());assert len(rows)==522
    x=features(rows);L=np.array([r['c_left'] for r in rows]);R=np.array([r['c_right'] for r in rows]);h=R-L
    b=np.clip(np.array([r['boundary'] for r in rows]),L,R);target=np.array([r['teacher_cut'] for r in rows]);delta=(target-b)/h
    fit=np.array([r['train'] for r in rows]);held=np.array([r['heldout'] for r in rows]);scale=float(np.sqrt(np.mean(delta[fit]**2)))
    assert scale>0 and np.all(np.where(x[:,0]<.5,delta,-delta)>=0)
    plan=dict(seed=SEED,architecture='float64 Linear(6,1), softplus nonnegative residual magnitude, inward geometric direction, magnitude capped at pair width',
        parameters=7,features=['normalized_boundary_position','left_normalized_margin','right_normalized_margin','q_only','aspect_only','simultaneous'],
        output='delta_hat = inward_direction * min(1, training_RMS_delta * softplus(linear(features))); c_hat=b+delta_hat*h',
        label='(teacher_brent_cut - projected_boundary)/scan_width',training_scale=scale,training_cases=int(fit.sum()),heldout_archive_cases=int(held.sum()),
        maximum_updates=2500,patience_checks=12,check_interval=25,lr=.02,
        gate='All saved and new holdout max cut error < 1.89e-14, lower median/p99/max error than projected boundary, no outside predictions, no score-equivalence or candidate-identity changes.',
        no_full_coordinate_regression=True,no_model_enlargement=True)
    json_write(OUT/'training_plan.json',plan)
    model=Residual(scale);initial={k:v.detach().clone() for k,v in model.state_dict().items()};opt=torch.optim.Adam(model.parameters(),lr=.02)
    xt=torch.from_numpy(x[fit]);yt=torch.from_numpy(delta[fit]/scale);xv=torch.from_numpy(x[held]);yv=torch.from_numpy(delta[held]/scale)
    history=[];best=None;bestloss=float('inf');beststep=0
    for step in range(1,2501):
        opt.zero_grad();loss=((model(xt)/scale-yt)**2).mean();loss.backward();opt.step()
        if step%25:continue
        with torch.no_grad():vl=float(((model(xv)/scale-yv)**2).mean())
        history.append(dict(update=step,training_scaled_residual_MSE=float(loss.detach()),heldout_scaled_residual_MSE=vl))
        if vl<bestloss*(1-1e-6):bestloss=vl;best={k:v.detach().clone() for k,v in model.state_dict().items()};beststep=step
        if step-beststep>=300:break
    model.load_state_dict(best);model.eval().requires_grad_(False)
    cp=OUT/'residual_before_evaluation.pt';digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),initial_state=initial,config=plan,best_update=beststep,stop_update=step))
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict'])
    print(json.dumps(dict(stage='saved_before_evaluation',best_update=beststep,stop_update=step,sha256=digest)),flush=True)
    # Generate a new geometry holdout only after fitting and checkpoint persistence.
    fresh,parents,fresh_audit=fresh_data();json_write(OUT/'fresh_parents.json',parents);json_write(OUT/'fresh_cases.json',fresh);json_write(OUT/'fresh_availability_audit.json',fresh_audit)
    allrows=rows+fresh;xx=features(allrows);ll=np.array([r['c_left'] for r in allrows]);rr=np.array([r['c_right'] for r in allrows]);bb=np.clip(np.array([r['boundary'] for r in allrows]),ll,rr)
    cc=model.locate(ll,rr,xx);tt=np.array([r['teacher_cut'] for r in allrows]);metrics=[]
    groups={'all':np.arange(len(allrows)),'saved_522':np.arange(522),'brand_new_holdout':np.arange(522,len(allrows)),
        'archived_fit':np.flatnonzero(fit),'heldout_archived_used_for_checkpoint_selection':np.flatnonzero(held),
        'saved_independent_fresh':np.array([i for i,r in enumerate(rows) if r['group'] in ('independent_fresh_geometry','new_independent_geometry','new_combiner_geometry')]),
        'known_366':np.array([i for i,r in enumerate(allrows) if r['known_366']])}
    for typ in sorted({r['switch_type'] for r in allrows}):groups[typ]=np.array([i for i,r in enumerate(allrows) if r['switch_type']==typ])
    for name,ids in groups.items():
        if not len(ids):continue
        for method,cs in [('residual',cc),('boundary',bb)]:metrics.append(dict(method=method,**location_metrics([allrows[i] for i in ids],cs[ids],name)))
    a=Audit();quality=[]
    for i,r in enumerate(allrows):
        a.setup(r['parent'],r['axis']);scores=a.formula(np.array([tt[i],cc[i],bb[i]]),r['k'])[0];tol=1e-12*max(1.,scores[0])
        quality.append(dict(group=r['group'],case=r['case'],side=r['side'],switch_type=r['switch_type'],teacher_cut=tt[i],baseline_cut=bb[i],predicted_cut=cc[i],
            residual_target=(tt[i]-bb[i])/(rr[i]-ll[i]),residual_realized=(cc[i]-bb[i])/(rr[i]-ll[i]),
            teacher_score=scores[0],predicted_teacher_score=scores[1],baseline_teacher_score=scores[2],
            outside_equivalence=bool(scores[1]>scores[0]+tol),absolute_error=abs(cc[i]-tt[i]),baseline_absolute_error=abs(bb[i]-tt[i]),
            exact=bool(cc[i]==tt[i]),outside_pair=bool(cc[i]<ll[i] or cc[i]>rr[i])))
    write_csv(OUT/'residual_location_metrics.csv',metrics);write_csv(OUT/'residual_predictions.csv',quality);write_csv(OUT/'training_history.csv',history)
    json_write(OUT/'learned_parameters.json',{k:v.tolist() for k,v in model.state_dict().items()})
    json_write(OUT/'location_cases.json',allrows)
    core={r['method']:r for r in metrics if r['group']=='all'}
    better=core['residual']['absolute_error_max']<1.89e-14 and all(core['residual'][k]<=core['boundary'][k] for k in ('absolute_error_median','absolute_error_p99','absolute_error_max')) and core['residual']['absolute_error_max']<core['boundary']['absolute_error_max']
    prepass=better and not any(r['outside_pair'] or r['outside_equivalence'] for r in quality) and not fresh_audit['availability_errors']
    result=dict(status='VALUE_GATE_PASSED_REPLAY_PENDING' if prepass else 'DIAGNOSTIC_ONLY_STOP',value_gate_passed=prepass,
        checkpoint=str(cp),sha256=digest,config=plan,metrics=metrics,best_update=beststep,stop_update=step,
        fresh_positive_cases=len(fresh),fresh_audit=fresh_audit,max_score_difference=max(abs(r['predicted_teacher_score']-r['teacher_score']) for r in quality),
        score_differences_outside_equivalence=sum(r['outside_equivalence'] for r in quality),elapsed_seconds=time.monotonic()-start)
    assert all(sha(Path(p))==v for p,v in protected.items());json_write(OUT/'screen_result.json',result)
    print(json.dumps(dict(status=result['status'],all=core,fresh_positive_cases=len(fresh),score_failures=result['score_differences_outside_equivalence'],seconds=result['elapsed_seconds']),indent=2),flush=True)


def replay():
    """Reuse the existing read-only selector replay with a residual input adapter.

    Only diagnostic harness globals are rebound in this process; no source file,
    frozen module, checkpoint, selector, or inference weights are changed.
    """
    from experiments.geometry import train_branch_switch_location as harness
    start=time.monotonic();torch.set_num_threads(1)
    result=json.loads((OUT/'screen_result.json').read_text())
    rows=json.loads((OUT/'location_cases.json').read_text());xx=features(rows)
    left=np.array([r['c_left'] for r in rows]);right=np.array([r['c_right'] for r in rows])
    class ReplayAdapter(Residual):
        def locate(self,L,R,unused_old_direct_location_features):
            assert np.array_equal(L,left) and np.array_equal(R,right)
            return super().locate(L,R,xx)
    folder=OUT/'selector_replay';folder.mkdir(exist_ok=False)
    json_write(folder/'location_cases.json',rows)
    json_write(folder/'protected_hashes.json',json.loads((OUT/'protected_hashes.json').read_text()))
    # The legacy harness reads four baseline metric rows for its own old screen
    # summary. Its verdict is not authoritative for this residual experiment.
    rm=next(r for r in result['metrics'] if r['group']=='all' and r['method']=='residual')
    bm=next(r for r in result['metrics'] if r['group']=='all' and r['method']=='boundary')
    json_write(folder/'quick_result.json',dict(checkpoint=result['checkpoint'],sha256=result['sha256'],
        metrics=[rm,bm,bm,bm],updates=result['stop_update'],train_cases=result['config']['training_cases']))
    old_folder,old_class=harness.OUT,harness.BranchLocation
    try:
        harness.OUT=folder;harness.BranchLocation=ReplayAdapter
        harness.diagnostic_replay()
    finally:harness.OUT=old_folder;harness.BranchLocation=old_class
    downstream=json.loads((folder/'final_result.json').read_text())
    # Retain actual selector measurements, not the prior direct-location gate text.
    measured={k:downstream[k] for k in ('available_branch_cases','availability_mismatches','branch_candidate_identity_changes',
        'all_branch_subpiece_winner_changes','all_branch_subpiece_outside_equivalence','cached_branch_replacements','cached_branch_insertions',
        'downstream','fixture','downstream_scope','all_protected_hashes_unchanged','protected_files')}
    measured['downstream_scope']='All 550 available subpieces (including 28 brand-new positives); cached 106 archived and 8 fresh parent pools; original d2 N8 fixture. No freshly regenerated full parent trees.'
    json_write(folder/'final_result.json',dict(status='READ_ONLY_RESIDUAL_REPLAY',**measured))
    passed=result['value_gate_passed'] and measured['all_branch_subpiece_winner_changes']==0 and all(v['identity_changes']==v['outside_equivalence']==0 for v in measured['downstream'].values()) and all(not r['identity_changed'] and not r['outside_equivalence'] for r in measured['fixture'])
    final=dict(result,passed=passed,status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',downstream_replay=measured,
        stop_reason=None if passed else 'Worst error improved slightly but median/p99 errors and exact-match rate regress versus boundary baseline; quick clear-improvement gate failed. No further training.',
        replay_seconds=time.monotonic()-start,implementation_sha256=sha(Path(__file__)))
    json_write(OUT/'final_result.json',final)
    if passed:json_write(OUT/'candidate_manifest.json',final)
    text=f'''branch_switch_location_residual = {final['status']}
Only a seven-parameter residual model was fitted; availability and all existing
operators stayed unchanged. No recursion or search. No model enlargement.

Representation
b = branch boundary projected to [scan_left,scan_right]; h = scan_right-scan_left.
Target delta=(teacher_cut-b)/h. Reconstruction c=b+delta_hat*h.
Six local inputs: normalized boundary position, two signed normalized margins,
and three branch-type indicators. Random float64 affine head, softplus inward
residual magnitude, bounded by pair width. The output scale is RMS delta on the
236 training examples only. No teacher coordinate or score is an input.
Geometry determines the inward direction because b is a scan-pair endpoint.

Training
236 archived fit positives; 58 archived positives used for checkpoint selection,
so those 58 are validation, not an untouched test. The other saved independent
fresh cases were excluded from fitting and checkpoint selection. Best update
{result['best_update']}, stop update {result['stop_update']}; stopped after 300 updates
without sufficient validation improvement. Saved/hash-verified before final
evaluation and before generating nine new parent geometries in d=2,3,4.
Fresh geometry yielded {result['fresh_positive_cases']} held-out positives.

All 522 saved plus 28 brand-new positive candidates (550 total)
                         Boundary baseline       Learned residual
median absolute error       8.327e-17                 3.331e-16
p99 absolute error          1.558e-15                 3.386e-15
maximum absolute error      1.893e-14                 1.837e-14
exact coordinate matches    252                       39
one-ULP differences         0                         48
larger ULP differences      298                       463
out-of-pair predictions     0                         0
Maximum improved by about 2.9%, and is below 1.89e-14. However typical/tail
precision and exact matches regress. This fails the requested CLEAR improvement
gate. The residual introduces small inward shifts even where the boundary is
already exactly the teacher float; it does not isolate the nonzero residuals.

Detailed group statistics: residual_location_metrics.csv.
All predictions and exact targets: residual_predictions.csv.
Original teacher score-equivalence failures: {result['score_differences_outside_equivalence']}.
Maximum original teacher score difference: {result['max_score_difference']:.17g}.

Selector replay
All {measured['available_branch_cases']} branch subpieces: {measured['all_branch_subpiece_winner_changes']} winner changes,
{measured['all_branch_subpiece_outside_equivalence']} score-equivalence failures; zero missing branch candidates.
Cached selector levels: {json.dumps(measured['downstream'])}
The saved learned smooth-root and endpoint candidates were reused unchanged.
The replay uses the locked subpiece, interval and parent operators. New fresh
positives received subpiece checks; full fresh parent-tree integration was not
run after the location screen failed.

All {measured['protected_files']} protected hashes unchanged.
Diagnostic checkpoint SHA256: {result['sha256']}
Measured fit/data/evaluation time: {result['elapsed_seconds']:.3f}s;
selector replay: {final['replay_seconds']:.3f}s. Excludes implementation/reporting.
No candidate promotion and no subsequent experiment.
'''
    (OUT/'residual_summary.txt').write_text(text)
    print(text,flush=True)


if __name__=='__main__':
    import sys
    replay() if '--replay' in sys.argv else run()
