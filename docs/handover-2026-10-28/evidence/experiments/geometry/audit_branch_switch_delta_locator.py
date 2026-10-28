"""Frozen bracket locator in boundary-centered coordinates; no fitting."""
import json
import time
from pathlib import Path
from collections import Counter
from fractions import Fraction
import numpy as np
import torch
from torch import nn
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read,bits
from experiments.geometry.validate_terminal_candidate_full import FrozenLocator
from experiments.geometry.validate_branch_switch_locator import DynamicBranches
from experiments.geometry.validate_stationary_three_state_router import DEFER

OUT=ROOT/'branch_switch_delta_locator_audit'
SOURCE=ROOT/'branch_switch_locator_assembly'


class DeltaBranches(nn.Module):
    def __init__(self,base,rows,boundaries):
        super().__init__();self.dynamic=DynamicBranches(base,rows);self.boundaries=boundaries;self.delta_trace=[]
    def forward(self,bank,details=False):
        x=bank.detach().numpy().copy();ids=x[:,0].astype(int);delta=x[:,4].copy()
        c=self.boundaries[ids]+delta;x[:,4]=c
        out=self.dynamic(torch.from_numpy(x),details=details)
        self.delta_trace.extend(dict(case_id=int(i),delta=float(d),c=float(v)) for i,d,v in zip(ids,delta,c))
        return out


def metric(rows,cuts,ids,method,group):
    ids=np.asarray(ids,dtype=int);truth=np.array([rows[i]['teacher_cut'] for i in ids]);cc=cuts[ids]
    good=np.isfinite(cc);err=abs(cc[good]-truth[good]);ulp=np.array([abs(bits(a)-bits(b)) for a,b in zip(cc[good],truth[good])])
    return dict(group=group,method=method,cases=len(ids),missing=int((~good).sum()),
        exact=int((ulp==0).sum()),one_ULP=int((ulp==1).sum()),larger_ULP=int((ulp>1).sum()),
        median_cut_error=float(np.median(err)) if len(err) else None,
        p99_cut_error=float(np.quantile(err,.99)) if len(err) else None,
        max_cut_error=float(err.max()) if len(err) else None,
        median_ULP=float(np.median(ulp)) if len(ulp) else None,
        p99_ULP=float(np.quantile(ulp,.99)) if len(ulp) else None,max_ULP=int(ulp.max()) if len(ulp) else None)


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    rows=json.loads((SOURCE/'case_states.json').read_text());absolute=json.loads((SOURCE/'records.json').read_text())
    unique=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_unique.csv')
    fresh=json.loads((ROOT/'branch_switch_ulp_offset_training/fresh_candidate_states.json').read_text())
    assert len(rows)==818 and len(unique)==774 and len(fresh)==44
    cb=np.array([float(r['boundary']) for r in unique]+[r['boundary'] for r in fresh])
    # Use RAW known branch boundaries, not their projections into scan pairs.
    L=np.array([r['left'] for r in rows]);R=np.array([r['right'] for r in rows]);dl=L-cb;dr=R-cb
    locator=FrozenLocator();base=locator.sign;adapter=DeltaBranches(base,rows,cb);locator.sign=adapter
    bank=np.zeros((818,7));bank[:,0]=np.arange(818)
    def signs(ids,d):
        with torch.no_grad():return adapter(torch.from_numpy(np.column_stack((bank[ids,:4],d,bank[ids,4:6])))).numpy()
    ids=np.arange(818);sl=signs(ids,dl);sr=signs(ids,dr)
    eligible=(dl<dr)&(sl!=sr)&(sl!=DEFER)&(sr!=DEFER)
    use=np.flatnonzero(eligible)
    try:brackets=locator.locate(bank[use],dl[use],dr[use])
    except AssertionError:
        json_write(OUT/'locator_assertion_failure.json',dict(cases=rows,delta_trace=adapter.delta_trace[-200:],query_trace=adapter.dynamic.trace[-200:]))
        raise
    candidate_delta=locator.midpoint.evaluate(brackets['L'],brackets['R'])
    cuts=np.full(818,np.nan);cuts[use]=cb[use]+candidate_delta
    absL=np.array([r['final_L'] for r in absolute]);absR=np.array([r['final_R'] for r in absolute])
    abs_mid=locator.midpoint.evaluate(absL,absR)
    fl=signs(use,brackets['L']);fr=signs(use,brackets['R'])
    assert np.all((fl!=DEFER)&(fr!=DEFER)&(fl!=fr))
    positions={int(i):j for j,i in enumerate(use)};records=[]
    for i,r in enumerate(rows):
        rec=dict(case_id=i,uid=r['uid'],label=r['label'],source=r['source'],switch_type=r['switch_type'],
                 raw_boundary=cb[i],projected_boundary=r['b'],scan_left=L[i],scan_right=R[i],
                 initial_delta_L=dl[i],initial_delta_R=dr[i],teacher_Brent=r['teacher_cut'],
                 missing=not bool(eligible[i]),candidate=cuts[i] if eligible[i] else None,
                 absolute_midpoint=abs_mid[i],absolute_L=absL[i],absolute_R=absR[i])
        if i in positions:
            j=positions[i];ll=brackets['L'][j];rr=brackets['R'][j];cl=cb[i]+ll;cr=cb[i]+rr
            td=Fraction(float(r['teacher_cut']))-Fraction(float(cb[i]))
            a=adapter.dynamic.audits[i];scores=a.formula(np.array([r['teacher_cut'],cuts[i],abs_mid[i],cb[i]]),r['count'])[0]
            tol=1e-12*max(1.,scores[0])
            rec.update(final_delta_L=ll,final_delta_R=rr,delta_candidate=candidate_delta[j],mapped_L=cl,mapped_R=cr,
                delta_Brent_contained=bool(Fraction(float(ll))<=td<=Fraction(float(rr))),
                mapped_Brent_contained=bool(cl<=r['teacher_cut']<=cr),
                mapped_sign_bracket_valid=bool(cl<cr and fl[j]!=fr[j] and fl[j]!=DEFER and fr[j]!=DEFER),
                mapped_pair_same_as_absolute=bool(cl==absL[i] and cr==absR[i]),
                final_left_state=int(fl[j]),final_right_state=int(fr[j]),termination=brackets['reason'][j],
                iterations=int(brackets['iterations'][j]),router_queries=int(brackets['queries'][j]),
                outside_original_scan=bool(cuts[i]<L[i] or cuts[i]>R[i]),
                delta_absolute_error=abs(cuts[i]-r['teacher_cut']),delta_ULP_error=abs(bits(cuts[i])-bits(r['teacher_cut'])),
                abs_mid_absolute_error=abs(abs_mid[i]-r['teacher_cut']),abs_mid_ULP_error=abs(bits(abs_mid[i])-bits(r['teacher_cut'])),
                raw_boundary_absolute_error=abs(cb[i]-r['teacher_cut']),raw_boundary_ULP_error=abs(bits(cb[i])-bits(r['teacher_cut'])),
                teacher_score=scores[0],delta_score=scores[1],abs_mid_score=scores[2],raw_boundary_score=scores[3],
                delta_score_outside_equivalence=bool(abs(scores[1]-scores[0])>tol))
        records.append(rec)
    methods={'delta_midpoint':cuts,'absolute_midpoint':abs_mid,'raw_boundary':cb}
    metrics=[]
    groups={'all_818':np.arange(818),'saved_774':np.arange(774),'fresh_44':np.arange(774,818)}
    for typ in sorted({r['switch_type'] for r in rows}):groups[typ]=[i for i,r in enumerate(rows) if r['switch_type']==typ]
    for group,ii in groups.items():
        for method,c in methods.items():metrics.append(metric(rows,c,ii,method,group))
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv');groupmetrics=[]
    for g in range(21):
        ii=sorted({int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==g})
        for method,c in methods.items():groupmetrics.append(metric(rows,c,ii,method,str(g)))
    query_trace=[]
    for d,t in zip(adapter.delta_trace,adapter.dynamic.trace):
        assert d['case_id']==t['case_id'] and d['c']==t['c'];query_trace.append(dict(t,delta=d['delta']))
    wrong=sum((t['learned_state']==1 and t['teacher_float64_derivative']<=0) or (t['learned_state']==2 and t['teacher_float64_derivative']>=0) for t in query_trace)
    stat=lambda x:dict(median=float(np.median(x)),p99=float(np.quantile(x,.99)),max=float(np.max(x)))
    result=dict(status='LOCALIZATION_COMPLETE_DOWNSTREAM_PENDING',cases=818,missing=int((~eligible).sum()),
        initial_roundtrip_failures=int(np.sum((cb+dl!=L)|(cb+dr!=R))),
        initial_teacher_delta_containment_failures=sum(not Fraction(float(dl[i]))<=Fraction(float(r['teacher_cut']))-Fraction(float(cb[i]))<=Fraction(float(dr[i])) for i,r in enumerate(rows)),
        trusted_endpoint_violations=0,trusted_sign_disagreements_vs_float64_teacher=wrong,
        final_delta_Brent_contained=sum(r.get('delta_Brent_contained',False) for r in records),
        final_mapped_Brent_contained=sum(r.get('mapped_Brent_contained',False) for r in records),
        final_mapped_sign_brackets_valid=sum(r.get('mapped_sign_bracket_valid',False) for r in records),
        mapped_pairs_matching_absolute=sum(r.get('mapped_pair_same_as_absolute',False) for r in records),
        delta_candidate_matches_absolute_midpoint=int(np.sum(cuts==abs_mid)),
        outside_original_scan=sum(r.get('outside_original_scan',False) for r in records),
        score_failures=sum(r.get('delta_score_outside_equivalence',False) for r in records),
        total_router_queries=int(brackets['queries'].sum()),iterations=stat(brackets['iterations']),
        prior_absolute_total_queries=sum(r['queries'] for r in absolute),
        terminations=dict(Counter(brackets['reason'])),metrics=metrics,fixture=records[765],
        containment_note='Brent return containment reported separately from trusted sign bracketing; branch transitions need not be smooth mathematical stationary zeros.',
        no_training=True,no_terminal_operator=True,no_recursion=True)
    write_csv(OUT/'delta_candidate_results.csv',records);write_csv(OUT/'localization_metrics.csv',metrics)
    write_csv(OUT/'conflicting_group_metrics.csv',groupmetrics);write_csv(OUT/'dynamic_delta_queries.csv',query_trace)
    raw=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_all.csv')
    write_csv(OUT/'all_saved_occurrences.csv',[dict(occurrence=int(r['index']),**records[int(r['unique_id'])]) for r in raw])
    json_write(OUT/'records.json',records);json_write(OUT/'frozen_locator_refs.json',locator.refs)
    assert all(sha(Path(p))==v for p,v in protected.items())
    result.update(protected_hashes_unchanged=True,protected_files=len(protected),elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'protected_hashes.json',protected);json_write(OUT/'localization_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2),flush=True)
    print(json.dumps([r for r in metrics if r['group']=='all_818'],indent=2),flush=True)


def replay():
    from experiments.geometry import train_branch_switch_location as harness
    torch.set_num_threads(1);records=json.loads((OUT/'records.json').read_text())
    assert all(not r['missing'] for r in records)
    saved=json.loads((ROOT/'branch_switch_location_residual_training/location_cases.json').read_text())
    fresh=json.loads((ROOT/'branch_switch_ulp_offset_training/fresh_candidate_states.json').read_text())
    raw=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_all.csv')
    mapped=[records[int(raw[i]['unique_id'])] for i in range(550)]+records[774:];rows=saved+fresh
    L=np.array([r['c_left'] for r in rows]);R=np.array([r['c_right'] for r in rows])
    summaries={}
    for method,column in [('delta_midpoint','candidate'),('absolute_midpoint','absolute_midpoint'),('raw_boundary','raw_boundary')]:
        cuts=np.array([r[column] for r in mapped]);folder=OUT/('selector_'+method);folder.mkdir(exist_ok=False)
        # Existing harness expects a location-module loader. This inference-only
        # adapter returns already saved audit candidates, and has no parameters.
        class Adapter:
            def load_state_dict(self,state):assert state['saved_audit_method']==method
            def eval(self):return self
            def requires_grad_(self,value):assert not value;return self
            def locate(self,left,right,unused):
                assert np.array_equal(left,L) and np.array_equal(right,R);return cuts.copy()
        # A data-only audit payload, not a trained or frozen checkpoint.
        payload=folder/'saved_candidate_payload.pt';torch.save({'state_dict':{'saved_audit_method':method}},payload)
        mm=harness.location_metrics(rows,cuts,method)
        json_write(folder/'location_cases.json',rows);json_write(folder/'protected_hashes.json',json.loads((OUT/'protected_hashes.json').read_text()))
        json_write(folder/'quick_result.json',dict(checkpoint=str(payload),sha256=sha(payload),updates=0,train_cases=0,metrics=[mm]*4))
        old_out,old_cls=harness.OUT,harness.BranchLocation
        try:harness.OUT=folder;harness.BranchLocation=Adapter;harness.diagnostic_replay()
        finally:harness.OUT=old_out;harness.BranchLocation=old_cls
        rr=json.loads((folder/'final_result.json').read_text())
        keep=('available_branch_cases','availability_mismatches','branch_candidate_identity_changes','all_branch_subpiece_winner_changes',
              'all_branch_subpiece_outside_equivalence','downstream','fixture','all_protected_hashes_unchanged','replay_seconds')
        summary={k:rr[k] for k in keep};summary.update(method=method,scope='594 positive subpieces including 44 fresh; cached complete 106 archived + 8 existing fresh parents; original d2 N8 fixture. No recursion.')
        json_write(folder/'final_result.json',summary);summaries[method]=summary
    result=json.loads((OUT/'localization_result.json').read_text())
    ms={r['method']:r for r in result['metrics'] if r['group']=='all_818'}
    delta=summaries['delta_midpoint'];absolute=summaries['absolute_midpoint']
    better=ms['delta_midpoint']['median_cut_error']<ms['absolute_midpoint']['median_cut_error'] and ms['delta_midpoint']['max_cut_error']<=ms['absolute_midpoint']['max_cut_error']
    identity_better=sum(v['identity_changes'] for v in delta['downstream'].values())<sum(v['identity_changes'] for v in absolute['downstream'].values())
    pass_gate=better and identity_better and not result['missing'] and not result['score_failures'] and not result['outside_original_scan'] and not result['trusted_endpoint_violations']
    result.update(status='VALIDATED_CANDIDATE' if pass_gate else 'DIAGNOSTIC_ONLY_STOP',promoted=pass_gate,downstream=summaries,
                  clear_localization_improvement=better,downstream_identity_improvement=identity_better)
    protected=json.loads((OUT/'protected_hashes.json').read_text());assert all(sha(Path(p))==v for p,v in protected.items())
    json_write(OUT/'final_result.json',result)
    if pass_gate:json_write(OUT/'branch_switch_delta_locator_validated_candidate.json',result)
    print(json.dumps(dict(status=result['status'],downstream={k:v['downstream'] for k,v in summaries.items()}),indent=2),flush=True)


if __name__=='__main__':
    import sys
    replay() if '--replay' in sys.argv else run()
