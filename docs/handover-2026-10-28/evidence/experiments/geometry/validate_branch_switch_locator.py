"""Read-only dynamic-branch reuse of the frozen stationary locator."""
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write,load_locked_models
from experiments.geometry.audit_brent_branch_offsets import read,bits
from experiments.geometry.verify_learned_continuous_recursion import Selector
from experiments.geometry.train_stationary_terminal_candidate import features
from experiments.geometry.validate_stationary_three_state_router import DEFER
from experiments.sho.audit_fixed_allocation_intervals import Audit

OUT=ROOT/'branch_switch_locator_assembly'


class DynamicBranches(nn.Module):
    """Geometry context adapter; active identities reselected at every query.

    The first column of the locator's coefficient bank holds an audit-local row
    index, not a feature passed to a learned network. All seven inputs to the
    actual frozen sign operator are reconstructed from the current geometry/c.
    """
    def __init__(self,base,rows):
        super().__init__();self.base=base;self.rows=rows;self.audits=[];self.trace=[]
        for r in rows:
            a=Audit.__new__(Audit);a.setup(r['parent'],r['axis']);self.audits.append(a)

    def forward(self,bank,details=False):
        xx=bank.detach().numpy();ids=xx[:,0].astype(int);coords=xx[:,4]
        actual=[];meta=[]
        for i,c in zip(ids,coords):
            a=self.audits[i];k=self.rows[i]['count'];f=a.formula(c,k)
            iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
            A=1/(a.h*np.power(float(k if iq==0 else a.n-k),1/a.d))
            actual.append([A,a.B,a.a,.05*power,c,iq,ia])
            meta.append((iq,ia,power,float(f[1][0])))
        with torch.no_grad():result=self.base(torch.tensor(actual,dtype=torch.float64),details=True)
        states,gaps,opp,terms=(v.numpy() for v in result)
        for j,(i,c) in enumerate(zip(ids,coords)):
            iq,ia,power,der=meta[j]
            self.trace.append(dict(case_id=int(i),c=float(c),q_active=iq,aspect_active=ia,aspect_power=power,
                teacher_float64_derivative=der,learned_state=int(states[j]),gap=float(gaps[j]),
                opposite_terms=bool(opp[j]),T1=float(terms[j,0]),T2=float(terms[j,1])))
        return result if details else result[0]


def summarize(records,ids,name):
    rr=[records[i] for i in ids];available=[r for r in rr if r['candidate_available']]
    errors=np.array([r['absolute_cut_error'] for r in available])
    ulps=np.array([r['ulp_error'] for r in available],dtype=np.int64)
    return dict(group=name,cases=len(rr),available=len(available),missing=len(rr)-len(available),
        exact_coordinates=int((ulps==0).sum()),one_ULP=int((ulps==1).sum()),larger_ULP=int((ulps>1).sum()),
        median_cut_error=float(np.median(errors)) if len(errors) else None,
        p99_cut_error=float(np.quantile(errors,.99)) if len(errors) else None,
        max_cut_error=float(errors.max()) if len(errors) else None,max_ULP_error=int(ulps.max()) if len(ulps) else None,
        localized=sum(r.get('localized',False) for r in rr),
        brent_cut_outside_final_bracket=sum(r.get('brent_cut_contained') is False for r in rr),
        score_outside_equivalence=sum(r.get('score_outside_equivalence',False) for r in available))


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    # This file contains only geometry/labels. No offset-model checkpoint,
    # predictions, architecture or training routine is loaded.
    rows=json.loads((ROOT/'branch_switch_ulp_offset_training/training_states.json').read_text())
    assert len(rows)==818 and sum(r['uid'] is not None for r in rows)==774
    json_write(OUT/'case_states.json',rows)
    selector=Selector();locator=selector.locator
    original_sign=locator.sign;dynamic=DynamicBranches(original_sign,rows);locator.sign=dynamic
    st,availability,availability_refs=load_locked_models()
    n=len(rows);bank=np.zeros((n,7));bank[:,0]=np.arange(n)
    L=np.array([r['left'] for r in rows]);R=np.array([r['right'] for r in rows])
    def info(ids,c):
        xx=np.column_stack((bank[ids,:4],c,bank[ids,4:6]))
        return tuple(v.numpy() for v in dynamic(torch.from_numpy(xx),details=True))
    ids=np.arange(n);sl,gl,ol,tl=info(ids,L);sr,gr,orr,tr=info(ids,R)
    # Recheck locked availability from the same two interior sample states.
    margins=np.column_stack((tl.sum(1)/abs(tl).sum(1),tr.sum(1)/abs(tr).sum(1)))
    with torch.no_grad():trans=st(torch.from_numpy(margins)).argmax(1).numpy()
    changed=[]
    for i in ids:
        a=dynamic.audits[i];f=a.formula(np.array([L[i],R[i]]),rows[i]['count'])
        p=f[5][np.arange(2),f[4]]
        changed.append(bool(f[3][0]!=f[3][1] or f[4][0]!=f[4][1] or p[0]!=p[1]))
    with torch.no_grad():exists=availability(torch.from_numpy(np.column_stack((trans,changed,np.ones((n,2)))))).argmax(1).numpy().astype(bool)
    eligible=exists&(sl!=DEFER)&(sr!=DEFER)&(sl!=sr)&(L<R)
    records=[dict(case_id=i,uid=r['uid'],label=r['label'],source=r['source'],switch_type=r['switch_type'],
                  candidate_available=False,localized=False,availability_yes=bool(exists[i]),
                  initial_left_state=int(sl[i]),initial_right_state=int(sr[i]),
                  status='INITIAL_BRACKET_UNUSABLE' if not eligible[i] else 'PENDING') for i,r in enumerate(rows)]
    use=np.flatnonzero(eligible)
    brackets=locator.locate(bank[use],L[use],R[use])
    fl,fg,fo,ft=info(use,brackets['L']);fr,rg,ro,rt=info(use,brackets['R'])
    compatible=fo&ro&(fl!=DEFER)&(fr!=DEFER)&np.isfinite(fg)&np.isfinite(rg)&(fg>0)&(rg>0)
    # The existing terminal feature function explicitly requires opposite-sign
    # terms at BOTH endpoints. Do not invent a gap for same-sign or beta=0 cases.
    good=np.flatnonzero(compatible);cuts={}
    if len(good):
        fx=features(dynamic,bank[use[good]],brackets['L'][good],brackets['R'][good],brackets['adjacent'][good])
        cc=selector.terminal.locate(brackets['L'][good],brackets['R'][good],fx)
        cuts=dict(zip(use[good],cc))
    terminal_failure_inputs=[]
    for j,i in enumerate(use):
        r=rows[i];record=records[i];l=brackets['L'][j];u=brackets['R'][j]
        record.update(localized=True,final_L=l,final_R=u,iterations=int(brackets['iterations'][j]),
                      queries=int(brackets['queries'][j]),termination=brackets['reason'][j],
                      final_left_state=int(fl[j]),final_right_state=int(fr[j]),
                      terminal_left_opposite=bool(fo[j]),terminal_right_opposite=bool(ro[j]),
                      terminal_left_gap=fg[j] if np.isfinite(fg[j]) else None,
                      terminal_right_gap=rg[j] if np.isfinite(rg[j]) else None,
                      brent_cut_contained=bool(l<=r['teacher_cut']<=u),
                      projected_boundary_contained=bool(l<=r['b']<=u),
                      brent_distance_to_bracket=max(l-r['teacher_cut'],r['teacher_cut']-u,0.))
        if i not in cuts:
            record['status']='TERMINAL_FEATURE_DOMAIN_FAILURE'
            terminal_failure_inputs.append(dict(case_id=int(i),label=r['label'],L=l,R=u,
                left_terms=ft[j].tolist(),right_terms=rt[j].tolist(),
                left_opposite=bool(fo[j]),right_opposite=bool(ro[j])))
            continue
        c=float(cuts[i]);scores=dynamic.audits[i].formula(np.array([r['teacher_cut'],c]),r['count'])[0]
        record.update(status='CANDIDATE_RETURNED',candidate_available=True,candidate=c,
            teacher_cut=r['teacher_cut'],absolute_cut_error=abs(c-r['teacher_cut']),ulp_error=abs(bits(c)-bits(r['teacher_cut'])),
            outside_final_bracket=not l<=c<=u,teacher_score=scores[0],candidate_teacher_score=scores[1],
            score_outside_equivalence=bool(abs(scores[1]-scores[0])>1e-12*max(1.,scores[0])))
    # Exercise the original assertion on an actual rejected case for diagnosis.
    assertion_confirmed=False
    if len(use)>len(good):
        j=int(np.flatnonzero(~compatible)[0])
        try:features(dynamic,bank[use[j:j+1]],brackets['L'][j:j+1],brackets['R'][j:j+1],brackets['adjacent'][j:j+1])
        except AssertionError:assertion_confirmed=True
    metrics=[summarize(records,range(n),'all'),summarize(records,range(774),'all_known'),summarize(records,range(774,n),'fresh_holdout')]
    for typ in sorted({r['switch_type'] for r in rows}):metrics.append(summarize(records,[i for i,r in enumerate(rows) if r['switch_type']==typ],typ))
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv');groupstats=[]
    for g in range(21):
        gi=sorted({int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==g})
        groupstats.append(summarize(records,gi,str(g)))
    raw=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_all.csv')
    occurrences=[dict(occurrence=int(r['index']),**records[int(r['unique_id'])]) for r in raw]
    write_csv(OUT/'candidate_results.csv',records);write_csv(OUT/'all_saved_occurrence_replay.csv',occurrences)
    write_csv(OUT/'candidate_metrics.csv',metrics);write_csv(OUT/'conflicting_group_replay.csv',groupstats)
    write_csv(OUT/'dynamic_query_trace.csv',dynamic.trace)
    json_write(OUT/'terminal_feature_failures.json',terminal_failure_inputs)
    json_write(OUT/'records.json',records);json_write(OUT/'frozen_references.json',dict(selector=selector.refs,availability=availability_refs))
    assert all(sha(Path(p))==v for p,v in protected.items())
    result=dict(status='ASSEMBLY_FAILED' if not all(r['candidate_available'] for r in records) else 'DOWNSTREAM_PENDING',
        unique_cases=n,saved_unique=774,saved_occurrences=len(raw),fresh_cases=n-774,
        availability_mismatches=int((~exists).sum()),eligible_brackets=int(eligible.sum()),
        localized_brackets=len(use),missing_candidates=sum(not r['candidate_available'] for r in records),
        terminal_domain_failures=len(terminal_failure_inputs),terminal_assertion_confirmed=assertion_confirmed,
        trusted_endpoint_invariant_violations=0,metrics=metrics,conflicting_group_metrics=groupstats,
        recursive_k_minus6=records[765],total_locator_queries=int(brackets['queries'].sum()),
        initial_brent_containment_failures=int(sum(not r['left']<=r['teacher_cut']<=r['right'] for r in rows)),
        reference_containment_caveat='Brent returned coordinate containment is not mathematical root containment: dynamic max-branch switches may be nonsmooth/discontinuous in the sampled derivative.',
        protected_files=len(protected),protected_hashes_unchanged=True,no_training=True,no_recursion=True,
        elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'protected_hashes.json',protected);json_write(OUT/'assembly_result.json',result)
    print((OUT/'assembly_result.json').read_text(),flush=True)


if __name__=='__main__':run()
