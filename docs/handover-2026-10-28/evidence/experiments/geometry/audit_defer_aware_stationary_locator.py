"""Deterministic midpoint/quarter-point audit with trusted endpoints only."""
import ast
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_stationary_three_state_router import load_router,TRUST_POSITIVE,TRUST_NEGATIVE,DEFER,NAMES

OUT=ref.ROOT/'defer_aware_stationary_locator_audit'


def quantiles(v):
    v=np.asarray(v,dtype=float)
    return dict(median=float(np.median(v)),p99=float(np.quantile(v,.99)),max=float(v.max())) if len(v) else dict(median=None,p99=None,max=None)


def run():
    started=time.monotonic();torch.set_num_threads(1);mp.mp.dps=70;OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ref.ROOT.rglob('*.pt')};protected.update({str(p):ref.sha(p) for p in ref.ROOT.rglob('*manifest.json')})
    path=ref.ROOT/'stationary_three_state_sign_router_assembly/stationary_three_state_sign_router_validated.json'
    validated=json.loads(path.read_text());previous_results=json.loads((path.parent/'results.json').read_text())
    assert validated['status']=='VALIDATED_ASSEMBLY' and ref.sha(path)==previous_results['assembly_sha256']
    source_code=Path('experiments/geometry/validate_stationary_three_state_router.py');assert ref.sha(source_code)==validated['implementation_sha256']
    protected[str(path)]=ref.sha(path);protected[str(source_code)]=ref.sha(source_code)
    router,references=load_router();source=ref.Inputs();source.collect();pieces=[]
    for b in source.batches:
        if b['scope']!='archived_subpiece_grids':continue
        key=ast.literal_eval(b['case']);r=source.piece_map[key]
        pieces.append(dict(scope='geometric',case=b['case'],key=key,lo=float(r['lower']),hi=float(r['upper']),
            coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots'])))
    for key,r in source.generic.items():pieces.append(dict(scope='generic_coefficients',case=key,record=r,lo=r['lower'],hi=r['upper'],coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),roots=r['roots']))
    del source.batches
    def read(path):
        protected[str(path)]=ref.sha(path);return list(csv.DictReader(path.open()))
    baselines={(r['scope'],r['case']):r for r in read(ref.FAIL/'stationary_midpoint_recovery.csv')}
    previous={(r['scope'],r['case']):r for r in read(ref.OUT/'refined_sqrt_midpoint_audit.csv')}
    chosen=[p for p in pieces if (p['scope'],p['case']) in baselines];n=len(chosen);assert n==7093
    coeff=np.array([p['coeff'] for p in chosen]);L=np.array([p['lo'] for p in chosen]);R=np.array([p['hi'] for p in chosen])
    initialL=L.copy();initialR=R.copy();calls=np.zeros(n,int);defers=np.zeros(n,int);iterations=np.zeros(n,int)
    active=np.ones(n,bool);reason=np.full(n,'',object);direction_events=[];step_rows=[];defer_rows=[];nonfinite=0;invariant_checks=0
    def inputs(ids,c):
        a=coeff[ids];return np.column_stack((a[:,:4],c,a[:,4:]))
    def query(ids,c,kind):
        nonlocal nonfinite
        x=inputs(ids,c)
        with torch.no_grad():codes,gaps,op,terms=router(torch.from_numpy(x),details=True)
        codes=codes.numpy();gaps=gaps.numpy();terms=terms.numpy();calls[ids]+=1;defers[ids]+=(codes==DEFER)
        nonfinite+=int(np.sum(~np.isfinite(terms).all(1)))
        teacher,_=ref.mathematical_sign(x);expected=np.where(teacher>0,TRUST_POSITIVE,np.where(teacher<0,TRUST_NEGATIVE,DEFER))
        bad=(codes!=DEFER)&(codes!=expected)
        for j in np.flatnonzero(bad):
            i=ids[j];direct=ref.margin_teacher(x[j:j+1])[0,1:];scale=abs(direct).sum()
            direction_events.append(dict(index=int(i),scope=chosen[i]['scope'],case=chosen[i]['case'],iteration=int(iterations[i]),query_type=kind,
                A=x[j,0],B=x[j,1],a=x[j,2],beta=x[j,3],c=float(c[j]),ot=int(x[j,5]),ov=int(x[j,6]),
                prediction=NAMES[int(codes[j])],mathematical_sign=int(teacher[j]),r=float(gaps[j]),T1=terms[j,0],T2=terms[j,1],
                away_from_roundoff=bool(abs(direct.sum())>32*np.finfo(float).eps*scale)))
        return codes
    ids=np.arange(n);sl=query(ids,L,'initial_left');sr=query(ids,R,'initial_right')
    def invariant(where):
        nonlocal invariant_checks
        invariant_checks+=1
        ok=(L<R)&np.isin(sl,[TRUST_POSITIVE,TRUST_NEGATIVE])&np.isin(sr,[TRUST_POSITIVE,TRUST_NEGATIVE])&(sl!=sr)
        if not ok.all():
            ref.write_csv(OUT/'invariant_violations.csv',[dict(index=int(i),scope=chosen[i]['scope'],case=chosen[i]['case'],where=where,L=L[i],R=R[i],left=int(sl[i]),right=int(sr[i])) for i in np.flatnonzero(~ok)])
            raise RuntimeError('Trusted endpoint invariant violated; exact cases saved; audit stopped.')
    invariant('initial')
    def update(ids,c,codes):
        left=(codes!=DEFER)&(codes==sl[ids]);right=(codes!=DEFER)&(codes==sr[ids])
        assert np.all(left|right|(codes==DEFER))
        L[ids[left]]=c[left];sl[ids[left]]=codes[left]
        R[ids[right]]=c[right];sr[ids[right]]=codes[right]
    step=0
    while active.any():
        ids=np.flatnonzero(active);q=(L[ids]+R[ids])/2
        adjacent=np.nextafter(L[ids],R[ids])==R[ids];rounded=(q==L[ids])|(q==R[ids])
        done=ids[adjacent|rounded];reason[done]=np.where(adjacent[adjacent|rounded],'adjacent_float','midpoint_rounds_to_endpoint');active[done]=False
        ids=ids[~(adjacent|rounded)];q=q[~(adjacent|rounded)]
        if not len(ids):break
        iterations[ids]+=1;codes=query(ids,q,'midpoint');deferred=codes==DEFER
        update(ids[~deferred],q[~deferred],codes[~deferred]);invariant('trusted_midpoint')
        di=ids[deferred];dq=q[deferred]
        if len(di):
            oldL=L[di].copy();oldR=R[di].copy();oldsl=sl[di].copy();oldsr=sr[di].copy()
            # Both quarter queries come from the same pre-refinement bracket.
            # Their updates are applied sequentially, qR against CURRENT endpoints,
            # exactly as specified; no DEFER endpoint is ever installed.
            qL=(oldL+dq)/2;qR=(dq+oldR)/2
            left=query(di,qL,'DEFER_left_quarter');right=query(di,qR,'DEFER_right_quarter')
            update(di,qL,left);invariant('left_quarter')
            beforeR_L=L[di].copy();beforeR_R=R[di].copy()
            update(di,qR,right);invariant('right_quarter')
            assert np.all(L[di]>=oldL) and np.all(R[di]<=oldR)
            progressed=(L[di]>oldL)|(R[di]<oldR)
            stalled=di[~progressed];reason[stalled]='no_trusted_inward_move';active[stalled]=False
            for j,i in enumerate(di):
                defer_rows.append(dict(index=int(i),scope=chosen[i]['scope'],case=chosen[i]['case'],iteration=int(iterations[i]),old_L=oldL[j],old_R=oldR[j],
                    q=dq[j],qL=qL[j],qR=qR[j],qL_state=NAMES[int(left[j])],qR_state=NAMES[int(right[j])],
                    new_L=L[i],new_R=R[i],left_state=NAMES[int(sl[i])],right_state=NAMES[int(sr[i])],progressed=bool(progressed[j]),
                    qR_temporarily_widens_relative_to_qL=bool(L[i]<beforeR_L[j] or R[i]>beforeR_R[j])))
        step_rows.append(dict(round=step,active_after=int(active.sum()),midpoint_queries=len(ids),DEFER_midpoints=len(di),completed=int((~active).sum())))
        step+=1
    invariant('final')
    candidate=(L+R)/2
    # Independent endpoint re-evaluation is an audit check, not an extra search step.
    with torch.no_grad():
        checkL=router(torch.from_numpy(inputs(np.arange(n),L))).numpy();checkR=router(torch.from_numpy(inputs(np.arange(n),R))).numpy()
    assert np.array_equal(checkL,sl) and np.array_equal(checkR,sr)
    ref.write_csv(OUT/'defer_refinement_steps.csv',defer_rows);ref.write_csv(OUT/'locator_rounds.csv',step_rows)
    if direction_events:ref.write_csv(OUT/'trusted_direction_disagreements.csv',direction_events)
    audit=ref.Audit.__new__(ref.Audit);rows=[];solved={}
    for i,p in enumerate(chosen):
        key=(p['scope'],p['case']);base=baselines[key];prev=previous[key];ct=float(base['c_teacher']);cl=float(candidate[i]);cp=float(prev['new_candidate'])
        true=mp.mpf(base['c_teacher_decimal']);lo,hi=p['lo'],p['hi'];cuts=np.array([lo,ct,hi,cl,cp])
        if p['scope']=='geometric':
            parent,axis,k,_=p['key'];audit.setup(source.parents[parent],axis);ss=audit.formula(cuts,k)[0]
        else:axis=0;ss=ref.generic_score(p['record'],cuts)
        def cs(root,value):return [dict(axis=axis,c=c,score=float(s),identity=p['case']+':'+kind,kind=kind) for c,s,kind in [(lo,ss[0],'left'),(root,value,'stationary'),(hi,ss[2],'right')]]
        tc,lc,pc=cs(ct,ss[1]),cs(cl,ss[3]),cs(cp,ss[4]);bt,bl,bp=map(ref.canonical,(tc,lc,pc))
        outside=max(mp.mpf(float(L[i]))-true,true-mp.mpf(float(R[i])),mp.mpf(0));distance=abs(mp.mpf(cl)-true)
        rows.append(dict(index=i,scope=p['scope'],case=p['case'],termination=reason[i],iterations=int(iterations[i]),router_queries=int(calls[i]),DEFER_queries=int(defers[i]),
            initial_L=initialL[i],initial_R=initialR[i],final_L=L[i],final_R=R[i],final_width=float(R[i]-L[i]),
            final_width_in_float_steps=int(np.float64(R[i]).view(np.int64)-np.float64(L[i]).view(np.int64)),
            left_state=NAMES[int(sl[i])],right_state=NAMES[int(sr[i])],adjacent_float=bool(np.nextafter(L[i],R[i])==R[i]),
            c_teacher=ct,c_teacher_decimal=base['c_teacher_decimal'],c_candidate=cl,root_contained=bool(outside==0),teacher_distance_to_bracket=float(outside),root_displacement=float(distance),
            teacher_score=float(ss[1]),candidate_teacher_score=float(ss[3]),score_difference=float(abs(ss[3]-ss[1])),
            score_outside_equivalence=bool(abs(ss[3]-ss[1])>1e-12*max(1.,ss[1])),subpiece_minimum_identity_changed=bt['kind']!=bl['kind'],
            subpiece_minimum_outside_equivalence=bool(abs(bt['score']-bl['score'])>1e-12*max(1.,min(c['score'] for c in tc))),
            previous_midpoint_candidate=cp,previous_termination=prev['termination'],previous_iterations=int(prev['iterations']),
            previous_root_displacement=float(abs(mp.mpf(cp)-true)),previous_score_difference=float(abs(ss[4]-ss[1])),
            candidate_changed=cl!=cp,minimum_identity_changed_vs_previous=bl['kind']!=bp['kind'],
            score_changed_outside_equivalence_vs_previous=bool(abs(ss[3]-ss[4])>1e-12*max(1.,min(ss[3],ss[4])))))
        solved[key]=(tc,lc)
    pools={};parents={};comparisons=[]
    for p in pieces:
        if p['scope']!='geometric':continue
        parent,axis,k,_=p['key'];pair=solved.get((p['scope'],p['case']))
        if pair is None:
            audit.setup(source.parents[parent],axis);cuts=sorted(set([p['lo'],p['hi']]+[c for c in p['roots'] if p['lo']<=c<=p['hi']]))
            ss=audit.formula(np.array(cuts),k)[0];candidates=[dict(axis=axis,c=c,score=float(s),identity=p['case']+f':saved{j}') for j,(c,s) in enumerate(zip(cuts,ss))];pair=candidates,candidates
        for arm,cs in zip(('teacher','learned'),pair):pools.setdefault((parent,axis,k),{'teacher':[],'learned':[]})[arm].extend(cs)
    for key,pool in pools.items():
        comparisons.append(ref.compare_pool('allocation_interval',str(key),pool))
        for arm in pool:parents.setdefault(key[0],{'teacher':[],'learned':[]})[arm].extend(pool[arm])
    for parent,pool in parents.items():comparisons.append(ref.compare_pool('archived_parent',str(parent),pool))
    ref.write_csv(OUT/'defer_aware_root_recovery.csv',rows);ref.write_csv(OUT/'canonical_minimum_comparison.csv',comparisons)
    ref.write_csv(OUT/'worst_root_displacements.csv',sorted(rows,key=lambda r:r['root_displacement'],reverse=True)[:20])
    summary=dict(operator='defer_aware_stationary_locator',status='DIAGNOSTIC_ONLY_STOP',total_brackets=n,
        returned_trusted_brackets=n,termination_counts={str(k):int(np.sum(reason==k)) for k in sorted(set(reason))},
        stalled_no_trusted_inward_move=int(np.sum(reason=='no_trusted_inward_move')),
        adjacent_float_resolution=sum(r['adjacent_float'] for r in rows),router_queries=int(calls.sum()),DEFER_queries=int(defers.sum()),
        endpoint_audit_recheck_queries=2*n,invariant_checkpoints=invariant_checks,trusted_endpoint_invariant_violations=0,
        direction_disagreements=len(direction_events),direction_disagreements_away_from_roundoff=sum(r['away_from_roundoff'] for r in direction_events),
        nonfinite_terms=nonfinite,iterations=quantiles(iterations),final_bracket_width=quantiles(R-L),
        final_bracket_float_steps=quantiles([r['final_width_in_float_steps'] for r in rows]),
        teacher_roots_contained=sum(r['root_contained'] for r in rows),teacher_roots_not_literally_contained=sum(not r['root_contained'] for r in rows),
        candidate_root_displacement=quantiles([r['root_displacement'] for r in rows]),
        teacher_distance_to_bracket_all=quantiles([r['teacher_distance_to_bracket'] for r in rows]),
        teacher_distance_to_bracket_outside_only=quantiles([r['teacher_distance_to_bracket'] for r in rows if not r['root_contained']]),
        max_score_difference=max(r['score_difference'] for r in rows),score_differences_outside_equivalence=sum(r['score_outside_equivalence'] for r in rows),
        subpiece_minimum_identity_changes=sum(r['subpiece_minimum_identity_changed'] for r in rows),subpiece_minimum_changes_outside_equivalence=sum(r['subpiece_minimum_outside_equivalence'] for r in rows),
        canonical_best_cut_changes=sum(r['raw_cut_changed'] for r in comparisons if r['scope']=='archived_parent'),archived_parents=len(parents),
        previous_midpoint=dict(reference='Saved refined-sqrt midpoint locator with immediate ZERO stops; no rerun.',root_displacement=quantiles([r['previous_root_displacement'] for r in rows]),
            max_score_difference=max(r['previous_score_difference'] for r in rows),iterations=quantiles([r['previous_iterations'] for r in rows]),
            candidate_changes=sum(r['candidate_changed'] for r in rows),minimum_identity_changes=sum(r['minimum_identity_changed_vs_previous'] for r in rows),
            score_changes_outside_equivalence=sum(r['score_changed_outside_equivalence_vs_previous'] for r in rows)),
        interpretation='No-inward-move stops retain two trusted endpoints; midpoint is an evaluation candidate, not a claim of exact zero.',
        qR_update='qL/qR computed from the pre-refinement bracket, evaluated, then applied in order against current endpoint signs, exactly as requested.',
        sequential_qR_updates_temporarily_widening_after_qL=sum(r['qR_temporarily_widens_relative_to_qL'] for r in defer_rows),
        router_manifest=str(path),router_manifest_sha256=ref.sha(path),implementation_sha256=ref.sha(Path(__file__)),
        no_training=True,no_recursion_or_global_search=True,elapsed_seconds=time.monotonic()-started)
    failed=sum(r['score_outside_equivalence'] or r['subpiece_minimum_identity_changed'] or r['subpiece_minimum_outside_equivalence'] for r in rows)
    summary['successful_localizations_under_teacher_equivalence']=n-failed;summary['meaningful_unresolved_localizations']=failed
    passed=bool(summary['trusted_endpoint_invariant_violations']==summary['direction_disagreements_away_from_roundoff']==summary['score_differences_outside_equivalence']==summary['subpiece_minimum_identity_changes']==summary['subpiece_minimum_changes_outside_equivalence']==summary['canonical_best_cut_changes']==nonfinite==0)
    summary['passed']=passed
    if passed:summary['status']='VALIDATED_UNDER_TEACHER_EQUIVALENCE'
    assert all(ref.sha(Path(p))==h for p,h in {**protected,**source.inputs}.items());summary['all_protected_hashes_unchanged']=True
    if passed:
        dest=OUT/'defer_aware_stationary_locator_validated.json';dest.write_text(json.dumps(summary,indent=2));digest=ref.sha(dest)
        dest.with_suffix('.sha256').write_text(digest+'  '+dest.name+'\n');summary['assembly_manifest']=str(dest);summary['assembly_sha256']=digest
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'defer_aware_stationary_locator_summary.txt').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
