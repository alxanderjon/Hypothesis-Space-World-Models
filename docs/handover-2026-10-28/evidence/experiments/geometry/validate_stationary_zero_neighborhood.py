"""Read-only midpoint replay with the user-specified adjacent-float ZERO rule."""
import ast
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from experiments.geometry import refine_sqrt_failure_cases as ref

OUT=ref.ROOT/'stationary_zero_neighborhood_audit'


def classification(sm,sp,sl,sr):
    # Reversed local bracket orientation is inconsistent with the maintained bracket.
    return np.select([(sm==sl)&(sp==sr),(sm==sl)&(sp==sl),(sm==sr)&(sp==sr)],
                     ['opposite_neighbors','continue_right','continue_left'],default='unresolved')


def run():
    started=time.monotonic();torch.set_num_threads(1);mp.mp.dps=60
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ref.ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ref.ROOT.rglob('*manifest.json')})
    manifest=json.loads((ref.OUT/'candidate_manifest.json').read_text())
    assert manifest['status']=='FROZEN_CANDIDATE'
    cp=Path(manifest['checkpoint']);assert ref.sha(cp)==manifest['sha256']
    sqrt=ref.ScalarPower();sqrt.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);sqrt.eval().requires_grad_(False)
    _,router=ref.routers(sqrt)
    selector=ref.BracketSelector();selector.load_state_dict(torch.load(ref.FAIL/'bracket_selector_before_evaluation.pt',weights_only=False)['state_dict'])
    selector.eval().requires_grad_(False)
    source=ref.Inputs();source.collect();pieces=[]
    for b in source.batches:
        if b['scope']!='archived_subpiece_grids':continue
        key=ast.literal_eval(b['case']);r=source.piece_map[key]
        pieces.append(dict(scope='geometric',case=b['case'],key=key,lo=float(r['lower']),hi=float(r['upper']),
            coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots'])))
    for key,r in source.generic.items():
        pieces.append(dict(scope='generic_coefficients',case=key,record=r,lo=r['lower'],hi=r['upper'],
            coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),roots=r['roots']))
    del source.batches
    def matrix(coeff,c):
        v=np.asarray(coeff);return np.column_stack((v[:,0],v[:,1],v[:,2],v[:,3],c,v[:,4],v[:,5]))
    def signs(x):
        with torch.no_grad():return router(torch.from_numpy(x)).numpy()
    ex=np.array([[*p['coeff'][:4],c,*p['coeff'][4:]] for p in pieces for c in (p['lo'],p['hi'])])
    ep=signs(ex).reshape(-1,2);truth,_=ref.mathematical_sign(ex);truth=truth.reshape(-1,2)
    endpoint_rows=[dict(scope=p['scope'],case=p['case'],left=int(a[0]),right=int(a[1]),teacher_left=int(t[0]),teacher_right=int(t[1]),
        changed=bool(np.any(a!=t)),missed=bool(t[0]*t[1]<0 and a[0]*a[1]>=0),added=bool(t[0]*t[1]>=0 and a[0]*a[1]<0)) for p,a,t in zip(pieces,ep,truth)]
    ref.write_csv(OUT/'endpoint_brackets.csv',endpoint_rows)
    baseline={(r['scope'],r['case']):r for r in csv.DictReader((ref.FAIL/'stationary_midpoint_recovery.csv').open())}
    selected=[i for i,p in enumerate(pieces) if (p['scope'],p['case']) in baseline];chosen=[pieces[i] for i in selected];n=len(chosen)
    assert n==7093 and np.all(ep[selected,0]*ep[selected,1]<0)
    coeff=np.array([p['coeff'] for p in chosen]);L=np.array([p['lo'] for p in chosen]);R=np.array([p['hi'] for p in chosen]);sl=ep[selected,0].copy();sr=ep[selected,1].copy()
    def inputs(ids,c):return matrix(coeff[ids],c)
    previous={(r['scope'],r['case']):r for r in csv.DictReader((ref.OUT/'refined_sqrt_midpoint_audit.csv').open())}
    oldzero=np.array([i for i,p in enumerate(chosen) if previous[(p['scope'],p['case'])]['termination']=='learned_ZERO'])
    assert len(oldzero)==2944
    prevq=np.array([float(previous[(chosen[i]['scope'],chosen[i]['case'])]['new_candidate']) for i in oldzero])
    qm=np.nextafter(prevq,L[oldzero]);qp=np.nextafter(prevq,R[oldzero])
    sm=signs(inputs(oldzero,qm));sp=signs(inputs(oldzero,qp));prevclass=classification(sm,sp,sl[oldzero],sr[oldzero])
    prevrows=[dict(index=int(i),scope=chosen[i]['scope'],case=chosen[i]['case'],q=float(q),q_minus=float(m),q_plus=float(p),
        sign_minus=int(a),sign_plus=int(b),classification=str(k)) for i,q,m,p,a,b,k in zip(oldzero,prevq,qm,qp,sm,sp,prevclass)]
    active=np.ones(n,bool);candidate=np.full(n,np.nan);stopq=np.full(n,np.nan);reason=np.full(n,'',object);iterations=np.zeros(n,int)
    zeroevents=[];directions=[];selector_errors=0;queries=0
    def audit_direction(ids,c,sgn,operation):
        if not len(ids):return
        x=inputs(ids,c);exact,_=ref.mathematical_sign(x);bad=(sgn!=exact)&(exact!=0)
        for j in np.flatnonzero(bad):
            i=ids[j];terms=ref.margin_teacher(x[j:j+1])[0,1:];scale=abs(terms).sum()
            directions.append(dict(index=int(i),scope=chosen[i]['scope'],case=chosen[i]['case'],iteration=int(iterations[i]),
                operation=operation,c=float(c[j]),learned_sign=int(sgn[j]),mathematical_sign=int(exact[j]),
                normalized_direct_margin=float(abs(terms.sum())/scale) if scale else 0.,
                away_from_roundoff=bool(abs(terms.sum())>32*np.finfo(float).eps*scale)))
    for step in range(256):
        ids=np.flatnonzero(active)
        if not len(ids):break
        q=(L[ids]+R[ids])/2;adj=(np.nextafter(L[ids],np.inf)>=R[ids])|(q<=L[ids])|(q>=R[ids])
        done=ids[adj];candidate[done]=q[adj];reason[done]='adjacent_float';active[done]=False
        ids=ids[~adj];q=q[~adj]
        if not len(ids):continue
        sq=signs(inputs(ids,q));queries+=len(ids);iterations[ids]+=1
        z=sq==0;zi=ids[z];zq=q[z]
        if len(zi):
            minus=np.nextafter(zq,L[zi]);plus=np.nextafter(zq,R[zi]);sm=signs(inputs(zi,minus));sp=signs(inputs(zi,plus));queries+=2*len(zi)
            cls=classification(sm,sp,sl[zi],sr[zi])
            for j,i in enumerate(zi):
                ct=mp.mpf(baseline[(chosen[i]['scope'],chosen[i]['case'])]['c_teacher_decimal'])
                zeroevents.append(dict(index=int(i),scope=chosen[i]['scope'],case=chosen[i]['case'],iteration=int(iterations[i]),L=float(L[i]),R=float(R[i]),
                    q=float(zq[j]),q_minus=float(minus[j]),q_plus=float(plus[j]),sign_L=int(sl[i]),sign_R=int(sr[i]),
                    sign_minus=int(sm[j]),sign_plus=int(sp[j]),classification=str(cls[j]),
                    teacher_root_inside_neighbor_interval=bool(mp.mpf(float(minus[j]))<=ct<=mp.mpf(float(plus[j])))))
            opposite=cls=='opposite_neighbors';done=zi[opposite];candidate[done]=zq[opposite];L[done]=minus[opposite];R[done]=plus[opposite];reason[done]='machine_precision_neighborhood';active[done]=False
            unresolved=cls=='unresolved';done=zi[unresolved];stopq[done]=zq[unresolved];reason[done]='unresolved_ZERO_neighborhood';active[done]=False
            right=cls=='continue_right';left=cls=='continue_left'
            audit_direction(zi[right],plus[right],sp[right],'ZERO_continue_right');audit_direction(zi[left],minus[left],sm[left],'ZERO_continue_left')
            L[zi[right]]=plus[right];sl[zi[right]]=sp[right];R[zi[left]]=minus[left];sr[zi[left]]=sm[left]
        live=ids[~z];cq=q[~z];ss=sq[~z]
        audit_direction(live,cq,ss,'nonzero_midpoint')
        trip=np.column_stack((sl[live],ss,sr[live]))
        with torch.no_grad():decision=np.sign(selector(torch.from_numpy(trip)).numpy())
        selector_errors+=int(np.sum(decision!=np.where(ss==sl[live],1.,-1.)))
        left=decision>0;L[live[left]]=cq[left];sl[live[left]]=ss[left];R[live[~left]]=cq[~left];sr[live[~left]]=ss[~left]
    reason[active]='iteration_limit';stopq[active]=(L[active]+R[active])/2
    print(json.dumps(dict(event='replay_complete',subpieces=n,zero_events=len(zeroevents),unresolved=int(np.sum(reason=='unresolved_ZERO_neighborhood')))),flush=True)
    ref.write_csv(OUT/'zero_neighborhood_events.csv',zeroevents)
    ref.write_csv(OUT/'unresolved_zero_neighborhoods.csv',[r for r in zeroevents if r['classification']=='unresolved'])
    ref.write_csv(OUT/'direction_disagreements.csv',directions or [dict(status='none')])
    audit=ref.Audit.__new__(ref.Audit);rootrows=[];solved={}
    for i,p in enumerate(chosen):
        br=baseline[(p['scope'],p['case'])];ct=float(br['c_teacher']);cl=candidate[i];found=np.isfinite(cl);lo,hi=p['lo'],p['hi']
        cdiag=float(cl if found else stopq[i])
        cuts=np.array([lo,ct,hi,cdiag])
        if p['scope']=='geometric':
            parent,axis,k,_=p['key'];audit.setup(source.parents[parent],axis);scores=audit.formula(cuts,k)[0]
        else:axis=0;scores=ref.generic_score(p['record'],cuts)
        def entries():return [dict(axis=axis,c=lo,score=float(scores[0]),identity=p['case']+':left',kind='left'),dict(axis=axis,c=hi,score=float(scores[2]),identity=p['case']+':right',kind='right')]
        tc=entries();lc=entries();tc.append(dict(axis=axis,c=ct,score=float(scores[1]),identity=p['case']+':stationary',kind='stationary'))
        if found:lc.append(dict(axis=axis,c=float(cl),score=float(scores[3]),identity=p['case']+':stationary',kind='stationary'))
        bt,bl=ref.canonical(tc),ref.canonical(lc)
        row=dict(index=i,scope=p['scope'],case=p['case'],termination=reason[i],found=bool(found),iterations=int(iterations[i]),
            L_final=float(L[i]),R_final=float(R[i]),c_teacher=ct,c_teacher_decimal=br['c_teacher_decimal'],
            c_learned=float(cl) if found else '',unresolved_q='' if found else cdiag,
            root_displacement=float(abs(mp.mpf(float(cl))-mp.mpf(br['c_teacher_decimal']))) if found else '',
            teacher_root_in_final_bracket=bool(mp.mpf(float(L[i]))<=mp.mpf(br['c_teacher_decimal'])<=mp.mpf(float(R[i]))),
            teacher_root_score=float(scores[1]),recovered_root_score=float(scores[3]) if found else '',
            score_difference=float(abs(scores[3]-scores[1])) if found else '',
            root_score_outside_equivalence=bool(abs(scores[3]-scores[1])>1e-12*max(1.,scores[1])) if found else '',
            unresolved_q_diagnostic_score_difference='' if found else float(abs(scores[3]-scores[1])),
            unresolved_q_diagnostic_outside_equivalence='' if found else bool(abs(scores[3]-scores[1])>1e-12*max(1.,scores[1])),
            teacher_minimum_kind=bt['kind'],available_minimum_kind=bl['kind'],minimum_identity_changed=bt['kind']!=bl['kind'],
            minimum_outside_equivalence=bool(abs(bt['score']-bl['score'])>1e-12*max(1.,min(x['score'] for x in tc))))
        rootrows.append(row);solved[(p['scope'],p['case'])]=(tc,lc)
    for r in prevrows:
        i=r['index'];r['final_termination']=str(reason[i]);r['eventually_recovered']=bool(np.isfinite(candidate[i]));r['immediate_resolved']=r['classification']!='unresolved'
    ref.write_csv(OUT/'previous_2944_zero_replay.csv',prevrows);ref.write_csv(OUT/'root_recovery.csv',rootrows)
    pools={};parents={};comparisons=[]
    for p in pieces:
        if p['scope']!='geometric':continue
        parent,axis,k,_=p['key'];pair=solved.get((p['scope'],p['case']))
        if pair is None:
            audit.setup(source.parents[parent],axis);cuts=sorted(set([p['lo'],p['hi']]+[c for c in p['roots'] if p['lo']<=c<=p['hi']]))
            ss=audit.formula(np.array(cuts),k)[0];cs=[dict(axis=axis,c=c,score=float(s),identity=p['case']+f':saved{j}') for j,(c,s) in enumerate(zip(cuts,ss))];pair=cs,cs
        for arm,cs in zip(('teacher','learned'),pair):pools.setdefault((parent,axis,k),{'teacher':[],'learned':[]})[arm].extend(cs)
    for key,pool in pools.items():
        comparisons.append(ref.compare_pool('allocation_interval',str(key),pool))
        for arm in pool:parents.setdefault(key[0],{'teacher':[],'learned':[]})[arm].extend(pool[arm])
    for parent,pool in parents.items():comparisons.append(ref.compare_pool('archived_parent',str(parent),pool))
    ref.write_csv(OUT/'canonical_minimum_comparison.csv',comparisons)
    recovered=[r for r in rootrows if r['found']];disp=np.array([r['root_displacement'] for r in recovered])
    results=dict(status='DIAGNOSTIC_ONLY_STOP',total_bracketed_subpieces=n,total_endpoint_subpieces=len(pieces),
        total_midpoint_and_neighbor_sign_queries=queries,total_routed_ZERO_events=len(zeroevents),
        zero_event_classification={k:sum(r['classification']==k for r in zeroevents) for k in ('opposite_neighbors','continue_left','continue_right','unresolved')},
        direction_mistakes=len(directions),direction_mistakes_away_from_roundoff=sum(r['away_from_roundoff'] for r in directions),
        direction_roundoff_diagnostic='Same previous diagnostic: |direct M| > 32*eps*(|T1|+|T2|). Never used for routing.',selector_errors=selector_errors,
        root_found=len(recovered),root_found_not_found_mismatches=n-len(recovered),
        termination_counts={k:int(np.sum(reason==k)) for k in sorted(set(reason))},
        false_termination_merely_on_ZERO=0,
        endpoint_changes=sum(r['changed'] for r in endpoint_rows),missed_brackets=sum(r['missed'] for r in endpoint_rows),added_brackets=sum(r['added'] for r in endpoint_rows),
        root_displacement_median=float(np.median(disp)),root_displacement_p99=float(np.quantile(disp,.99)),root_displacement_max=float(disp.max()),
        recovered_max_teacher_score_difference=max(r['score_difference'] for r in recovered),recovered_score_differences_outside_equivalence=sum(r['root_score_outside_equivalence'] for r in recovered),
        recovered_subpiece_minimum_identity_changes=sum(r['minimum_identity_changed'] for r in recovered),
        all_subpiece_minimum_identity_changes=sum(r['minimum_identity_changed'] for r in rootrows),
        all_subpiece_minimum_changes_outside_equivalence=sum(r['minimum_outside_equivalence'] for r in rootrows),
        canonical_parent_best_cut_changes=sum(r['raw_cut_changed'] for r in comparisons if r['scope']=='archived_parent'),
        canonical_parent_score_changes_outside_equivalence=sum(r['outside_equivalence'] for r in comparisons if r['scope']=='archived_parent'),
        archived_parents=len(parents),
        previous_2944_immediate_resolved=sum(r['immediate_resolved'] for r in prevrows),previous_2944_eventually_recovered=sum(r['eventually_recovered'] for r in prevrows),
        previous_2944_classification={k:sum(r['classification']==k for r in prevrows) for k in ('opposite_neighbors','continue_left','continue_right','unresolved')},
        opposite_neighbor_localizations_not_containing_high_precision_root=sum(r['classification']=='opposite_neighbors' and not r['teacher_root_inside_neighbor_interval'] for r in zeroevents),
        unresolved_q_diagnostics_outside_equivalence=sum(r['unresolved_q_diagnostic_outside_equivalence']==True for r in rootrows if not r['found']),
        unresolved_policy='No recovered root is returned. Only original subpiece endpoints remain as minimum candidates; unresolved q score is diagnostic only, never silently accepted.',
        opposite_neighbor_policy='Record midpoint as representative of the two-ULP neighborhood, not as proof of an exact root. Reversed neighbor orientation is unresolved.',
        sqrt_checkpoint=str(cp),sqrt_sha256=manifest['sha256'],implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started)
    required=('root_found_not_found_mismatches','direction_mistakes_away_from_roundoff','selector_errors','endpoint_changes','missed_brackets','added_brackets',
              'recovered_score_differences_outside_equivalence','all_subpiece_minimum_identity_changes','all_subpiece_minimum_changes_outside_equivalence','canonical_parent_best_cut_changes')
    passed=all(results[k]==0 for k in required);results['passed']=passed
    if passed:results.update(status='VALIDATED_SCAFFOLD',midpoint_stationary_locator='VALIDATED_UNDER_TEACHER_EQUIVALENCE')
    assert all(ref.sha(Path(p))==h for p,h in {**protected,**source.inputs}.items())
    results['protected_hashes_unchanged']=True
    results=json.loads(json.dumps(results,default=lambda v:v.item()))
    (OUT/'results.json').write_text(json.dumps(results,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:(OUT/'validated_scaffold_manifest.json').write_text(json.dumps(results,indent=2))
    (OUT/'stationary_zero_neighborhood_summary.txt').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':run()
