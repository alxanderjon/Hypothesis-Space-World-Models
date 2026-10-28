"""Learn sign-triple routing only; deterministic midpoint evaluation, no query model."""
import ast
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn

from experiments.geometry.validate_full_derivative import Inputs,case_matrix
from experiments.geometry.validate_stationary_margin import load_model as load_terms,margin_teacher
from experiments.geometry.validate_stationary_sign import StationarySign,mathematical_sign,term_teacher
from experiments.geometry.train_stationary_magnitude_margin import MagnitudeMargin
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.audit_stationary_point_impact import canonical,compare_pool
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.train_subpiece_stationary_solver import roots_high_precision,score as generic_score
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'stationary_bracket_update_smoke'
SEED=27552026


class BracketSelector(nn.Module):
    def __init__(self):
        super().__init__();self.weights=nn.Parameter(.3*torch.randn(7,dtype=torch.float64))

    def features(self,s):
        l,q,r=s.unbind(1)
        return torch.stack((torch.ones_like(l),l,q,r,l*q,l*r,q*r),1)

    def forward(self,s):return self.features(s)@self.weights


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=60
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    base,refs=load_terms();parts=[]
    for folder,cls in [('stationary_magnitude_margin_smoke',MagnitudeMargin),('stationary_magnitude_threshold_smoke/zero_anchor_converged',MagnitudeThreshold)]:
        p=ROOT/folder/'candidate_manifest.json';ref=json.loads(p.read_text());cp=Path(ref['checkpoint'])
        assert ref['status']=='FROZEN_CANDIDATE' and sha(cp)==ref['sha256']
        m=cls();m.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);parts.append(m.eval().requires_grad_(False))
        protected[str(p)]=sha(p)
    for file in ('experiments/geometry/validate_stationary_sign.py','experiments/geometry/validate_stationary_margin.py'):
        protected[file]=sha(Path(file))
    sign=StationarySign(base.components,*parts).eval().requires_grad_(False)
    triples=np.array([[l,q,-l] for l in (-1.,1.) for q in (-1.,1.)])
    labels=np.where(triples[:,1]==triples[:,0],1.,-1.)
    selector=BracketSelector();initial=selector.weights.detach().numpy().copy()
    tx=torch.from_numpy(triples);phi=selector.features(tx).numpy();history=[]
    for update in range(8):
        with torch.no_grad():pred=selector(tx).numpy()
        err=labels-pred;w=selector.weights.detach().numpy().copy()
        history.append(dict(update=update,rmse=float(np.sqrt(np.mean(err*err))),max_error=float(abs(err).max()),weights=json.dumps(w.tolist())))
        if abs(err).max()<=8*np.finfo(float).eps:break
        delta=np.linalg.lstsq(phi,err,rcond=None)[0]
        with torch.no_grad():selector.weights.add_(torch.from_numpy(delta))
    config=dict(seed=SEED,dtype='float64',inputs=['sign_L','sign_q','sign_R'],features=['1','L','q','R','L*q','L*R','q*R'],
        initial_weights=initial.tolist(),optimizer='Supervised least-squares residual updates from random initialization',
        label_encoding='+1 retains [q,R]; -1 retains [L,q]',truth_table='All four valid nonzero sign triples; no unseen valid triple exists.',
        zero_query='Record deterministic midpoint as candidate and stop; ZERO does not certify a mathematical root.',
        midpoint='(L+R)/2',termination='Learned ZERO, adjacent float endpoints / stagnating midpoint, or 256 iteration safeguard',
        sign_operator_status='Existing DIAGNOSTIC_ONLY sign assembly, used unchanged under explicit user authorization',no_coefficient_snapping=True)
    checkpoint=OUT/'bracket_selector_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=selector.state_dict(),config=config,training_history=history,status='SAVED_BEFORE_EVALUATION'))
    selector.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);selector.eval().requires_grad_(False)
    with torch.no_grad():tt=selector(tx).numpy()
    truth_errors=int(np.sum(np.sign(tt)!=labels))
    write_csv(OUT/'bracket_selector_truth_table.csv',[dict(sign_L=l,sign_q=q,sign_R=r,teacher_keep='q_R' if y>0 else 'L_q',learned_logit=p,correct=bool(np.sign(p)==y)) for (l,q,r),y,p in zip(triples,labels,tt)])
    write_csv(OUT/'training_history.csv',history);(OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(event='selector_saved',weights=selector.weights.tolist(),truth_table_errors=truth_errors,sha256=digest)),flush=True)
    if truth_errors:raise RuntimeError('Bracket truth-table failed; no iteration authorized.')
    source=Inputs();source.collect();pieces=[]
    for b in source.batches:
        if b['scope']!='archived_subpiece_grids':continue
        key=ast.literal_eval(b['case']);r=source.piece_map[key]
        pieces.append(dict(scope='geometric',case=b['case'],key=key,lo=float(r['lower']),hi=float(r['upper']),
            coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots'])))
    for key,r in source.generic.items():pieces.append(dict(scope='generic_coefficients',case=key,record=r,lo=r['lower'],hi=r['upper'],
        coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),roots=r['roots']))
    del source.batches
    endpoints={r['case']:r for r in csv.DictReader((ROOT/'stationary_sign_assembly/stationary_sign_endpoints.csv').open())}
    chosen=[p for p in pieces if endpoints[p['scope']+':'+p['case']]['prior_bracket']=='True']; n=len(chosen)
    teacher_roots=[];root_sources=[]
    for p in chosen:
        if p['scope']=='generic_coefficients' and p['record'].get('root_decimal'):
            root=mp.mpf(p['record']['root_decimal']);root_sources.append('saved_high_precision_decimal')
        else:
            A,B,a,beta,ot,ov=p['coeff'];r=dict(A=A,B=B,a=a,beta=beta,t=ot,v=ov)
            candidates=[x for x in roots_high_precision(r) if mp.mpf(p['lo'])<x<mp.mpf(p['hi'])]
            assert len(candidates)==1
            root=candidates[0];root_sources.append('existing_high_precision_teacher_routine')
        assert mp.mpf(p['lo'])<root<mp.mpf(p['hi']);teacher_roots.append(root)
    coeff=np.array([p['coeff'] for p in chosen]);L=np.array([p['lo'] for p in chosen]);R=np.array([p['hi'] for p in chosen])
    def inputs(ids,c):
        v=coeff[ids];return np.column_stack((v[:,0],v[:,1],v[:,2],v[:,3],c,v[:,4],v[:,5]))
    with torch.no_grad():
        sl=sign(torch.from_numpy(inputs(np.arange(n),L))).numpy();sr=sign(torch.from_numpy(inputs(np.arange(n),R))).numpy()
    assert np.all(sl*sr<0)
    active=np.ones(n,bool);candidate=np.full(n,np.nan);reason=np.full(n,'',object);steps=np.zeros(n,int)
    routing_errors=np.zeros(n,int);directions=np.zeros(n,int);zero_false=np.zeros(n,bool);events=[];selector_calls=0
    for iteration in range(1,257):
        ids=np.flatnonzero(active)
        if not len(ids):break
        q=(L[ids]+R[ids])/2
        adjacent=(np.nextafter(L[ids],np.inf)>=R[ids])|(q<=L[ids])|(q>=R[ids])
        done=ids[adjacent];candidate[done]=q[adjacent];reason[done]='adjacent_float';active[done]=False
        ids=ids[~adjacent];q=q[~adjacent]
        if not len(ids):continue
        x=inputs(ids,q);xt=torch.from_numpy(x)
        with torch.no_grad():terms=sign.terms(xt);sq=sign.route_terms(terms).numpy()
        exact,_=mathematical_sign(x);steps[ids]+=1
        zero=sq==0;done=ids[zero];candidate[done]=q[zero];reason[done]='learned_ZERO';active[done]=False;zero_false[done]=exact[zero]!=0
        nonzero=~zero;live=ids[nonzero]
        trip=np.column_stack((sl[live],sq[nonzero],sr[live]))
        with torch.no_grad():decision=np.sign(selector(torch.from_numpy(trip)).numpy())
        selector_calls+=len(live);teacher_decision=np.where(trip[:,1]==trip[:,0],1.,-1.)
        selector_bad=decision!=teacher_decision;routing_errors[live]+=selector_bad
        wrong_direction=(sq[nonzero]!=exact[nonzero])&(exact[nonzero]!=0)
        directions[live]+=wrong_direction
        problematic=(sq!=exact)
        # Diagnostic interventions do not affect the actual midpoint trajectory.
        for j in np.flatnonzero(problematic):
            i=ids[j];p=chosen[i];row=x[j:j+1];tt=terms[j:j+1]
            with torch.no_grad():
                A,B,a,beta,c,ot,ov=torch.from_numpy(row).unbind(1);m=sign.components
                t=m['orientation'](torch.stack((c,ot),1));v=m['orientation'](torch.stack((c,ov),1));w=m['orientation'].weights;vp=w[2]+w[3]*ov
                qq=m['quadratic'](torch.stack((B,a,t),1));rr=m['sqrt'](qq)
                rt=torch.from_numpy(np.sqrt(qq.numpy()))
                numer=m['multiply'](torch.stack((beta,vp),1));newterm=m['multiply'](torch.stack((numer,rt),1))
                corrected=torch.stack((tt[:,0],newterm),1);counter=int(sign.route_terms(corrected)[0])
            direct=margin_teacher(row)[0,1:];directsign=int(term_teacher(direct[None,:])[0])
            scale=float(abs(direct).sum());summed=float(direct.sum());meaningful=abs(summed)>32*np.finfo(float).eps*scale
            events.append(dict(index=int(i),scope=p['scope'],case=p['case'],iteration=iteration,L=float(L[i]),R=float(R[i]),q=float(q[j]),q_hex=float(q[j]).hex(),
                A=float(row[0,0]),B=float(row[0,1]),a=float(row[0,2]),beta=float(row[0,3]),orientation_t=int(row[0,5]),orientation_v=int(row[0,6]),
                learned_sign=int(sq[j]),mathematical_sign=int(exact[j]),event='false_learned_ZERO' if sq[j]==0 else 'wrong_direction',
                meaningful_away_from_roundoff=meaningful,sqrt_input=float(qq[0]),sqrt_input_hex=float(qq[0]).hex(),sqrt_target=float(rt[0]),sqrt_learned=float(rr[0]),
                learned_term1=float(tt[0,0]),learned_term2=float(tt[0,1]),sqrt_only_replacement_sign=counter,
                sqrt_only_replacement_fixes=counter==int(exact[j]),float_teacher_term_sign=directsign,
                likely_cause='sqrt_approximation' if counter==int(exact[j]) else 'remaining_float64_arithmetic_or_input_rounding',
                teacher_term1=float(direct[0]),teacher_term2=float(direct[1])))
        replace_left=decision>0;L[live[replace_left]]=q[nonzero][replace_left];sl[live[replace_left]]=sq[nonzero][replace_left]
        R[live[~replace_left]]=q[nonzero][~replace_left];sr[live[~replace_left]]=sq[nonzero][~replace_left]
        if iteration%16==0:print(json.dumps(dict(event='midpoints',iteration=iteration,active=int(active.sum()),zero_stops=int(np.sum(reason=='learned_ZERO')),direction_errors=int(directions.sum()))),flush=True)
    if active.any():candidate[active]=(L[active]+R[active])/2;reason[active]='iteration_limit'
    audit=Audit.__new__(Audit);root_rows=[];solved={}
    for i,p in enumerate(chosen):
        ct=float(teacher_roots[i]);cl=float(candidate[i]);lo,hi=p['lo'],p['hi']
        if p['scope']=='geometric':
            parent,axis,k,_=p['key'];audit.setup(source.parents[parent],axis);scores=audit.formula(np.array([lo,ct,hi,cl]),k)[0]
        else:axis=0;scores=generic_score(p['record'],np.array([lo,ct,hi,cl]))
        st,sc=float(scores[1]),float(scores[3]);tol=1e-12*max(1.,st)
        def candidates(root,value):return [dict(axis=axis,c=c,score=float(s),identity=p['case']+':'+kind,kind=kind) for c,s,kind in [(lo,scores[0],'left'),(root,value,'stationary'),(hi,scores[2],'right')]]
        tc,lc=candidates(ct,st),candidates(cl,sc);bt,bl=canonical(tc),canonical(lc)
        displacement=float(abs(mp.mpf(cl)-teacher_roots[i]));outside=abs(sc-st)>tol
        r=dict(index=i,scope=p['scope'],case=p['case'],L_initial=lo,R_initial=hi,L_final=float(L[i]),R_final=float(R[i]),
            c_teacher=ct,c_teacher_decimal=mp.nstr(teacher_roots[i],60),teacher_reference_source=root_sources[i],c_learned=cl,
            termination=reason[i],iterations=int(steps[i]),false_learned_ZERO=bool(zero_false[i]),root_displacement=displacement,
            final_bracket_contains_teacher=bool(mp.mpf(L[i])<=teacher_roots[i]<=mp.mpf(R[i])),
            selector_errors=int(routing_errors[i]),direction_mistakes=int(directions[i]),teacher_score=st,learned_candidate_teacher_score=sc,
            score_difference=abs(sc-st),outside_teacher_equivalence=outside,teacher_minimum_kind=bt['kind'],learned_minimum_kind=bl['kind'],
            minimum_identity_changed=bt['kind']!=bl['kind'],minimum_outside_equivalence=abs(bt['score']-bl['score'])>1e-12*max(1.,min(x['score'] for x in tc)))
        root_rows.append(r);solved[(p['scope'],p['case'])]=(tc,lc)
    pools={};parentpools={}
    for p in pieces:
        if p['scope']!='geometric':continue
        parent,axis,k,_=p['key'];pair=solved.get((p['scope'],p['case']))
        if pair is None:
            audit.setup(source.parents[parent],axis);cuts=sorted(set([p['lo'],p['hi']]+[c for c in p['roots'] if p['lo']<=c<=p['hi']]))
            scores=audit.formula(np.array(cuts),k)[0]
            cs=[dict(axis=axis,c=c,score=float(s),identity=p['case']+f':saved{j}') for j,(c,s) in enumerate(zip(cuts,scores))];pair=(cs,cs)
        dest=pools.setdefault((parent,axis,k),{'teacher':[],'learned':[]})
        for arm,cs in zip(('teacher','learned'),pair):dest[arm].extend(cs)
    comparisons=[]
    for key,pool in pools.items():
        comparisons.append(compare_pool('allocation_interval',str(key),pool));parent=key[0]
        for arm in pool:parentpools.setdefault(parent,{'teacher':[],'learned':[]})[arm].extend(pool[arm])
    for parent,pool in parentpools.items():comparisons.append(compare_pool('archived_parent',str(parent),pool))
    disp=np.array([r['root_displacement'] for r in root_rows]);parentchanges=sum(r['raw_cut_changed'] for r in comparisons if r['scope']=='archived_parent')
    success=sum(r['termination']!='iteration_limit' and not r['outside_teacher_equivalence'] and not r['minimum_identity_changed'] for r in root_rows)
    passed=bool(truth_errors==0 and routing_errors.sum()==0 and directions.sum()==0 and success==n and parentchanges==0)
    status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP'
    summary=dict(operator='stationary_bracket_update_operator',status=status,gate_passed=passed,checkpoint=str(checkpoint),sha256=digest,
        learned_weights=selector.weights.tolist(),features=config['features'],truth_table_errors=truth_errors,heldout_routing_errors=int(routing_errors.sum()),
        heldout_selector_queries=selector_calls,bracketed_subpieces=n,successful_teacher_equivalent_candidates=success,
        zero_triggered_terminations=int(np.sum(reason=='learned_ZERO')),false_zero_terminations=int(zero_false.sum()),
        adjacent_float_terminations=int(np.sum(reason=='adjacent_float')),iteration_limit_failures=int(np.sum(reason=='iteration_limit')),
        root_displacement_median=float(np.median(disp)),root_displacement_p99=float(np.quantile(disp,.99)),root_displacement_max=float(disp.max()),
        direction_mistakes=int(directions.sum()),subpieces_with_direction_mistakes=int(np.sum(directions>0)),
        direction_mistakes_away_from_roundoff=sum(e['event']=='wrong_direction' and e['meaningful_away_from_roundoff'] for e in events),
        maximum_score_difference=max(r['score_difference'] for r in root_rows),score_differences_outside_equivalence=sum(r['outside_teacher_equivalence'] for r in root_rows),
        subpiece_minimum_identity_changes=sum(r['minimum_identity_changed'] for r in root_rows),canonical_parent_best_cut_changes=parentchanges,
        diagnosis=dict(events=len(events),sqrt_only_replacement_fixes=sum(e['sqrt_only_replacement_fixes'] for e in events),
            wrong_direction_sqrt_only_fixes=sum(e['event']=='wrong_direction' and e['sqrt_only_replacement_fixes'] for e in events),
            false_zero_sqrt_only_fixes=sum(e['event']=='false_learned_ZERO' and e['sqrt_only_replacement_fixes'] for e in events)),
        zero_policy=config['zero_query'],teacher_references=dict(saved_decimal=root_sources.count('saved_high_precision_decimal'),existing_teacher_routine=root_sources.count('existing_high_precision_teacher_routine')),
        no_existing_operator_modified=True,no_query_model=True,no_recursion_or_global_search=True,elapsed_seconds=time.monotonic()-start)
    unchanged=all(sha(Path(p))==h for p,h in {**protected,**source.inputs}.items());assert unchanged
    summary['all_prior_hashes_unchanged']=True
    write_csv(OUT/'stationary_midpoint_recovery.csv',root_rows);write_csv(OUT/'canonical_minimum_comparison.csv',comparisons)
    write_csv(OUT/'direction_and_zero_failures.csv',events or [dict(status='none')])
    write_csv(OUT/'sqrt_failure_inputs_targets.csv',[e for e in events if e['sqrt_only_replacement_fixes']] or [dict(status='none')])
    write_csv(OUT/'zero_terminated_cases.csv',[r for r in root_rows if r['termination']=='learned_ZERO'] or [dict(status='none')])
    # Counts involving numpy booleans can be numpy integer scalars.
    summary=json.loads(json.dumps(summary,default=lambda value:value.item(),indent=2))
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(source.inputs,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps(dict(summary,midpoint_root_location='VALIDATED_UNDER_TEACHER_EQUIVALENCE',module='experiments.geometry.train_stationary_bracket_update',model='BracketSelector',implementation_sha256=sha(Path(__file__))),indent=2))
    (OUT/'stationary_bracket_update_summary.txt').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
