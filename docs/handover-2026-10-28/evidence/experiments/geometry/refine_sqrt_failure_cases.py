"""Refine only ScalarPower on saved sqrt failures; all downstream work is evaluation."""
import ast
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch

from experiments.geometry.train_sqrt_operator import ScalarPower,predict,metrics,RESERVED
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.geometry.validate_stationary_margin import load_model as load_terms,margin_teacher
from experiments.geometry.validate_stationary_sign import StationarySign,mathematical_sign,term_teacher
from experiments.geometry.train_stationary_magnitude_margin import MagnitudeMargin
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_stationary_bracket_update import BracketSelector
from experiments.geometry.validate_full_derivative import Inputs
from experiments.geometry.audit_stationary_point_impact import canonical,compare_pool
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.train_subpiece_stationary_solver import score as generic_score

OUT=ROOT/'sqrt_operator_failure_refine'
SOURCE=ROOT/'sqrt_operator_smoke'
FAIL=ROOT/'stationary_bracket_update_smoke'
SEED=27562026


def routers(refined):
    base,refs=load_terms();parts=[]
    for folder,cls in [('stationary_magnitude_margin_smoke',MagnitudeMargin),('stationary_magnitude_threshold_smoke/zero_anchor_converged',MagnitudeThreshold)]:
        manifest=json.loads((ROOT/folder/'candidate_manifest.json').read_text());cp=Path(manifest['checkpoint']);assert sha(cp)==manifest['sha256']
        m=cls();m.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);parts.append(m.eval().requires_grad_(False))
    old=StationarySign(base.components,*parts).eval().requires_grad_(False)
    new=StationarySign(base.components,*parts).eval().requires_grad_(False)
    new.components['sqrt']=refined
    return old,new


def replay(old,new):
    rows=list(csv.DictReader((FAIL/'direction_and_zero_failures.csv').open()))
    x=np.array([[float(r[k]) for k in ('A','B','a','beta','q','orientation_t','orientation_v')] for r in rows])
    target=np.array([int(r['mathematical_sign']) for r in rows]);before=np.array([int(r['learned_sign']) for r in rows])
    with torch.no_grad():
        actual_before=old(torch.from_numpy(x)).numpy();after=new(torch.from_numpy(x)).numpy()
    assert np.array_equal(before,actual_before)
    direct=np.array([int(r['sqrt_only_replacement_sign']) for r in rows]);known=np.array([r['sqrt_only_replacement_fixes']=='False' for r in rows])
    direction=np.array([r['event']=='wrong_direction' for r in rows]);zero=~direction
    assert direction.sum()==5330 and zero.sum()==2908 and known.sum()==625
    output=[]
    for i,r in enumerate(rows):
        output.append(dict(r,refined_sign=int(after[i]),fixed=bool(after[i]==target[i]),known_direct_float64_case=bool(known[i]),
            agrees_with_direct_sqrt_replay=bool(after[i]==direct[i])))
    write_csv(OUT/'sqrt_downstream_failure_replay.csv',output)
    remaining=(after!=target)&~known
    write_csv(OUT/'remaining_sqrt_attributable_cases.csv',[r for r,keep in zip(output,remaining) if keep] or [dict(status='none')])
    return dict(saved_occurrences=len(rows),original_direction_disagreements=int(direction.sum()),
        direction_disagreements_fixed=int(np.sum(direction&(after==target))),direction_disagreements_remaining=int(np.sum(direction&(after!=target))),
        remaining_directions_known_625=int(np.sum(direction&known&(after!=target))),
        remaining_directions_sqrt_attributable=int(np.sum(direction&~known&(after!=target))),
        false_ZERO_stops_before=int(zero.sum()),previous_false_ZERO_stops_still_ZERO=int(np.sum(zero&(after==0))),
        previous_false_ZERO_cases_corrected=int(np.sum(zero&(after==target))),
        previous_false_ZERO_cases_wrong_nonzero=int(np.sum(zero&(after!=target)&(after!=0))),
        known_625_fixed=int(np.sum(known&(after==target))),all_replay_matches_direct_sqrt=bool(np.array_equal(after,direct)),
        total_remaining_sqrt_attributable=int(remaining.sum()))


def archived_audit(old,new):
    """Frozen selector and deterministic midpoint; no new training or root teacher solve."""
    source=Inputs();source.collect();pieces=[]
    for b in source.batches:
        if b['scope']!='archived_subpiece_grids':continue
        key=ast.literal_eval(b['case']);r=source.piece_map[key]
        pieces.append(dict(scope='geometric',case=b['case'],key=key,lo=float(r['lower']),hi=float(r['upper']),
            coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots'])))
    for key,r in source.generic.items():pieces.append(dict(scope='generic_coefficients',case=key,record=r,lo=r['lower'],hi=r['upper'],
        coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),roots=r['roots']))
    del source.batches
    endpoint_x=[]
    for p in pieces:
        A,B,a,beta,ot,ov=p['coeff']
        endpoint_x.extend([[A,B,a,beta,p['lo'],ot,ov],[A,B,a,beta,p['hi'],ot,ov]])
    x=np.array(endpoint_x)
    with torch.no_grad():before=old(torch.from_numpy(x)).numpy().reshape(-1,2);after=new(torch.from_numpy(x)).numpy().reshape(-1,2)
    exact,_=mathematical_sign(x);exact=exact.reshape(-1,2)
    endpoint_rows=[]
    for p,b,a,t in zip(pieces,before,after,exact):
        old_bracket=b[0]*b[1]<0;new_bracket=a[0]*a[1]<0
        endpoint_rows.append(dict(scope=p['scope'],case=p['case'],old_left=int(b[0]),old_right=int(b[1]),new_left=int(a[0]),new_right=int(a[1]),
            teacher_left=int(t[0]),teacher_right=int(t[1]),changed=bool(np.any(b!=a)),teacher_disagreement=bool(np.any(a!=t)),
            missed_bracket=bool(old_bracket and not new_bracket),added_bracket=bool(new_bracket and not old_bracket)))
    write_csv(OUT/'refined_sqrt_endpoint_audit.csv',endpoint_rows)
    baseline=list(csv.DictReader((FAIL/'stationary_midpoint_recovery.csv').open()))
    baseline_map={(r['scope'],r['case']):r for r in baseline}
    chosen=[p for p in pieces if (p['scope'],p['case']) in baseline_map];n=len(chosen)
    selector=BracketSelector();selector.load_state_dict(torch.load(FAIL/'bracket_selector_before_evaluation.pt',weights_only=False)['state_dict']);selector.eval().requires_grad_(False)
    L=np.array([p['lo'] for p in chosen]);R=np.array([p['hi'] for p in chosen]);coeff=np.array([p['coeff'] for p in chosen])
    def inputs(ids,c):
        v=coeff[ids];return np.column_stack((v[:,0],v[:,1],v[:,2],v[:,3],c,v[:,4],v[:,5]))
    with torch.no_grad():sl=new(torch.from_numpy(inputs(np.arange(n),L))).numpy();sr=new(torch.from_numpy(inputs(np.arange(n),R))).numpy()
    if not np.all(sl*sr<0):return dict(passed=False,endpoint_failure=True),source.inputs
    active=np.ones(n,bool);candidate=np.full(n,np.nan);reason=np.full(n,'',object);steps=np.zeros(n,int)
    direction=np.zeros(n,int);false_zero=np.zeros(n,bool);queries=0;selector_errors=0;changed_queries=0;sqrt_mismatch=0
    for iteration in range(256):
        ids=np.flatnonzero(active)
        if not len(ids):break
        q=(L[ids]+R[ids])/2;adjacent=(np.nextafter(L[ids],np.inf)>=R[ids])|(q<=L[ids])|(q>=R[ids])
        done=ids[adjacent];candidate[done]=q[adjacent];reason[done]='adjacent_float';active[done]=False
        ids=ids[~adjacent];q=q[~adjacent]
        if not len(ids):continue
        x=inputs(ids,q);tx=torch.from_numpy(x)
        with torch.no_grad():
            sq=new(tx).numpy();prior=old(tx).numpy()
            m=new.components;A,B,a,beta,c,ot,ov=tx.unbind(1);t=m['orientation'](torch.stack((c,ot),1))
            qq=m['quadratic'](torch.stack((B,a,t),1));rr=m['sqrt'](qq).numpy()
        sqrt_mismatch+=int(np.sum(rr!=np.sqrt(qq.numpy())))
        exact,_=mathematical_sign(x);queries+=len(ids);changed_queries+=int(np.sum(sq!=prior));steps[ids]+=1
        zero=sq==0;done=ids[zero];candidate[done]=q[zero];reason[done]='learned_ZERO';active[done]=False;false_zero[done]=exact[zero]!=0
        live=ids[~zero];trip=np.column_stack((sl[live],sq[~zero],sr[live]))
        with torch.no_grad():decision=np.sign(selector(torch.from_numpy(trip)).numpy())
        selector_errors+=int(np.sum(decision!=np.where(trip[:,1]==trip[:,0],1.,-1.)))
        direction[live]+=(sq[~zero]!=exact[~zero])&(exact[~zero]!=0)
        left=decision>0;L[live[left]]=q[~zero][left];sl[live[left]]=sq[~zero][left]
        R[live[~left]]=q[~zero][~left];sr[live[~left]]=sq[~zero][~left]
    if active.any():candidate[active]=(L[active]+R[active])/2;reason[active]='iteration_limit'
    audit=Audit.__new__(Audit);rows=[];solved={};mp.mp.dps=60
    for i,p in enumerate(chosen):
        br=baseline_map[(p['scope'],p['case'])];ct=float(br['c_teacher']);cl=float(candidate[i]);lo,hi=p['lo'],p['hi']
        if p['scope']=='geometric':
            parent,axis,k,_=p['key'];audit.setup(source.parents[parent],axis);score=audit.formula(np.array([lo,ct,hi,cl]),k)[0]
        else:axis=0;score=generic_score(p['record'],np.array([lo,ct,hi,cl]))
        def candidates(root,value):return [dict(axis=axis,c=c,score=float(s),identity=p['case']+':'+kind,kind=kind) for c,s,kind in [(lo,score[0],'left'),(root,value,'stationary'),(hi,score[2],'right')]]
        tc,lc=candidates(ct,score[1]),candidates(cl,score[3]);bt,bl=canonical(tc),canonical(lc)
        row=dict(scope=p['scope'],case=p['case'],c_teacher=ct,c_teacher_decimal=br['c_teacher_decimal'],old_candidate=float(br['c_learned']),new_candidate=cl,
            candidate_changed=cl!=float(br['c_learned']),termination=reason[i],iterations=int(steps[i]),false_zero=bool(false_zero[i]),direction_mistakes=int(direction[i]),
            root_displacement=float(abs(mp.mpf(cl)-mp.mpf(br['c_teacher_decimal']))),teacher_score=float(score[1]),candidate_teacher_score=float(score[3]),
            score_difference=float(abs(score[3]-score[1])),score_outside_equivalence=bool(abs(score[3]-score[1])>1e-12*max(1.,score[1])),
            minimum_identity_changed=bt['kind']!=bl['kind'],minimum_outside_equivalence=bool(abs(bt['score']-bl['score'])>1e-12*max(1.,min(x['score'] for x in tc))))
        rows.append(row);solved[(p['scope'],p['case'])]=(tc,lc)
    pools={};parents={};comparisons=[]
    for p in pieces:
        if p['scope']!='geometric':continue
        parent,axis,k,_=p['key'];pair=solved.get((p['scope'],p['case']))
        if pair is None:
            audit.setup(source.parents[parent],axis);cuts=sorted(set([p['lo'],p['hi']]+[c for c in p['roots'] if p['lo']<=c<=p['hi']]))
            ss=audit.formula(np.array(cuts),k)[0];cs=[dict(axis=axis,c=c,score=float(s),identity=p['case']+f':saved{j}') for j,(c,s) in enumerate(zip(cuts,ss))];pair=(cs,cs)
        for arm,cs in zip(('teacher','learned'),pair):pools.setdefault((parent,axis,k),{'teacher':[],'learned':[]})[arm].extend(cs)
    for key,pool in pools.items():
        comparisons.append(compare_pool('allocation_interval',str(key),pool))
        for arm in pool:parents.setdefault(key[0],{'teacher':[],'learned':[]})[arm].extend(pool[arm])
    for parent,pool in parents.items():comparisons.append(compare_pool('archived_parent',str(parent),pool))
    write_csv(OUT/'refined_sqrt_midpoint_audit.csv',rows);write_csv(OUT/'refined_sqrt_canonical_comparison.csv',comparisons)
    result=dict(endpoint_subpieces=len(endpoint_rows),endpoint_changes=sum(r['changed'] for r in endpoint_rows),missed_brackets=sum(r['missed_bracket'] for r in endpoint_rows),
        added_brackets=sum(r['added_bracket'] for r in endpoint_rows),endpoint_teacher_disagreements=sum(r['teacher_disagreement'] for r in endpoint_rows),
        midpoint_subpieces=n,midpoint_queries=queries,query_sign_changes_vs_original=changed_queries,
        candidate_changes_vs_original=sum(r['candidate_changed'] for r in rows),remaining_direction_disagreements=int(direction.sum()),
        learned_ZERO_stops=int(np.sum(reason=='learned_ZERO')),false_ZERO_stops=int(false_zero.sum()),adjacent_float_stops=int(np.sum(reason=='adjacent_float')),
        iteration_limit_failures=int(np.sum(reason=='iteration_limit')),sqrt_output_mismatches_against_direct_sqrt=sqrt_mismatch,selector_errors=selector_errors,
        max_score_difference=max(r['score_difference'] for r in rows),score_differences_outside_equivalence=sum(r['score_outside_equivalence'] for r in rows),
        subpiece_minimum_identity_changes=sum(r['minimum_identity_changed'] for r in rows),subpiece_minimum_changes_outside_equivalence=sum(r['minimum_outside_equivalence'] for r in rows),
        canonical_parent_best_cut_changes=sum(r['raw_cut_changed'] for r in comparisons if r['scope']=='archived_parent'),archived_parents=len(parents),
        reference='Previously saved high-precision roots; frozen existing bracket selector and deterministic midpoint. No root model or selector training.')
    result['passed']=all(result[k]==0 for k in ('endpoint_changes','missed_brackets','added_brackets','endpoint_teacher_disagreements','iteration_limit_failures','selector_errors',
        'score_differences_outside_equivalence','subpiece_minimum_identity_changes','subpiece_minimum_changes_outside_equivalence','canonical_parent_best_cut_changes'))
    return result,source.inputs


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    for p in ROOT.rglob('*manifest.json'):protected[str(p)]=sha(p)
    original=json.loads((SOURCE/'candidate_manifest.json').read_text());cp=Path(original['checkpoint']);assert sha(cp)==original['sha256']
    old=ScalarPower();old.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);old.eval().requires_grad_(False)
    model=ScalarPower();model.load_state_dict(old.state_dict());initial=model.weights.detach().numpy().copy()
    failures=np.load(FAIL/'sqrt_failure_inputs_targets.npz');fq=failures['inputs'];fy=failures['targets'];uq=failures['unique_inputs']
    assert len(fq)==7613 and len(uq)==5678 and np.array_equal(fy,np.sqrt(fq))
    data=np.load(SOURCE/'training_data.npz');held=np.load(SOURCE/'heldout_data.npz')
    rng=np.random.default_rng(SEED);exponent=rng.uniform(-300,300,8192);keep=np.ones(len(exponent),bool)
    for lo,hi in RESERVED:keep&=~((exponent>=lo)&(exponent<=hi))
    retention=np.r_[data['q'],10.**exponent[keep]]
    adjacent=np.r_[np.nextafter(uq,0.),np.nextafter(uq,np.inf)]
    groups=[('saved_failures',fq,fy,.5),('adjacent_failure_floats',adjacent,np.sqrt(adjacent),.25),('broad_retention',retention,np.sqrt(retention),.25)]
    q=np.concatenate([g[1] for g in groups]);target=np.concatenate([g[2] for g in groups]);weight=np.concatenate([np.full(len(g[1]),np.sqrt(g[3]/len(g[1]))) for g in groups])
    logq=np.log(q);history=[];best=(np.inf,initial.copy())
    for update in range(13):
        pred=predict(model,q);res=(pred-target)/target;loss=float(np.sum((weight*res)**2));w=model.weights.detach().numpy().copy()
        history.append(dict(update=update,p=w[0],b=w[1],gain=float(np.exp(w[1])),weighted_relative_rmse=np.sqrt(loss),max_relative_error=float(abs(res).max()),exact_matches=int(np.sum(pred==target))))
        if loss<best[0]:best=(loss,w.copy())
        if loss==0 or update==12:break
        jac=(pred/target)[:,None]*np.column_stack((logq,np.ones(len(q))))
        delta=np.linalg.lstsq(jac*weight[:,None],-res*weight,rcond=None)[0]
        accepted=False
        for step in (1.,.5,.25):
            proposed=w+step*delta
            if proposed[0]<=0 or np.array_equal(proposed,w):continue
            with torch.no_grad():model.weights.copy_(torch.from_numpy(proposed))
            error=(predict(model,q)-target)/target;newloss=float(np.sum((weight*error)**2))
            if newloss<loss:accepted=True;break
        if not accepted:
            with torch.no_grad():model.weights.copy_(torch.from_numpy(w))
            break
    with torch.no_grad():model.weights.copy_(torch.from_numpy(best[1]))
    config=dict(seed=SEED,architecture='UNCHANGED ScalarPower: exp(b)*pow(q,p)',initial_weights=initial.tolist(),
        optimizer='Supervised relative-output least-squares residual refinement; analytic Jacobian, training-loss backtracking, no parameter snapping',
        objective_group_weights={g[0]:g[3] for g in groups},training_group_counts={g[0]:len(g[1]) for g in groups},
        retention='All 4096 original training inputs plus fresh broad [-300,300] log10 samples excluding original held-out magnitude bands.',
        original_validation_used_for_training=False,neighbor_training='Immediate nextafter on either side of unique failure inputs',
        original_checkpoint=str(cp),original_sha256=sha(cp),no_other_training=True)
    path=OUT/'sqrt_refined_before_evaluation.pt';digest=save_checkpoint(path,dict(state_dict=model.state_dict(),config=config,training_history=history,status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'training_history.csv',history);(OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(event='saved_before_evaluation',parameters=best[1].tolist(),sha256=digest,updates=len(history)-1)),flush=True)
    model.load_state_dict(torch.load(path,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    bits=uq.view(np.uint64)[:,None]+np.arange(-8,9,dtype=np.int64).astype(np.uint64)[None,:]
    nearby=np.unique(bits.ravel().view(np.float64));nearby=nearby[(nearby>0)&np.isfinite(nearby)]
    evalsets=[('saved_7613_occurrences',fq),('unique_5678_failures',uq),('nextafter_neighborhoods',nearby),('original_training',data['q']),('original_validation',held['q'])]
    for tag in sorted(set(held['group'])):evalsets.append(('original_validation_'+str(tag),held['q'][held['group']==tag]))
    evalsets.append(('fresh_broad_heldout',10.**rng.uniform(-300,300,8192)))
    rows=[]
    for name,v in evalsets:
        truth=np.sqrt(v)
        for arm,m in [('before',old),('after',model)]:rows.append(dict(group=name,arm=arm,**metrics(truth,predict(m,v))))
    ordered=np.unique(np.r_[held['q'],nearby,np.logspace(-300,300,20000)])
    monotonic={arm:int(np.sum(np.diff(predict(m,ordered))<0)) for arm,m in [('before',old),('after',model)]}
    prior=predict(old,held['q']);now=predict(model,held['q']);truth=held['target']
    regression=dict(original_validation_cases=len(truth),absolute_error_worse=int(np.sum(abs(now-truth)>abs(prior-truth))),
        previously_exact_became_inexact=int(np.sum((prior==truth)&(now!=truth))),monotonicity_reversals=monotonic)
    scaling=np.load(SOURCE/'scaling_data.npz');scale_rows=[]
    for arm,m in [('before',old),('after',model)]:scale_rows.append(dict(arm=arm,**metrics(abs(scaling['k'])*predict(m,scaling['q']),predict(m,scaling['scaled_q']))))
    write_csv(OUT/'sqrt_before_after.csv',rows);write_csv(OUT/'sqrt_scaling_retention.csv',scale_rows)
    (OUT/'global_regression.json').write_text(json.dumps(regression,indent=2))
    oldrouter,newrouter=routers(model);replayed=replay(oldrouter,newrouter)
    print(json.dumps(dict(event='saved_failure_replay_complete',**replayed)),flush=True)
    # A remaining mismatch to direct teacher sqrt is a stop, not permission to tune again.
    failure_exact=np.array_equal(predict(model,uq),np.sqrt(uq))
    broad_after=next(r for r in rows if r['group']=='original_validation' and r['arm']=='after')
    broad_before=next(r for r in rows if r['group']=='original_validation' and r['arm']=='before')
    global_ok=bool(broad_after['max_relative_error']<=broad_before['max_relative_error'] and monotonic['after']==0 and broad_after['nonfinite']==0)
    if global_ok and failure_exact and replayed['total_remaining_sqrt_attributable']==0:
        bracket,archive_hashes=archived_audit(oldrouter,newrouter)
    else:bracket=dict(passed=False,skipped='Primary sqrt/replay gate failed; no further audit.');archive_hashes={}
    passed=bool(global_ok and failure_exact and replayed['total_remaining_sqrt_attributable']==0 and bracket['passed'])
    assert all(sha(Path(p))==h for p,h in {**protected,**archive_hashes}.items())
    result=dict(operator='sqrt_operator_refined',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        before_parameters=dict(p=float(initial[0]),b=float(initial[1]),gain=float(np.exp(initial[1]))),
        after_parameters=dict(p=float(model.weights[0]),b=float(model.weights[1]),gain=float(torch.exp(model.weights[1]))),
        checkpoint=str(path),sha256=digest,original_checkpoint=str(cp),original_sha256=sha(cp),
        module='experiments.geometry.train_sqrt_operator',model='ScalarPower',architecture_sha256=sha(Path('experiments/geometry/train_sqrt_operator.py')),
        refinement_implementation_sha256=sha(Path(__file__)),primary_metrics=rows,regression=regression,replay=replayed,bracket_audit=bracket,
        all_prior_hashes_unchanged=True,original_checkpoint_not_overwritten=True,elapsed_seconds=time.monotonic()-start)
    result=json.loads(json.dumps(result,default=lambda x:x.item()))
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k!='primary_metrics'},indent=2))
    (OUT/'sqrt_refinement_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('primary_metrics',)},indent=2),flush=True)


if __name__=='__main__':run()
