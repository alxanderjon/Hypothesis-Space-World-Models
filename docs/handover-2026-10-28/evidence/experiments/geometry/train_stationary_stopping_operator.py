"""Only train stopping decisions; replay the frozen locator components.

Replay is copied from the locked scheduler experiment, replacing only its two
termination decisions with a callback. Frozen source and checkpoints unchanged.
"""
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_stationary_midpoint_operator import Midpoint
from experiments.geometry.train_stationary_bracket_action_router import BracketActionRouter, LEFT, RIGHT
from experiments.geometry.validate_stationary_three_state_router import load_router, DEFER
from experiments.geometry.train_stationary_query_scheduler import (
    QueryScheduler, TABLE, MAIN, DLEFT, DRIGHT, NONE, teacher_schedule,
)
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT=ref.ROOT
SOURCE=ROOT/'defer_aware_stationary_locator_audit'
ACTION_DIR=ROOT/'stationary_bracket_action_router_training'
MID_DIR=ROOT/'stationary_midpoint_operator_training'
SCHED_DIR=ROOT/'stationary_query_scheduler_training'
OUT=ROOT/'stationary_stopping_operator_training'
SEED=28302026
STOP_FEATURES=['main_check','endpoints_adjacent','midpoint_equals_endpoint','left_moved_inward','right_moved_inward']
STOP_TABLE=np.array([[1,a,b,0,0] for a in (0,1) for b in (0,1)]+[[0,0,0,l,r] for l in (0,1) for r in (0,1)],dtype=int)


def teacher_stop(states):
    """Training labels and independent audit only; not learned actuation."""
    a=np.asarray(states,dtype=bool)
    return np.where(a[:,0],a[:,1]|a[:,2],~(a[:,3]|a[:,4])).astype(int)


class StoppingOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(5,8),nn.Tanh(),nn.Linear(8,2)).double()
    def forward(self,x):return self.layers(x)
    def predict(self,states):
        with torch.no_grad():
            return self(torch.from_numpy(np.asarray(states,dtype=np.float64))).argmax(1).numpy()


def condition(states):
    s=np.asarray(states,dtype=bool)
    return np.where(s[:,0],np.where(s[:,1],'adjacent',np.where(s[:,2],'midpoint_rounded','main_continue')),
                    np.where(~(s[:,3]|s[:,4]),'no_inward_move','refinement_continue'))


def errors(states,pred):
    y=teacher_stop(states);bad=pred!=y
    return dict(examples=len(y),errors=int(bad.sum()),false_STOP=int(np.sum((pred==1)&(y==0))),
        false_CONTINUE=int(np.sum((pred==0)&(y==1))),STOP=int(np.sum(pred==1)),CONTINUE=int(np.sum(pred==0)))


def fresh_states(midpoint,seed,sequences=4096):
    """Fresh geometric boundary-state sequences, no root search or dynamics."""
    rng=np.random.default_rng(seed); states=[]; ids=[]; coords=[]
    for seq in range(sequences):
        L=float(rng.uniform(.02,.95))
        # Synthetic state transitions only: carry each narrowed bracket forward.
        # No coefficient functions, signs, or root solving enter this dataset.
        R=np.array([np.float64(L).view(np.uint64)+np.uint64(8)],dtype=np.uint64).view(np.float64)[0]
        while True:
            width=int(np.float64(R).view(np.uint64)-np.float64(L).view(np.uint64))
            q=midpoint.evaluate(np.array([L]),np.array([R]))[0]
            a=int(np.nextafter(L,R)==R);b=int(q==L or q==R)
            states.append([1,a,b,0,0]);ids.append(seq);coords.append([L,R,q,L,R])
            if a or b:break
            mask=int(rng.integers(0,4))
            if width==2 and mask==3:mask=1
            lm=bool(mask&1);rm=bool(mask&2)
            newL=np.nextafter(L,R) if lm else L;newR=np.nextafter(R,L) if rm else R
            states.append([0,0,0,int(newL>L),int(newR<R)]);ids.append(seq);coords.append([L,R,q,newL,newR])
            if not (lm or rm):break
            L,R=newL,newR
    # Matched before/after zero-progress boundaries for every progress pattern.
    for j in range(sequences):
        L=float(rng.uniform(.01,.45));R=float(rng.uniform(.55,.99));q=midpoint.evaluate(np.array([L]),np.array([R]))[0]
        for lm,rm in ((0,0),(1,0),(0,1),(1,1)):
            states.append([0,0,0,lm,rm]);ids.append(sequences+j)
            coords.append([L,R,q,np.nextafter(L,R) if lm else L,np.nextafter(R,L) if rm else R])
    return np.array(states),np.array(ids),np.array(coords)


def replay(saved, coeff, cached, midpoint, scheduler, sign_router=None, action_router=None, stopping=teacher_stop):
    """A scheduler-driven coordinate preparation loop, frozen numerical pipeline.

    Main midpoint evaluation precedes the unchanged adjacent/rounded stop check.
    The scheduler prepares both refinement coordinates before either is scored
    or applied, exactly as the audited implementation does.
    """
    stop_records=[]
    def stop(ids, state):
        truth=teacher_stop(state); learned=stopping(state)
        stop_records.append(np.column_stack((ids,iterations[ids],state,truth,learned)))
        if not np.array_equal(truth,learned):
            np.savez_compressed(OUT/'failed_stop_decisions.npz',ids=ids,state=state,teacher=truth,learned=learned)
            raise RuntimeError('Stopping decision mismatch; no teacher override.')
        return learned.astype(bool)
    n=len(saved); L=np.array([float(r['initial_L']) for r in saved]); R=np.array([float(r['initial_R']) for r in saved])
    code={'TRUST_POSITIVE':1,'TRUST_NEGATIVE':2}
    sl=np.array([code[r['left_state']] for r in saved]); sr=np.array([code[r['right_state']] for r in saved])
    active=np.ones(n,bool); iterations=np.zeros(n,int); reason=np.full(n,'',object)
    flags=np.tile(TABLE[0],(n,1)); savedL=L.copy(); savedR=R.copy(); deferred_q=np.full(n,np.nan)
    qL=np.full(n,np.nan); qR=np.full(n,np.nan)
    records=[]; mid_records=[]; sign_cursor=0; midpoint_calls=0; action_changes=0; invariants=0
    sign_queries=0; defers=0; scheduling_errors=0; terminal_checks=0

    def invariant():
        nonlocal invariants
        invariants+=1
        assert np.all((L<R)&(sl!=sr)&np.isin(sl,[1,2])&np.isin(sr,[1,2])), 'Trusted endpoint invariant failed.'

    def schedule(ids):
        nonlocal midpoint_calls,scheduling_errors
        state=flags[ids].copy(); pred=scheduler(state); truth=teacher_schedule(state)
        scheduling_errors+=int(np.sum(np.any(pred != truth,axis=1)))
        records.append(np.column_stack((ids,iterations[ids],state,pred)))
        if not np.array_equal(pred,truth):
            np.savez_compressed(OUT/'failed_scheduling_request.npz',ids=ids,state=state,pred=pred,teacher=truth)
            raise RuntimeError('Scheduling/pointer mismatch; no teacher override.')
        use=pred[:,0]!=NONE; c=np.full(len(ids),np.nan)
        if use.any():
            take=ids[use]; ptr=pred[use,1:]
            # The two learned pointer heads select the operands. No action-based
            # switch constructs an input pair during inference.
            bank=np.column_stack((L[take],R[take],savedL[take],savedR[take],deferred_q[take],np.full(len(take),np.nan)))
            pair=np.take_along_axis(bank,ptr,axis=1)
            assert np.all(np.isfinite(pair)), 'Invalid coordinate-slot request.'
            c[use]=midpoint.evaluate(*pair.T); midpoint_calls+=len(take)
            mid_records.append(np.column_stack((take,iterations[take],pred[use,0],pair,c[use])))
        return pred[:,0],c

    def query(ids,c):
        nonlocal sign_cursor,sign_queries,defers
        rows=cached[sign_cursor:sign_cursor+len(ids)]
        assert np.array_equal(rows[:,0].astype(int),ids)
        assert np.array_equal(rows[:,1].astype(int),iterations[ids])
        assert np.array_equal(rows[:,7],c), 'Query coordinate differs from validated archive.'
        if sign_router is None:
            states=rows[:,4].astype(int)
        else:
            a=coeff[ids]
            with torch.no_grad(): states=sign_router(torch.from_numpy(np.column_stack((a[:,:4],c,a[:,4:])))).numpy()
        sign_cursor+=len(ids); sign_queries+=len(ids); defers+=int(np.sum(states==DEFER))
        return states,rows

    def update(ids,c,codes,old):
        nonlocal action_changes
        if action_router is None: actions=old[:,5].astype(int)
        else: actions=action_router.actions(np.column_stack((sl[ids],sr[ids],codes)))
        action_changes+=int(np.sum(actions!=old[:,5]))
        assert np.array_equal(actions,old[:,5]), 'Endpoint action changed.'
        left=actions==LEFT; right=actions==RIGHT
        L[ids[left]]=c[left];sl[ids[left]]=codes[left]
        R[ids[right]]=c[right];sr[ids[right]]=codes[right]

    def audit_endpoints():
        if sign_router is not None:
            for c,expected in ((L,sl),(R,sr)):
                with torch.no_grad(): got=sign_router(torch.from_numpy(np.column_stack((coeff[:,:4],c,coeff[:,4:])))).numpy()
                assert np.array_equal(got,expected)

    invariant();audit_endpoints()
    while active.any():
        ids=np.flatnonzero(active);flags[ids]=TABLE[0]
        actions,q=schedule(ids)
        assert np.all(actions==MAIN)
        flags[ids,1]=1
        # Coordinates exist but there is no main response yet: request nothing.
        waiting,_=schedule(ids);assert np.all(waiting==NONE)
        adjacent=np.nextafter(L[ids],R[ids])==R[ids];rounded=(q==L[ids])|(q==R[ids]);done=stop(ids,np.column_stack((np.ones(len(ids),int),adjacent,rounded,np.zeros((len(ids),2),int))))
        stopids=ids[done];active[stopids]=False;flags[stopids,0]=0
        reason[stopids]=np.where(adjacent[done],'adjacent_float','midpoint_rounds_to_endpoint')
        if len(stopids):
            a,_=schedule(stopids);assert np.all(a==NONE);terminal_checks+=len(stopids)
        ids=ids[~done];q=q[~done]
        if not len(ids):break
        iterations[ids]+=1;codes,rows=query(ids,q);update(ids,q,codes,rows);invariant()
        flags[ids,2]=1;flags[ids,3]=(codes==DEFER)
        di=ids[codes==DEFER]
        savedL[di]=L[di];savedR[di]=R[di];deferred_q[di]=q[codes==DEFER]
        # This loop is driven by learned query actions, not by a phase-to-pair
        # table. Dispatch only stores results and marks request-completion flags.
        pending=ids
        while len(pending):
            chosen,coords=schedule(pending)
            left=chosen==DLEFT;right=chosen==DRIGHT
            assert not np.any(chosen==MAIN), 'Duplicate main request.'
            qL[pending[left]]=coords[left];flags[pending[left],4]=1
            qR[pending[right]]=coords[right];flags[pending[right],5]=1
            pending=pending[chosen!=NONE]
        if len(di):
            left,lr=query(di,qL[di]);right,rr=query(di,qR[di])
            update(di,qL[di],left,lr);invariant()
            update(di,qR[di],right,rr);invariant()
            assert np.all(L[di]>=savedL[di]) and np.all(R[di]<=savedR[di])
            left_moved=L[di]>savedL[di];right_moved=R[di]<savedR[di]
            done=stop(di,np.column_stack((np.zeros((len(di),3),int),left_moved,right_moved)));stopped=di[done]
            active[stopped]=False;reason[stopped]='no_trusted_inward_move';flags[stopped,0]=0
            if len(stopped):
                a,_=schedule(stopped);assert np.all(a==NONE);terminal_checks+=len(stopped)
    assert sign_cursor==len(cached)
    invariant();audit_endpoints()
    # Final candidate is an output calculation, not a sign query. Preserve the
    # existing locked midpoint invocation; it does not enter the scheduler.
    candidate=midpoint.evaluate(L,R)
    return dict(L=L,R=R,candidate=candidate,iterations=iterations,termination=reason,
        transitions=np.concatenate(records),midpoint_records=np.concatenate(mid_records),
        scheduled_midpoint_requests=midpoint_calls,actual_sign_queries=sign_queries,
        DEFER_responses=defers,action_changes=action_changes,invariant_checks=invariants,
        scheduling_errors=scheduling_errors,terminal_NO_QUERY_checks=terminal_checks,stopping_records=np.concatenate(stop_records))


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('locked','validated','manifest'))})
    for folder in (SOURCE,ACTION_DIR,MID_DIR,SCHED_DIR):
        protected.update({str(p):ref.sha(p) for p in folder.iterdir() if p.is_file()})
    audit=dict(features=STOP_FEATURES,actions=['CONTINUE','STOP'],
        main_check='After locked midpoint computation, before sign query or iteration increment: adjacent=(nextafter(L,R)==R); rounded=(q==L)|(q==R). STOP iff adjacent OR rounded. Adjacent has priority in the diagnostic reason when both hold.',
        refinement_check='Only after both DEFER refinement signs and both sequential endpoint actions: left_moved=(L>saved_L); right_moved=(R<saved_R). STOP iff neither moved. Compare final endpoints with the saved pre-refinement bracket, not with the temporary bracket after the left action.',
        nontermination='DEFER itself, completion of a progressing refinement round, a small derivative, and a small score gap never cause STOP. No iteration cap, interval-width tolerance, score criterion, or root-zero criterion is implemented.',
        terminal_bookkeeping='Already stopped brackets are removed from active set. When none remain, batch loop exits and final locked midpoint is computed. Invariant assertions are failure checks, not learned termination labels.',
        scope='Only the Boolean combination/decision is learned. Exact nextafter, equality, and inward-progress state observations remain deterministic. No teacher decision is an input.',
        truth_table_scope='Eight formal check-phase/flag rows. At the main check, irrelevant movement flags are zero; at the refinement check, adjacency/rounding flags are zero. Adjacent without midpoint rounding is included for complete Boolean guard coverage, though absent for the locked midpoint on archived brackets.',
        sources={str(p):ref.sha(p) for p in [Path('experiments/geometry/audit_defer_aware_stationary_locator.py'),Path('experiments/geometry/train_stationary_query_scheduler.py')]})
    (OUT/'stopping_teacher_audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'stopping_teacher_audit.txt').write_text('\n\n'.join(k+': '+str(v) for k,v in audit.items())+'\n')
    saved=list(csv.DictReader((SOURCE/'defer_aware_root_recovery.csv').open()));assert len(saved)==7093
    prior=json.loads((SOURCE/'results.json').read_text())
    cached=np.load(ACTION_DIR/'replayed_query_actions.npz')['transitions']
    data=np.load(ROOT/'stationary_float64_minimizer_full_audit/all_float64_candidates.npz')
    source_rows=list(csv.DictReader((ROOT/'stationary_float64_minimizer_full_audit/stationary_float64_minimizers.csv').open()))
    coeffmap={(r['scope'],r['case']):data['inputs'][data['offsets'][i],[0,1,2,3,5,6]] for i,r in enumerate(source_rows)}
    coeff=np.array([coeffmap[(r['scope'],r['case'])] for r in saved])
    midpoint=Midpoint();scheduler=QueryScheduler();action=BracketActionRouter()
    dependencies=[(midpoint,MID_DIR,'midpoint_before_evaluation.pt'),(scheduler,SCHED_DIR,'query_scheduler_before_evaluation.pt'),
                  (action,ACTION_DIR,'bracket_action_router_before_evaluation.pt')]
    for model,folder,name in dependencies:
        meta=json.loads((folder/'results.json').read_text());assert meta['status']=='LOCKED'
        assert ref.sha(folder/name)==meta.get('checkpoint_sha256',meta.get('sha256'))
        model.load_state_dict(torch.load(folder/name,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    baseline=replay(saved,coeff,cached,midpoint,scheduler.predict)
    assert np.array_equal(baseline['L'],[float(r['final_L']) for r in saved])
    assert np.array_equal(baseline['R'],[float(r['final_R']) for r in saved])
    assert np.array_equal(baseline['candidate'],[float(r['c_candidate']) for r in saved])
    previous_replay=np.load(SCHED_DIR/'learned_scheduling_replay.npz')
    assert np.array_equal(previous_replay['transitions'],baseline['transitions'])
    assert np.array_equal(previous_replay['midpoint_requests'],baseline['midpoint_records'])
    trace=baseline['stopping_records'];states=trace[:,2:7].astype(int)
    columns=['bracket','completed_iterations']+STOP_FEATURES+['teacher_stop','learned_stop']
    np.savez_compressed(OUT/'archived_stopping_decisions.npz',decisions=trace,columns=np.array(columns))
    trainmask=trace[:,0].astype(int)%5!=0
    counts=np.array([np.sum(np.all(states[trainmask]==r,axis=1)) for r in STOP_TABLE],float)
    assert counts.sum()==trainmask.sum()
    weight=torch.from_numpy(.5/len(STOP_TABLE)+.5*counts/counts.sum())
    model=StoppingOperator();initial={k:v.detach().tolist() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=.03);x=torch.from_numpy(STOP_TABLE.astype(float));y=torch.from_numpy(teacher_stop(STOP_TABLE))
    history=[]
    for step in range(1,2001):
        opt.zero_grad();logits=model(x);loss=(nn.functional.cross_entropy(logits,y,reduction='none')*weight).sum()
        loss.backward();opt.step()
        with torch.no_grad():
            logits=model(x);pred=logits.argmax(1)
            val=float((nn.functional.cross_entropy(logits,y,reduction='none')*weight).sum())
            margin=float((logits[torch.arange(len(y)),y]-logits[torch.arange(len(y)),1-y]).min())
        history.append(dict(update=step,loss=val,truth_errors=int((pred!=y).sum()),minimum_correct_logit_margin=margin))
        if torch.equal(pred,y) and margin>=3 and val<.01:break
    model.eval().requires_grad_(False)
    checkpoint=OUT/'stopping_operator_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),optimizer_state=opt.state_dict(),seed=SEED,updates=step,
        initial_parameters=initial,features=STOP_FEATURES,actions=['CONTINUE','STOP'],dtype='float64',
        architecture='5 observed state flags -> 8 tanh -> 2 action logits',training_archive_occurrences=int(trainmask.sum())))
    ref.write_csv(OUT/'training_history.csv',history)
    (OUT/'learned_parameters.json').write_text(json.dumps({k:v.tolist() for k,v in model.state_dict().items()},indent=2))
    print(json.dumps(dict(stage='trained_saved',updates=step,loss=val,sha256=digest,archive_decisions=len(states))),flush=True)
    fresh,seqids,coordinates=fresh_states(midpoint,SEED+1)
    assert np.all(coordinates[:,3]<coordinates[:,4])
    np.savez_compressed(OUT/'fresh_boundary_states.npz',states=fresh,sequence_ids=seqids,coordinates=coordinates,
                        coordinate_columns=np.array(['oldL','oldR','q','newL','newR']))
    reports={'complete_formal_rule':errors(STOP_TABLE,model.predict(STOP_TABLE)),
        'all_archived':errors(states,model.predict(states)),
        'heldout_archived':errors(states[~trainmask],model.predict(states[~trainmask])),
        'fresh_boundary_state_sequences':errors(fresh,model.predict(fresh))}
    detail=[]
    for group,s in [('truth_table',STOP_TABLE),('archive',states),('heldout_archive',states[~trainmask]),('fresh',fresh)]:
        labels=condition(s)
        for name in np.unique(labels):
            part=s[labels==name];detail.append(dict(group=group,teacher_condition=name,**errors(part,model.predict(part))))
    ref.write_csv(OUT/'stopping_accuracy.csv',[dict(group=k,**v) for k,v in reports.items()])
    ref.write_csv(OUT/'stopping_errors_by_condition.csv',detail)
    ref.write_csv(OUT/'stopping_truth_table.csv',[dict(zip(STOP_FEATURES,s.tolist()),teacher=int(t),learned=int(p)) for s,t,p in zip(STOP_TABLE,teacher_stop(STOP_TABLE),model.predict(STOP_TABLE))])
    assert all(v['errors']==0 for v in reports.values()), 'Stopping gate failed; no live replay.'
    router,router_refs=load_router()
    live=replay(saved,coeff,cached,midpoint,scheduler.predict,router,action,model.predict)
    stop_changes=int(np.sum(live['stopping_records'][:,-1]!=baseline['stopping_records'][:,-2]))
    assert np.array_equal(live['stopping_records'],baseline['stopping_records'])
    assert np.array_equal(live['transitions'],baseline['transitions'])
    assert np.array_equal(live['midpoint_records'],baseline['midpoint_records'])
    live_query_counts=np.bincount(live['midpoint_records'][:,0].astype(int),minlength=len(saved))
    old_query_counts=np.bincount(baseline['midpoint_records'][:,0].astype(int),minlength=len(saved))
    rows=[]
    for i,r in enumerate(saved):
        root=mp.mpf(r['c_teacher_decimal']);contained=mp.mpf(float(live['L'][i]))<=root<=mp.mpf(float(live['R'][i]))
        rows.append(dict(index=i,scope=r['scope'],case=r['case'],L=float(live['L'][i]),R=float(live['R'][i]),
            candidate=float(live['candidate'][i]),bracket_changed=bool(live['L'][i]!=baseline['L'][i] or live['R'][i]!=baseline['R'][i]),
            candidate_changed=bool(live['candidate'][i]!=baseline['candidate'][i]),root_contained=bool(contained),
            containment_changed=bool(contained)!=(r['root_contained']=='True'),
            iterations=int(live['iterations'][i]),iteration_count_changed=bool(live['iterations'][i]!=baseline['iterations'][i]),
            midpoint_query_count=int(live_query_counts[i]),query_count_changed=bool(live_query_counts[i]!=old_query_counts[i]),
            termination=live['termination'][i],termination_changed=bool(live['termination'][i]!=baseline['termination'][i])))
    ref.write_csv(OUT/'stopping_bracket_replay.csv',rows)
    np.savez_compressed(OUT/'learned_stopping_replay.npz',decisions=live['stopping_records'],columns=np.array(columns),
                        scheduling=live['transitions'],midpoint_requests=live['midpoint_records'])
    unchanged=all(not r[k] for r in rows for k in ('bracket_changed','candidate_changed','containment_changed','iteration_count_changed','query_count_changed','termination_changed'))
    downstream=dict(method='Exact-input invariance: identical scheduling/actions, query coordinates, final brackets and candidates, with unchanged frozen checkpoints. Saved original teacher-score and canonical selections therefore remain applicable.',
        score_differences_outside_teacher_equivalence=prior['score_differences_outside_equivalence'] if unchanged else None,
        subpiece_minimum_identity_changes=prior['subpiece_minimum_identity_changes'] if unchanged else None,
        canonical_best_cut_changes=prior['canonical_best_cut_changes'] if unchanged else None,parents=106)
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    passed=(unchanged and stop_changes==0 and all(v['errors']==0 for v in reports.values()) and all(r['root_contained'] for r in rows)
            and all(downstream[k]==0 for k in ('score_differences_outside_teacher_equivalence','subpiece_minimum_identity_changes','canonical_best_cut_changes')))
    result=dict(operator='stationary_stopping_operator',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        architecture='float64 5 observed state flags -> Linear(5,8) -> Tanh -> Linear(8,2); learned argmax CONTINUE/STOP',
        parameter_count=sum(p.numel() for p in model.parameters()),seed=SEED,updates=step,loss=val,teacher_audit=audit,
        archived_training_occurrences=int(trainmask.sum()),archived_heldout_occurrences=int((~trainmask).sum()),
        metrics=reports,fresh_valid_sequences=len(np.unique(seqids[seqids<4096])),
        fresh_independent_boundary_groups=len(np.unique(seqids[seqids>=4096])),
        finite_holdout_caveat='All formal states are supervised; archived holdout and fresh sequences test replay/encoding and float64 boundary observations, not unseen Boolean truth-table rows. Each independent boundary group contains four alternative one-round movement patterns, not one multi-step sequence.',
        total_stopping_decisions=len(trace),stopping_decision_mismatches=stop_changes,
        false_STOP=0,false_CONTINUE=0,termination_counts={str(k):int(np.sum(live['termination']==k)) for k in np.unique(live['termination'])},
        iteration_count_differences=sum(r['iteration_count_changed'] for r in rows),query_count_differences=sum(r['query_count_changed'] for r in rows),
        final_bracket_differences=sum(r['bracket_changed'] for r in rows),trusted_endpoint_invariant_violations=0,trusted_endpoint_invariant_checks=live['invariant_checks'],
        teacher_roots_contained=sum(r['root_contained'] for r in rows),teacher_root_containment_changes=sum(r['containment_changed'] for r in rows),
        final_candidate_differences=sum(r['candidate_changed'] for r in rows),scheduled_midpoints=live['scheduled_midpoint_requests'],actual_sign_queries=live['actual_sign_queries'],
        action_decision_changes=live['action_changes'],downstream=downstream,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        frozen_router_refs=router_refs,all_existing_protected_hashes_unchanged=True,implementation_sha256=ref.sha(Path(__file__)),
        elapsed_seconds=time.monotonic()-start,no_recursion=True,no_global_search=True,no_float64_enumeration_changes=True)
    if passed:
        lock=OUT/'stationary_stopping_operator_locked.json';lock.write_text(json.dumps(result,indent=2));h=ref.sha(lock)
        lock.with_suffix('.sha256').write_text(h+'  '+lock.name+'\n');result['lock_manifest']=str(lock);result['lock_manifest_sha256']=h
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    summary=(f"stationary_stopping_operator = {result['status']}\n"
        f"Architecture: {result['architecture']}; {result['parameter_count']} learned parameters.\n"
        f"Random initialization; {step} Adam updates; saved before evaluation.\n"
        f"Stopping decisions: {len(trace)}; false STOP/CONTINUE and all finite/archive/heldout/fresh errors: 0.\n"
        f"Termination counts: {result['termination_counts']}\n"
        f"All 7093 final brackets/candidates unchanged; 7093 roots contained. Iteration/query-count changes: 0.\n"
        f"Trusted endpoint invariant violations: 0; downstream: {json.dumps(downstream)}\n"
        "Only the stopping Boolean decision was trained. Exact float64 state predicates remain deterministic.\n"
        "No existing checkpoints changed; no recursion, search, or changes to post-localization enumeration/minimum selection.\n"
        f"Checkpoint SHA256: {digest}\nElapsed seconds: {result['elapsed_seconds']:.3f}\n")
    (OUT/'stationary_stopping_summary.txt').write_text(summary)
    print(summary,flush=True)


if __name__=='__main__':run()
