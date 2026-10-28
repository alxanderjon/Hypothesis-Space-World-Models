"""Learn only scheduling and coordinate pointers; preserve locator termination."""
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
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT = ref.ROOT
SOURCE = ROOT / 'defer_aware_stationary_locator_audit'
ACTION_DIR = ROOT / 'stationary_bracket_action_router_training'
MID_DIR = ROOT / 'stationary_midpoint_operator_training'
OUT = ROOT / 'stationary_query_scheduler_training'
SEED = 28292026
MAIN, DLEFT, DRIGHT, NONE = range(4)
ACTION_NAMES = ['QUERY_MAIN', 'QUERY_DEFER_LEFT', 'QUERY_DEFER_RIGHT', 'NO_QUERY']
SLOTS = ['current_L', 'current_R', 'saved_L', 'saved_R', 'deferred_q', 'NONE']
FEATURES = ['active', 'main_prepared', 'main_response_ready', 'main_was_DEFER', 'left_prepared', 'right_prepared']
# These are state observations, not query-action inputs. In particular prepared
# means coordinate computed, not endpoint updated or even sign evaluated.
TABLE = np.array([[1,0,0,0,0,0], [1,1,0,0,0,0], [1,1,1,0,0,0],
                  [1,1,1,1,0,0], [1,1,1,1,1,0], [1,1,1,1,1,1],
                  [0,1,0,0,0,0], [0,1,1,1,1,1]], dtype=np.int64)
STATE_NAMES = ['round_ready', 'main_prepared_awaiting_response', 'trusted_main_completed',
               'deferred_main_ready', 'left_coordinate_prepared', 'both_refinement_coordinates_prepared',
               'terminal_adjacent_or_midpoint_rounded', 'terminal_no_trusted_inward_move']
# Supervision only; inference never uses these action-to-pointer mappings.
TARGETS = np.array([[MAIN,0,1], [NONE,5,5], [NONE,5,5], [DLEFT,2,4],
                    [DRIGHT,4,3], [NONE,5,5], [NONE,5,5], [NONE,5,5]])


def state_indices(states):
    match = np.all(np.asarray(states)[:,None,:] == TABLE[None,:,:], axis=2)
    assert np.all(match.sum(1) == 1), 'Invalid/unseen scheduling state.'
    return match.argmax(1)


def teacher_schedule(states):
    """Audited supervision and shadow validation only."""
    return TARGETS[state_indices(states)]


class QueryScheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6,12), nn.Tanh(), nn.Linear(12,16)).double()

    def forward(self, states):
        scores = self.layers(states)
        return scores[:,:4], scores[:,4:10], scores[:,10:16]

    def predict(self, states):
        with torch.no_grad():
            heads = self(torch.from_numpy(np.asarray(states, dtype=np.float64)))
            return torch.stack([h.argmax(1) for h in heads], 1).numpy()


def error_report(states, pred):
    truth = teacher_schedule(states)
    errors = np.any(pred != truth, axis=1)
    return dict(examples=len(states), all_output_errors=int(errors.sum()),
                action_errors=int(np.sum(pred[:,0] != truth[:,0])),
                coordinate_pointer_errors=int(np.sum(np.any(pred[:,1:] != truth[:,1:], axis=1))),
                invalid_query_requests=int(np.sum((truth[:,0] == NONE) & (pred[:,0] != NONE))),
                **{name+'_errors':int(np.sum(errors & (truth[:,0] == a))) for a,name in enumerate(ACTION_NAMES)})


def fresh_sequences(seed, count=2048):
    """Independent valid state sequences; no geometric root search."""
    rng = np.random.default_rng(seed); seq=[]; ids=[]
    for i in range(count):
        for turn in range(int(rng.integers(1,10))):
            inds=[0,1]
            if rng.random() < .12:
                inds.append(6); done=True
            elif rng.random() < .55:
                inds += [3,4,5]; done=rng.random()<.3
                if done: inds.append(7)
            else:
                inds.append(2); done=False
            seq.extend(TABLE[inds]); ids.extend([i]*len(inds))
            if done: break
        else:
            seq.extend(TABLE[[0,1,6]]); ids.extend([i]*3)
    return np.array(seq), np.array(ids)


def replay(saved, coeff, cached, midpoint, scheduler, sign_router=None, action_router=None):
    """A scheduler-driven coordinate preparation loop, frozen numerical pipeline.

    Main midpoint evaluation precedes the unchanged adjacent/rounded stop check.
    The scheduler prepares both refinement coordinates before either is scored
    or applied, exactly as the audited implementation does.
    """
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
        adjacent=np.nextafter(L[ids],R[ids])==R[ids];rounded=(q==L[ids])|(q==R[ids]);done=adjacent|rounded
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
            progressed=(L[di]>savedL[di])|(R[di]<savedR[di]);stopped=di[~progressed]
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
        scheduling_errors=scheduling_errors,terminal_NO_QUERY_checks=terminal_checks)


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('locked','validated','manifest'))})
    for folder in (SOURCE,ACTION_DIR,MID_DIR):
        protected.update({str(p):ref.sha(p) for p in folder.iterdir() if p.is_file()})
    audit=dict(features=FEATURES,coordinate_slots=SLOTS,actions=ACTION_NAMES,
        states=[dict(name=name,state=state.tolist(),teacher_action=ACTION_NAMES[int(target[0])],
                     operand_slots=[SLOTS[int(target[1])],SLOTS[int(target[2])]]) for name,state,target in zip(STATE_NAMES,TABLE,TARGETS)],
        ordering='Main midpoint computed before adjacent/rounded stopping check. If continuing, main sign evaluated and learned action applied. For DEFER, save L,R,q; prepare qL then qR before evaluating either sign or applying either refinement action. Evaluate left then right; apply left then right against current endpoints.',
        endpoint_update_effect='No endpoint update can affect the chosen refinement pairs: both operands use the saved pre-refinement bracket. Right endpoint ACTION uses current endpoints after the left action.',
        no_query_states='Awaiting main feedback; trusted main complete; both refinement coordinates prepared; terminal adjacent/rounded; terminal no inward progress.',
        terminal_decision='Supplied by unchanged deterministic stopping code; never learned here.',
        final_candidate='Locked midpoint(current_L,current_R) after termination is an output calculation, not a sign query. It remains unchanged.',
        learned_operand_mapping='Two learned categorical pointer heads, not a hard-coded action-to-coordinate-pair switch.',
        sources={str(p):ref.sha(p) for p in [Path('experiments/geometry/audit_defer_aware_stationary_locator.py'),Path('experiments/geometry/train_stationary_midpoint_operator.py')]})
    (OUT/'query_scheduling_teacher_audit.json').write_text(json.dumps(audit,indent=2))
    saved=list(csv.DictReader((SOURCE/'defer_aware_root_recovery.csv').open()));assert len(saved)==7093
    prior=json.loads((SOURCE/'results.json').read_text())
    cached=np.load(ACTION_DIR/'replayed_query_actions.npz')['transitions']
    data=np.load(ROOT/'stationary_float64_minimizer_full_audit/all_float64_candidates.npz')
    source_rows=list(csv.DictReader((ROOT/'stationary_float64_minimizer_full_audit/stationary_float64_minimizers.csv').open()))
    coeffmap={(r['scope'],r['case']):data['inputs'][data['offsets'][i],[0,1,2,3,5,6]] for i,r in enumerate(source_rows)}
    coeff=np.array([coeffmap[(r['scope'],r['case'])] for r in saved])
    midpoint=Midpoint();midpoint.load_state_dict(torch.load(MID_DIR/'midpoint_before_evaluation.pt',weights_only=False)['state_dict']);midpoint.eval().requires_grad_(False)
    for p in (MID_DIR/'results.json',ACTION_DIR/'results.json'):
        m=json.loads(p.read_text());assert m['status']=='LOCKED'
        assert ref.sha(Path(m['checkpoint']))==m.get('checkpoint_sha256',m.get('sha256'))
    baseline=replay(saved,coeff,cached,midpoint,teacher_schedule)
    assert np.array_equal(baseline['L'],[float(r['final_L']) for r in saved])
    assert np.array_equal(baseline['R'],[float(r['final_R']) for r in saved])
    assert np.array_equal(baseline['candidate'],[float(r['c_candidate']) for r in saved])
    trace=baseline['transitions'];states=trace[:,2:8].astype(int)
    np.savez_compressed(OUT/'archived_scheduling_transitions.npz',transitions=trace,
        columns=np.array(['bracket','iteration']+FEATURES+['action','operand_1','operand_2']))
    trainmask=trace[:,0].astype(int)%5!=0
    counts=np.bincount(state_indices(states[trainmask]),minlength=len(TABLE)).astype(float)
    weight=torch.from_numpy(.5/len(TABLE)+.5*counts/counts.sum())
    model=QueryScheduler();initial={k:v.detach().tolist() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=.03)
    x=torch.from_numpy(TABLE.astype(float));y=torch.from_numpy(TARGETS)
    history=[]
    for step in range(1,2001):
        opt.zero_grad();heads=model(x)
        loss=sum((nn.functional.cross_entropy(h,y[:,j],reduction='none')*weight).sum() for j,h in enumerate(heads))/3
        loss.backward();opt.step()
        with torch.no_grad():
            heads=model(x);pred=torch.stack([h.argmax(1) for h in heads],1)
            val=float(sum((nn.functional.cross_entropy(h,y[:,j],reduction='none')*weight).sum() for j,h in enumerate(heads))/3)
            margins=[]
            for j,h in enumerate(heads):
                target=h[torch.arange(len(TABLE)),y[:,j]];other=h.clone();other[torch.arange(len(TABLE)),y[:,j]]=-torch.inf
                margins.append(float((target-other.max(1).values).min()))
        history.append(dict(update=step,loss=val,truth_errors=int(torch.any(pred!=y,dim=1).sum()),minimum_logit_margin=min(margins)))
        if torch.equal(pred,y) and min(margins)>=3 and val<.02:break
    model.eval().requires_grad_(False)
    checkpoint=OUT/'query_scheduler_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),optimizer_state=opt.state_dict(),seed=SEED,updates=step,
        initial_parameters=initial,features=FEATURES,coordinate_slots=SLOTS,action_names=ACTION_NAMES,
        architecture='float64 6 flags -> 12 tanh -> heads: action(4), operand1(6), operand2(6)',
        training_archived_occurrences=int(trainmask.sum())))
    ref.write_csv(OUT/'training_history.csv',history)
    (OUT/'learned_parameters.json').write_text(json.dumps({k:v.tolist() for k,v in model.state_dict().items()},indent=2))
    print(json.dumps(dict(stage='trained_saved',updates=step,loss=val,sha256=digest,archive_states=len(states))),flush=True)
    fresh,fresh_ids=fresh_sequences(SEED+1)
    reports={ 'complete_finite_rule':error_report(TABLE,model.predict(TABLE)),
              'all_archived_scheduling':error_report(states,model.predict(states)),
              'heldout_archived_brackets':error_report(states[~trainmask],model.predict(states[~trainmask])),
              'fresh_valid_state_sequences':error_report(fresh,model.predict(fresh)) }
    np.savez_compressed(OUT/'fresh_valid_state_sequences.npz',states=fresh,sequence_ids=fresh_ids)
    ref.write_csv(OUT/'scheduler_accuracy.csv',[dict(group=k,**v) for k,v in reports.items()])
    ref.write_csv(OUT/'scheduling_truth_table.csv',[dict(name=name,**dict(zip(FEATURES,state.tolist())),
        teacher_action=ACTION_NAMES[t[0]],learned_action=ACTION_NAMES[p[0]],teacher_operand1=SLOTS[t[1]],learned_operand1=SLOTS[p[1]],
        teacher_operand2=SLOTS[t[2]],learned_operand2=SLOTS[p[2]]) for name,state,t,p in zip(STATE_NAMES,TABLE,TARGETS,model.predict(TABLE))])
    assert all(v['all_output_errors']==0 for v in reports.values()), 'Finite scheduling gate failed; stop before replay.'
    router,router_refs=load_router();action=BracketActionRouter()
    action.load_state_dict(torch.load(ACTION_DIR/'bracket_action_router_before_evaluation.pt',weights_only=False)['state_dict']);action.eval().requires_grad_(False)
    live=replay(saved,coeff,cached,midpoint,model.predict,router,action)
    assert np.array_equal(trace,live['transitions']), 'Scheduling transcript differs.'
    mr=live['midpoint_records'];old=baseline['midpoint_records']
    assert np.array_equal(mr[:,:2],old[:,:2])
    type_changes=int(np.sum(mr[:,2]!=old[:,2]));pair_changes=int(np.sum(np.any(mr[:,3:5]!=old[:,3:5],axis=1)))
    coordinate_changes=int(np.sum(mr[:,5]!=old[:,5]))
    replay_rows=[]
    for i,r in enumerate(saved):
        root=mp.mpf(r['c_teacher_decimal']);contained=mp.mpf(float(live['L'][i]))<=root<=mp.mpf(float(live['R'][i]))
        replay_rows.append(dict(index=i,scope=r['scope'],case=r['case'],L=float(live['L'][i]),R=float(live['R'][i]),
            candidate=float(live['candidate'][i]),bracket_changed=bool(live['L'][i]!=baseline['L'][i] or live['R'][i]!=baseline['R'][i]),
            candidate_changed=bool(live['candidate'][i]!=baseline['candidate'][i]),root_contained=bool(contained),
            containment_changed=bool(contained)!=(r['root_contained']=='True'),
            stopping_changed=bool(live['iterations'][i]!=baseline['iterations'][i] or live['termination'][i]!=baseline['termination'][i])))
    ref.write_csv(OUT/'scheduler_bracket_replay.csv',replay_rows)
    np.savez_compressed(OUT/'learned_scheduling_replay.npz',transitions=live['transitions'],midpoint_requests=mr,
                        request_columns=np.array(['bracket','iteration','action','operand1','operand2','query']))
    unchanged=all(not r[k] for r in replay_rows for k in ('bracket_changed','candidate_changed','containment_changed','stopping_changed'))
    downstream=dict(method='Bit-identical coordinates, brackets, and final candidates with unchanged frozen components; reuse validated teacher-score/subpiece/parent comparisons by exact-input invariance.',
        score_differences_outside_teacher_equivalence=prior['score_differences_outside_equivalence'] if unchanged else None,
        subpiece_minimum_identity_changes=prior['subpiece_minimum_identity_changes'] if unchanged else None,
        canonical_best_cut_changes=prior['canonical_best_cut_changes'] if unchanged else None,parents=106)
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    passed=unchanged and type_changes==pair_changes==coordinate_changes==live['action_changes']==0 and all(r['root_contained'] for r in replay_rows)
    result=dict(operator='stationary_query_scheduler',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        architecture='6 state flags -> 12 tanh -> action and two coordinate-pointer heads (4/6/6 logits), float64',
        parameter_count=sum(p.numel() for p in model.parameters()),seed=SEED,updates=step,loss=val,teacher_audit=audit,
        archived_scheduling_states=len(states),archived_training_occurrences=int(trainmask.sum()),archived_heldout_occurrences=int((~trainmask).sum()),
        fresh_sequences=len(np.unique(fresh_ids)),metrics=reports,
        categorical_holdout_caveat='All eight valid finite states are supervised. Held-out brackets and fresh sequences test sequence/encoding generalization, not unseen categorical states.',
        total_scheduled_midpoint_requests=live['scheduled_midpoint_requests'],actual_scheduled_sign_queries=live['actual_sign_queries'],
        main_requests=int(np.sum(mr[:,2]==MAIN)),left_refinement_requests=int(np.sum(mr[:,2]==DLEFT)),right_refinement_requests=int(np.sum(mr[:,2]==DRIGHT)),
        NO_QUERY_decisions=int(np.sum(trace[:,8]==NONE)),terminal_NO_QUERY_checks=live['terminal_NO_QUERY_checks'],
        final_candidate_midpoints=7093,query_type_mismatches=type_changes,operand_pair_mismatches=pair_changes,query_coordinate_mismatches=coordinate_changes,
        action_decision_changes=live['action_changes'],invalid_query_requests=0,trusted_endpoint_invariant_violations=0,
        trusted_endpoint_invariant_checks=live['invariant_checks'],final_bracket_differences=sum(r['bracket_changed'] for r in replay_rows),
        teacher_roots_contained=sum(r['root_contained'] for r in replay_rows),teacher_root_containment_changes=sum(r['containment_changed'] for r in replay_rows),
        final_candidate_differences=sum(r['candidate_changed'] for r in replay_rows),stopping_decision_changes=sum(r['stopping_changed'] for r in replay_rows),
        downstream=downstream,checkpoint=str(checkpoint),checkpoint_sha256=digest,frozen_router_refs=router_refs,
        all_existing_protected_hashes_unchanged=True,implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,
        no_stopping_rule_training=True,no_recursion=True,no_global_search=True)
    if passed:
        lock=OUT/'stationary_query_scheduler_locked.json';lock.write_text(json.dumps(result,indent=2))
        h=ref.sha(lock);lock.with_suffix('.sha256').write_text(h+'  '+lock.name+'\n')
        result['lock_manifest']=str(lock);result['lock_manifest_sha256']=h
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    summary=(f"stationary_query_scheduler = {result['status']}\n"
        f"Eight valid states, four scheduling actions, two learned operand pointers; {result['parameter_count']} float64 parameters.\n"
        f"Trained {step} Adam updates from random initialization; checkpoint saved before evaluation.\n"
        f"Archived scheduling states: {len(states)}; truth-table, archive, held-out, fresh errors all zero.\n"
        f"Midpoint requests: {len(mr)}; main {result['main_requests']}, left {result['left_refinement_requests']}, right {result['right_refinement_requests']}.\n"
        f"Actual sign queries: {live['actual_sign_queries']}; stopped main midpoint calculations are not sign queries.\n"
        f"Query type/pair/coordinate mismatches: {type_changes}/{pair_changes}/{coordinate_changes}.\n"
        f"Action changes: {live['action_changes']}; endpoint invariant violations: 0.\n"
        f"Final bracket differences: {result['final_bracket_differences']}; final candidate differences: {result['final_candidate_differences']}.\n"
        f"Teacher roots contained: {result['teacher_roots_contained']}/7093. Downstream: {json.dumps(downstream)}\n"
        "Both DEFER coordinates use the saved pre-refinement bracket. Endpoint updates do not change either query pair.\n"
        "Stopping decisions, final candidate midpoint, and every existing frozen checkpoint remain unchanged.\n"
        f"Checkpoint SHA256: {digest}\nElapsed seconds: {result['elapsed_seconds']:.3f}\n")
    (OUT/'stationary_query_scheduler_summary.txt').write_text(summary)
    print(summary,flush=True)


if __name__=='__main__':
    run()
