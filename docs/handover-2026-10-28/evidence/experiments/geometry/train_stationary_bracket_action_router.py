"""Only train the finite bracket-action classifier; deterministic query replay."""
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_stationary_three_state_router import load_router,TRUST_POSITIVE,TRUST_NEGATIVE,DEFER,NAMES
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT=ref.ROOT;SOURCE=ROOT/'defer_aware_stationary_locator_audit';OUT=ROOT/'stationary_bracket_action_router_training'
LEFT=0;RIGHT=1;NO_UPDATE=2
ACTIONS={LEFT:'MOVE_LEFT_ENDPOINT',RIGHT:'MOVE_RIGHT_ENDPOINT',NO_UPDATE:'NO_UPDATE'}
SEED=28272026


def encode(states):
    """Categorical inputs: two endpoint signs and a distinct three-state query."""
    states=np.asarray(states,dtype=np.int64).reshape(-1,3)
    assert np.all(np.isin(states[:,:2],[TRUST_POSITIVE,TRUST_NEGATIVE]))
    assert np.all(states[:,0]!=states[:,1]) and np.all(np.isin(states[:,2],[TRUST_POSITIVE,TRUST_NEGATIVE,DEFER]))
    return torch.from_numpy(np.concatenate((np.eye(2)[states[:,0]-1],np.eye(2)[states[:,1]-1],np.eye(3)[states[:,2]-1]),axis=1))


class BracketActionRouter(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(7,8),nn.Tanh(),nn.Linear(8,3)).double()
    def forward(self,x):return self.layers(x)
    def actions(self,states):
        with torch.no_grad():return self(encode(states)).argmax(1).numpy()


def teacher_actions(states):
    """Supervision/audit only. Not called by the learned endpoint actuation."""
    a=np.asarray(states);return np.where(a[:,2]==DEFER,NO_UPDATE,np.where(a[:,2]==a[:,0],LEFT,RIGHT)).astype(np.int64)


def errors(states,pred):
    truth=teacher_actions(states);bad=pred!=truth
    return dict(examples=len(truth),errors=int(bad.sum()),MOVE_LEFT_errors=int(np.sum(bad&(truth==LEFT))),
        MOVE_RIGHT_errors=int(np.sum(bad&(truth==RIGHT))),NO_UPDATE_errors=int(np.sum(bad&(truth==NO_UPDATE))),
        DEFER_to_endpoint_move_errors=int(np.sum((states[:,2]==DEFER)&(pred!=NO_UPDATE))))


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80;OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if 'manifest' in p.name or 'validated' in p.name})
    for p in SOURCE.iterdir():
        if p.is_file():protected[str(p)]=ref.sha(p)
    prior=json.loads((SOURCE/'results.json').read_text());manifest=SOURCE/'defer_aware_stationary_locator_validated.json'
    assert prior['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE' and ref.sha(manifest)==prior['assembly_sha256']
    saved=list(csv.DictReader((SOURCE/'defer_aware_root_recovery.csv').open()));assert len(saved)==7093
    archive=list(csv.DictReader((SOURCE/'defer_refinement_steps.csv').open()));reverse={v:k for k,v in NAMES.items()}
    archive_states=[];archive_ids=[]
    for r in archive:
        L,R=reverse[r['left_state']],reverse[r['right_state']]
        for q in (DEFER,reverse[r['qL_state']],reverse[r['qR_state']]):archive_states.append((L,R,q));archive_ids.append(int(r['index']))
    archive_states=np.array(archive_states);archive_ids=np.array(archive_ids);training=archive_ids%5!=0
    table=np.array([(l,r,q) for l,r in [(1,2),(2,1)] for q in (1,2,3)])
    table_y=torch.from_numpy(teacher_actions(table));table_x=encode(table)
    # Aggregation of repeated identical state vectors exactly preserves the
    # archived cross-entropy objective without thousands of redundant forwards.
    weights=np.array([np.sum(np.all(archive_states[training]==r,axis=1)) for r in table],float)
    weights=.5/6+.5*weights/weights.sum();weights=torch.from_numpy(weights)
    model=BracketActionRouter();initial={k:v.detach().tolist() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=.03);history=[];best_loss=float('inf');best=None
    for update in range(1,2001):
        opt.zero_grad();logits=model(table_x);loss=(nn.functional.cross_entropy(logits,table_y,reduction='none')*weights).sum();loss.backward();opt.step()
        with torch.no_grad():
            logits=model(table_x);loss=float((nn.functional.cross_entropy(logits,table_y,reduction='none')*weights).sum());pred=logits.argmax(1)
            correct=logits[torch.arange(6),table_y];other=logits.clone();other[torch.arange(6),table_y]=-torch.inf;minimum_margin=float((correct-other.max(1).values).min())
        if loss<best_loss:best_loss=loss;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        history.append(dict(update=update,loss=loss,truth_errors=int((pred!=table_y).sum()),minimum_correct_logit_margin=minimum_margin))
        if torch.equal(pred,table_y) and minimum_margin>=3 and loss<=.02:break
    model.load_state_dict(best);model.eval().requires_grad_(False)
    checkpoint=OUT/'bracket_action_router_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),optimizer_state=opt.state_dict(),seed=SEED,updates=update,
        architecture='7 categorical one-hot inputs -> Linear(7,8) -> Tanh -> Linear(8,3), float64',
        state_codes=NAMES,action_codes=ACTIONS,initial_parameters=initial,training_archive_manifest=str(manifest)))
    ref.write_csv(OUT/'training_history.csv',history)
    truth_rows=[]
    with torch.no_grad():logits=model(table_x).numpy()
    for s,t,logit in zip(table,teacher_actions(table),logits):
        truth_rows.append(dict(left=NAMES[s[0]],right=NAMES[s[1]],query=NAMES[s[2]],teacher_action=ACTIONS[int(t)],learned_action=ACTIONS[int(logit.argmax())],logits=json.dumps(logit.tolist())))
    ref.write_csv(OUT/'truth_table.csv',truth_rows)
    rng=np.random.default_rng(SEED+1);fresh=table[rng.integers(0,6,12288)]
    metrics=dict(truth_table=errors(table,model.actions(table)),archived_saved_transitions=errors(archive_states,model.actions(archive_states)),
        heldout_archived_brackets=errors(archive_states[~training],model.actions(archive_states[~training])),fresh_synthetic=errors(fresh,model.actions(fresh)))
    assert all(r['errors']==0 for r in metrics.values()), 'Finite router failed; no locator replay allowed.'
    print(json.dumps(dict(stage='trained_and_saved',updates=update,checkpoint_sha256=digest,metrics=metrics)),flush=True)
    # Coefficients cached by the previous float64 audit; initial brackets and
    # termination/query counts come from the original validated locator itself.
    pool=ROOT/'stationary_float64_minimizer_full_audit/all_float64_candidates.npz';protected[str(pool)]=ref.sha(pool)
    data=np.load(pool);source_rows=list(csv.DictReader((pool.parent/'stationary_float64_minimizers.csv').open()))
    coeffmap={(r['scope'],r['case']):data['inputs'][data['offsets'][i],[0,1,2,3,5,6]] for i,r in enumerate(source_rows)}
    coeff=np.array([coeffmap[(r['scope'],r['case'])] for r in saved]);n=len(saved)
    L=np.array([float(r['initial_L']) for r in saved]);R=np.array([float(r['initial_R']) for r in saved]);router,router_refs=load_router()
    calls=np.zeros(n,int);defers=np.zeros(n,int);iterations=np.zeros(n,int);active=np.ones(n,bool);reason=np.full(n,'',object)
    transition_records=[];step_rows=[];invariant_checks=0;invariant_violations=0;action_errors=0;failures=[];quarter_steps=[]
    def query(ids,c):
        a=coeff[ids];x=np.column_stack((a[:,:4],c,a[:,4:]))
        with torch.no_grad():out=router(torch.from_numpy(x)).numpy()
        calls[ids]+=1;defers[ids]+=(out==DEFER);return out
    ids=np.arange(n);sl=query(ids,L);sr=query(ids,R)
    def invariant(where):
        nonlocal invariant_checks,invariant_violations
        invariant_checks+=1;valid=(L<R)&(sl!=sr)&np.isin(sl,[1,2])&np.isin(sr,[1,2]);invariant_violations+=int((~valid).sum())
        if not valid.all():raise RuntimeError('Trusted endpoint invariant violated at '+where)
    def apply_action(ids,c,codes,kind):
        nonlocal action_errors
        if not len(ids):return
        states=np.column_stack((sl[ids],sr[ids],codes));actions=model.actions(states)
        # Independent shadow action only checks the model; it never overrides it.
        truth=teacher_actions(states);bad=actions!=truth;action_errors+=int(bad.sum())
        transition_records.append(np.column_stack((ids,iterations[ids],states,actions,truth,c)))
        if bad.any():
            failures.extend(dict(index=int(ids[j]),kind=kind,query=float(c[j]),states=states[j].tolist(),learned_action=int(actions[j]),teacher_action=int(truth[j])) for j in np.flatnonzero(bad))
        left=actions==LEFT;right=actions==RIGHT
        L[ids[left]]=c[left];sl[ids[left]]=codes[left]
        R[ids[right]]=c[right];sr[ids[right]]=codes[right]
        if bad.any():raise RuntimeError('Action mismatch in learned replay; no teacher fallback.')
    invariant('initial');step=0
    while active.any():
        ids=np.flatnonzero(active);q=(L[ids]+R[ids])/2
        adjacent=np.nextafter(L[ids],R[ids])==R[ids];rounded=(q==L[ids])|(q==R[ids]);done=adjacent|rounded
        reason[ids[done]]=np.where(adjacent[done],'adjacent_float','midpoint_rounds_to_endpoint');active[ids[done]]=False
        ids=ids[~done];q=q[~done]
        if not len(ids):break
        iterations[ids]+=1;codes=query(ids,q);deferred=codes==DEFER
        # The action network sees ALL midpoint states, including DEFER. Query
        # refinement remains the existing deterministic DEFER control flow.
        apply_action(ids,q,codes,'midpoint');invariant('midpoint')
        di=ids[deferred];dq=q[deferred]
        if len(di):
            oldL=L[di].copy();oldR=R[di].copy();qL=(oldL+dq)/2;qR=(dq+oldR)/2
            left=query(di,qL);right=query(di,qR)
            apply_action(di,qL,left,'left_quarter');invariant('left_quarter')
            apply_action(di,qR,right,'right_quarter');invariant('right_quarter')
            assert np.all(L[di]>=oldL) and np.all(R[di]<=oldR)
            progressed=(L[di]>oldL)|(R[di]<oldR);stalled=di[~progressed];reason[stalled]='no_trusted_inward_move';active[stalled]=False
            quarter_steps.extend(dict(index=int(i),iteration=int(iterations[i]),q=float(dq[j]),qL=float(qL[j]),qR=float(qR[j])) for j,i in enumerate(di))
        step_rows.append(dict(round=step,active_after=int(active.sum()),midpoint_queries=len(ids),DEFER_midpoints=len(di),completed=int((~active).sum())));step+=1
    invariant('final');candidate=(L+R)/2
    # Endpoint re-evaluation verifies trusted state, independently of action IDs.
    before_calls=calls.copy();before_defers=defers.copy();checkL=query(np.arange(n),L);checkR=query(np.arange(n),R);calls=before_calls;defers=before_defers
    assert np.array_equal(checkL,sl) and np.array_equal(checkR,sr)
    transitions=np.concatenate(transition_records);states=transitions[:,2:5].astype(int);actions=transitions[:,5].astype(int)
    metrics['full_locator_replay_actions']=errors(states,actions)
    np.savez_compressed(OUT/'replayed_query_actions.npz',transitions=transitions,columns=np.array(['index','iteration','left_state','right_state','query_state','learned_action','teacher_action','query_coordinate']))
    archived_quarters={(int(r['index']),int(r['iteration'])):r for r in archive};query_location_changes=0
    for row in quarter_steps:
        priorrow=archived_quarters[(row['index'],row['iteration'])]
        query_location_changes+=int(any(row[k]!=float(priorrow[k]) for k in ('q','qL','qR')))
    assert len(quarter_steps)==len(archive)
    old_steps=list(csv.DictReader((SOURCE/'locator_rounds.csv').open()))
    assert len(old_steps)==len(step_rows)
    round_trace_changes=sum(any(int(old[k])!=new[k] for k in new) for old,new in zip(old_steps,step_rows))
    replay_rows=[]
    for i,r in enumerate(saved):
        root=mp.mpf(r['c_teacher_decimal']);contained=mp.mpf(float(L[i]))<=root<=mp.mpf(float(R[i]))
        replay_rows.append(dict(index=i,scope=r['scope'],case=r['case'],final_L=float(L[i]),final_R=float(R[i]),
            final_bracket_changed=L[i]!=float(r['final_L']) or R[i]!=float(r['final_R']),candidate=float(candidate[i]),candidate_changed=candidate[i]!=float(r['c_candidate']),
            root_contained=bool(contained),root_containment_changed=bool(contained)!=(r['root_contained']=='True'),
            iterations=int(iterations[i]),iteration_count_changed=iterations[i]!=int(r['iterations']),router_queries=int(calls[i]),query_count_changed=calls[i]!=int(r['router_queries']),
            DEFER_queries=int(defers[i]),DEFER_count_changed=defers[i]!=int(r['DEFER_queries']),termination=reason[i],termination_changed=reason[i]!=r['termination']))
    unchanged=all(not r[k] for r in replay_rows for k in ('final_bracket_changed','candidate_changed','root_containment_changed','iteration_count_changed','query_count_changed','DEFER_count_changed','termination_changed'))
    # Identical candidate coordinates + unchanged frozen scores imply identical
    # full subpiece/interval/parent selection. Audit this from existing manifests;
    # do not rerun recursion or introduce new geometry/minimum computation.
    canonical_rows=list(csv.DictReader((SOURCE/'canonical_minimum_comparison.csv').open()))
    downstream=dict(method='Exact-input invariance: all replayed brackets/candidates bit-identical; original frozen hashes unchanged. Reuse saved teacher/minimum comparison results.',
        score_differences_outside_teacher_equivalence=prior['score_differences_outside_equivalence'] if unchanged else None,
        subpiece_minimum_identity_changes=prior['subpiece_minimum_identity_changes'] if unchanged else None,
        canonical_best_cut_changes=prior['canonical_best_cut_changes'] if unchanged else None,
        changes_relative_to_validated_locator=0 if unchanged else None,archived_parents=prior['archived_parents'],saved_canonical_comparisons=len(canonical_rows))
    ref.write_csv(OUT/'bracket_replay.csv',replay_rows);ref.write_csv(OUT/'replay_rounds.csv',step_rows)
    ref.write_csv(OUT/'action_validation.csv',[dict(group=k,**v) for k,v in metrics.items()])
    passed=unchanged and action_errors==invariant_violations==query_location_changes==round_trace_changes==0 and all(v['errors']==0 for v in metrics.values())
    summary=dict(operator='stationary_bracket_action_router',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        architecture='float64 MLP: 7 one-hot inputs -> 8 tanh units -> 3 logits; argmax action; 91 trainable parameters',seed=SEED,updates=update,
        training_loss=best_loss,training_truth_table_states=6,archived_training_transition_occurrences=int(training.sum()),archived_heldout_transition_occurrences=int((~training).sum()),
        discrete_holdout_caveat='There are only six valid categorical combinations, all supervised. Fresh examples and held-out brackets test replay/encoding, not unseen truth-table states.',
        parameters={k:v.detach().tolist() for k,v in model.state_dict().items()},metrics=metrics,
        bracket_searches=n,total_router_queries=int(calls.sum()),DEFER_queries=int(defers.sum()),endpoint_audit_queries=2*n,
        learned_action_decisions=len(actions),action_counts={ACTIONS[k]:int(np.sum(actions==k)) for k in ACTIONS},
        trusted_endpoint_invariant_checks=invariant_checks,trusted_endpoint_invariant_violations=invariant_violations,direction_action_changes=action_errors,
        final_bracket_differences=sum(r['final_bracket_changed'] for r in replay_rows),teacher_root_containment_changes=sum(r['root_containment_changed'] for r in replay_rows),
        teacher_roots_contained=sum(r['root_contained'] for r in replay_rows),query_location_changes=query_location_changes,round_trace_changes=round_trace_changes,
        downstream=downstream,checkpoint=str(checkpoint),sha256=digest,source_locator_manifest=str(manifest),source_locator_sha256=ref.sha(manifest),
        source_archive_transition_sha256=ref.sha(SOURCE/'defer_refinement_steps.csv'),sign_router_references=router_refs,
        implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,no_recursive_partitioning=True,no_global_search=True,no_query_location_training=True)
    assert all(ref.sha(Path(p))==h for p,h in protected.items());summary['all_existing_protected_hashes_unchanged']=True
    summary=json.loads(json.dumps(summary,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    if passed:
        dest=OUT/'stationary_bracket_action_router_locked.json';dest.write_text(json.dumps(summary,indent=2));h=ref.sha(dest);dest.with_suffix('.sha256').write_text(h+'  '+dest.name+'\n')
        summary['lock_manifest']=str(dest);summary['lock_manifest_sha256']=h
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'bracket_action_router_summary.txt').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
