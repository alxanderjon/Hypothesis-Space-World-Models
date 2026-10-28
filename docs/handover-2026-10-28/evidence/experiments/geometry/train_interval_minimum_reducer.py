"""Learn only interval KEEP/REPLACE routing under global canonical equivalence."""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_subpiece_minimum_selector import SubpieceSelector,canonical
from experiments.sho.train_finite_split_score_smoke import load_frozen
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT=ref.ROOT;SOURCE=ROOT/'subpiece_minimum_selector_training';OUT=ROOT/'interval_minimum_reducer_training'
SEED=28332026
KEEP=0;REPLACE=1
TABLE=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64)
LABELS=np.array([KEEP,REPLACE,KEEP,KEEP])


class IntervalMinimumReducer(nn.Module):
    """Existing frozen min/equivalence pass followed by learned binary routing."""
    def __init__(self,comparison):
        super().__init__();self.head=nn.Linear(2,2,dtype=torch.float64);self.comparison=comparison
    def actions(self,features):
        with torch.no_grad():return self.head(torch.from_numpy(np.asarray(features,dtype=float))).argmax(1).numpy()
    def prepare(self,sequences):
        lengths=np.array([len(s) for s in sequences]);assert np.all(lengths>=1)
        scores=np.zeros((len(sequences),int(lengths.max())))
        valid=np.arange(scores.shape[1])[None,:]<lengths[:,None]
        for i,s in enumerate(sequences):scores[i,:len(s)]=s
        assert np.isfinite(scores).all()
        best=scores[:,0].copy()
        for j in range(1,scores.shape[1]):
            take=valid[:,j]&self.comparison.less(best,scores[:,j]);best[take]=scores[take,j]
        # Preserve the exact absolute tolerance floor below 1, as audited for
        # the locked subpiece selector. Binary scaling is exact on this domain.
        ex=np.zeros(len(best),int);large=best>=1;ex[large]=np.frexp(best[large])[1]-1
        normalized=np.ldexp(scores,-ex[:,None]);nb=np.ldexp(best,-ex)
        bound=nb+1e-12*np.maximum(1.,nb)
        eligible=valid&~self.comparison.less(normalized,np.broadcast_to(bound[:,None],scores.shape))
        assert np.all(eligible.any(1))
        return scores,valid,best,eligible
    def reduce(self,sequences,shadow=False,teacher_actions=False):
        scores,valid,best,eligible=self.prepare(sequences)
        current=np.zeros(len(sequences),int);records=[];normalization_errors=0
        if shadow:
            tb=np.where(valid,scores,np.inf).min(1)
            te=valid&(scores<=tb[:,None]+1e-12*np.maximum(1.,tb[:,None]))
            normalization_errors=int(np.sum(eligible!=te))
            assert normalization_errors==0, 'Frozen equivalence feature disagrees with teacher.'
        for j in range(1,scores.shape[1]):
            ids=np.flatnonzero(valid[:,j]);cur=current[ids].copy()
            state=np.column_stack((eligible[ids,cur],eligible[ids,j])).astype(float)
            truth=((~eligible[ids,cur])&eligible[ids,j]).astype(int)
            action=truth if teacher_actions else self.actions(state)
            if shadow or teacher_actions:
                sa=scores[ids,cur];sb=scores[ids,j]
                tie=sa==sb
                ulp=(np.nextafter(sa,np.inf)==sb)|(np.nextafter(sb,np.inf)==sa)
                # Exact float neighborhoods at the global eligibility boundary.
                boundary=best[ids]+1e-12*np.maximum(1.,best[ids]);near=np.zeros(len(ids),bool)
                for values in (sa,sb):
                    near|=(values==boundary)|(values==np.nextafter(boundary,-np.inf))|(values==np.nextafter(boundary,np.inf))
                records.append(np.column_stack((ids,np.full(len(ids),j),cur,state,truth,action,tie,ulp,near)))
            move=action==REPLACE;current[ids[move]]=j
        trace=np.concatenate(records) if records else np.empty((0,10))
        if shadow:
            expected=np.argmax(te,axis=1)
            assert np.array_equal(current,expected), 'Learned reduction changed canonical winner.'
        return current,trace,dict(comparison_feature_errors=normalization_errors,
            frozen_minimum_comparisons=int(np.sum(valid)-len(sequences)),routing_transitions=int(np.sum(valid)-len(sequences)))


def transition_metrics(trace):
    if not len(trace):return dict(transitions=0,total_errors=0,KEEP_errors=0,REPLACE_errors=0,exact_tie_errors=0,one_ULP_errors=0,equivalence_boundary_errors=0)
    t=trace[:,5].astype(int);p=trace[:,6].astype(int);bad=t!=p
    return dict(transitions=len(trace),total_errors=int(bad.sum()),KEEP_errors=int(np.sum(bad&(t==KEEP))),
        REPLACE_errors=int(np.sum(bad&(t==REPLACE))),KEEP_count=int(np.sum(t==KEEP)),REPLACE_count=int(np.sum(t==REPLACE)),
        exact_tie_transitions=int(trace[:,7].sum()),exact_tie_errors=int(np.sum(bad&trace[:,7].astype(bool))),
        one_ULP_transitions=int(trace[:,8].sum()),one_ULP_errors=int(np.sum(bad&trace[:,8].astype(bool))),
        equivalence_boundary_transitions=int(trace[:,9].sum()),equivalence_boundary_errors=int(np.sum(bad&trace[:,9].astype(bool))))


def synthetic_sequences(seed,n):
    rng=np.random.default_rng(seed);seq=[];groups=[]
    for i in range(n):
        length=int(rng.choice([1,2,3,4,8,16,31,64]));base=float(10**rng.uniform(-9,9));mode=i%6
        if mode==0:s=10**rng.uniform(-9,9,length);name='ordinary'
        elif mode==1:s=np.full(length,base);name='exact_ties'
        elif mode==2:
            s=np.full(length,base)
            for j in range(1,length):s[j]=np.nextafter(s[j-1],np.inf)
            rng.shuffle(s);name='one_ULP'
        elif mode==3:
            bound=base+1e-12*max(1.,base)
            s=rng.choice([base,np.nextafter(bound,-np.inf),bound,np.nextafter(bound,np.inf)],length)
            s[-1]=base;name='equivalence_boundary'
        elif mode==4:
            # Include a deliberately non-transitive chain: memoryless
            # pairwise tolerance reduction picks the wrong canonical identity.
            s=np.array([base+1.5e-12*max(1.,base),base+.75e-12*max(1.,base),base]);name='tolerance_chain'
        else:s=10**rng.uniform(-9,9,length)*rng.choice([-1.,1.],length);name='signed_generic'
        seq.append(np.asarray(s,float));groups.append(name)
    return seq,np.array(groups)


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('manifest','locked','validated'))})
    meta=json.loads((SOURCE/'results.json').read_text());assert meta['status']=='LOCKED'
    cp=Path(meta['checkpoint']);assert ref.sha(cp)==meta['checkpoint_sha256']
    _,execution,refs=load_frozen();subpiece=SubpieceSelector(execution)
    subpiece.route.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);subpiece.eval().requires_grad_(False)
    reducer=IntervalMinimumReducer(subpiece)
    allrows=[]
    for r in csv.DictReader((SOURCE/'frozen_candidate_scores.csv').open()):
        if r['group'] not in ('archive_geometry','fresh_geometry'):continue
        e=dict(r)
        for k in ('parent','axis','k'):e[k]=int(e[k])
        for k in ('cut','teacher_score','score'):e[k]=float(e[k])
        allrows.append(e)
    subpools=defaultdict(list)
    for e in allrows:subpools[(e['group'],e['case'])].append(e)
    ordered_sub=[];s=[];v=[]
    for key,pool in subpools.items():
        pool=sorted(pool,key=lambda e:(e['cut'],e['identity']));assert len(pool) in (2,3)
        slots=[pool[0],pool[-1],pool[1] if len(pool)==3 else None];ordered_sub.append((key,slots))
        s.append([e['score'] if e else 0 for e in slots]);v.append([e is not None for e in slots])
    s=np.array(s);v=np.array(v);subwinner=subpiece.select(subpiece.normalize(s,v),v)
    full=defaultdict(list);reduced=defaultdict(list)
    for e in allrows:full[(e['group'],e['parent'],e['axis'],e['k'])].append(e)
    for (key,slots),j in zip(ordered_sub,subwinner):
        e=slots[j];assert e is not None
        reduced[(e['group'],e['parent'],e['axis'],e['k'])].append(e)
    keys=sorted(reduced);lists=[sorted(reduced[k],key=lambda e:(e['axis'],e['cut'],e['identity'])) for k in keys]
    seq=[np.array([e['score'] for e in rr]) for rr in lists]
    archive=np.array([k[0]=='archive_geometry' for k in keys]);archive_ids=np.flatnonzero(archive);fresh_ids=np.flatnonzero(~archive)
    assert len(archive_ids)==4356 and len(fresh_ids)==124
    # Explicitly trace all 46 generic two-root diagnostics to their source.
    legacy=[];legacy_lists=[];legacyrows=[]
    for r in csv.DictReader((SOURCE/'excluded_multi_root_intervals.csv').open()):
        stem,position=r['case'].rsplit(':',1);path=ROOT/'split_subpiece_stationary_solver_smoke'/(stem+'.json')
        record=json.loads(path.read_text())[int(position)];assert len(record['roots'])==2
        assert not any(k in record for k in ('parent','axis','interval','child_counts'))
        cuts=sorted(set([float(record['lower']),float(record['upper'])]+[float(c) for c in record['roots']]))
        scores=ref.generic_score(record,np.array(cuts));legacy.append(scores);legacy_lists.append(cuts)
        legacyrows.append(dict(case=r['case'],classification='legacy_generic_diagnostic_only',source=str(path),source_sha256=ref.sha(path),
            active_parent_mapping=False,candidates=len(cuts),cuts=json.dumps(cuts),scores=json.dumps(scores.tolist())))
    assert len(legacy)==46
    ref.write_csv(OUT/'two_root_case_audit.csv',legacyrows)
    chain=np.array([1+1.5e-12,1+.75e-12,1.]);globalwinner=int(np.flatnonzero(chain<=chain.min()+1e-12)[0]);pairwinner=0
    for j in range(1,3):
        b=min(chain[pairwinner],chain[j])
        if chain[pairwinner]>b+1e-12*max(1.,b):pairwinner=j
    audit=dict(order='Subpiece winners are sorted by (axis, cut ascending, identity lexicographic) before interval selection; archive insertion order is not the final ordering.',
        duplicates='No new deduplication. Shared endpoint winners retain distinct subpiece identities; equal cut/score uses the first identity in canonical order.',
        teacher_rule='First determine global strict minimum best_score. Then choose first ordered candidate with score <= best_score + 1e-12*max(1,best_score).',
        sequential_action='With global-best eligibility supplied by frozen comparison machinery: REPLACE iff current is ineligible and challenger is eligible; otherwise KEEP. Once eligible, current survives every later equivalent challenger.',
        required_context='This is a two-pass canonical reducer: frozen global minimum/equivalence followed by learned sequential KEEP/REPLACE. A memoryless local pairwise tolerance rule is not the existing teacher and is not substituted.',
        counterexample=dict(scores=chain.tolist(),teacher_global_canonical_index=globalwinner,incorrect_local_pairwise_index=pairwinner),
        two_root_cases='All 46 originate in generic stationary-solver diagnostic data, with no parent/axis/allocation mapping. The active geometric pipeline only assembles scope=geometric into interval/parent pools. Audit all complete four-candidate lists and replay them as supplementary reducer checks.',
        supported_domain='Active geometric interval lists plus arbitrary ordered finite diagnostic score sequences; candidate validity/coordinates and upstream subpiece routing remain unchanged.',
        source_hashes={str(p):ref.sha(p) for p in [Path('experiments/geometry/train_subpiece_minimum_selector.py'),Path('experiments/geometry/validate_terminal_candidate_full.py'),Path('experiments/geometry/audit_stationary_point_impact.py')]})
    (OUT/'interval_teacher_audit.json').write_text(json.dumps(audit,indent=2));(OUT/'interval_teacher_audit.txt').write_text('\n\n'.join(k+': '+str(v) for k,v in audit.items()))
    _,arch_trace,_=reducer.reduce(seq,shadow=True,teacher_actions=True)
    heldout_interval=(np.arange(len(keys))%5==0)&archive
    train_transition=archive[arch_trace[:,0].astype(int)]&~heldout_interval[arch_trace[:,0].astype(int)]
    train_syn,train_kind=synthetic_sequences(SEED+1,4096);_,syn_trace,_=reducer.reduce(train_syn,shadow=True,teacher_actions=True)
    x=np.concatenate((TABLE,arch_trace[train_transition,3:5],syn_trace[:,3:5]));y=np.concatenate((LABELS,arch_trace[train_transition,5].astype(int),syn_trace[:,5].astype(int)))
    unique,inverse=np.unique(np.column_stack((x,y)),axis=0,return_inverse=True);counts=np.bincount(inverse).astype(float)
    weights=.5*counts/counts.sum()+.5/len(counts)
    xt=torch.from_numpy(unique[:,:2]);yt=torch.from_numpy(unique[:,2].astype(int));wt=torch.from_numpy(weights)
    initial={k:v.detach().tolist() for k,v in reducer.head.state_dict().items()};opt=torch.optim.Adam(reducer.head.parameters(),lr=.06);history=[]
    for update in range(1,2001):
        opt.zero_grad();logits=reducer.head(xt);loss=(nn.functional.cross_entropy(logits,yt,reduction='none')*wt).sum();loss.backward();opt.step()
        with torch.no_grad():
            logits=reducer.head(xt);value=float((nn.functional.cross_entropy(logits,yt,reduction='none')*wt).sum());err=int((logits.argmax(1)!=yt).sum())
        if update%10==0:history.append(dict(update=update,loss=value,truth_errors=err))
        if err==0 and value<.01:break
    reducer.eval().requires_grad_(False)
    checkpoint=OUT/'interval_reducer_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=reducer.head.state_dict(),optimizer_state=opt.state_dict(),initial_parameters=initial,seed=SEED,updates=update,
        architecture='Frozen global minimum/equivalence -> learned float64 Linear(2,2) -> KEEP/REPLACE',
        features=['current_eligible_against_global_best','challenger_eligible_against_global_best'],actions=['KEEP_CURRENT','REPLACE_CURRENT'],
        frozen_subpiece_checkpoint=str(cp),frozen_subpiece_sha256=meta['checkpoint_sha256']))
    ref.write_csv(OUT/'training_history.csv',history)
    print(json.dumps(dict(stage='trained_saved',updates=update,sha256=digest,archived_transitions=int(np.sum(archive[arch_trace[:,0].astype(int)])))),flush=True)
    chosen,trace,stats=reducer.reduce(seq,shadow=True)
    fresh_syn,fresh_kind=synthetic_sequences(SEED+2,8192);synchosen,st,_=reducer.reduce(fresh_syn,shadow=True)
    legacychosen,lt,_=reducer.reduce(legacy,shadow=True)
    ac=archive[trace[:,0].astype(int)];hc=heldout_interval[trace[:,0].astype(int)]
    reports={'complete_rule':dict(transitions=4,total_errors=int(np.sum(reducer.actions(TABLE)!=LABELS))),
        'archived':transition_metrics(trace[ac]),'heldout_archived':transition_metrics(trace[hc]),
        'fresh_geometric':transition_metrics(trace[~ac]),'fresh_synthetic':transition_metrics(st),
        'legacy_two_root_diagnostics':transition_metrics(lt)}
    for group in np.unique(fresh_kind):reports['synthetic_'+group]=transition_metrics(st[fresh_kind[st[:,0].astype(int)]==group])
    np.savez_compressed(OUT/'interval_reduction_transitions.npz',transitions=trace,
        columns=np.array(['interval_index','challenger_index','current_index','current_eligible','challenger_eligible','teacher_action','learned_action','exact_tie','one_ULP','equivalence_boundary']),
        interval_keys=np.array([str(k) for k in keys]),synthetic_transitions=st,legacy_transitions=lt)
    ref.write_csv(OUT/'reducer_accuracy.csv',[dict(group=k,**v) for k,v in reports.items()])
    ref.write_csv(OUT/'reduction_truth_table.csv',[dict(current_eligible=int(s[0]),challenger_eligible=int(s[1]),teacher=int(t),learned=int(p)) for s,t,p in zip(TABLE,LABELS,reducer.actions(TABLE))])
    intervals=[];parentteacher=defaultdict(list);parentlearned=defaultdict(list);ordered_rows=[]
    for i,(key,pool) in enumerate(zip(keys,lists)):
        t=canonical(full[key]);l=pool[chosen[i]];best=min(e['teacher_score'] for e in full[key]);outside=l['teacher_score']>best+1e-12*max(1.,best)
        gap=abs(int(np.float64(t['cut']).view(np.uint64))-int(np.float64(l['cut']).view(np.uint64)))
        intervals.append(dict(group=key[0],parent=key[1],axis=key[2],allocation=key[3],candidate_count=len(pool),teacher_identity=t['identity'],learned_identity=l['identity'],
            exact_winner=t['identity']==l['identity'] and t['cut']==l['cut'],identity_changed=t['identity']!=l['identity'],coordinate_changed=t['cut']!=l['cut'],
            teacher_cut=t['cut'],learned_cut=l['cut'],coordinate_ULP_difference=gap,score_penalty=max(0.,l['teacher_score']-best),outside_equivalence=outside))
        parentteacher[(key[0],key[1])].extend(full[key]);parentlearned[(key[0],key[1])].append(l)
        ordered_rows.extend(dict(interval_index=i,order=j,**e) for j,e in enumerate(pool))
    parents=[]
    for key,pool in parentteacher.items():
        t=canonical(pool);l=canonical(parentlearned[key],'score');best=min(e['teacher_score'] for e in pool)
        parents.append(dict(group=key[0],parent=key[1],teacher_axis=t['axis'],learned_axis=l['axis'],teacher_allocation=t['k'],learned_allocation=l['k'],
            teacher_identity=t['identity'],learned_identity=l['identity'],teacher_cut=t['cut'],learned_cut=l['cut'],
            axis_changed=t['axis']!=l['axis'],allocation_changed=t['k']!=l['k'],identity_changed=t['identity']!=l['identity'],
            canonical_best_cut_changed=t['axis']!=l['axis'] or t['cut']!=l['cut'],score_penalty=max(0.,l['teacher_score']-best),
            outside_equivalence=l['teacher_score']>best+1e-12*max(1.,best)))
    for i,r in enumerate(legacyrows):
        y=legacy[i];best=float(y.min());expected=int(np.flatnonzero(y<=best+1e-12*max(1.,best))[0])
        r.update(teacher_winner=expected,learned_winner=int(legacychosen[i]),winner_error=bool(expected!=legacychosen[i]))
    ref.write_csv(OUT/'two_root_case_audit.csv',legacyrows)
    ref.write_csv(OUT/'ordered_interval_candidates.csv',ordered_rows)
    ref.write_csv(OUT/'interval_winner_replay.csv',intervals);ref.write_csv(OUT/'parent_validation.csv',parents)
    ref.write_csv(OUT/'learned_parameters.csv',[dict(parameter=k,values=json.dumps(v.tolist())) for k,v in reducer.head.state_dict().items()])
    passed=(all(v['total_errors']==0 for v in reports.values()) and not any(r['outside_equivalence'] for r in intervals+parents)
        and not any(r['canonical_best_cut_changed'] or r['identity_changed'] or r['allocation_changed'] for r in parents)
        and not any(r['winner_error'] for r in legacyrows))
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    def summarize_intervals(group):
        rr=[r for r in intervals if r['group']==group]
        return dict(intervals=len(rr),exact_winner_matches=sum(r['exact_winner'] for r in rr),equivalent_coordinate_differences=sum(r['coordinate_changed'] and not r['outside_equivalence'] for r in rr),
            candidate_identity_changes=sum(r['identity_changed'] for r in rr),outside_equivalence=sum(r['outside_equivalence'] for r in rr),max_coordinate_ULP_difference=max(r['coordinate_ULP_difference'] for r in rr),
            candidate_list_length_min=min(r['candidate_count'] for r in rr),candidate_list_length_max=max(r['candidate_count'] for r in rr))
    def summarize_parents(group):
        rr=[r for r in parents if r['group']==group]
        return dict(parents=len(rr),axis_changes=sum(r['axis_changed'] for r in rr),allocation_changes=sum(r['allocation_changed'] for r in rr),
            canonical_best_cut_changes=sum(r['canonical_best_cut_changed'] for r in rr),identity_changes=sum(r['identity_changed'] for r in rr),outside_equivalence=sum(r['outside_equivalence'] for r in rr))
    result=dict(operator='interval_minimum_reducer',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        architecture='Frozen global minimum/equivalence pass then learned Linear(2,2) KEEP/REPLACE router; 6 new float64 parameters',
        parameters={k:v.tolist() for k,v in reducer.head.state_dict().items()},updates=update,seed=SEED,audit=audit,
        metrics=reports,archive_intervals=summarize_intervals('archive_geometry'),fresh_intervals=summarize_intervals('fresh_geometry'),
        archive_parents=summarize_parents('archive_geometry'),fresh_parents=summarize_parents('fresh_geometry'),
        legacy_two_root_cases=46,legacy_two_root_winner_errors=sum(r['winner_error'] for r in legacyrows),
        training_archived_transitions=int(train_transition.sum()),heldout_archived_transitions=int(hc.sum()),
        synthetic_train_sequences=len(train_syn),synthetic_fresh_sequences=len(fresh_syn),synthetic_list_length_min=min(map(len,fresh_syn)),synthetic_list_length_max=max(map(len,fresh_syn)),
        frozen_comparison_stats=stats,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        all_existing_protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start,implementation_sha256=ref.sha(Path(__file__)),
        no_parent_training=True,no_recursion=True,no_global_search=True)
    result=json.loads(json.dumps(result,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    if passed:
        manifest=OUT/'interval_minimum_reducer_locked.json';manifest.write_text(json.dumps(result,indent=2));h=ref.sha(manifest)
        manifest.with_suffix('.sha256').write_text(h+'  '+manifest.name+'\n');result['lock_manifest_sha256']=h
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'interval_reducer_summary.txt').write_text(json.dumps(result,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
