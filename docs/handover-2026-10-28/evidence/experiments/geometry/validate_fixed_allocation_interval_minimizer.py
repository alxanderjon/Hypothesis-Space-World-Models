"""Frozen score/minimum assembly on archived interval candidates, no new roots."""
import ast
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.sho.train_finite_split_score_smoke import load_frozen,FiniteSplitScore,upstream,predict

ROOT=ref.ROOT
SOURCE=ROOT/'stationary_float64_minimizer_full_audit'
OUT=ROOT/'fixed_allocation_interval_minimizer_audit'


def bitgap(a,b):return abs(int(np.float64(a).view(np.uint64))-int(np.float64(b).view(np.uint64)))


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*manifest.json')})
    for p in SOURCE.iterdir():
        if p.is_file():protected[str(p)]=ref.sha(p)
    manifest=SOURCE/'stationary_float64_minimizer_validated.json';prior=json.loads((SOURCE/'results.json').read_text())
    assert json.loads(manifest.read_text())['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE' and ref.sha(manifest)==prior['assembly_sha256']
    stationary={(r['scope'],r['case']):r for r in csv.DictReader((SOURCE/'stationary_float64_minimizers.csv').open())}
    models,execution,score_refs=load_frozen()
    lockpath=ROOT/'split_validity_smoke/split_score_operator_locked.pt';lock=torch.load(lockpath,weights_only=False)
    fm=lock['components']['finite_score'];fp=Path(fm['checkpoint'])
    fmanifest=json.loads((ROOT/'finite_split_score_smoke/candidate_manifest.json').read_text());assert ref.sha(fp)==fmanifest['sha256']
    finite=FiniteSplitScore();finite.load_state_dict(torch.load(fp,weights_only=False)['state_dict']);finite.eval().requires_grad_(False)
    source=ref.Inputs();source.collect();audit=ref.Audit.__new__(ref.Audit)
    candidates=[];pools=defaultdict(list);shared=0;stationary_count=0;local_duplicate_removals=0
    for b in source.batches:
        if b['scope']!='archived_subpiece_grids':continue
        key=ast.literal_eval(b['case']);parent,axis,k,piece=key;sp=source.piece_map[key]
        lo,hi=float(sp['lower']),float(sp['upper']);root=stationary.get(('geometric',b['case']))
        if root:
            entries=[(lo,lo,'left'),(float(root['c_teacher_min']),float(root['c_learned_min']),'stationary'),(hi,hi,'right')]
            stationary_count+=1
        else:
            roots=json.loads(sp['stationary_points'])+json.loads(sp['quartic_stationary_roots'])
            raw=[lo,hi]+[c for c in roots if lo<=c<=hi];cuts=sorted(set(raw));local_duplicate_removals+=len(raw)-len(cuts)
            entries=[(c,c,f'saved{j}') for j,c in enumerate(cuts)]
        audit.setup(source.parents[parent],axis)
        for ct,cl,kind in entries:
            assert lo<=ct<=hi and lo<=cl<=hi
            idx=len(candidates);pools[(parent,axis,k)].append(idx)
            candidates.append(dict(index=idx,parent=parent,axis=axis,interval=k,piece=piece,identity=b['case']+':'+kind,kind=kind,
                dimension=audit.d,budget=audit.n,teacher_cut=ct,learned_cut=cl,
                teacher_score=float(audit.formula(np.array([ct]),k)[0][0]),
                learned_cut_teacher_score=float(audit.formula(np.array([cl]),k)[0][0])))
    del source.batches
    assert len(pools)==4356 and stationary_count==370
    for ids in pools.values():shared+=len(ids)-len(set(candidates[i]['teacher_cut'] for i in ids))
    # Same frozen complete geometric score, using fixed interval counts. No
    # physical-coordinate reallocation that could change interval ownership.
    grouped=defaultdict(list)
    for r in candidates:grouped[r['dimension']].append(r)
    for d,rows in grouped.items():
        parents=np.array([np.asarray(source.parents[r['parent']]['upper'])-source.parents[r['parent']]['lower'] for r in rows])
        budgets=np.array([r['budget'] for r in rows]);counts=np.array([[r['interval'],r['budget']-r['interval']] for r in rows])
        for cutfield,outfield in [('learned_cut','learned_score'),('teacher_cut','same_cut_learned_score')]:
            children=np.tile(parents[:,None,:],(1,2,1));ids=np.arange(len(rows));axes=np.array([r['axis'] for r in rows]);cuts=np.array([r[cutfield] for r in rows])
            children[ids,0,axes]=parents[ids,axes]*cuts;children[ids,1,axes]=parents[ids,axes]*(1-cuts)
            assert np.all(children>0) and np.all(counts>=1)
            features=upstream(models,execution,parents,budgets,children,counts);scores=predict(finite,features)
            assert np.all(np.isfinite(scores)&(scores>0))
            for r,s in zip(rows,scores):r[outfield]=float(s)
    comparison_rows=[]
    def compare(current,incoming,scope,case):
        current=np.asarray(current,float);incoming=np.asarray(incoming,float);active=current!=incoming
        replace=np.zeros(current.shape,bool);logits=np.full(current.shape,np.nan)
        if active.any():
            logits[active]=execution.logits(current[active],incoming[active]);replace[active]=logits[active]>0
        # Teacher comparison is audit only, never used for routing.
        comparison_rows.append(dict(scope=scope,case=str(case),comparisons=int(current.size),exact_ties=int((~active).sum()),
            mistakes=int(np.sum(replace!=(incoming<current))),non_tied_zero_logits=int(np.sum(active&(logits==0)))))
        return replace
    def canonical(ids,learned,scope,case,scorefield=None):
        field=scorefield or ('learned_score' if learned else 'teacher_score');cf='learned_cut' if learned else 'teacher_cut'
        order=sorted(ids,key=lambda i:(candidates[i]['axis'],candidates[i][cf],candidates[i]['identity']))
        values=np.array([candidates[i][field] for i in order]);winner=0
        if learned:
            for j in range(1,len(values)):
                if compare(values[winner:winner+1],values[j:j+1],scope,case)[0]:winner=j
            best=float(values[winner])
        else:best=float(values.min())
        bound=best+1e-12*max(1.,best)
        eligible=~compare(values,np.full(len(values),bound),scope+'_equivalence',case) if learned else values<=bound
        assert eligible.any()
        selected=order[int(np.flatnonzero(eligible)[0])]
        return selected,best,order
    oldcomparisons={(r['scope'],r['case']):r for r in csv.DictReader((SOURCE/'interval_parent_comparison.csv').open())}
    intervals=[];pairs=[];teacher_winners=defaultdict(list);learned_winners=defaultdict(list);parent_ids=defaultdict(list);fails=[]
    for key,ids in pools.items():
        ti,tbest,torder=canonical(ids,False,'interval',key);li,lbest,_=canonical(ids,True,'interval',key)
        iso,_,_=canonical(ids,True,'interval_same_coordinates',key,'same_cut_learned_score')
        t,l=candidates[ti],candidates[li];old=oldcomparisons[('allocation_interval',str(key))]
        assert t['identity']==old['teacher_identity'] and t['teacher_cut']==float(old['teacher_cut']) and t['teacher_score']==float(old['teacher_score'])
        tol=1e-12*max(1.,tbest);penalty=l['learned_cut_teacher_score']-tbest;outside=penalty>tol
        # Exact all-pair statistics include shared endpoints, just as the teacher pool does.
        ii,jj=np.triu_indices(len(ids),1);ty=np.array([candidates[i]['teacher_score'] for i in ids]);ly=np.array([candidates[i]['learned_score'] for i in ids])
        same=np.array([candidates[i]['same_cut_learned_score'] for i in ids]);st=np.sign(ty[ii]-ty[jj]);sl=np.sign(ly[ii]-ly[jj]);ss=np.sign(same[ii]-same[jj])
        pairtol=1e-12*np.maximum(1,np.minimum(ty[ii],ty[jj]));gaps=abs(ty[ii]-ty[jj]);rev=(st*sl)<0
        row=dict(parent=key[0],axis=key[1],interval=key[2],candidates=len(ids),unique_teacher_cuts=len(set(candidates[i]['teacher_cut'] for i in ids)),
            teacher_identity=t['identity'],learned_identity=l['identity'],teacher_cut=t['teacher_cut'],learned_cut=l['learned_cut'],
            winning_candidate_exact=t['identity']==l['identity'] and t['teacher_cut']==l['learned_cut'],identity_changed=t['identity']!=l['identity'],
            cut_exact=t['teacher_cut']==l['learned_cut'],coordinate_ulp_gap=bitgap(t['teacher_cut'],l['learned_cut']),
            teacher_best_score=tbest,teacher_selected_score=t['teacher_score'],learned_selected_score=l['learned_score'],
            learned_cut_teacher_score=l['learned_cut_teacher_score'],teacher_score_penalty=max(0.,penalty),outside_equivalence=bool(outside),
            identity_changed_outside_equivalence=bool(outside and t['identity']!=l['identity']),
            teacher_equivalent_coordinate_different=bool(not outside and t['teacher_cut']!=l['learned_cut']),
            exact_teacher_score_tie_pairs=int(np.sum(st==0)),teacher_exact_minimum_count=int(np.sum(ty==tbest)),
            learned_ties_replacing_teacher_strict=int(np.sum((st!=0)&(sl==0))),learned_strict_reversals=int(rev.sum()),
            strict_reversals_outside_equivalence=int(np.sum(rev&(gaps>pairtol))),
            same_coordinate_reversals=int(np.sum((st*ss)<0)),same_coordinate_tie_collapses=int(np.sum((st!=0)&(ss==0))),
            same_coordinate_winner_identity_changed=candidates[iso]['identity']!=t['identity'])
        intervals.append(row)
        for j in np.flatnonzero(rev|((st!=0)&(sl==0))):
            pairs.append(dict(parent=key[0],axis=key[1],interval=key[2],left_identity=candidates[ids[ii[j]]]['identity'],right_identity=candidates[ids[jj[j]]]['identity'],
                teacher_left=ty[ii[j]],teacher_right=ty[jj[j]],learned_left=ly[ii[j]],learned_right=ly[jj[j]],kind='reversal' if rev[j] else 'collapsed_tie',outside_equivalence=bool(gaps[j]>pairtol[j])))
        teacher_winners[key[0]].append(ti);learned_winners[key[0]].append(li);parent_ids[key[0]].extend(ids)
        if outside:fails.extend(dict(r,failure_interval=str(key)) for r in (candidates[i] for i in ids))
    parents=[]
    for parent,ids in parent_ids.items():
        ti,tbest,_=canonical(ids,False,'parent_teacher_full',parent)
        th,_,_=canonical(teacher_winners[parent],False,'parent_teacher_interval_winners',parent)
        li,_,_=canonical(learned_winners[parent],True,'parent_learned_interval_winners',parent)
        lf,_,_=canonical(ids,True,'parent_learned_full_reference',parent)
        t,l=candidates[ti],candidates[li];old=oldcomparisons[('archived_parent',str(parent))]
        assert t['identity']==old['teacher_identity'] and t['teacher_cut']==float(old['teacher_cut'])
        outside=l['learned_cut_teacher_score']>tbest+1e-12*max(1.,tbest)
        parents.append(dict(parent=parent,intervals=len(learned_winners[parent]),teacher_identity=t['identity'],learned_identity=l['identity'],
            teacher_axis=t['axis'],learned_axis=l['axis'],teacher_cut=t['teacher_cut'],learned_cut=l['learned_cut'],
            raw_cut_changed=t['axis']!=l['axis'] or t['teacher_cut']!=l['learned_cut'],identity_changed=t['identity']!=l['identity'],
            canonical_best_cut_changed=t['axis']!=l['axis'] or t['teacher_cut']!=l['learned_cut'],
            teacher_selected_score=t['teacher_score'],learned_cut_teacher_score=l['learned_cut_teacher_score'],
            teacher_score_penalty=max(0.,l['learned_cut_teacher_score']-tbest),outside_equivalence=bool(outside),
            teacher_interval_summary_vs_full_changed=th!=ti,learned_interval_summary_vs_full_changed=lf!=li))
        if outside:fails.extend(dict(candidates[i],failure_interval='parent:'+str(parent)) for i in ids)
    assert len(parents)==106
    preserved_codes=[Path('experiments/geometry/audit_stationary_float64_minimizer.py'),Path('experiments/geometry/audit_stationary_point_impact.py'),
        Path('experiments/sho/audit_fixed_allocation_intervals.py'),Path('experiments/sho/fast_nearest_argmin.py')]
    summary=dict(operator='fixed_allocation_interval_minimizer',status='DIAGNOSTIC_ONLY_STOP',intervals=len(intervals),parents=len(parents),
        candidate_entries=len(candidates),stationary_minimizers_used=stationary_count,subpiece_local_duplicates_removed=local_duplicate_removals,
        cross_subpiece_duplicate_coordinates_retained=shared,
        exact_winning_candidate_matches=sum(r['winning_candidate_exact'] for r in intervals),exact_cut_matches=sum(r['cut_exact'] for r in intervals),
        teacher_equivalent_coordinate_different=sum(r['teacher_equivalent_coordinate_different'] for r in intervals),
        interval_identity_changes=sum(r['identity_changed'] for r in intervals),interval_identity_changes_outside_equivalence=sum(r['identity_changed_outside_equivalence'] for r in intervals),
        interval_score_differences_outside_equivalence=sum(r['outside_equivalence'] for r in intervals),
        exact_teacher_score_tie_pairs=sum(r['exact_teacher_score_tie_pairs'] for r in intervals),intervals_with_exact_teacher_minimum_ties=sum(r['teacher_exact_minimum_count']>1 for r in intervals),
        learned_ties_replacing_teacher_strict=sum(r['learned_ties_replacing_teacher_strict'] for r in intervals),
        learned_strict_reversals=sum(r['learned_strict_reversals'] for r in intervals),strict_reversals_outside_equivalence=sum(r['strict_reversals_outside_equivalence'] for r in intervals),
        max_coordinate_ulp_gap=max(r['coordinate_ulp_gap'] for r in intervals),max_teacher_score_penalty=max(r['teacher_score_penalty'] for r in intervals),
        parent_raw_cut_changes=sum(r['raw_cut_changed'] for r in parents),parent_identity_changes=sum(r['identity_changed'] for r in parents),
        parent_score_changes_outside_equivalence=sum(r['outside_equivalence'] for r in parents),canonical_parent_best_cut_changes=sum(r['canonical_best_cut_changed'] for r in parents),
        teacher_hierarchical_vs_flat_changes=sum(r['teacher_interval_summary_vs_full_changed'] for r in parents),
        learned_hierarchical_vs_flat_changes=sum(r['learned_interval_summary_vs_full_changed'] for r in parents),
        scalar_comparison_mistakes=sum(r['mistakes'] for r in comparison_rows),non_tied_zero_logits=sum(r['non_tied_zero_logits'] for r in comparison_rows),
        scalar_comparisons=sum(r['comparisons'] for r in comparison_rows),margin_weights=execution.w.tolist(),threshold_theta=execution.theta,threshold_alpha=execution.alpha,
        teacher_reconstruction_matches_previous_intervals=len(intervals),teacher_reconstruction_matches_previous_parents=len(parents),
        maximum_score_error=float(max(abs(r['learned_score']-r['learned_cut_teacher_score']) for r in candidates)),
        score_components=score_refs,finite_score_checkpoint=str(fp),finite_score_sha256=ref.sha(fp),split_score_lock_sha256=ref.sha(lockpath),
        stationary_minimizer_manifest_sha256=ref.sha(manifest),implementation_sha256=ref.sha(Path(__file__)),
        teacher_source_hashes={str(p):ref.sha(p) for p in preserved_codes},elapsed_seconds=time.monotonic()-start,
        no_training=True,no_root_search=True,no_recursive_splitting=True,no_global_search=True)
    passed=all(summary[k]==0 for k in ('interval_identity_changes_outside_equivalence','interval_score_differences_outside_equivalence',
        'parent_score_changes_outside_equivalence','canonical_parent_best_cut_changes','scalar_comparison_mistakes',
        'teacher_hierarchical_vs_flat_changes','learned_hierarchical_vs_flat_changes'))
    summary['passed']=passed
    assert all(ref.sha(Path(p))==h for p,h in {**protected,**source.inputs}.items());summary['protected_hashes_unchanged']=True
    ref.write_csv(OUT/'interval_minimum_validation.csv',intervals);ref.write_csv(OUT/'parent_minimum_validation.csv',parents)
    ref.write_csv(OUT/'complete_candidate_sets.csv',candidates);ref.write_csv(OUT/'scalar_comparison_audit.csv',comparison_rows)
    if pairs:ref.write_csv(OUT/'pairwise_score_order_changes.csv',pairs)
    if fails:ref.write_csv(OUT/'meaningful_failure_candidate_sets.csv',fails)
    trace=['TEACHER CANDIDATE/SELECTION AUDIT',
        'Authoritative 4,356-interval baseline: audit_stationary_float64_minimizer.py, with canonical() from audit_stationary_point_impact.py.',
        'The earlier audit_fixed_allocation_intervals.py used np.unique at whole-axis recovery. The later validated subpiece/interval audit retains cross-subpiece duplicate entries; that is the exact baseline reconstructed here.',
        'Bracketed subpiece: left endpoint, saved discrete stationary minimum, right endpoint in that insertion order; identities are case:left/stationary/right.',
        'Other subpieces: sorted(set([left,right]+in-range saved stationary/quartic roots)); identities case:saved{j}. No new root solving.',
        'Subpiece entries are concatenated in archived source order. Shared endpoints remain separate entries. No new interval-level deduplication.',
        'Canonical candidate order: (axis ascending, cut ascending, identity lexicographic). Exact duplicate score/cut ties retain first identity in this order.',
        'Find strict minimum score best, then accept score <= best + 1e-12*max(1,best); select first accepted entry in canonical order.',
        'Learned execution uses existing FastNearestArgmin.logits to find the strict minimum AND compare each score with the equivalence bound. Exact equal input scores retain the existing first-entry bookkeeping.',
        'Teacher/reference comparisons are audit labels only; they do not supply learned selection decisions.',
        'Fixed-count learned scores: complete frozen q/h/radius/aspect/final-sum chain, with child sides parent_sides except axis scaled by c and 1-c. Counts stay [k,n-k]; allocator is not rerun at float boundaries.',
        'Teacher scores preserve Audit.formula execution order. This fixed-count context matches the saved interval audit; physical-coordinate allocation is not substituted.',
        'Parent assembly receives only interval canonical winners. Both teacher and learned hierarchical selections are checked against full original parent candidate pools to detect non-associativity of tolerance selection.',
        f"Reconstructed previous teacher decisions exactly: {len(intervals)} intervals and {len(parents)} parents."]
    (OUT/'teacher_candidate_selection_audit.txt').write_text('\n'.join(trace)+'\n')
    if passed:
        summary['status']='VALIDATED_UNDER_TEACHER_EQUIVALENCE';dest=OUT/'fixed_allocation_interval_minimizer_validated.json'
        dest.write_text(json.dumps(summary,indent=2));digest=ref.sha(dest);dest.with_suffix('.sha256').write_text(digest+'  '+dest.name+'\n')
        summary['assembly_manifest']=str(dest);summary['assembly_sha256']=digest
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'fixed_allocation_interval_minimizer_summary.txt').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
