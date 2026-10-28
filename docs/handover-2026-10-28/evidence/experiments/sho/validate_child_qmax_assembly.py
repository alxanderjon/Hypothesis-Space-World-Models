"""Read-only assembled q_max validation; frozen q and existing learned max.

The radius division below is ONLY a teacher-held-fixed ranking attribution.
No quotient network, training, recursion, or search is introduced.
"""
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sho.train_child_normalized_diameter_smoke import ChildNormalizedDiameter,target
from experiments.sho.train_characteristic_spacing_smoke import inputs,metrics,write_csv,sha
from experiments.sho.fast_nearest_argmin import FastNearestArgmin
from experiments.sho.run_stage1b_nearest_threshold import NonzeroThreshold
from experiments.sho.train_geometry_allocation_curriculum import GeometryAllocator,feature_tensor
from experiments.sho.hierarchical_allocation_poc import conserve
from experiments.sho.audit_split_geometry_score import reconstruct

ROOT=Path('outputs/sho/stage1B');OUT=ROOT/'child_qmax_assembly'
QPATH=ROOT/'child_normalized_diameter_smoke/child_normalized_diameter_before_evaluation.pt'
OPERATOR=ROOT/'nearest_argmin_assembled/nearest_argmin_operator_locked.pt'
ALLOCATION=ROOT/'geometry_allocation_curriculum/stage_4_best.pt'
PRECISION=256*np.finfo(np.float64).eps


def bucket(n):return '1' if n==1 else '2-5' if n<=5 else '6-10' if n<=10 else '11-20' if n<=20 else '>20'


def predict_q(model,lengths,counts,d):
    log=np.log(lengths.reshape(-1,d));n=np.asarray(counts).reshape(-1)
    result=[]
    with torch.no_grad():
        for start in range(0,len(n),32768):
            z=torch.from_numpy(log[start:start+32768]);ln=torch.from_numpy(np.log(n[start:start+32768]))
            result.append(model(z,torch.ones_like(z,dtype=torch.bool),ln,torch.full_like(ln,float(d))).numpy())
    return np.concatenate(result).reshape(np.shape(counts))


def reduce_max(execution,values):
    """Existing larger() controls every replacement; equal values keep first."""
    current=values[:,0].copy();winner=np.zeros(len(values),dtype=int)
    stat=dict(comparisons=0,exact_input_ties=0,comparison_mistakes=0,non_tied_zero_logits=0)
    for child in range(1,values.shape[1]):
        incoming=values[:,child]
        replace=execution.larger(current,incoming)
        # Audit only: none of these reference checks supplies a decision.
        active=current!=incoming
        stat['comparisons']+=len(values);stat['exact_input_ties']+=int((~active).sum())
        stat['comparison_mistakes']+=int(np.sum(replace!=(incoming>current)))
        if active.any():stat['non_tied_zero_logits']+=int(np.sum(execution.logits(current[active],incoming[active])==0))
        current=np.where(replace,incoming,current);winner=np.where(replace,child,winner)
    return current,winner,stat


def row_batch(execution,truth,predicted,source,dimension,budgets,ids,rows,child_counts=None):
    actual,owner,stat=reduce_max(execution,predicted)
    isolated,isolated_owner,isolated_stat=reduce_max(execution,truth)
    # The exact teacher is an audit label only.
    reference=truth.max(1);teacher_owner=truth.argmax(1)
    counts=np.sum(truth==reference[:,None],axis=1)
    chosen_truth=truth[np.arange(len(truth)),owner]
    sorted_q=np.sort(truth,axis=1)
    top_gap=sorted_q[:,-1]-sorted_q[:,-2] if truth.shape[1]>1 else np.zeros(len(truth))
    bits=sorted_q.view(np.uint64)
    top_ulp=bits[:,-1]-bits[:,-2] if truth.shape[1]>1 else np.zeros(len(truth),dtype=np.uint64)
    loss=(reference-chosen_truth)/reference
    for i in range(len(truth)):
        rows.append(dict(source=source,case=str(ids[i]),dimension=dimension,budget=int(budgets[i]),
            budget_bucket=bucket(int(budgets[i])),children=truth.shape[1],teacher_qmax=float(reference[i]),
            learned_qmax=float(actual[i]),absolute_error=float(actual[i]-reference[i]),
            relative_error=float((actual[i]-reference[i])/reference[i]),teacher_owner=int(teacher_owner[i]),learned_owner=int(owner[i]),
            owner_changed=bool(owner[i]!=teacher_owner[i]),teacher_optimal_child=bool(chosen_truth[i]==reference[i]),
            meaningful_owner_change=bool(loss[i]>PRECISION),selected_teacher_relative_deficit=float(loss[i]),
            exact_teacher_max_tie=bool(counts[i]>1),teacher_top_gap=float(top_gap[i]),teacher_top_ulp_gap=int(top_ulp[i]),
            one_ulp_unique_top=bool(top_ulp[i]==1 and counts[i]==1),
            near_tie_unique_top=bool(counts[i]==1 and top_gap[i]/reference[i]<=PRECISION),
            isolated_max_exact_value=bool(isolated[i]==reference[i]),isolated_max_exact_owner=bool(isolated_owner[i]==teacher_owner[i]),
            minimum_child_count=int(child_counts[i].min()) if child_counts is not None else '',
            maximum_child_count=int(child_counts[i].max()) if child_counts is not None else '',
            winning_child_count=int(child_counts[i,owner[i]]) if child_counts is not None else ''))
    return actual,owner,dict(actual=stat,isolated=isolated_stat)


def ranking_changes(old,new,label,source,parent,cuts,axes,change_rows):
    finite=np.isfinite(old);ids=np.flatnonzero(finite);a=old[ids];b=new[ids]
    assert np.array_equal(np.isfinite(old),np.isfinite(new))
    old_order=np.argsort(old,kind='stable');new_order=np.argsort(new,kind='stable')
    threshold=a.min()+1e-12*max(1,float(a.min()))
    new_threshold=b.min()+1e-12*max(1,float(b.min()))
    old_equiv=a<=threshold;new_equiv=b<=new_threshold
    canonical=ids[np.flatnonzero(old_equiv)[0]];new_canonical=ids[np.flatnonzero(new_equiv)[0]]
    ii,jj=np.triu_indices(len(ids),1);sign_a=np.sign(a[ii]-a[jj]);sign_b=np.sign(b[ii]-b[jj])
    reversal=(sign_a*sign_b)<0;collapse=(sign_a!=0)&(sign_b==0);tie_break=(sign_a==0)&(sign_b!=0)
    gap=abs(a[ii]-a[jj]);eqtol=1e-12*np.maximum(1,np.minimum(a[ii],a[jj]))
    significant=reversal&(gap>eqtol)
    for pair in np.flatnonzero(reversal|collapse|tie_break):
        i,j=ids[ii[pair]],ids[jj[pair]]
        change_rows.append(dict(component=label,source=source,parent=parent,
            candidate_i=int(i),candidate_j=int(j),axis_i=int(axes[i]),axis_j=int(axes[j]),
            cut_i=float(cuts[i]),cut_j=float(cuts[j]),teacher_i=float(old[i]),teacher_j=float(old[j]),
            substituted_i=float(new[i]),substituted_j=float(new[j]),teacher_gap=float(gap[pair]),
            strict_reversal=bool(reversal[pair]),strict_order_collapsed=bool(collapse[pair]),exact_tie_broken=bool(tie_break[pair]),
            within_existing_teacher_equivalence=bool(gap[pair]<=eqtol[pair])))
    return dict(component=label,source=source,parent=parent,ranking_order_changed=bool(np.any(old_order!=new_order)),
        raw_best_changed=bool(old_order[0]!=new_order[0]),canonical_best_changed=bool(canonical!=new_canonical),
        best_equivalence_mask_changed=bool(np.any(old_equiv!=new_equiv)),strict_reversals=int(reversal.sum()),
        strict_order_collapses=int(collapse.sum()),exact_ties_broken=int(tie_break.sum()),
        reversals_outside_existing_tolerance=int(significant.sum()))


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'child_normalized_diameter_smoke/candidate_manifest.json').read_text())
    assert sha(QPATH)==manifest['sha256']
    qmodel=ChildNormalizedDiameter();qmodel.load_state_dict(torch.load(QPATH,weights_only=False)['state_dict']);qmodel.eval().requires_grad_(False)
    op=torch.load(OPERATOR,weights_only=False);assert op['status']=='LOCKED'
    for name in ('margin','threshold'):assert sha(op[name+'_checkpoint'])==op[name+'_sha256']
    margin=nn.Linear(2,1,bias=False,dtype=torch.float64);margin.load_state_dict(torch.load(op['margin_checkpoint'],weights_only=False)['state_dict'])
    threshold=NonzeroThreshold();threshold.load_state_dict(torch.load(op['threshold_checkpoint'],weights_only=False)['state_dict'])
    margin.eval().requires_grad_(False);threshold.eval().requires_grad_(False)
    execution=FastNearestArgmin(None,margin,threshold,None)
    paths=sorted((ROOT/'recursive_region_splitter_poc').glob('*.npz'))+sorted((ROOT/'split_quality_smoke').glob('*.npz'))
    protected={str(p):sha(p) for p in paths+[QPATH,OPERATOR,ALLOCATION,Path(op['margin_checkpoint']),Path(op['threshold_checkpoint']),
        Path('experiments/sho/fast_nearest_argmin.py'),Path('experiments/sho/train_child_normalized_diameter_smoke.py'),
        ROOT/'characteristic_spacing_smoke/characteristic_spacing_before_evaluation.pt']}
    details=[];rank_rows=[];changes=[];stats=[];invalid=0;archive_count=0;fixture_invalid=0
    for path in paths:
        a=np.load(path);legacy='scores' in a.files;source=path.parent.name+'/'+path.stem
        if legacy:
            k=14;pi=np.repeat(np.arange(len(a['budget'])),k)
            axes=np.tile(np.repeat(np.arange(2),7),len(a['budget']))
            cuts=np.tile(np.array([.2,.3,.4,.5,.6,.7,.8]*2),len(a['budget']))
            saved=a['scores'].reshape(-1);allocation=a['counts'].reshape(-1,2)
        else:
            pi=a['parent_index'];axes=a['axis'][pi];cuts=a['cut'];saved=a['y'];allocation=a['counts'];k=len(pi)//len(a['budget'])
        score,valid,values,diagnostics,length=reconstruct(a['lower'],a['upper'],a['budget'],pi,axes,cuts,allocation)
        assert np.array_equal(score.view(np.uint64),saved.view(np.uint64))
        truth=values['child_diameter_over_sqrt_count'] # EXACT audited 2D q evaluation
        prediction=predict_q(qmodel,length[valid],allocation[valid],2)
        actual,owners,st=row_batch(execution,truth,prediction,source,2,a['budget'][pi[valid]],np.flatnonzero(valid),details,allocation[valid])
        stats.append(dict(source=source,**st));invalid+=int((~valid).sum());archive_count+=len(score)
        old_q=np.full(len(score),np.inf);new_q=old_q.copy();new_score=score.copy()
        old_q[valid]=truth.max(1);new_q[valid]=actual
        # Diagnostic attribution only: teacher h and aspect penalty remain exact.
        new_score[valid]=actual/values['parent_spacing']+values['weighted_aspect_penalty']
        for parent in range(len(a['budget'])):
            sl=slice(parent*k,(parent+1)*k)
            for label,old,new in [('qmax_only',old_q,new_q),('teacher_score_qmax_substitution',score,new_score)]:
                rank_rows.append(ranking_changes(old[sl],new[sl],label,source,parent,cuts[sl],axes[sl],changes))
        print(json.dumps(dict(event='archived_pool',source=source,candidates=len(score))),flush=True)
    assert archive_count==117511
    # Existing multi-child partition geometries in 1D-3D, plus explicit test-only
    # extrusion of archived 1D partitions into 4D/5D (no new learned mechanism).
    raw=json.loads((ROOT/'geometry_allocation_curriculum/stage_4_partitions.json').read_text())['heldout']
    records=[]
    for r in raw:
        rr={k:np.array(v) if k in ('lower','upper','parent_lower','parent_upper','target') else v for k,v in r.items()}
        rr['fixture']='archived_partition';records.append(rr)
    saved_boxes=np.load(ROOT/'child_normalized_diameter_smoke/heldout_boxes.npz')
    for d in (4,5):
        box_ids=np.flatnonzero(saved_boxes['d']==d)
        for i,r in enumerate([r for r in raw if r['dimension']==1]):
            extent=saved_boxes['sides'][box_ids[i%len(box_ids)],:d]
            lo=np.zeros((len(r['target']),d));hi=np.broadcast_to(extent,(len(r['target']),d)).copy()
            lo[:,0]=np.array(r['lower'])[:,0]*extent[0];hi[:,0]=np.array(r['upper'])[:,0]*extent[0]
            records.append(dict(lower=lo,upper=hi,parent_lower=np.zeros(d),parent_upper=extent,
                target=np.array(r['target']),budget=r['budget'],dimension=d,name=f'extrude{d}_{i}',fixture='archived_partition_extrusion'))
    allocator=GeometryAllocator();allocator.load_state_dict(torch.load(ALLOCATION,weights_only=False)['state_dict']);allocator.eval().requires_grad_(False)
    f,m,t=feature_tensor(records)
    with torch.no_grad():fractions=allocator(f,m).numpy()
    batches=defaultdict(list)
    fixture_archive=[]
    for i,r in enumerate(records):
        d=r['dimension'];k=len(r['target']);counts=conserve(r['budget'],fractions[i,:k])
        assert counts.sum()==r['budget']
        if np.min(counts)<1:fixture_invalid+=1;continue
        lengths=r['upper']-r['lower'];truth=np.array([target(ell,int(n),d) for ell,n in zip(lengths,counts)])
        batches[(r['fixture'],d,k)].append((r['name'],r['budget'],lengths,counts,truth))
    # Geometry stress fixtures: n_child=1, equal boxes and controlled nextafter
    # changes. Reversed child order makes collapsed strict maxima observable.
    for d in range(1,6):
        for scale in (1e-9,.137,1.,3.,1e9):
            a=np.full(d,scale)
            for steps in (0,1,2,8,64):
                b=a.copy()
                for _ in range(steps):b[0]=np.nextafter(b[0],np.inf)
                for reverse in (False,True):
                    ls=np.array([b,a] if reverse else [a,b]);ns=np.ones(2,dtype=int)
                    truth=np.array([target(ell,1,d) for ell in ls])
                    batches[('geometry_tie_ulp_stress',d,2)].append((f'{scale}_{steps}_{reverse}',2,ls,ns,truth))
        # A small-count two-child diagnostic at every parent budget 2..20.
        for n in range(2,21):
            ns=np.array([n//2,n-n//2]);extent=saved_boxes['sides'][np.flatnonzero(saved_boxes['d']==d)[n],:d]
            ls=np.tile(extent,(2,1));ls[:,0]*=ns/n
            truth=np.array([target(ell,int(c),d) for ell,c in zip(ls,ns)])
            batches[('archived_box_small_count_fixture',d,2)].append((f'n{n}',n,ls,ns,truth))
    for (source,d,k),items in batches.items():
        names,budgets,lengths,child_counts,truth=zip(*items)
        ls=np.array(lengths);nc=np.array(child_counts);tr=np.array(truth)
        predicted=predict_q(qmodel,ls,nc,d)
        _,_,st=row_batch(execution,tr,predicted,source,d,np.array(budgets),names,details,nc)
        stats.append(dict(source=f'{source}_d{d}_k{k}',**st))
        for name,n,l,c in zip(names,budgets,lengths,child_counts):
            fixture_archive.append(dict(source=source,case=name,dimension=d,budget=int(n),lengths=l.tolist(),counts=c.tolist()))
    # Isolate learned max precision on exact scalar one-ULP inputs, independently
    # of whether log/exp in the child network can preserve such differences.
    scalar=[]
    for x in (1e-12,.137,1.,3.,1e12):
        for steps in (0,1,2,8,64):
            y=x
            for _ in range(steps):y=np.nextafter(y,np.inf)
            scalar.extend(([x,y],[y,x]))
    scalar=np.array(scalar);_,_,scalar_stat=row_batch(execution,scalar,scalar,'isolated_scalar_max_stress',0,
        np.full(len(scalar),2),np.arange(len(scalar)),details)
    stats.append(dict(source='isolated_scalar_max_stress',**scalar_stat))
    # Summaries preserve raw strict decisions rather than silently treating a
    # changed one-ULP candidate ordering as exact agreement.
    summary=[]
    def summarize(items,**key):
        if not items:return
        tr=np.array([r['teacher_qmax'] for r in items]);pr=np.array([r['learned_qmax'] for r in items])
        summary.append(dict(record_type='qmax_metrics',**key,**metrics(tr,pr),
            owner_changes=sum(r['owner_changed'] for r in items),nonoptimal_teacher_children=sum(not r['teacher_optimal_child'] for r in items),
            meaningful_owner_changes=sum(r['meaningful_owner_change'] for r in items),
            exact_teacher_ties=sum(r['exact_teacher_max_tie'] for r in items),
            one_ulp_unique_tops=sum(r['one_ulp_unique_top'] for r in items),
            isolated_value_failures=sum(not r['isolated_max_exact_value'] for r in items),
            isolated_owner_failures=sum(not r['isolated_max_exact_owner'] for r in items)))
    actual=[r for r in details if r['dimension']>0]
    summarize(actual,source='all',dimension='all',budget_bucket='all')
    for d in range(1,6):
        for b in ('all','2-5','6-10','11-20','>20'):
            summarize([r for r in actual if r['dimension']==d and (b=='all' or r['budget_bucket']==b)],source='all',dimension=d,budget_bucket=b)
    for source in sorted({r['source'] for r in details}):summarize([r for r in details if r['source']==source],source=source,dimension='all',budget_bucket='all')
    ranking_totals={}
    for component in ('qmax_only','teacher_score_qmax_substitution'):
        rr=[r for r in rank_rows if r['component']==component]
        total={k:sum(r[k] for r in rr) for k in ('ranking_order_changed','raw_best_changed','canonical_best_changed','best_equivalence_mask_changed',
            'strict_reversals','strict_order_collapses','exact_ties_broken','reversals_outside_existing_tolerance')}
        ranking_totals[component]=total;summary.append(dict(record_type='candidate_ranking',source=component,parents=len(rr),**total))
    full=summary[0]
    rt=ranking_totals['teacher_score_qmax_substitution']
    passed=(full['max_relative_error']<=PRECISION and full['meaningful_owner_changes']==0
            and rt['ranking_order_changed']==0 and rt['strict_reversals']==0 and rt['strict_order_collapses']==0
            and rt['exact_ties_broken']==0 and rt['canonical_best_changed']==0)
    assert all(sha(p)==h for p,h in protected.items())
    write_csv(OUT/'qmax_assembly_validation.csv',summary)
    write_csv(OUT/'qmax_child_witnesses.csv',details)
    write_csv(OUT/'qmax_candidate_rankings.csv',rank_rows)
    write_csv(OUT/'qmax_ranking_changes.csv',changes)
    (OUT/'validation_fixtures.json').write_text(json.dumps(fixture_archive))
    result=dict(status='VALIDATED_EXISTING_OPERATORS' if passed else 'NOT_VALIDATED',passed=passed,
        metrics=full,ranking_totals=ranking_totals,archived_candidates=archive_count,archived_invalid_splits=invalid,
        additional_invalid_partitions=fixture_invalid,protected_hashes=protected,checkpoints_unchanged=True,
        elapsed_seconds=time.monotonic()-started,comparison_stats=stats,
        precision_relative_threshold=PRECISION,tie_rule='existing larger(): equal learned values keep current/first child',
        quotient_note='audited teacher division ONLY for fixed-other-terms attribution; no learned quotient',
        no_training=True,no_recursion=True,no_search=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if passed:(OUT/'validated_assembly.json').write_text(json.dumps(dict(q_max_assembly='VALIDATED_USING_EXISTING_LEARNED_OPERATORS',**result),indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('protected_hashes','comparison_stats')},indent=2),flush=True)


if __name__=='__main__':run()
