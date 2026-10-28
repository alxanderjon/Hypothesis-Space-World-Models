"""Independent float64 expansion of archived geometry scores; read-only audit."""
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'split_geometry_score_audit'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def reconstruct(lower,upper,budgets,parent_index,axes,cuts,counts):
    """No calls to teacher quality(), children(), learned models, or allocator."""
    lo=lower[parent_index];hi=upper[parent_index];parent_length=hi-lo
    n=budgets[parent_index];m=len(n);idx=np.arange(m)
    offset=cuts*(hi[idx,axes]-lo[idx,axes])
    coordinate=lo[idx,axes]+offset
    cl=np.repeat(lo[:,None,:],2,axis=1);ch=np.repeat(hi[:,None,:],2,axis=1)
    ch[idx,0,axes]=coordinate;cl[idx,1,axes]=coordinate
    length=ch-cl
    valid=counts.min(1)>=1
    v=np.flatnonzero(valid);lv=length[v];cv=counts[v]
    # Match the original ordered ufunc operations, not an algebraic rewrite.
    area=np.multiply.reduce(parent_length[v],axis=1)
    area_per_point=np.array([float(a)/int(k) for a,k in zip(area,n[v])])
    spacing=np.array([math.sqrt(float(a)) for a in area_per_point])
    squares=lv*lv
    squared_diameter=np.add.reduce(squares,axis=2)
    diameter=np.sqrt(squared_diameter)
    sqrt_count=np.sqrt(cv)
    scaled_diameter=diameter/sqrt_count
    worst_scaled=np.maximum.reduce(scaled_diameter,axis=1)
    radius=worst_scaled/spacing
    longest=np.maximum.reduce(lv,axis=2);shortest=np.minimum.reduce(lv,axis=2)
    aspect=longest/shortest
    log_aspect=np.log(aspect)
    worst_log=np.maximum.reduce(log_aspect,axis=1)
    penalty=.05*worst_log
    score=radius+penalty
    full=np.full(m,np.inf);full[v]=score
    values=dict(parent_side=parent_length[v],cut_fraction=cuts[v],cut_offset=offset[v],
                cut_coordinate=coordinate[v],child_side=lv,parent_area=area,
                parent_area_per_point=area_per_point,parent_spacing=spacing,
                child_side_squared=squares,child_diameter_squared=squared_diameter,
                child_diameter=diameter,child_count=cv,child_sqrt_count=sqrt_count,
                child_diameter_over_sqrt_count=scaled_diameter,worst_child_diameter_over_sqrt_count=worst_scaled,
                radius_term=radius,child_longest_side=longest,child_shortest_side=shortest,
                child_aspect=aspect,child_log_aspect=log_aspect,worst_child_log_aspect=worst_log,
                weighted_aspect_penalty=penalty,final_score=score)
    diagnostics=dict(parent_aspect=parent_length[v].max(1)/parent_length[v].min(1),
        minimum_mass_fraction=cv.min(1)/n[v],count_imbalance=cv.max(1)/cv.min(1),
        max_child_aspect=aspect.max(1),radius_witness=scaled_diameter.argmax(1),aspect_witness=log_aspect.argmax(1),
        combined_child_max=np.max(scaled_diameter/spacing[:,None]+.05*log_aspect,axis=1),
        child_volume=np.multiply.reduce(lv,axis=2))
    diagnostics['child_volume_per_point']=diagnostics['child_volume']/cv
    diagnostics['separate_maxima_gap']=score-diagnostics['combined_child_max']
    return full,valid,values,diagnostics,length


def run():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    paths=sorted((ROOT/'recursive_region_splitter_poc').glob('*.npz'))+sorted((ROOT/'split_quality_smoke').glob('*.npz'))
    protected={str(p):sha(p) for p in paths}
    for folder in ('recursive_region_splitter_poc','split_quality_smoke','geometry_allocation_curriculum'):
        for p in (ROOT/folder).glob('*.pt'):protected[str(p)]=sha(p)
    archived=list(csv.DictReader((ROOT/'split_quality_smoke/all_parent_rankings.csv').open()))
    ranks={(r['group'],int(r['parent'])):r for r in archived}
    learned_candidates=list(csv.DictReader((ROOT/'split_quality_smoke/split_quality_small_budget.csv').open()))
    pred={(r['group'],int(r['parent']),float(r['cut'])):float(r['learned_score'])
          for r in learned_candidates if r['record_type']=='candidate' and r['valid']=='True'}
    distribution=[];examples=[];pools=[];global_values=defaultdict(list);small_values=defaultdict(list)
    counts=defaultdict(int);budget_stats=defaultdict(lambda:defaultdict(int));cluster=[];example_seen=set();failure_rows=[]
    for path in paths:
        a=np.load(path);legacy='scores' in a.files
        dataset='original_splitter' if legacy else 'continuous_scorer'
        group=path.stem.replace('heldout_','')
        if legacy:
            k=14;parent_index=np.repeat(np.arange(len(a['budget'])),k)
            axes=np.tile(np.repeat(np.arange(2),7),len(a['budget']))
            cuts=np.tile(np.array([.2,.3,.4,.5,.6,.7,.8]*2),len(a['budget']))
            target=a['scores'].reshape(-1);allocation=a['counts'].reshape(-1,2)
        else:
            parent_index=a['parent_index'];axes=a['axis'][parent_index];cuts=a['cut']
            target=a['y'];allocation=a['counts'];k=len(cuts)//len(a['budget'])
        score,valid,values,diagnostics,length=reconstruct(a['lower'],a['upper'],a['budget'],parent_index,axes,cuts,allocation)
        bits_equal=score.view(np.uint64)==target.view(np.uint64)
        assert bits_equal.all(),(str(path),np.flatnonzero(~bits_equal)[:10].tolist())
        finite_indices=np.flatnonzero(valid);finite_position=np.full(len(score),-1);finite_position[finite_indices]=np.arange(valid.sum())
        n_candidates=a['budget'][parent_index]
        counts['candidates']+=len(score);counts['parents']+=len(a['budget']);counts['finite']+=int(valid.sum())
        counts['invalid']+=int((~valid).sum());counts['bit_exact_scores']+=len(score)
        for name,arr in values.items():
            global_values[name].append(arr.reshape(-1));small_values[name].append(arr[n_candidates[valid]<=20].reshape(-1))
            distribution.append(dict(record_type='quantity_range',dataset=dataset,pool=group,quantity=name,
                samples=arr.size,minimum=float(arr.min()),maximum=float(arr.max()),mean=float(arr.mean())))
        for name,arr in diagnostics.items():
            global_values['diagnostic_only_'+name].append(arr.reshape(-1))
        witness_different=diagnostics['radius_witness']!=diagnostics['aspect_witness']
        counts['different_max_witnesses']+=int(witness_different.sum())
        counts['strict_separate_maxima_effect']+=int((diagnostics['separate_maxima_gap']>1e-12).sum())
        poolcounts=defaultdict(int);gap_lists=defaultdict(list)

        def example(ci,reason,extras=None):
            pos=finite_position[ci];parent=int(parent_index[ci])
            row=dict(record_type='calculation',dataset=dataset,pool=group,parent=parent,reason=reason,
                n=int(n_candidates[ci]),axis=int(axes[ci]),cut=float(cuts[ci]),valid=bool(valid[ci]),
                count_0=int(allocation[ci,0]),count_1=int(allocation[ci,1]),score=float(score[ci]),
                score_hex=float(score[ci]).hex(),parent_lower=json.dumps(a['lower'][parent].tolist()),
                parent_upper=json.dumps(a['upper'][parent].tolist()))
            if pos>=0:
                for name,arr in values.items():
                    v=arr[pos]
                    if np.ndim(v)==0:row[name]=float(v)
                    else:
                        for ix in np.ndindex(v.shape):row[name+'_'+ '_'.join(map(str,ix))]=float(v[ix])
                for name,arr in diagnostics.items():
                    v=arr[pos]
                    if np.ndim(v)==0:row['diagnostic_'+name]=float(v)
                row['learned_score']=pred.get((group,parent,float(cuts[ci])), '')
            if extras:row.update(extras)
            examples.append(row)

        for parent,n in enumerate(a['budget']):
            start=parent*k;stop=start+k;ids=np.arange(start,stop);fi=ids[valid[ids]]
            y=score[fi];saved=target[fi]
            order=np.argsort(y,kind='stable');saved_order=np.argsort(saved,kind='stable')
            assert np.array_equal(order,saved_order)
            best=float(y.min());tolerance=1e-12*max(1,best)
            equiv=y<=best+tolerance;canonical=fi[np.flatnonzero(equiv)[0]]
            assert np.array_equal(equiv,saved<=saved.min()+1e-12*max(1,float(saved.min())))
            poolcounts['exact_rank_reconstruction']+=1;counts['exact_rank_reconstruction']+=1
            if canonical!=fi[order[0]]:poolcounts['canonical_differs_from_raw_argmin']+=1
            if legacy:
                assert canonical-start==int(a['axis'][parent])*7+int(a['cut'][parent])
                assert np.array_equal(score[ids]<=best+tolerance,a['equivalent'][parent])
                counts['stored_best_indices_checked']+=1
            elif (group,parent) in ranks:
                assert float(cuts[canonical])==float(ranks[group,parent]['teacher_cut'])
                counts['stored_best_indices_checked']+=1
            gap_lists['best_second_gap'].append(float(y[order[1]]-best))
            positive=y[y>best]-best
            if len(positive):gap_lists['best_next_positive_gap'].append(float(positive.min()))
            strict=y[~equiv]-best
            if len(strict):gap_lists['best_next_outside_tolerance_gap'].append(float(strict.min()))
            ii,jj=np.triu_indices(len(y),1)
            gaps=abs(y[ii]-y[jj]);lower=np.minimum(y[ii],y[jj]);relative_tolerance=1e-12*np.maximum(1,lower)
            bitdistance=np.maximum(y[ii].view(np.uint64),y[jj].view(np.uint64))-np.minimum(y[ii].view(np.uint64),y[jj].view(np.uint64))
            pairstats=dict(finite_score_pairs=len(gaps),exact_tie_pairs=int((gaps==0).sum()),
                one_ulp_pairs=int((bitdistance==1).sum()),positive_pairs_within_existing_tolerance=int(((gaps>0)&(gaps<=relative_tolerance)).sum()),
                positive_pairs_within_1e_minus_6_relative=int(((gaps>0)&(gaps<=1e-6*np.maximum(1,lower))).sum()),
                parents_with_exact_best_ties=int(np.sum(y==best)>1),parents_with_multiple_equivalent_best=int(equiv.sum()>1))
            for name,v in pairstats.items():poolcounts[name]+=v;counts[name]+=v
            if n<=20:
                bs=budget_stats[int(n)];bs['parents']+=1;bs['candidates']+=k;bs['invalid_candidates']+=int((~valid[ids]).sum())
                for axis in np.unique(axes[ids]):
                    aa=ids[axes[ids]==axis]
                    for prev,cur in zip(aa[:-1],aa[1:]):
                        bs['adjacent_cut_pairs']+=1
                        if np.any(allocation[prev]!=allocation[cur]):bs['allocation_changes']+=1
                if int(n) not in example_seen:
                    for ci in ids:example(ci,'representative_small_budget_parent')
                    example_seen.add(int(n))
            if not legacy and group=='previous_n4_failure':
                for ci in ids:example(ci,'exact_archived_n4_failure')
            if not legacy and (group,parent) in ranks and group!='previous_n4_failure':
                r=ranks[group,parent];wrong=r['teacher_best_agreement']=='False'
                pos=finite_position[canonical]
                entry=dict(dataset=dataset,pool=group,parent=parent,n=int(n),wrong=wrong,
                    strict_margin=float(strict.min()) if len(strict) else np.nan,
                    radius_term=float(values['radius_term'][pos]),aspect_penalty=float(values['weighted_aspect_penalty'][pos]),
                    parent_aspect=float(diagnostics['parent_aspect'][pos]),max_child_aspect=float(diagnostics['max_child_aspect'][pos]),
                    minimum_mass_fraction=float(diagnostics['minimum_mass_fraction'][pos]),
                    count_imbalance=float(diagnostics['count_imbalance'][pos]),
                    different_max_witness=bool(witness_different[pos]))
                cluster.append(entry)
                if wrong:
                    chosen=ids[np.flatnonzero(cuts[ids]==float(r['learned_cut']))[0]]
                    example(canonical,'strict_error_teacher_winner',entry)
                    example(chosen,'strict_error_learned_winner',dict(relative_penalty=float(r['relative_penalty']),strict_margin=entry['strict_margin']))
                    failure_rows.append(dict(**entry,teacher_cut=float(cuts[canonical]),learned_cut=float(cuts[chosen]),
                        teacher_score=float(score[canonical]),chosen_score=float(score[chosen]),
                        learned_teacher_score=pred[(group,parent,float(cuts[canonical]))],
                        learned_chosen_score=pred[(group,parent,float(cuts[chosen]))]))
        for name,vals in gap_lists.items():
            distribution.append(dict(record_type='score_gap',dataset=dataset,pool=group,quantity=name,samples=len(vals),
                minimum=float(np.min(vals)),median=float(np.median(vals)),maximum=float(np.max(vals))))
        row=dict(record_type='pool_verification',dataset=dataset,pool=group,parents=len(a['budget']),candidates=len(score),
                 finite=int(valid.sum()),invalid=int((~valid).sum()),bit_exact_scores=int(bits_equal.sum()),**poolcounts)
        distribution.append(row);pools.append(row)
        print(json.dumps(dict(event='verified',pool=str(path),candidates=len(score))),flush=True)
    for scope,container in [('all_candidates',global_values),('budgets_2_to_20',small_values)]:
        for name,parts in container.items():
            z=np.concatenate(parts);z=z[np.isfinite(z)]
            distribution.append(dict(record_type='quantity_range',dataset='combined',pool=scope,quantity=name,samples=len(z),
                minimum=float(z.min()),maximum=float(z.max()),mean=float(z.mean()),median=float(np.median(z)),p95=float(np.quantile(z,.95))))
    for n,stats in sorted(budget_stats.items()):distribution.append(dict(record_type='small_budget_allocation',n=n,**stats))
    small_cluster=[r for r in cluster if r['n']<=20]
    for n in range(2,21):
        items=[r for r in small_cluster if r['n']==n]
        if items:distribution.append(dict(record_type='scorer_error_by_budget',n=n,parents=len(items),wrong=sum(r['wrong'] for r in items)))
    for name in ('strict_margin','radius_term','aspect_penalty','parent_aspect','max_child_aspect','minimum_mass_fraction','count_imbalance'):
        v=np.array([r[name] for r in small_cluster]);cuts=np.unique(np.quantile(v[np.isfinite(v)],[0,.25,.5,.75,1]))
        for bin_id,(left,right) in enumerate(zip(cuts[:-1],cuts[1:])):
            items=[r for r in small_cluster if r[name]>=left and (r[name]<right or (bin_id==len(cuts)-2 and r[name]<=right))]
            distribution.append(dict(record_type='scorer_error_feature_bin',quantity=name,bin=bin_id,lower=left,upper=right,
                parents=len(items),wrong=sum(r['wrong'] for r in items),error_rate=sum(r['wrong'] for r in items)/len(items) if items else 0.))
        for wrong in (False,True):
            vals=[r[name] for r in small_cluster if r['wrong']==wrong and np.isfinite(r[name])]
            distribution.append(dict(record_type='scorer_error_feature_range',quantity=name,wrong=wrong,parents=len(vals),
                minimum=float(np.min(vals)),median=float(np.median(vals)),maximum=float(np.max(vals))))
    for r in failure_rows:distribution.append(dict(record_type='strict_scorer_failure',**r))
    assert all(sha(p)==h for p,h in protected.items())
    distribution.append(dict(record_type='audit_totals',**counts))
    write_csv(OUT/'split_geometry_score_examples.csv',examples)
    write_csv(OUT/'split_geometry_score_distribution.csv',distribution)
    result=dict(counts=dict(counts),pools=pools,failures=failure_rows,protected_hashes=protected,
                artifacts_unchanged=True,seconds=time.monotonic()-started)
    (OUT/'audit_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='finished',seconds=result['seconds'],counts=dict(counts))),flush=True)


if __name__=='__main__':run()
