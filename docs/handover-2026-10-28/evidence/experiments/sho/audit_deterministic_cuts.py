"""Read-only cut-menu, allocation landscape, canonical selection and thinning audit."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv
from experiments.sho.train_recursive_region_splitter_poc import LockedAllocation,CUTS as LEGACY_CUTS
from experiments.sho.train_finite_split_score_smoke import load_frozen,FiniteSplitScore,upstream,predict
from experiments.sho.train_split_validity_smoke import ValidityMargin,ValidityThreshold,route

ROOT=Path('outputs/sho/stage1B');SOURCE=ROOT/'learned_split_proposer_smoke';OUT=ROOT/'deterministic_cut_audit'
CUTS=np.arange(1,200,dtype=np.float64)/200;K=len(CUTS)


def canonical(scores):
    best=float(np.min(scores));assert np.isfinite(best)
    return int(np.flatnonzero(scores<=best+1e-12*max(1,best))[0])


def bucket(n):return '2-5' if n<=5 else '6-10' if n<=10 else '11-20' if n<=20 else '>20'


def local_minima(values,tol):
    """Merge adjacent numerical plateaus; invalid intervals separate basins."""
    groups=[];i=0
    while i<len(values):
        if not np.isfinite(values[i]):i+=1;continue
        j=i+1
        while j<len(values) and np.isfinite(values[j]) and abs(values[j]-values[i])<=tol:j+=1
        left=values[i-1] if i else np.inf;right=values[j] if j<len(values) else np.inf
        val=float(np.min(values[i:j]))
        if val<=left+tol and val<=right+tol:groups.append((i,j-1))
        i=j
    return groups


def adaptive_indices(scores,beam):
    """Cached-score simulation: decisions read only indices already visited.

    Each axis receives the same coarse start. At step 8,4,2,1, select the best
    beam visited cuts per axis and request their +/-step neighbors. No lookahead
    at unvisited values, allocation transitions, or dense-menu optimum.
    """
    d=len(scores)//K;s=scores.reshape(d,K)
    start=set(range(99%16,K,16))|{0,99,198};visited=[set(start) for _ in range(d)]
    for step in (8,4,2,1):
        for axis in range(d):
            seen=sorted(visited[axis]);good=sorted(seen,key=lambda j:(s[axis,j],j))[:beam]
            add={j+delta for j in good for delta in (-step,step) if 0<=j+delta<K}
            visited[axis].update(add)
    return np.array(sorted(axis*K+j for axis in range(d) for j in visited[axis]),dtype=int)


def menu_indices(name,scores):
    d=len(scores)//K
    if name=='full_199':idx=np.arange(K)
    elif name=='validated_7':idx=np.rint(LEGACY_CUTS*200).astype(int)-1
    elif name.startswith('raw_stride_'):
        stride=int(name.rsplit('_',1)[1]);idx=np.arange(0,K,stride)
    elif name.startswith('centered_stride_'):
        stride=int(name.rsplit('_',1)[1]);idx=np.unique(np.r_[np.arange(99%stride,K,stride),0,198,99])
    elif name.startswith('adaptive_beam_'):return adaptive_indices(scores,int(name.rsplit('_',1)[1]))
    else:raise ValueError(name)
    return np.concatenate([axis*K+idx for axis in range(d)])


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    inputs_files=list(SOURCE.glob('*reference.npz'))+list(SOURCE.glob('*regions.json'))+[SOURCE/'worst_case.json',
        ROOT/'recursive_split_assembly/recursive_trees.json',ROOT/'split_quality_smoke/previous_n4_failure.npz']
    protected={str(p):sha(p) for p in list(ROOT.glob('*/*.pt'))+inputs_files}
    code=['train_recursive_region_splitter_poc.py','validate_recursive_split_assembly.py','train_split_proposer_smoke.py',
          'hierarchical_allocation_poc.py','train_finite_split_score_smoke.py','train_split_validity_smoke.py']
    for name in code:
        path=Path('experiments/sho')/name;protected[str(path)]=sha(path)
    models,execution,_=load_frozen();allocator=LockedAllocation()
    lockpath=ROOT/'split_validity_smoke/split_score_operator_locked.pt'
    lock=torch.load(lockpath,weights_only=False);assert lock['status']=='LOCKED'
    for c in lock['components'].values():assert sha(c['checkpoint'])==c['sha256']
    finite=FiniteSplitScore();finite.load_state_dict(torch.load(lock['components']['finite_score']['checkpoint'],weights_only=False)['state_dict'])
    gate=nn.ModuleDict(dict(margin=ValidityMargin(),threshold=ValidityThreshold()))
    gate.load_state_dict(torch.load(lock['components']['validity']['checkpoint'],weights_only=False)['state_dict'])
    for m in (finite,gate):m.eval().requires_grad_(False)
    records=[];cache=[];cached_equiv=[]
    for group in ('train','heldout'):
        rs=json.loads((SOURCE/(group+'_regions.json')).read_text());a=np.load(SOURCE/(group+'_reference.npz'))
        for i,r in enumerate(rs):records.append(dict(r,source=group,source_case=i,reason=''))
        cache.extend(a['scores']);cached_equiv.extend(a['optimal_set'])
    old=np.load(ROOT/'split_quality_smoke/previous_n4_failure.npz')
    records.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist(),source='previous_n4_failure',source_case=0,reason='previous n=4 failure'))
    n3=json.loads((SOURCE/'worst_case.json').read_text())['result']['case']+1280
    records[n3]['reason']='previous n=3 failure'
    representative=set([n3,len(records)-1])
    for d in range(1,6):
        for n in list(range(2,21))+[29,73,251]:
            match=[i for i,r in enumerate(records) if r['d']==d and r['n']==n and r['source']=='heldout']
            if match:representative.add(match[0])
    scores=np.full((len(records),5*K),np.inf);counts=np.full((len(records),5*K,2),-1,dtype=np.int64)
    branches=np.full((len(records),5*K,6),-1,dtype=np.int64)
    bit_exact=0;finite_total=0
    for d in range(1,6):
        ids=np.array([i for i,r in enumerate(records) if r['d']==d]);p=d*K
        for start in range(0,len(ids),16):
            ix=ids[start:start+16];rs=[records[i] for i in ix];b=len(rs)
            lo=np.repeat(np.array([r['lower'] for r in rs]),p,0);hi=np.repeat(np.array([r['upper'] for r in rs]),p,0)
            n=np.repeat(np.array([r['n'] for r in rs]),p);axes=np.tile(np.repeat(np.arange(d),K),b);cuts=np.tile(CUTS,d*b);ii=np.arange(len(axes))
            cl=np.repeat(lo[:,None,:],2,1);ch=np.repeat(hi[:,None,:],2,1)
            coord=lo[ii,axes]+cuts*(hi[ii,axes]-lo[ii,axes]);ch[ii,0,axes]=coord;cl[ii,1,axes]=coord
            lengths=ch-cl;parent=hi-lo;cc,_,_=allocator.allocate(lengths/parent[:,None,:],n)
            _,_,_,valid=route(execution,gate['margin'],gate['threshold'],cc)
            s=np.full(len(n),np.inf)
            if valid.any():s[valid]=predict(finite,upstream(models,execution,parent[valid],n[valid],lengths[valid],cc[valid]))
            scores[ix,:p]=s.reshape(b,p);counts[ix,:p]=cc.reshape(b,p,2);finite_total+=int(valid.sum())
            # Diagnostic branch identities from the audited geometric quantities;
            # they never choose a candidate or replace frozen model outputs.
            br=np.full((len(n),6),-1,dtype=np.int64);ell=lengths[valid]
            q=np.sqrt(np.add.reduce(ell*ell,axis=2))/np.power(cc[valid].astype(float),1/d)
            logs=np.log(ell.max(2)/ell.min(2))
            br[valid]=np.column_stack((q.argmax(1),logs.argmax(1),ell[:,0].argmax(1),ell[:,0].argmin(1),ell[:,1].argmax(1),ell[:,1].argmin(1)))
            branches[ix,:p]=br.reshape(b,p,6)
            for j,i in enumerate(ix):
                if i<len(cache):
                    assert np.array_equal(scores[i,:p].view(np.uint64),cache[i][:p].view(np.uint64)),(i,'score reconstruction')
                    best=scores[i,:p].min();eq=scores[i]<=best+1e-12*max(1,float(best))
                    assert np.array_equal(eq,cached_equiv[i]);bit_exact+=p
        print(json.dumps(dict(event='dense_menus_reconstructed',dimension=d,parents=len(ids))),flush=True)
    np.savez_compressed(OUT/'audited_dense_menus.npz',scores=scores,counts=counts,branches=branches)
    # Independent canonical reconstruction of both saved recursive traces; no recursion.
    trees=json.loads((ROOT/'recursive_split_assembly/recursive_trees.json').read_text());tree_checks=[]
    for case,tree in trees.items():
        for mode in ('teacher_nodes','learned_nodes'):
            for path,node in tree[mode].items():
                s=np.array(node['candidate_scores']);chosen=canonical(s)
                assert chosen==node['candidate'] and chosen//7==node['axis'] and float(LEGACY_CUTS[chosen%7])==node['cut']
                assert node['all_counts'][chosen]==node['child_counts']
                tree_checks.append(dict(case=case,mode=mode,path=path,exact=True,axis=node['axis'],cut=node['cut']))
    write_csv(OUT/'cut_selection_reconstruction.csv',tree_checks)

    landscapes=[];axis_rows=[];selection_rows=[];thinning=[]
    methods=['full_199','validated_7']+[f'raw_stride_{k}' for k in (2,4,8)]+[f'centered_stride_{k}' for k in (2,4,8)]+[f'adaptive_beam_{k}' for k in (1,2,4)]
    for ri,r in enumerate(records):
        d,n=r['d'],r['n'];s=scores[ri,:d*K];cc=counts[ri,:d*K];choice=canonical(s);best=float(s.min());tol=1e-12*max(1,best)
        eq=np.flatnonzero(s<=best+tol);rev_choice=np.flatnonzero(s[::-1]<=best+tol)[0];reverse_index=len(s)-1-int(rev_choice)
        eqaxes=eq//K;eqcuts=CUTS[eq%K]
        sr=dict(parent=ri,source=r['source'],source_case=r['source_case'],dimension=d,budget=n,budget_bucket=bucket(n),
            canonical_axis=choice//K,canonical_cut=float(CUTS[choice%K]),best_score=best,equivalent_count=len(eq),
            equivalent_axes=len(np.unique(eqaxes)),ordering_changes_witness=bool(choice!=reverse_index),
            mirrored_optimum=bool(any(t%K!=99 and (t//K*K+198-t%K) in eq for t in eq)))
        for name,value in [('quarter',.25),('third',1/3),('half',.5),('two_thirds',2/3),('three_quarters',.75)]:
            sr['canonical_distance_'+name]=float(abs(CUTS[choice%K]-value));sr['nearest_equivalent_distance_'+name]=float(abs(eqcuts-value).min())
        selection_rows.append(sr)
        for method in methods:
            visited=menu_indices(method,s);ss=s[visited];j=canonical(ss);chosen=int(visited[j]);value=float(ss.min())
            thinning.append(dict(record_type='parent',method=method,parent=ri,source=r['source'],dimension=d,budget=n,budget_bucket=bucket(n),
                teacher_equivalent=bool(value<=best+tol),canonical_recovered=bool(chosen==choice),relative_penalty=value/best-1,
                absolute_penalty=value-best,score_requests=len(visited),finite_score_evaluations=int(np.isfinite(ss).sum()),
                dense_score_requests=d*K,dense_finite_evaluations=int(np.isfinite(s).sum()),selected_axis=chosen//K,selected_cut=float(CUTS[chosen%K])))
        for axis in range(d):
            v=s[axis*K:(axis+1)*K];c=cc[axis*K:(axis+1)*K];br=branches[ri,axis*K:(axis+1)*K]
            change=np.r_[False,np.any(c[1:]!=c[:-1],axis=1)];starts=np.r_[0,np.flatnonzero(change)];ends=np.r_[starts[1:],K]
            minima=local_minima(v,tol);locations={j for a,b in minima for j in range(a,b+1)}
            mirror_local=any(j!=99 and 198-j in locations and abs(v[j]-v[198-j])<=tol for j in locations)
            adjacent_valid=np.isfinite(v[1:])&np.isfinite(v[:-1]);same=~change[1:]&adjacent_valid
            switches=np.any(br[1:]!=br[:-1],axis=1)&same
            region_minima=[len(local_minima(v[a:b],tol)) for a,b in zip(starts,ends) if np.isfinite(v[a:b]).any()]
            symvalid=np.isfinite(v)&np.isfinite(v[::-1]);mirrorerr=abs(v[symvalid]-v[::-1][symvalid])
            ar=dict(parent=ri,source=r['source'],dimension=d,budget=n,budget_bucket=bucket(n),axis=axis,
                allocation_regions=len(starts),unique_allocations=len(np.unique(c,axis=0)),valid_allocation_regions=len(region_minima),
                transition_count=int(change.sum()),allocation_monotone=bool(np.all(np.diff(c[:,0])>=0)),invalid_cuts=int((~np.isfinite(v)).sum()),
                local_minima=len(minima),mirrored_local_minimum_pair=mirror_local,within_region_branch_switches=int(switches.sum()),
                max_local_minima_inside_one_allocation_region=max(region_minima,default=0),
                mirrored_score_max_error=float(mirrorerr.max()) if len(mirrorerr) else 0.,
                mirrored_validity_mismatches=int(np.sum(np.isfinite(v)!=np.isfinite(v[::-1]))),
                mirrored_scores_outside_equivalence=int(np.sum(mirrorerr>tol)))
            axis_rows.append(ar)
            if ri in representative:
                allocation_region=np.cumsum(change)
                for j,cut in enumerate(CUTS):
                    landscapes.append(dict(parent=ri,source=r['source'],source_case=r['source_case'],reason=r['reason'],
                        dimension=d,budget=n,axis=axis,cut=float(cut),count_left=int(c[j,0]),count_right=int(c[j,1]),
                        allocation_region=int(allocation_region[j]),allocation_changed=bool(change[j]),
                        transition_bracket_lower=float(CUTS[j-1]) if change[j] else '',transition_bracket_upper=float(cut) if change[j] else '',
                        valid=bool(np.isfinite(v[j])),score=float(v[j]),teacher_equivalent=bool(v[j]<=best+tol),
                        global_canonical=bool(axis*K+j==choice),local_minimum_plateau=bool(j in locations),
                        radius_active_child=int(br[j,0]),aspect_active_child=int(br[j,1]),
                        branch_switch_inside_allocation=bool(j and switches[j-1]),
                        mirrored_score=float(v[198-j]),parent_lower=json.dumps(r['lower']),parent_upper=json.dumps(r['upper'])))
    write_csv(OUT/'cut_score_landscapes.csv',landscapes);write_csv(OUT/'cut_landscape_summary.csv',axis_rows)
    write_csv(OUT/'cut_selection_summary.csv',selection_rows);write_csv(OUT/'cut_menu_thinning_cases.csv',thinning)
    agg=[]
    for method in methods:
        for source in ('all','train','heldout','previous_n4_failure'):
            base=[r for r in thinning if r['method']==method and (source=='all' or r['source']==source)]
            if not base:continue
            for band in ('all','n<=20','2-5','6-10','11-20','>20'):
                rs=[r for r in base if band=='all' or (band=='n<=20' and r['budget']<=20) or r['budget_bucket']==band]
                if not rs:continue
                agg.append(dict(method=method,source=source,budget_bucket=band,parents=len(rs),
                    teacher_equivalent_recovery=float(np.mean([r['teacher_equivalent'] for r in rs])),
                    canonical_recovery=float(np.mean([r['canonical_recovered'] for r in rs])),
                    worst_relative_penalty=max(r['relative_penalty'] for r in rs),mean_relative_penalty=float(np.mean([r['relative_penalty'] for r in rs])),
                    worst_absolute_penalty=max(r['absolute_penalty'] for r in rs),
                    average_score_requests=float(np.mean([r['score_requests'] for r in rs])),max_score_requests=max(r['score_requests'] for r in rs),
                    average_finite_score_evaluations=float(np.mean([r['finite_score_evaluations'] for r in rs])),
                    comparison_reduction=sum(r['dense_score_requests'] for r in rs)/sum(r['score_requests'] for r in rs)))
    write_csv(OUT/'cut_menu_thinning.csv',agg)
    small=[r for r in axis_rows if r['budget']<=20];unchanged=all(sha(p)==h for p,h in protected.items())
    result=dict(dense_parents=len(records),dense_existing_candidates_recomputed_bit_exact=bit_exact,
        new_n4_dense_candidates=2*K,finite_candidates=finite_total,saved_recursive_nodes_checked=len(tree_checks),
        saved_recursive_decisions_exact=True,dense_equivalence_masks_exact=True,representative_parents=len(representative),
        landscape_rows=len(landscapes),small_budget_axes=len(small),
        small_allocation_regions_mean=float(np.mean([r['allocation_regions'] for r in small])),
        small_allocation_regions_range=[min(r['allocation_regions'] for r in small),max(r['allocation_regions'] for r in small)],
        small_local_minima_mean=float(np.mean([r['local_minima'] for r in small])),small_local_minima_max=max(r['local_minima'] for r in small),
        small_mirrored_local_minimum_fraction=float(np.mean([r['mirrored_local_minimum_pair'] for r in small])),
        small_max_minima_inside_allocation_region=max(r['max_local_minima_inside_one_allocation_region'] for r in small),
        axes_with_within_region_branch_switches=sum(r['within_region_branch_switches']>0 for r in axis_rows),
        global_mirrored_optimum_fraction=float(np.mean([r['mirrored_optimum'] for r in selection_rows])),
        order_dependent_witnesses=sum(r['ordering_changes_witness'] for r in selection_rows),
        allocator_stats=allocator.stats(),protected_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-started)
    assert unchanged
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    lines=[
        'Deterministic candidate-cut audit. No training, checkpoint edits, recursion, or search.',
        '', '1. TWO DISTINCT MENUS',
        'Validated recursive scaffold imports CUTS=[.2,.3,.4,.5,.6,.7,.8] from train_recursive_region_splitter_poc.py:24.',
        'For every non-leaf n>1: axes repeat(arange(d),7), cuts tile(CUTS,d). Order: axis 0 ascending cuts, then axis 1, etc.',
        'This is 7d proposals; fixed fractions exclude cuts below .2 and above .8. n==1 stops before proposals.',
        'The later proposer labeling grid (train_split_proposer_smoke.py:19) is arange(1,200,dtype=float64)/200: 199 fixed fractions .005,.010,...,.995 per axis.',
        'Its 199 is not budget-, aspect-, depth-, or allocation-dependent. It was never installed in the validated recursion.',
        'Dimension changes only the number of axes. Parent lengths affect physical coordinates; n/shape affect allocation, validity and score only.',
        'Physical cut coordinate executes lo[axis] + cut*(hi[axis]-lo[axis]). The left upper and right lower coordinates become that same value.',
        '', '2. FILTER AND INTEGER COUNTS',
        'For every proposal compute child lengths / parent lengths; feed their dimension-general geometry features and budget to the locked allocator.',
        'Normalize predicted fractions by their sum; quotas=n*f; floor quotas; give remaining units to descending fractional remainders, tie broken by smaller child index.',
        'There is no geometry pre-pruning or exact-distance search. Learned min -> learned validity margin -> threshold accepts only positive child counts.',
        'Invalid proposals retain +inf at their original index and bypass finite-score arithmetic. They are not renumbered.',
        'Finite proposals run the locked q_i -> q_max, spacing, radius, aspect penalty, final sum chain. No new score formula is used for decisions.',
        'For ideal binary volume allocation, count-left transitions are near (k+1/2)/n. Locked fractions and float64 largest-remainder ties control exact boundary ownership.',
        '', '3. SELECTION',
        'best=min(scores); equivalent=score <= best + 1e-12*max(1,best); choose first equivalent index.',
        'Thus lowest axis then lowest fraction wins a numerical tie, even if another equivalent candidate is the raw floating-point argmin.',
        'No special mirrored-cut branch exists. Mirrored scores are often equivalent; increasing-cut ordering then favors the smaller fraction.',
        'Integer rounding and translated-coordinate roundoff can make mirrored counts/scores differ. The audit tests actual masks and scores rather than assuming symmetry.',
        'The chosen child geometry is reallocated in the audited recursion, and counts must agree with proposal-time allocation.',
        '', '4. LANDSCAPE STRUCTURE',
        'With fixed integer counts, each child normalized diameter is smooth in positive side lengths, and each fixed aspect branch is smooth.',
        'The maxima over children and side-length extrema introduce kinks inside an allocation interval. Therefore the score is continuous/piecewise smooth there, not necessarily differentiable everywhere.',
        'Count transitions change denominators discretely and can introduce jumps. Invalid intervals are +inf. There can be multiple local minima along an axis.',
        'Local minima in this audit merge adjacent score plateaus using the same parent-best 1e-12 scale. Transition locations are sampled brackets of width .005, not claimed exact thresholds.',
        'Branch identities in landscape CSVs are mathematical diagnostics from the audited geometric quantities; they do not override frozen decisions.',
        json.dumps(result,indent=2),
        '', '5. OFFLINE MENU TESTS',
        'raw_stride_k: indices 0,k,2k,...; these all omit index 99 (cut .5).',
        'centered_stride_k: indices congruent to 99 mod k, plus endpoints 0/198 and midpoint. Mirrored support is preserved.',
        'adaptive_beam_b: start centered stride 16 plus endpoints/midpoint on EVERY axis; retain b best visited cuts per axis and evaluate +/-8, +/-4, +/-2, +/-1 index neighbors. Only visited scores choose refinements.',
        'Counts report unique requested candidates and separately finite neural score calls. Invalids still incur proposal/allocation/validity work. This is a cached offline cost simulation, not a timing benchmark.',
        'method | parents | best recovery | canonical recovery | worst relative penalty | avg requests | avg finite scores | reduction']
    for r in agg:
        if r['source']=='all' and r['budget_bucket']=='all':lines.append(f"{r['method']} | {r['parents']} | {r['teacher_equivalent_recovery']:.6g} | {r['canonical_recovery']:.6g} | {r['worst_relative_penalty']:.6g} | {r['average_score_requests']:.6g} | {r['average_finite_score_evaluations']:.6g} | {r['comparison_reduction']:.6g}")
    lines+=['', '6. PREVIOUS FAILURES']
    for ri in (n3,len(records)-1):
        r=records[ri];s=scores[ri,:r['d']*K];j=canonical(s);eq=np.flatnonzero(s<=s.min()+1e-12*max(1,float(s.min())))
        lines.append(json.dumps(dict(reason=r['reason'],dimension=r['d'],budget=r['n'],lower=r['lower'],upper=r['upper'],
            canonical_axis=j//K,canonical_cut=float(CUTS[j%K]),score=float(s[j]),counts=counts[ri,j].tolist(),
            equivalent_splits=[(int(t//K),float(CUTS[t%K])) for t in eq])))
        for c in ([.335,.405,.5,.595,.665] if ri==n3 else [.2,.5]):
            q=int(round(c*200))-1;token=j//K*K+q
            lines.append(f'  axis {j//K}, cut {CUTS[q]:.17g}: counts {counts[ri,token].tolist()}, score {scores[ri,token]:.17g}')
    lines+=['', 'The original scaffold uses a seven-cut heuristic; dense brute-force enumeration is confined to the recent offline proposal-labeling reference.',
        'Allocation intervals, mirrored basins, and within-interval extrema are exploitable structure. Thinning outcomes quantify whether simple local refinement captures it on this finite audit pool; they do not prove global optimality between grid points.',
        'No learned architecture is selected or trained by this audit. Production/scaffold behavior remains unchanged.']
    (OUT/'deterministic_cut_audit.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
