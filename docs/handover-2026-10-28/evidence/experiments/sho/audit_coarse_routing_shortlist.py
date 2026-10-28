"""Count-only uniform-grid ring routing audit; no model or search execution."""
import argparse,csv,json,math,signal,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_fifo_vs_ranker as source

OUT=Path('outputs/sho/stage1B/coarse_routing_shortlist')
KS=(4,8,16,32,64)
OCCUPANCIES=(1,4,16)
REGIONS=('all','interior','boundary_band','exact_boundary','sparse','dense','middle_density')
class TimeLimit(RuntimeError):pass
def timeout(*a):raise TimeLimit('wall-clock safety deadline')
def bins(points,g):return np.minimum(np.floor(points*g).astype(np.int64),g-1)
def integral_grid(sites,g):
    b=bins(sites,g);counts=np.bincount(b[:,0]*g+b[:,1],minlength=g*g).reshape(g,g)
    return np.pad(counts.cumsum(0).cumsum(1),((1,0),(1,0))),b
def count_square(integral,b,r,g):
    lo=np.maximum(b-r[:,None],0);hi=np.minimum(b+r[:,None]+1,g)
    return integral[hi[:,0],hi[:,1]]-integral[lo[:,0],hi[:,1]]-integral[hi[:,0],lo[:,1]]+integral[lo[:,0],lo[:,1]]

def route(sites,queries,k,g,self_query):
    """No distances or labels. Complete Chebyshev rings, own bin = ring zero.

    Binary search of integral-grid counts returns exactly the first qualifying
    ring, avoiding a Python loop through empty rings. Self is excluded from size.
    Membership is every site whose bin lies inside the returned square.
    """
    integral,site_bins=integral_grid(sites,g);qb=bins(queries,g)
    self_count=int(self_query);target=min(k,len(sites)-self_count)
    low=np.full(len(queries),-1,dtype=np.int64);high=np.full(len(queries),g-1,dtype=np.int64)
    while np.any(high-low>1):
        unresolved=high-low>1
        mid=np.maximum((high+low)//2,0)
        enough=count_square(integral,qb,mid,g)-self_count>=target
        high=np.where(unresolved&enough,mid,high);low=np.where(unresolved&~enough,mid,low)
    sizes=count_square(integral,qb,high,g)-self_count
    assert np.all(high>=0) and np.all(sizes>=target)
    expanded=high>0
    assert np.all(count_square(integral,qb[expanded],high[expanded]-1,g)-self_count<target)
    return qb,site_bins,high,sizes

def regions(sites,queries,self_query):
    # Fixed across all tested router resolutions: approx four sites/density bin.
    g=max(1,math.ceil(math.sqrt(len(sites)/4)))
    integral,_=integral_grid(sites,g);qb=bins(queries,g);r=np.ones(len(queries),dtype=np.int64)
    occupancy=count_square(integral,qb,r,g)-int(self_query)
    lo=np.maximum(qb-1,0);hi=np.minimum(qb+2,g)
    area=(hi[:,0]-lo[:,0])*(hi[:,1]-lo[:,1])/g**2
    density=occupancy/area
    q25,q75=np.quantile(density,[.25,.75])
    sparse=density<=q25;dense=(density>=q75)&~sparse
    boundary=np.any((queries<=.05)|(queries>=.95),axis=1)
    exact=np.any((queries==0)|(queries==1),axis=1)
    return dict(all=np.ones(len(queries),bool),interior=~boundary,boundary_band=boundary,
        exact_boundary=exact,sparse=sparse,dense=dense,middle_density=~(sparse|dense)),density,g,float(q25),float(q75)

def summarize(count,miss,ties,sizes,rings,exhaustive):
    n=len(sizes);pairs=int(sizes.sum());comparisons=int(np.maximum(sizes-1,0).sum())
    misses=int(miss.sum());non=~ties
    return dict(query_count=n,miss_count=misses,tie_query_count=int(ties.sum()),tie_misses=int((miss&ties).sum()),
        non_tied_queries=int(non.sum()),non_tied_misses=int((miss&non).sum()),
        shortlist_pairs=pairs,learned_comparisons=comparisons,exhaustive_pairs=exhaustive*n,
        exhaustive_comparisons=max(exhaustive-1,0)*n,shortlist_max=int(sizes.max(initial=0)),
        ring_sum=int(rings.sum()),ring_max=int(rings.max(initial=0)),
        count_only_stopping=True,final_ring_truncated=False)

def finalize(r):
    n=r['query_count'];m=r['miss_count'];non=r['non_tied_queries']
    r.update(recall=1-m/n if n else '',miss_fraction=m/n if n else '',
        non_tied_recall=1-r['non_tied_misses']/non if non else '',
        average_shortlist=r['shortlist_pairs']/n if n else '',average_rings=r['ring_sum']/n if n else '',
        distance_pair_reduction=r['exhaustive_pairs']/r['shortlist_pairs'] if r['shortlist_pairs'] else '',
        comparison_reduction=r['exhaustive_comparisons']/r['learned_comparisons'] if r['learned_comparisons'] else '')
    return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--started',type=float,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.01,args.deadline-time.time()))
    torch.set_num_threads(1)
    protected=json.loads(Path('outputs/sho/stage1B/fast_argmin/fast_execution_manifest.json').read_text())['protected_hashes']
    payload=torch.load(source.cells.base.turn.RAW_TRACE_PATH,weights_only=False)
    trajectories=source.cells.base.ordinary_data._trajectory_map(payload)
    indices=torch.load(source.nearest.OPERATOR,weights_only=False)['held_out_state_indices']
    primary=[17105,5660,17179];indices=primary+[i for i in indices if i not in primary]
    records=[];miss_records=[];aggregates={};completed=[];error=None
    try:
        for state in indices:
            if args.deadline-time.time()<45:raise TimeLimit('stopped before next state to leave reporting time')
            sites,probes,record=source.cells.base.distance_metric._state_geometry(payload,trajectories,state)
            assert np.all((sites>=0)&(sites<=1)) and np.all((probes>=0)&(probes<=1))
            n=len(sites);tree=cKDTree(sites)
            # Teacher calls are audit labels only; none are passed into route().
            probe_d,probe_owner=tree.query(probes,k=1);pd,_=tree.query(probes,k=2)
            sd,si=tree.query(sites,k=3)
            spacing_d,spacing_owner=tree.query(sites,k=2);spacing_d=spacing_d[:,1];spacing_owner=spacing_owner[:,1]
            state_rows=[];state_misses=[]
            size_bin=source.cells.base.turn._state_bin(n)
            for kind,q,owner,td,ties in [('probe',probes,probe_owner,probe_d,pd[:,0]==pd[:,1]),
                    ('spacing',sites,spacing_owner,spacing_d,sd[:,1]==sd[:,2])]:
                self_query=kind=='spacing';masks,density,dg,dq25,dq75=regions(sites,q,self_query)
                for occ in OCCUPANCIES:
                    g=max(1,math.ceil(math.sqrt(n/occ)))
                    for k in KS:
                        t=time.perf_counter();qb,sb,radii,sizes=route(sites,q,k,g,self_query)
                        routing_seconds=time.perf_counter()-t
                        delta=np.abs(qb-sb[owner]);bin_separation=delta.max(1)
                        miss=bin_separation>radii
                        assert np.all(sizes>=min(k,n-int(self_query)))
                        for region in REGIONS:
                            mask=masks[region]
                            row=dict(record_type='state',source_state_index=state,trajectory_id=int(record['trajectory_id']),
                                primary_profile_state=state in primary,site_count=n,probe_count=len(probes),site_count_bin=size_bin,
                                query_kind=kind,target_k=k,grid_target_occupancy=occ,grid_bins_per_axis=g,region=region,
                                density_grid_bins=dg,density_q25=dq25,density_q75=dq75,routing_seconds=routing_seconds,
                                **summarize(int(mask.sum()),miss[mask],ties[mask],sizes[mask],radii[mask],n-int(self_query)))
                            state_rows.append(finalize(row))
                        # Deterministic, explicitly sampled miss inspection: up to 8 per state/config/kind.
                        mi=np.flatnonzero(miss)
                        sample=mi[np.unique(np.linspace(0,len(mi)-1,min(8,len(mi))).astype(int))] if len(mi) else []
                        for j in sample:
                            best=int(owner[j]);state_misses.append(dict(source_state_index=state,site_count=n,site_count_bin=size_bin,
                                query_kind=kind,target_k=k,grid_target_occupancy=occ,grid_bins_per_axis=g,
                                query_index=int(j),query_x=float(q[j,0]),query_y=float(q[j,1]),
                                true_site_index=best,true_site_x=float(sites[best,0]),true_site_y=float(sites[best,1]),
                                teacher_distance=float(td[j]),teacher_exact_tie=bool(ties[j]),
                                query_bin_x=int(qb[j,0]),query_bin_y=int(qb[j,1]),true_bin_x=int(sb[best,0]),true_bin_y=int(sb[best,1]),
                                bin_dx=int(delta[j,0]),bin_dy=int(delta[j,1]),true_site_ring=int(bin_separation[j]),
                                stopping_ring=int(radii[j]),actual_shortlist_size=int(sizes[j]),
                                boundary_band=bool(masks['boundary_band'][j]),sparse=bool(masks['sparse'][j]),dense=bool(masks['dense'][j]),
                                local_density=float(density[j]),total_misses_this_state_config_kind=len(mi),
                                reason='Target count reached in completed square ring; teacher nearest lies in a later ring. Count is not a geometric distance certificate.',
                                sampling='up to 8 evenly spaced missed query indices per state/config/kind; total counts in audit CSV'))
            records.extend(state_rows);miss_records.extend(state_misses);completed.append(state)
            for row in state_rows:
                for scope in ('all_states',size_bin,'primary_three' if state in primary else None):
                    if scope is None:continue
                    for kind in (row['query_kind'],'combined'):
                        key=(scope,kind,row['target_k'],row['grid_target_occupancy'],row['region'])
                        if key not in aggregates:
                            aggregates[key]=dict(record_type='aggregate',scope=scope,query_kind=kind,target_k=row['target_k'],
                                grid_target_occupancy=row['grid_target_occupancy'],region=row['region'],query_count=0,miss_count=0,
                                tie_query_count=0,tie_misses=0,non_tied_queries=0,non_tied_misses=0,shortlist_pairs=0,
                                learned_comparisons=0,exhaustive_pairs=0,exhaustive_comparisons=0,shortlist_max=0,ring_sum=0,ring_max=0)
                        dest=aggregates[key]
                        for name in ('query_count','miss_count','tie_query_count','tie_misses','non_tied_queries','non_tied_misses',
                                     'shortlist_pairs','learned_comparisons','exhaustive_pairs','exhaustive_comparisons','ring_sum'):
                            dest[name]+=row[name]
                        for name in ('shortlist_max','ring_max'):dest[name]=max(dest[name],row[name])
            if len(completed)%10==0 or state in primary:
                print(f'Completed {len(completed)}/{len(indices)} states; sites={n}; elapsed={time.time()-args.started:.1f}s',flush=True)
                source.write_csv(OUT/'coarse_routing_shortlist_audit.csv',records)
                source.write_csv(OUT/'coarse_routing_misses.csv',miss_records or [dict(record_type='no_misses_so_far')])
    except Exception:error=traceback.format_exc();print(error,flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        aggregate_rows=[finalize(r) for r in aggregates.values()]
        source.write_csv(OUT/'coarse_routing_shortlist_audit.csv',aggregate_rows+records)
        source.write_csv(OUT/'coarse_routing_misses.csv',miss_records or [dict(record_type='no_observed_misses')])
        unchanged=all(source.sha(p)==h for p,h in protected.items())
        global_rows=[r for r in aggregate_rows if r['scope']=='all_states' and r['query_kind']=='combined' and r['region']=='all']
        exact=[r for r in global_rows if r['miss_count']==0]
        best=min(global_rows,key=lambda r:(r['miss_fraction'],r['average_shortlist'])) if global_rows else None
        chosen=min(exact,key=lambda r:(r['average_shortlist'],r['target_k'])) if exact else None
        lines=['COARSE SPATIAL ROUTING SHORTLIST AUDIT',
            f'Completed states: {len(completed)}/{len(indices)}; elapsed={time.time()-args.started:.3f}s; stop={error}',
            'No training, checkpoint changes, objective evaluations, learned-router construction, or trajectory search.',
            'The three prior profile states plus all remaining 77 saved held-out nearest-operator states are audited.',
            'Grid g=ceil(sqrt(site_count / target_occupancy)), target occupancy in {1,4,16}. Coordinates equal to 1 enter the last bin.',
            'Start ring=0 (own bin), expand complete Chebyshev square rings; exclude self for spacing; stop at min(k, available sites).',
            'Integral-bin occupancy plus integer radius binary search implements exactly the same complete-ring stopping rule. No distance or teacher-owner information enters route().',
            'Entire final ring retained; actual shortlist can greatly exceed target k. No distance-based or index-based truncation.',
            'Boundary band is within 0.05 of any box face; interior is its complement. Exact boundary is an additional subset.',
            'Sparse/dense use lower/upper query-density quartiles from a fixed per-state g=ceil(sqrt(n/4)) density grid, measuring 3x3-bin site count / clipped bin area, excluding self.',
            'Density labels are independent of router resolution. Sparse takes precedence if quartile thresholds coincide; middle density also reported.',
            'Authoritative cKDTree owners and exact k=2/k=3 distance ties are audit labels only. Primary recall counts production-owner inclusion, including exact ties; non-tied recall is separate.',
            'Each missed-case CSV sample contains coordinates, bins, stopping radius and true-site ring. Up to 8 misses per state/config/kind are sampled; all miss counts remain in the main CSV.',
            '', 'k | occupancy | recall | misses/queries | mean shortlist | max shortlist | mean/max rings | pair reduction | comparison reduction']
        for r in sorted(global_rows,key=lambda r:(r['target_k'],r['grid_target_occupancy'])):
            lines.append(f"{r['target_k']} | {r['grid_target_occupancy']} | {r['recall']:.10%} | {r['miss_count']}/{r['query_count']} | "
                f"{r['average_shortlist']:.3f} | {r['shortlist_max']} | {r['average_rings']:.3f}/{r['ring_max']} | "
                f"{r['distance_pair_reduction']:.3f}x | {r['comparison_reduction']:.3f}x")
        lines+=['', 'Smallest mean shortlist among zero-miss tested configurations: '+json.dumps(chosen),
            'Lowest observed miss fraction: '+json.dumps(best),
            '100% observed recall on these states would be empirical, not a general guarantee: fixed-count stopping alone cannot certify absence of a closer site just outside a ring.',
            'Protected hashes unchanged: '+str(unchanged)]
        (OUT/'coarse_routing_summary.txt').write_text('\n'.join(lines)+'\n')
        (OUT/'audit_manifest.json').write_text(json.dumps(dict(completed_states=completed,planned_states=indices,
            started=args.started,finished=time.time(),elapsed_seconds=time.time()-args.started,error=error,
            protected_hashes_unchanged=unchanged,protected_hashes=protected,
            implementation_sha256=source.sha(__file__),best=best,zero_miss_choice=chosen),indent=2)+'\n')
        print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
