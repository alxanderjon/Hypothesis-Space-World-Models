"""Full saved held-out pool audit of the geometric grid stopping certificate."""
import argparse,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import audit_coarse_routing_shortlist as coarse
from experiments.sho.grid_branch_bound import GridBranchBound

OUT=Path('outputs/sho/stage1B/grid_branch_bound')
def hist_median(hist):
    n=hist.sum();cs=np.cumsum(hist)
    return float((np.searchsorted(cs,(n-1)//2+1)+np.searchsorted(cs,n//2+1))/2) if n else 0.
def add_hist(a,b):
    result=np.zeros(max(len(a),len(b)),dtype=np.int64);result[:len(a)]+=a;result[:len(b)]+=b;return result
def aggregate_finish(r):
    r=r.copy();n=r['queries'];bh=r.pop('bin_hist');sh=r.pop('site_hist');rh=r.pop('ring_hist')
    r.update(recall=1-r['misses']/n if n else '',bins_mean=float(bh@np.arange(len(bh))/n) if n else '',
        bins_median=hist_median(bh),bins_max=int(np.flatnonzero(bh)[-1]) if n else 0,
        sites_mean=r['learned_distance_evaluations']/n if n else '',sites_median=hist_median(sh),
        sites_max=int(np.flatnonzero(sh)[-1]) if n else 0,
        ring0=int(rh[0]) if len(rh)>0 else 0,ring1=int(rh[1]) if len(rh)>1 else 0,
        ring2=int(rh[2]) if len(rh)>2 else 0,rings_larger=int(rh[3:].sum()),ring_max=int(np.flatnonzero(rh)[-1]) if n else 0,
        distance_reduction=r['exhaustive_distance_pairs']/r['learned_distance_evaluations'] if r['learned_distance_evaluations'] else '',
        comparison_reduction=r['exhaustive_candidate_comparisons']/r['learned_candidate_comparisons'] if r['learned_candidate_comparisons'] else '')
    return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--started',type=float,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,coarse.timeout);signal.setitimer(signal.ITIMER_REAL,max(.01,args.deadline-time.time()))
    source=coarse.source;torch.set_num_threads(1);rows=[];aggregates={};completed=[];error=None;failure=None
    protected=json.loads(Path('outputs/sho/stage1B/fast_argmin/fast_execution_manifest.json').read_text())['protected_hashes']
    models=source.FrozenProposals();router=GridBranchBound(models.distance,models.margin,models.threshold)
    payload=torch.load(source.cells.base.turn.RAW_TRACE_PATH,weights_only=False);trajectories=source.cells.base.ordinary_data._trajectory_map(payload)
    indices=torch.load(source.nearest.OPERATOR,weights_only=False)['held_out_state_indices'];primary=[17105,5660,17179]
    indices=primary+[i for i in indices if i not in primary]
    try:
        for state in indices:
            if args.deadline-time.time()<40:raise coarse.TimeLimit('stopping before next state to leave reporting time')
            sites,probes,record=source.cells.base.distance_metric._state_geometry(payload,trajectories,state);n,p=len(sites),len(probes)
            output=router.search(sites,probes)
            # Independent reference labels are evaluated only AFTER routing.
            tree=cKDTree(sites);pd,po=tree.query(probes,k=1);sd,so=tree.query(sites,k=2)
            truth=np.r_[po,so[:,1]];distance=np.r_[pd,sd[:,1]];miss=output['owner']!=truth
            if np.any(miss):
                j=int(np.flatnonzero(miss)[0]);queries=np.concatenate((probes,sites));kind='probe' if j<p else 'spacing'
                failure=dict(source_state_index=state,trajectory_id=int(record['trajectory_id']),query_kind=kind,
                    query_index=j if j<p else j-p,query=queries[j].tolist(),teacher_owner=int(truth[j]),
                    teacher_site=sites[truth[j]].tolist(),predicted_owner=int(output['owner'][j]),
                    predicted_site=sites[output['owner'][j]].tolist(),teacher_distance=float(distance[j]),
                    learned_distance=float(output['distance'][j]),stop_bound=float(output['stop_bound'][j]),
                    stop_ring=int(output['stop_ring'][j]),sites_evaluated=int(output['sites_evaluated'][j]),
                    exact_tie=bool(output['tied'][j]))
                torch.save(dict(sites=sites,probes=probes,record=record,output=output,failure=failure),OUT/'failing_state_query.pt')
                source.write_csv(OUT/'misses.csv',[failure]);print('MISS '+json.dumps(failure),flush=True)
            assert np.all(output['stop_bound']>output['distance'])
            state_row=dict(record_type='state',source_state_index=state,site_count=n,probe_count=p,
                grid_bins=output['grid_bins'],queries=p+n,misses=int(miss.sum()),recall=float(np.mean(~miss)),
                learned_distance_evaluations=int(output['sites_evaluated'].sum()),exhaustive_distance_pairs=p*n+n*(n-1),
                runtime_seconds=output['runtime_seconds'],setup_and_tie_seconds=output['setup_and_tie_seconds'],
                learned_distance_seconds=output['learned_distance_seconds'],distance_forward_calls=output['distance_forward_calls'],
                learned_distance_mismatches=int(np.sum(output['distance']!=distance)),
                equality_bound_continuations=output['equality_bound_continuations'])
            rows.append(state_row)
            for kind,query,sl in [('probe',probes,slice(0,p)),('spacing',sites,slice(p,None))]:
                masks,*_=coarse.regions(sites,query,kind=='spacing')
                for region,mask in masks.items():
                    for label in (kind,'combined'):
                        key=(label,region)
                        if key not in aggregates:aggregates[key]=dict(record_type='aggregate',query_kind=label,region=region,
                            queries=0,misses=0,tie_queries=0,learned_distance_evaluations=0,learned_candidate_comparisons=0,
                            exhaustive_distance_pairs=0,exhaustive_candidate_comparisons=0,bin_hist=np.zeros(1,dtype=np.int64),
                            site_hist=np.zeros(1,dtype=np.int64),ring_hist=np.zeros(1,dtype=np.int64))
                        a=aggregates[key];qcount=int(mask.sum());counts=output['sites_evaluated'][sl][mask]
                        a['queries']+=qcount;a['misses']+=int(miss[sl][mask].sum());a['tie_queries']+=int(output['tied'][sl][mask].sum())
                        a['learned_distance_evaluations']+=int(counts.sum());a['learned_candidate_comparisons']+=int(np.maximum(counts-1,0).sum())
                        available=n-int(kind=='spacing');a['exhaustive_distance_pairs']+=qcount*available
                        a['exhaustive_candidate_comparisons']+=qcount*max(available-1,0)
                        for keyname,field in [('bin_hist','bins_searched'),('site_hist','sites_evaluated'),('ring_hist','stop_ring')]:
                            a[keyname]=add_hist(a[keyname],np.bincount(output[field][sl][mask]))
            completed.append(state)
            source.write_csv(OUT/'grid_branch_bound_audit.csv',[aggregate_finish(a) for a in aggregates.values()]+rows)
            print(f"State {len(completed)}/80 n={n} queries={p+n} pairs={state_row['learned_distance_evaluations']} runtime={output['runtime_seconds']:.3f}s misses={int(miss.sum())} elapsed={time.time()-args.started:.1f}s",flush=True)
            if failure:break
    except Exception:error=traceback.format_exc();print(error,flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        unchanged=all(source.sha(path)==h for path,h in protected.items())
        final=[aggregate_finish(a) for a in aggregates.values()]
        exact=len(completed)==len(indices) and failure is None and error is None and unchanged
        source.write_csv(OUT/'grid_branch_bound_audit.csv',final+rows if final or rows else [dict(status='stopped',error=error)])
        summary=dict(status='DETERMINISTIC_CANDIDATE_SCAFFOLD' if exact else 'STOPPED',states_completed=len(completed),
            states_planned=len(indices),elapsed_seconds=time.time()-args.started,
            router_runtime_seconds=sum(r['runtime_seconds'] for r in rows),error=error,failure=failure,protected_hashes_unchanged=unchanged)
        lines=['EXACT GRID BRANCH-AND-BOUND AUDIT',json.dumps(summary),
            'Grid g=ceil(sqrt(site_count)); complete clipped Chebyshev rings. Site-spacing queries exclude self.',
            'All site comparisons use the locked distance metric and independent-weight margin/threshold; no distance argmin substitutes.',
            'The metric matrix was verified to be exactly the Euclidean identity.',
            'Unseen-region bound is min distance to each exposed side of the searched rectangle; domain edges with no unseen bins are ignored.',
            'Grid membership uses explicit representable grid edges. Two nextafter steps round the computed bound downward conservatively; no arbitrary tolerance.',
            'Stop strictly when r_unseen_min > r_best. Equality continues. Empty bins are included in bins-searched counts.',
            'Batched cKDTree precheck marks only exact nearest ties. Production ownership is used only in comparisons of equal learned distances for those tied queries.',
            'All query types, including ties, undergo branch-and-bound; no tied-query bypass inflates reduction.',
            'Teacher owners are separate audit labels; no reference validation occurs inside the router.',
            'Boundary band: within 0.05 of a face. Density strata match the previous audit, using independent 3x3-bin density quartiles.',
            '', 'kind / region | queries | recall | misses | bins mean/median/max | sites mean/median/max | pair reduction']
        for r in final:
            if r['region']=='all' or r['query_kind']=='combined':
                lines.append(f"{r['query_kind']} / {r['region']} | {r['queries']} | {r['recall']} | {r['misses']} | "
                    f"{r['bins_mean']}/{r['bins_median']}/{r['bins_max']} | {r['sites_mean']}/{r['sites_median']}/{r['sites_max']} | {r['distance_reduction']}")
        allrow=next((r for r in final if r['query_kind']=='combined' and r['region']=='all'),None)
        lines+=['','ALL QUERIES '+json.dumps(allrow),
            'No training, checkpoint modification, objective evaluations, trajectory search, or learned-router fitting.']
        if exact:
            manifest=dict(**summary,implementation='experiments/sho/grid_branch_bound.py',
                implementation_sha256=source.sha('experiments/sho/grid_branch_bound.py'),protected_hashes=protected,
                validated_states=completed,summary=allrow,exact_ties='TEMPORARY DETERMINISTIC SCAFFOLDING')
            (OUT/'grid_branch_bound_candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (OUT/'grid_branch_bound_summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
