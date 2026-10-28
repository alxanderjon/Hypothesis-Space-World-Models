"""Bounded saved-state validation and timing of fast nearest execution."""
import argparse,csv,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_fifo_vs_ranker as run
from experiments.sho import run_stage1b_spacefill_generator_smoke as geometry
from experiments.sho import screen_stage1b_spacefill_downstream as proposals
from experiments.sho.fast_nearest_argmin import FastNearestArgmin

OUT=Path('outputs/sho/stage1B/fast_argmin')
class Deadline(RuntimeError):pass
def timeout(*a):raise Deadline('wall-clock safety deadline')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--started',type=float,required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.01,args.deadline-time.time()))
    torch.set_num_threads(1);profiles=[];validation=[];error=None
    protected=json.loads((run.OUT/'plan.json').read_text())['protected_hashes']
    repair=Path('outputs/sho/stage1B/slope_sign_repair_runtime/hull_slope_sign_repaired.pt')
    protected[str(repair)]=run.sha(repair)
    prior=list(csv.DictReader(Path('outputs/sho/stage1B/slope_sign_repair_runtime/pipeline_runtime_profile.csv').open()))
    frozen=run.FrozenProposals();fast=FastNearestArgmin(frozen.distance,frozen.margin,frozen.threshold,frozen.half)
    payload=torch.load(run.cells.base.turn.RAW_TRACE_PATH,weights_only=False)
    trajectories=run.cells.base.ordinary_data._trajectory_map(payload)
    original_load=torch.load
    def use_saved_sign(path,*a,**kw):
        if isinstance(path,(str,Path)) and Path(path).resolve()==Path('outputs/sho/stage1B/hull_slope_sign/hull_slope_sign_locked.pt').resolve():path=repair
        return original_load(path,*a,**kw)
    try:
        for N in (1000,8600,32768):
            if args.deadline-time.time()<65:raise Deadline('insufficient safe time for next static state')
            signal.setitimer(signal.ITIMER_REAL,min(59,args.deadline-time.time()))
            old=next(r for r in prior if r['record_type']=='comparison' and int(r['N'])==N)
            state=int(old['source_state_index']);record=payload['states'][state];n=int(record['history_count'])
            tr=trajectories[int(record['trajectory_id'])]
            sites=np.asarray(tr['evaluated_hypotheses_normalized'][:n],dtype=np.float64)
            losses=np.asarray(tr['evaluated_prefix_losses'][:n],dtype=np.float64)
            t=time.perf_counter();probe=run.edirect._candidate_pool(n,int(record['generator_id']),int(record['iteration']));generation_ms=1000*(time.perf_counter()-t)
            # The only timed inference call: no external truth arrays supplied.
            output=fast.cells(sites,probe);p=len(probe)
            # Everything below is a distinct validation mode, outside inference timing.
            t=time.perf_counter()
            captured,_=run.cells.capture_nearest(sites,probe,frozen.distance,frozen.margin,frozen.threshold,state)
            reference_scan_ms=1000*(time.perf_counter()-t)
            owner=captured['winner'][:p];dist=captured['winner_distance'][:p];spacing=captured['winner_distance'][p:]
            radius,witness,_=run.cells.assigned_radius(owner,dist,n,frozen.margin,frozen.threshold)
            half=run.cells.predict_half(frozen.half,spacing)
            replace,_,_,_=run.cells.larger_candidate(frozen.margin,frozen.threshold,radius,half)
            sizes=np.where(replace,half,radius)
            row=dict(N=N,source_state_index=state,site_count=n,probe_count=p)
            for name,a,b in [('probe_owner',output['probe_owner'],owner),('spacing_owner',output['spacing_owner'],captured['winner'][p:]),
                    ('probe_distance',output['probe_distance'],dist),('assigned_radius',output['radius'],radius),
                    ('radius_witness',output['radius_witness'],witness),('spacing',output['spacing'],spacing),
                    ('half_spacing',output['half'],half),('cell_size',output['sizes'],sizes)]:
                row[name+'_mismatches']=int(np.sum(a!=b))
                row[name+'_max_error']=float(np.max(np.abs(a-b)))
            torch.load=use_saved_sign
            before=geometry.locked_trace(sizes,losses,record,state,frozen.locked)
            t=time.perf_counter();after=geometry.locked_trace(output['sizes'],losses,record,state,frozen.locked)
            downstream_ms=1000*(time.perf_counter()-t)
            for name in ('groups','representatives','hull','po'):
                row[name+'_changes']=len(before[name]^after[name])
            # Static proposal generation is checked too; no objective or state update.
            b=proposals.select_proposals(sites,losses,probe,owner,dist,sizes,before['po'],record,frozen.score,frozen.ranker)
            t=time.perf_counter()
            a=proposals.select_proposals(sites,losses,probe,output['probe_owner'],output['probe_distance'],output['sizes'],after['po'],record,frozen.score,frozen.ranker)
            proposal_ms=1000*(time.perf_counter()-t)
            torch.load=original_load
            row['proposal_changes']=int(a['proposals']!=b['proposals']);row['static_final_choice_changes']=int(a['final_choice']!=b['final_choice'])
            passed=all(v==0 for k,v in row.items() if k.endswith(('_mismatches','_changes')))
            row['passed']=passed;validation.append(row)
            t=time.perf_counter();tree=cKDTree(sites);tree.query(probe,k=1);tree.query(sites,k=2)
            kd_ms=1000*(time.perf_counter()-t)
            stats=output['stats']
            old_geometry=sum(float(r['seconds'])*1000 for r in prior if r.get('record_type')=='component' and int(r['N'])==N and r['component'] in
                ('learned_distance','argmin_reduction_and_validation','tie_handling_and_reference_audit','assigned_radius_max','half_spacing','final_cell_size_max'))
            profile=dict(N=N,source_state_index=state,site_count=n,probe_count=p,**stats,
                probe_generation_ms=generation_ms,old_profile_total_ms=float(old['static_total_seconds'])*1000,
                old_matched_geometry_ms=old_geometry,matched_geometry_speedup=old_geometry/stats['geometry_total_ms'],
                old_nearest_ms=float(old['learned_argmin_inclusive_seconds'])*1000,
                nearest_speedup=float(old['learned_argmin_inclusive_seconds'])*1000/stats['nearest_total_ms'],
                contemporaneous_reference_scan_ms=reference_scan_ms,ckdtree_ms=kd_ms,nearest_to_ckdtree_ratio=stats['nearest_total_ms']/kd_ms,
                unchanged_downstream_validation_ms=downstream_ms,static_proposals_and_ranker_ms=proposal_ms,
                full_static_composition_ms=generation_ms+stats['geometry_total_ms']+downstream_ms+proposal_ms,
                composition_note='fast inference through cell sizes + separately timed unchanged downstream validation helpers and proposal/ranker; not all audit-free',
                inference_scope='probe/site nearest + radius + half-spacing + final max; no teacher checks',passed=passed)
            profile['old_total_to_static_composition_ratio']=profile['old_profile_total_ms']/profile['full_static_composition_ms']
            profiles.append(profile)
            run.write_csv(OUT/'fast_argmin_profile.csv',profiles);run.write_csv(OUT/'fast_argmin_validation.csv',validation)
            print('STATE '+json.dumps(dict(profile=profile,validation=row)),flush=True)
            if not passed:raise RuntimeError('correctness gate failed')
        assert all(run.sha(p)==h for p,h in protected.items())
        manifest=dict(status='FAST_EXECUTION_PATH',component='existing_locked_nearest_argmin_operator',
            implementation='experiments/sho/fast_nearest_argmin.py',implementation_sha256=run.sha('experiments/sho/fast_nearest_argmin.py'),
            entry_point='FastNearestArgmin.nearest; FastNearestArgmin.cells for cell-size composition',
            weights_retrained=False,checkpoint_hashes_changed=False,protected_hashes=protected,
            validated_states=[r['source_state_index'] for r in profiles],validation_gate='all reported output arrays and downstream sets exactly equal',
            exact_tie_handling='TEMPORARY DETERMINISTIC SCAFFOLDING; one batched precheck phase; conditional production-owner queries only for tied rows',
            non_tied_selection='locked independent-weight margin + locked positive-alpha threshold; batched tournament',
            default_pair_block=262144,default_query_block=1024,
            elapsed_seconds=time.time()-args.started)
        (OUT/'fast_execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    finally:
        torch.load=original_load;signal.setitimer(signal.ITIMER_REAL,0)
        if not profiles:run.write_csv(OUT/'fast_argmin_profile.csv',[dict(status='stopped',reason=error)])
        if not validation:run.write_csv(OUT/'fast_argmin_validation.csv',[dict(status='stopped',reason=error)])
        lines=['FAST NEAREST/ARGMIN EXECUTION TEST',f'Elapsed seconds: {time.time()-args.started:.3f}',f'Stop/error: {error}',
            'No retraining, objective evaluations, rescue, trajectory search, or checkpoint modifications.',
            'Normal fast inference contains no reference checking. Only its initial batched exact-tie phase uses cKDTree.',
            'Conditional k=1 probe/k=2 site queries run only for tied rows to preserve production tie ownership; never inside comparisons.',
            'Float64, CPU, one Torch/BLAS thread, one timed pass per saved state. No warm-up or timing confidence intervals.',
            'Distance blocks: at most 262144 pairs per neural forward, at most 1024 query rows; intermediates O(block_pairs), not O(all_query_site_pairs).',
            'component | small ms | medium ms | large ms']
        for key in ('tie_precheck_ms','distance_ms','tournament_ms','assigned_radius_ms','remaining_cell_size_ms','geometry_total_ms','full_static_composition_ms'):
            lines.append(key+' | '+' | '.join(f'{r[key]:.4f}' for r in profiles))
        lines+=['Geometry total is the audit-free inference scope. Full static composition also includes separately timed unchanged downstream audit helpers and static proposal/ranker; comparisons to previous full totals are approximate in scope.',
            'Matched geometry and nearest-only speedups are reported separately for comparable scopes.',
            'All nearest-owner/radius/spacing/cell-size array comparisons and downstream group/representative/hull/PO/proposal comparisons are outside normal inference timing.']
        for r in profiles:lines.append(json.dumps(r))
        lines.append('All original protected hashes unchanged: '+str(all(run.sha(p)==h for p,h in protected.items())))
        (OUT/'fast_argmin_summary.txt').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
