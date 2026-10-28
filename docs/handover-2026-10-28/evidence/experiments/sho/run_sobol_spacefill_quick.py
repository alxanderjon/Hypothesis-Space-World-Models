"""Time-bounded generator-only test; no persistent production modifications."""
import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_generator_smoke as smoke
from experiments.sho.grid_branch_bound import GridBranchBound

OUT=Path('outputs/sho/stage1B/sobol_vs_spacefill_quick')
CSV=OUT/'sobol_vs_spacefill_quick_closedloop.csv'
SUMMARY=OUT/'sobol_vs_spacefill_quick_summary.txt'
im,ed=src.im,src.edirect

class Deadline(Exception): pass
class RecoveryLoss(Exception): pass

def run(started, deadline):
    OUT.mkdir(parents=True,exist_ok=True)
    def alarm(*args): raise Deadline('wall-clock cutoff')
    signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.time()))
    torch.set_num_threads(1)
    manifest=json.loads(Path('outputs/sho/stage1B/grid_branch_bound/grid_branch_bound_candidate.json').read_text())
    protected={p:src.sha(p) for p in manifest['protected_hashes']}
    candidate=Path('outputs/sho/stage1B/spacefill_downstream_screen/spacefill_generator_candidate.pt')
    protected[str(candidate)]=src.sha(candidate)
    protected[ed.__file__]=src.sha(ed.__file__)
    generator=smoke.Generator(2)
    generator.load_state_dict(torch.load(candidate,weights_only=False)['state_dict'])
    generator.eval().requires_grad_(False)
    distance,_=src.cells.base._load_locked_distance()
    op=torch.load(src.nearest.OPERATOR,weights_only=False)
    margin=nn.Linear(2,1,bias=False,dtype=torch.float64)
    margin.load_state_dict(torch.load(op['margin_checkpoint'],weights_only=False)['state_dict'])
    threshold=src.nearest.threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(op['threshold_checkpoint'],weights_only=False)['state_dict'])
    for model in (distance,margin,threshold): model.eval().requires_grad_(False)
    router=GridBranchBound(distance,margin,threshold)
    datasets,_=src.data_and_plan()
    archived=json.loads(Path('outputs/sho/stage1B/spacefill_2x2/closedloop_plan.json').read_text())['cases']
    cases=[dict(c,split='test',generator_id=200000+c['trajectory_id']) for c in archived[:6]]
    original_pool,original_cell=ed._candidate_pool,ed._cell_data
    current={};cache={};runs=[];status='COMPLETE';error=None
    timing=dict(router_seconds=0.,probe_seconds=0.,proposal_seconds=0.,greedy_seconds=0.,polish_seconds=0.)
    def pool(n,tid,iteration):
        t=time.perf_counter()
        if current['arm']=='A': result=original_pool(n,tid,iteration)
        else:
            budget=min(ed.MC_FACTOR_PER_SITE*n,ed.MC_CANDIDATE_CAP)
            if budget not in cache:
                result,_=generator.generate(budget,[0.,0.],[1.,1.])
                # Assert exactly the production face support, including endpoints.
                baseline=original_pool(n,tid,iteration)
                boundary=lambda x:x[np.any((x==0.)|(x==1.),axis=1)]
                assert np.array_equal(boundary(result),boundary(baseline))
                cache[budget]=result
            result=cache[budget]
        timing['probe_seconds']+=time.perf_counter()-t
        return result
    def cell(points,losses,tid,iteration):
        probes=pool(len(points),tid,iteration)
        routed=router.search(points,probes);timing['router_seconds']+=routed['runtime_seconds']
        p=len(probes);owners=routed['owner'][:p];distances=routed['distance'][:p]
        sizes=np.zeros(len(points));np.maximum.at(sizes,owners,distances)
        sizes=np.maximum(sizes,.5*routed['distance'][p:])
        selected=ed._potentially_optimal_cells(sizes,losses)
        current['last_po']=len(selected);current['po_batches']+=len(selected)
        current['distance_pairs']+=int(routed['sites_evaluated'].sum())
        return ed.CellProposalData(probes,owners,distances,sizes,selected)
    def add(r,point,value,kind,iteration,mode='normal'):
        r['events'].append(dict(evaluation=len(r['events'])+1,point=np.asarray(point).tolist(),loss=float(value),
            kind=kind,iteration=iteration,mode=mode,PO_cells_in_batch=r.get('last_po',0)))
        if kind=='cell':r['po_evaluated']+=1
    def normal_mode(r):
        return im.ModeRun('normal_edirect',src.ordinary.FULL_BOUNDS.copy(),r['sites'],r['losses'],len(r['losses']),bool(r['losses'].min()<=src.GATE))
    def complete(r):
        modes=r['modes'];best=min(modes,key=lambda m:m.best_loss)
        r['success']=any(m.success for m in modes)
        if r['success']:
            r['status']='POLISHING';r['polish_status']='RUNNING'
            print(f"Handoff {r['case']['trajectory_id']} {r['arm']} eval={len(r['events'])} loss={best.best_loss:.4g}",flush=True)
            save()
            t=time.perf_counter()
            result=src.finish_recovery(dict(case=r['case'],final=best,modes=modes,success=True),datasets['test'])
            timing['polish_seconds']+=time.perf_counter()-t;r.update(result)
        r['status']='COMPLETE'
        save()
    def divergence(a,b):
        for x,y in zip(a['events'],b['events']):
            if np.linalg.norm(np.asarray(x['point'])-y['point'])>1e-12:
                return dict(first_divergence_evaluation=x['evaluation'],first_divergence_iteration_A=x['iteration'],
                    first_divergence_iteration_B=y['iteration'],first_divergence_A_point=json.dumps(x['point']),
                    first_divergence_B_point=json.dumps(y['point']),first_divergence_A_loss=x['loss'],first_divergence_B_loss=y['loss'])
        return {}
    def report_rows():
        rows=[]
        for r in runs:
            losses=[v['loss'] for v in r['events']];hits=[i+1 for i,v in enumerate(losses) if v<=src.GATE]
            rows.append(dict(record_type='trajectory',**r['case'],arm=r['arm'],generator='Sobol' if r['arm']=='A' else 'spacefill',
                status=r['status'],handoff_reached=bool(hits),evaluations_to_handoff=hits[0] if hits else '',
                global_evaluations=len(losses),best_prefix_loss=min(losses),rescue_needed=r['rescue'],
                PO_proposals_evaluated=r['po_evaluated'],PO_cells_selected_across_batches=r['po_batches'],
                learned_distance_evaluations=r['distance_pairs'],iterations=r['iteration'],
                **{k:v for k,v in r.items() if k.startswith(('final_','rollout_','reaches_','local_','newton_','polish_'))}))
        for case in cases:
            pair=[r for r in runs if r['case']['trajectory_id']==case['trajectory_id']]
            if len(pair)==2:
                a,b=pair;ha=[e['evaluation'] for e in a['events'] if e['loss']<=src.GATE];hb=[e['evaluation'] for e in b['events'] if e['loss']<=src.GATE]
                rows.append(dict(record_type='pair',**case,matched_complete=all(r['status']=='COMPLETE' for r in pair),
                    handoff_A=bool(ha),handoff_B=bool(hb),evaluations_B_minus_A=hb[0]-ha[0] if ha and hb else '',**divergence(a,b)))
        return rows
    def save():
        src.write_csv(CSV,report_rows())
    try:
        for case in cases:
            sites=ed._initial_design().copy();position=datasets['test']['p'][case['trajectory_id']]
            losses=ed._prefix_losses_numpy(ed._physical(sites),position)
            for arm in ('A','B'):
                r=dict(case=case,arm=arm,sites=sites.copy(),losses=losses.copy(),events=[],iteration=0,status='RUNNING',
                    rescue=False,po_evaluated=0,po_batches=0,distance_pairs=0,modes=[],polish_status='NOT_REACHED')
                for point,loss in zip(sites,losses):add(r,ed._physical(point),loss,'initial',-1)
                runs.append(r)
        ed._candidate_pool=pool;ed._cell_data=cell
        save();print('Six matched cases initialized; production budgets, FIFO, common boundary support.',flush=True)
        last_notice=time.time()
        while any(r['status']!='COMPLETE' for r in runs):
            for r in runs:
                if r['status']=='COMPLETE':continue
                current=r;case=r['case'];position=datasets['test']['p'][case['trajectory_id']]
                if r['losses'].min()<=src.GATE:
                    r['modes']=[normal_mode(r)];complete(r);continue
                if len(r['sites'])>=im.NORMAL_CAP:
                    r['rescue']=True;r['modes']=[normal_mode(r)];save()
                    rescued,_=im._run_boundary_rescue(split='test',trajectory_id=case['trajectory_id'],generator_id=case['generator_id'],position=position,normal=r['modes'][0],collect_trace=False)
                    r['modes'].append(rescued)
                    for pt,val in zip(im._physical(rescued.points[-rescued.new_evaluations:],rescued.bounds),rescued.losses[-rescued.new_evaluations:]):add(r,pt,val,'rescue','',rescued.mode)
                    if not rescued.success:
                        nested=im._run_nested_rescue(split='test',trajectory_id=case['trajectory_id'],position=position,boundary_run=rescued,collect_trace=False)
                        r['modes'].append(nested)
                        for pt,val in zip(im._physical(nested.points[-nested.new_evaluations:],nested.bounds),nested.losses[-nested.new_evaluations:]):add(r,pt,val,'rescue','',nested.mode)
                    complete(r)
                else:
                    t=time.perf_counter();pending=ed._cell_proposals(r['sites'],r['losses'],case['generator_id'],r['iteration'])
                    timing['proposal_seconds']+=time.perf_counter()-t
                    for point in pending:
                        if len(r['sites'])>=im.NORMAL_CAP or r['losses'].min()<=src.GATE:break
                        value=float(ed._prefix_losses_numpy(ed._physical(point[None,:]),position)[0])
                        r['sites']=np.vstack((r['sites'],point));r['losses']=np.r_[r['losses'],value]
                        add(r,ed._physical(point),value,'cell',r['iteration'])
                    if len(r['sites'])<im.NORMAL_CAP and r['losses'].min()>src.GATE:
                        t=time.perf_counter();point=ed._greedy_proposal(r['sites'],r['losses'],case['generator_id'],r['iteration'])
                        timing['greedy_seconds']+=time.perf_counter()-t
                        if point is not None:
                            value=float(ed._prefix_losses_numpy(ed._physical(point[None,:]),position)[0])
                            r['sites']=np.vstack((r['sites'],point));r['losses']=np.r_[r['losses'],value]
                            add(r,ed._physical(point),value,'greedy',r['iteration'])
                    r['iteration']+=1
                    if r['losses'].min()<=src.GATE:r['modes']=[normal_mode(r)];complete(r)
                save()
                pair=[v for v in runs if v['case']==case]
                if all(v['status']=='COMPLETE' for v in pair) and pair[0].get('success') and not pair[1].get('success'):
                    torch.save(dict(case=case,runs=pair),OUT/'recovery_loss_state.pt');raise RecoveryLoss('Sobol succeeds, spacefill exhausts recovery')
            if time.time()-last_notice>15:
                print('Progress '+json.dumps([(r['case']['trajectory_id'],r['arm'],len(r['events']),float(r['losses'].min()),r['status']) for r in runs]),flush=True)
                last_notice=time.time()
            # Five-minute target; hard alarm remains in force during components.
            if time.time()-started>=300:
                raise Deadline('five-minute target reached at matched sweep boundary')
    except Deadline as exc:status='PARTIAL_WALL_CLOCK_STOP';error=str(exc)
    except RecoveryLoss as exc:status='STOP_RECOVERY_LOSS';error=str(exc)
    except Exception as exc:status='STOP_ERROR';error=repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        ed._candidate_pool=original_pool;ed._cell_data=original_cell
        for r in runs:
            if r['status']!='COMPLETE':r['status']='PARTIAL_WALL_CLOCK_STOP' if status=='PARTIAL_WALL_CLOCK_STOP' else status
        save()
        torch.save(dict(runs=runs,status=status,timing=timing),OUT/'quick_screen_states.pt')
        unchanged=all(src.sha(p)==h for p,h in protected.items())
        rows=report_rows();pairs=[r for r in rows if r['record_type']=='pair'];tr=[r for r in rows if r['record_type']=='trajectory']
        differences=[r['evaluations_B_minus_A'] for r in pairs if r['evaluations_B_minus_A']!='']
        summary=dict(status=status,error=error,elapsed_seconds=time.time()-started,planned_matched_trajectories=6,
            completed_matched_trajectories=sum(r['matched_complete'] for r in pairs),
            A_handoffs=sum(r['handoff_reached'] for r in tr if r['arm']=='A'),B_handoffs=sum(r['handoff_reached'] for r in tr if r['arm']=='B'),
            spacefill_wins=sum(d<0 for d in differences),ties=sum(d==0 for d in differences),spacefill_losses=sum(d>0 for d in differences),
            median_evaluations_B_minus_A=float(np.median(differences)) if differences else None,
            worst_evaluations_B_minus_A=max(differences) if differences else None,protected_hashes_unchanged=unchanged,timing=timing)
        lines=['QUICK SOBOL VS SPACEFILL CLOSED-LOOP SCREEN',json.dumps(summary,indent=2),
            'Only probe generation differs. Both use original teacher scoring/PO/FIFO/greedy/rescue/refinement rules.',
            'Spacefill replaces the interior pool in normal cell and greedy generation, and rescue if reached; boundary coordinates are asserted identical.',
            'Grid branch-and-bound uses locked distance/comparator and exact cKDTree tie bookkeeping.',
            f'Unchanged production normal cap {im.NORMAL_CAP}, rescue cap {im.RESCUE_CAP}; no budget-truncated failure claims.',
            'Partial arms are censored by wall clock and are NOT recovery failures. No reliability conclusion from unfinished pairs.',
            'Candidate checkpoint unchanged; no training, promotion, or production source changes.',
            'Timing proposal_seconds includes router/probe work; timing categories are not additive.',
            'trajectory | category | A handoff/evaluations/best | B handoff/evaluations/best | B-A | status']
        for case in cases:
            rr=[r for r in tr if r['trajectory_id']==case['trajectory_id']]
            if len(rr)==2:
                a,b=rr;pair=next(r for r in pairs if r['trajectory_id']==case['trajectory_id'])
                lines.append(f"{case['trajectory_id']} | {case['stratum']} | {a['handoff_reached']}/{a['evaluations_to_handoff'] or a['global_evaluations']}/{a['best_prefix_loss']:.8g} | {b['handoff_reached']}/{b['evaluations_to_handoff'] or b['global_evaluations']}/{b['best_prefix_loss']:.8g} | {pair['evaluations_B_minus_A']} | {'complete' if pair['matched_complete'] else 'partial'}")
        SUMMARY.write_text('\n'.join(lines)+'\n');print(SUMMARY.read_text(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--started',type=float,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();run(args.started,args.deadline)
