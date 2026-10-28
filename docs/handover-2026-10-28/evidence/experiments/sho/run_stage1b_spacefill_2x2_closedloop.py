"""18-case bounded screen, with a mandatory fidelity stop on new states."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.sho import run_stage1b_spacefill_2x2 as test

OUT=test.OUT
base,cells,nearest,down=test.base,test.cells,test.nearest,test.down
selection,edirect=test.selection,test.edirect
imitation=selection.imitation
NORMAL_CAP,RESCUE_CAP,REFINE_CAP=160,40,40
GATE=1e-4


class FidelityStop(Exception):pass
class RefinementBudget(Exception):pass


def run():
    torch.set_num_threads(1)
    static=json.loads((OUT/'static_summary.json').read_text())
    assert not static['summary']['distribution_fidelity_failures']
    assert not test.CLOSED.exists(), 'Do not repeat an existing closed-loop run'
    candidate_path=test.screen.OUT/'spacefill_generator_candidate.pt'
    checkpoint=torch.load(candidate_path,weights_only=False)
    generator=test.smoke.Generator(2)
    generator.load_state_dict(checkpoint['state_dict'])
    metric,_=base._load_locked_distance()
    op=torch.load(nearest.OPERATOR,weights_only=False)
    margin=nn.Linear(2,1,bias=False,dtype=torch.float64)
    margin.load_state_dict(torch.load(op['margin_checkpoint'],weights_only=False)['state_dict'])
    threshold=nearest.threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(op['threshold_checkpoint'],weights_only=False)['state_dict'])
    half=nn.Linear(1,1,bias=False,dtype=torch.float64)
    half.load_state_dict(torch.load(cells.OUT/'half_spacing_operator_locked.pt',weights_only=False)['state_dict'])
    models=(metric,margin,threshold,half)
    score,selector=selection._load_search_models()
    for m in (generator,*models,score,selector):m.eval().requires_grad_(False)
    locked=down._load_downstream()
    positions,_,hidden=selection._make_exact_labeled_split(1024,imitation.TEST_SEED)
    positions=positions.cpu().numpy()
    truth=np.column_stack((-np.ones(len(hidden)),hidden.cpu().numpy()))
    edge=np.min(np.column_stack((truth-edirect.LOWER,edirect.UPPER-truth)),axis=1)
    # Truth is used ONLY to stratify reporting/choose cases, never by a search.
    strata={'interior':np.flatnonzero(edge>=.1),
        'near_boundary':np.flatnonzero((edge>=.01)&(edge<.1)),
        'very_near_boundary':np.flatnonzero((edge>=0)&(edge<.01))}
    chosen={}
    for name,ids in strata.items():
        assert len(ids)>=6,(name,len(ids))
        order=ids[np.argsort(edge[ids],kind='stable')]
        chosen[name]=order[np.linspace(0,len(order)-1,6).round().astype(int)].tolist()
    cases=[(name,chosen[name][i]) for i in range(6) for name in strata]
    plan=dict(planned_trajectories=18,cases=[dict(stratum=s,trajectory_id=i,boundary_distance=float(edge[i])) for s,i in cases],
        data_seed=imitation.TEST_SEED,source_pool_size=1024,normal_cap=NORMAL_CAP,boundary_rescue_cap=RESCUE_CAP,
        nested_refinement_cap=REFINE_CAP,maximum_evaluations_per_arm=240,success_gate=GATE,
        initial_design='identical original deterministic ten points',
        rescue='common original deterministic strip rescue and nested refinement, truncated identically; production Sobol retained in common rescue',
        learned_generator_scope='normal-mode Voronoi probe sets; greedy/local candidates and rescue/refinement unchanged',
        fidelity='C/D independently compute each visited state; identical objective observations reused only after choices agree; any changed proposal stops before evaluation')
    (OUT/'closedloop_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[];completed=[];failure=None
    def log_eval(case,arm,mode,count,point,loss,best,kind,po_count=0):
        rows.append(dict(record_type='evaluation',trajectory_id=case[1],stratum=case[0],arm=arm,mode=mode,
            evaluation_count=count,point=json.dumps(point.tolist()),loss=float(loss),best_loss=float(best),
            reached_gate=best<=GATE,candidate_type=kind,PO_cells_in_batch=po_count))
    def rescue(case,position,sites,losses,arms):
        # Identical original rescue algorithms for all arms; only caps shorten.
        normal=imitation.ModeRun('normal_edirect',selection.FULL_BOUNDS.copy(),sites,losses,len(losses),False)
        physical=imitation._physical(normal.points,normal.bounds)
        best=physical[normal.best_index]
        _,axis,is_lower,_=imitation.boundary._nearest_face(best)
        spacing=imitation.boundary._local_normal_spacing(physical,best,axis)
        bounds,_=imitation.boundary._strip_bounds(best,axis,is_lower,spacing)
        inside=np.all((physical>=bounds[:,0]-1e-14)&(physical<=bounds[:,1]+1e-14),axis=1)
        seeds=(physical[inside]-bounds[:,0])/(bounds[:,1]-bounds[:,0])
        mode=imitation._run_edirect_mode(split='test',trajectory_id=case[1],generator_id=200000+case[1],position=position,
            bounds=bounds,cap=RESCUE_CAP,mode='boundary_rescue',selector='teacher',model=None,
            seed_points=seeds,seed_losses=normal.losses[inside],collect_trace=False)
        best_loss=float(losses.min());count=len(losses)
        for point,value in zip(imitation._physical(mode.points[-mode.new_evaluations:],bounds),mode.losses[-mode.new_evaluations:]):
            count+=1;best_loss=min(best_loss,float(value))
            for arm in arms:log_eval(case,arm,'boundary_rescue',count,point,value,best_loss,'common_rescue')
        if best_loss<=GATE:return count,best_loss
        captures=[];active={};code=edirect._prefix_losses_numpy.__code__;previous=sys.getprofile()
        def profile(frame,event,arg):
            if frame.f_code is code:
                if event=='call':
                    if len(captures)>=REFINE_CAP:raise RefinementBudget()
                    active['theta']=np.asarray(frame.f_locals['theta']).copy()
                elif event=='return' and arg is not None:
                    captures.append((active['theta'].reshape(-1,2)[0],float(np.asarray(arg)[0])))
        try:
            sys.setprofile(profile)
            imitation._run_nested_rescue(split='test',trajectory_id=case[1],position=position,boundary_run=mode,collect_trace=False)
        except RefinementBudget:pass
        finally:sys.setprofile(previous)
        for point,value in captures:
            count+=1;best_loss=min(best_loss,value)
            for arm in arms:log_eval(case,arm,'nested_refinement',count,point,value,best_loss,'common_refinement')
        return count,best_loss
    try:
        for case in cases:
            label,trajectory=case;position=positions[trajectory]
            print(f'Closed-loop case {len(completed)+1}/18 {label} id={trajectory}',flush=True)
            initial=edirect._initial_design()
            initial_losses=edirect._prefix_losses_numpy(edirect._physical(initial),position)
            for arm in ('sobol_teacher','spacefill_teacher','spacefill_learned'):
                for i,(point,loss) in enumerate(zip(initial,initial_losses),1):
                    log_eval(case,arm,'normal',i,point,loss,float(initial_losses[:i].min()),'initial')
            # First audit/evolve C and D together. Independent decisions must
            # agree before observations are shared; no teacher choice override.
            arm_results=[]
            for use_learned_generator,arms in ((True,('spacefill_teacher','spacefill_learned')),(False,('sobol_teacher',))):
                sites=initial.copy();losses=initial_losses.copy();iteration=0;po_evaluated=0
                while len(losses)<NORMAL_CAP and float(losses.min())>GATE:
                    record=dict(trajectory_id=trajectory,iteration=iteration)
                    state=10000000+trajectory*1000+iteration
                    probes=generator.generate(min(100*len(sites),32768),[0.,0.],[1.,1.])[0] if use_learned_generator else edirect._candidate_pool(len(sites),200000+trajectory,iteration)
                    exact=test.trace_geometry(sites,losses,probes,record,state,False,models,locked)
                    pending,local=test.proposals(sites,losses,probes,exact['owners'],exact['distances'],exact['sizes'],exact['po'])
                    if use_learned_generator:
                        learned=test.trace_geometry(sites,losses,probes,record,state,True,models,locked)
                        other,learned_local=test.proposals(sites,losses,probes,learned['owners'],learned['distances'],learned['sizes'],learned['po'],score)
                        discrepancies=[key for key in ('owners','sizes','membership','representatives','hull','po') if not np.array_equal(exact[key],learned[key])]
                        if not np.array_equal(pending,other):discrepancies.append('proposals')
                        rows.append(dict(record_type='visited_state_fidelity',trajectory_id=trajectory,stratum=label,
                            iteration=iteration,history_count=len(sites),N=min(100*len(sites),32768),
                            exact=not discrepancies,discrepancies=json.dumps(discrepancies),
                            teacher_PO=json.dumps(exact['po'].tolist()),learned_PO=json.dumps(learned['po'].tolist())))
                        if discrepancies:
                            failure=dict(trajectory_id=trajectory,stratum=label,iteration=iteration,history_count=len(sites),
                                discrepancies=discrepancies,teacher_proposals=pending.tolist(),learned_proposals=other.tolist())
                            for r in learned_local:
                                if not r.get('empty') and r['teacher_probe_index']!=r['chosen_probe_index']:
                                    # Evaluation of these alternatives is diagnostic only and is not a search update.
                                    losses_pair=edirect._prefix_losses_numpy(edirect._physical(np.asarray([r['teacher_coordinate'],r['chosen_coordinate']])),position)
                                    rows.append(dict(record_type='divergence_diagnosis',trajectory_id=trajectory,stratum=label,iteration=iteration,
                                        history_count=len(sites),first_divergent_operator='ordinary_cell_score_network',
                                        teacher_objective_loss=float(losses_pair[0]),learned_objective_loss=float(losses_pair[1]),**r))
                            np.savez_compressed(OUT/'closedloop_failure_state.npz',sites=sites,losses=losses,probes=probes,position=position)
                            raise FidelityStop()
                    while len(pending) and len(losses)<NORMAL_CAP and float(losses.min())>GATE:
                        if use_learned_generator:
                            features=imitation._global_features(sites,losses,len(pending),len(sites),NORMAL_CAP,iteration,selection.FULL_BOUNDS,'normal_edirect')
                            choice,_=imitation._model_choice(selector,sites,losses,pending,'cell',features)
                            if choice!=0:
                                failure=dict(trajectory_id=trajectory,stratum=label,iteration=iteration,history_count=len(sites),
                                    discrepancies=['final_selector'],teacher_choice=0,learned_choice=int(choice))
                                raise FidelityStop()
                        proposal=pending[0]
                        value=float(edirect._prefix_losses_numpy(edirect._physical(proposal[None,:]),position)[0])
                        sites=np.vstack((sites,proposal));losses=np.r_[losses,value];po_evaluated+=1
                        for arm in arms:log_eval(case,arm,'normal',len(losses),proposal,value,float(losses.min()),'cell',len(exact['po']))
                        pending=pending[1:]
                    if len(losses)<NORMAL_CAP and float(losses.min())>GATE:
                        greedy=edirect._greedy_proposal(sites,losses,200000+trajectory,iteration)
                        if greedy is not None:
                            value=float(edirect._prefix_losses_numpy(edirect._physical(greedy[None,:]),position)[0])
                            sites=np.vstack((sites,greedy));losses=np.r_[losses,value]
                            for arm in arms:log_eval(case,arm,'normal',len(losses),greedy,value,float(losses.min()),'greedy')
                    iteration+=1
                    base._write_csv(test.CLOSED,rows)
                count,best=len(losses),float(losses.min())
                if best>GATE:count,best=rescue(case,position,sites,losses,arms)
                for arm in arms:
                    observations=[r for r in rows if r.get('record_type')=='evaluation' and r.get('trajectory_id')==trajectory and r.get('arm')==arm]
                    hits=[r['evaluation_count'] for r in observations if r['best_loss']<=GATE]
                    result=dict(record_type='trajectory_summary',trajectory_id=trajectory,stratum=label,arm=arm,
                        success=best<=GATE,evaluations=count,first_gate_evaluation=min(hits) if hits else '',best_loss=best,
                        PO_cells_evaluated=po_evaluated,stalled_last20=bool(len(observations)>=20 and observations[-1]['best_loss']==observations[-20]['best_loss']))
                    rows.append(result);arm_results.append(result)
            completed.append(trajectory)
            base._write_csv(test.CLOSED,rows)
    except FidelityStop:
        rows.append(dict(record_type='hard_stop',status='STOPPED_DOWNSTREAM_DIVERGENCE',details=json.dumps(failure),
            completed_matched_trajectories=len(completed),planned_trajectories=18,
            efficiency_conclusion='NOT_JUDGED: downstream fidelity failed'))
    finally:
        for path,digest in static['protected_hashes'].items():assert base._sha256(Path(path))==digest,path
        base._write_csv(test.CLOSED,rows)
    summary=dict(status='STOPPED_DOWNSTREAM_DIVERGENCE' if failure else 'COMPLETE_BOUNDED_SCREEN',
        planned_trajectories=18,completed_matched_trajectories=len(completed),failure=failure,
        visited_learned_states=sum(r['record_type']=='visited_state_fidelity' for r in rows),
        protected_hashes_unchanged=True,no_retraining=True,production_Sobol_unchanged=True)
    (OUT/'closedloop_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
