"""Resume the saved bounded plan using the endpoint-exact structured scorer."""
import json
import sys
import time
import traceback
import inspect
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.sho import run_stage1b_spacefill_2x2 as test

OUT=Path("outputs/sho/stage1B/spacefill_closedloop_structured")
COMPARISON=OUT/"spacefill_closedloop_comparison.csv"
FIDELITY=OUT/"spacefill_fidelity_trace.csv"
base,cells,nearest,down=test.base,test.cells,test.nearest,test.down
selection,edirect=test.selection,test.edirect
imitation=selection.imitation
NORMAL_CAP,RESCUE_CAP,REFINE_CAP=160,40,40
GATE=1e-4


class FidelityStop(Exception):pass
class RefinementBudget(Exception):pass


def run():
    torch.set_num_threads(1)
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    static=json.loads((test.OUT/'static_summary.json').read_text())
    saved_plan=json.loads((test.OUT/'closedloop_plan.json').read_text())
    assert not static['summary']['distribution_fidelity_failures']
    assert not COMPARISON.exists(), 'Do not repeat an existing closed-loop run'
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
    from experiments.sho.run_stage1b_ordinary_bilinear_candidate import BilinearScore
    score_path=Path('outputs/sho/stage1B/ordinary_score_invariance_refine/ordinary_score_invariance_candidate.pt')
    assert base._sha256(score_path)==score_path.with_suffix('.sha256').read_text().split()[0]
    score_checkpoint=torch.load(score_path,weights_only=False)
    assert score_checkpoint['status']=='CANDIDATE'
    score=BilinearScore()
    score.load_state_dict(score_checkpoint['state_dict'])
    assert torch.equal(score.W.detach(),torch.tensor([[0.,-1.],[1.,1.]],dtype=torch.float64))
    _,selector=selection._load_search_models()
    static['protected_hashes'][str(score_path)]=base._sha256(score_path)
    static['protected_hashes'][str(test.OUT/'closedloop_plan.json')]=base._sha256(test.OUT/'closedloop_plan.json')
    for path,digest in static['protected_hashes'].items():assert base._sha256(Path(path))==digest,path
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
    assert plan['cases']==saved_plan['cases'] and plan['data_seed']==saved_plan['data_seed']
    plan['scorer_checkpoint']=str(score_path)
    plan['scorer_sha256']=base._sha256(score_path)
    plan['arms']={'A':'Sobol + deterministic','B':'spacefill + deterministic','C':'spacefill + learned / structured score'}
    (OUT/'closedloop_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[];completed=[];failure=None;fidelity_rows=[];context={}
    def score_hook(module,args,output):
        x=args[0].detach();actual=output.detach().numpy().ravel()
        expected=(x[:,2]*x[:,0]-(1-x[:,2])*x[:,1]).numpy()
        endpoint_errors=[];endpoint_variations=[]
        with torch.no_grad():
            for endpoint in (0.,1.):
                z=x.clone();z[:,2]=endpoint
                p=module.forward(z).numpy().ravel()
                target=(-z[:,1] if endpoint==0 else z[:,0]).numpy()
                altered=z.clone()
                axis=0 if endpoint==0 else 1
                altered[:,axis]=torch.flip(altered[:,axis],dims=(0,))
                pp=module.forward(altered).numpy().ravel()
                endpoint_errors.append(float(np.max(np.abs(p-target))))
                endpoint_variations.append(float(np.max(np.abs(p-pp))))
        exact=np.array_equal(actual,expected)
        fidelity_rows.append(dict(record_type='scorer_batch',**context,candidates=len(x),
            score_exact=exact,score_max_error=float(np.max(np.abs(actual-expected))),
            implied_pairwise_rank_comparisons=len(x)*(len(x)-1)//2,
            all_rankings_exact_from_equal_scores=exact,
            actual_u0_count=int(torch.sum(x[:,2]==0)),actual_u1_count=int(torch.sum(x[:,2]==1)),
            endpoint_u0_error=endpoint_errors[0],endpoint_u1_error=endpoint_errors[1],
            endpoint_u0_variation=endpoint_variations[0],endpoint_u1_variation=endpoint_variations[1]))
        if not exact or any(endpoint_errors) or any(endpoint_variations):
            raise FidelityStop('structured_scorer_exactness_failure')
    hook=score.register_forward_hook(score_hook)
    def log_eval(case,arm,mode,count,point,loss,best,kind,po_count=0):
        rows.append(dict(record_type='evaluation',trajectory_id=case[1],stratum=case[0],arm=arm,arm_label={'sobol_teacher':'A','spacefill_teacher':'B','spacefill_learned':'C'}[arm],mode=mode,coordinate_system='normalized' if mode=='normal' else 'physical',
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
        fidelity_rows.append(dict(record_type='common_rescue',trajectory_id=case[1],stratum=case[0],arms=json.dumps(arms),mode='boundary_rescue',evaluations=mode.new_evaluations,shared_unchanged_deterministic_machinery=True))
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
        fidelity_rows.append(dict(record_type='common_rescue',trajectory_id=case[1],stratum=case[0],arms=json.dumps(arms),mode='nested_refinement',evaluations=len(captures),shared_unchanged_deterministic_machinery=True))
        return count,best_loss
    try:
        for case in cases:
            label,trajectory=case;position=positions[trajectory]
            print(f'Closed-loop case {len(completed)+1}/18 {label} id={trajectory} elapsed={time.monotonic()-started:.1f}s',flush=True)
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
                        context.update(trajectory_id=trajectory,stratum=label,iteration=iteration,history_count=len(sites),mode='normal')
                        learned=test.trace_geometry(sites,losses,probes,record,state,True,models,locked)
                        other,learned_local=test.proposals(sites,losses,probes,learned['owners'],learned['distances'],learned['sizes'],learned['po'],score)
                        discrepancies=[key for key in ('owners','sizes','membership','representatives','hull','po') if not np.array_equal(exact[key],learned[key])]
                        if not np.array_equal(pending,other):discrepancies.append('proposals')
                        fidelity_rows.append(dict(record_type='visited_state_fidelity',trajectory_id=trajectory,stratum=label,
                            iteration=iteration,history_count=len(sites),N=min(100*len(sites),32768),
                            exact=not discrepancies,discrepancies=json.dumps(discrepancies),
                            teacher_PO=json.dumps(exact['po'].tolist()),learned_PO=json.dumps(learned['po'].tolist())))
                        for stage in ('owners','distances','spacing','radius','half_spacing','sizes','membership','representatives','hull','po'):
                            matching,total=test.exact_counts(exact[stage],learned[stage])
                            fidelity_rows.append(dict(record_type='stage_agreement',**context,stage=stage,exact_matches=matching,total=total,exact=matching==total))
                        matching,total=test.exact_counts(pending,other)
                        fidelity_rows.append(dict(record_type='stage_agreement',**context,stage='proposals',exact_matches=matching,total=total,exact=matching==total))
                        for r in learned_local:
                            if not r.get('empty'):
                                fidelity_rows.append(dict(record_type='ordinary_winner' if not r['largest'] else 'largest_winner',**context,
                                    site=r['site'],teacher_probe=r['teacher_probe_index'],learned_probe=r['chosen_probe_index'],
                                    strict_teacher_decision=r['teacher_top2_gap']>0,teacher_margin=r['teacher_top2_gap'],
                                    exact=r['teacher_probe_index']==r['chosen_probe_index']))
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
                            fidelity_rows.append(dict(record_type='next_point',trajectory_id=trajectory,stratum=label,iteration=iteration,
                                history_count=len(sites),teacher_choice=0,learned_choice=int(choice),
                                teacher_point=json.dumps(pending[0].tolist()),learned_point=json.dumps(pending[choice].tolist()),exact=choice==0))
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
                    base._write_csv(COMPARISON,rows)
                    base._write_csv(FIDELITY,fidelity_rows)
                    if use_learned_generator and iteration%5==0:print(f'  id={trajectory} iteration={iteration} evaluations={len(losses)} best={losses.min():.6g} exact scorer/pipeline so far',flush=True)
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
            base._write_csv(COMPARISON,rows)
            base._write_csv(FIDELITY,fidelity_rows)
            print(f'Completed matched case {len(completed)}/18: '+json.dumps(arm_results),flush=True)
    except FidelityStop as exc:
        if failure is None:failure=dict(reason=str(exc),**context)
        context.update({key:failure[key] for key in ('trajectory_id','stratum','iteration','history_count') if key in failure})
        fidelity_rows.append(dict(record_type='first_divergence',details=json.dumps(failure),**context))
        rows.append(dict(record_type='hard_stop',status='STOPPED_DOWNSTREAM_DIVERGENCE',details=json.dumps(failure),
            completed_matched_trajectories=len(completed),planned_trajectories=18,
            efficiency_conclusion='NOT_JUDGED: downstream fidelity failed'))
    except Exception as exc:
        failure=dict(reason=repr(exc),traceback=traceback.format_exc(),**context)
        fidelity_rows.append(dict(record_type='first_divergence_or_audit_error',details=json.dumps(failure)))
        print(traceback.format_exc(),flush=True)
    finally:
        hook.remove()
        for path,digest in static['protected_hashes'].items():assert base._sha256(Path(path))==digest,path
        base._write_csv(COMPARISON,rows)
        base._write_csv(FIDELITY,fidelity_rows)
    summary=dict(status='STOPPED_DOWNSTREAM_DIVERGENCE' if failure else 'COMPLETE_BOUNDED_SCREEN',
        planned_trajectories=18,completed_matched_trajectories=len(completed),failure=failure,
        visited_learned_states=sum(r['record_type']=='visited_state_fidelity' for r in fidelity_rows),
        scorer_batches=sum(r['record_type']=='scorer_batch' for r in fidelity_rows),
        elapsed_seconds=time.monotonic()-started,
        protected_hashes_unchanged=True,no_retraining=True,production_Sobol_unchanged=True)
    (OUT/'protected_hashes.json').write_text(json.dumps(static['protected_hashes'],indent=2)+'\n')
    (OUT/'closedloop_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
