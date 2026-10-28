"""Matched 2x2 static fidelity gate before any bounded trajectory experiment."""
import csv
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from experiments.sho import screen_stage1b_spacefill_downstream as screen

base,cells,nearest,down=screen.base,screen.cells,screen.nearest,screen.downstream
smoke,selection=screen.smoke,screen.selection
edirect=down.edirect
OUT=Path('outputs/sho/stage1B/spacefill_2x2')
STATIC=OUT/'spacefill_2x2_static.csv'
CLOSED=OUT/'spacefill_2x2_closedloop.csv'


def proposals(sites,losses,points,owners,distances,sizes,po,score=None):
    surface=edirect.GaussianLocalResponse(sites,losses)
    selected_sizes=sizes[po]
    weights=.5*edirect._normalize(selected_sizes)+.5*edirect._normalize(surface.loo_errors(po))
    largest=int(np.argmax(selected_sizes))
    weights[largest]=1.
    chosen=[]
    records=[]
    for pos,site in enumerate(po):
        indices=np.flatnonzero(owners==site)
        if not len(indices):
            records.append(dict(site=int(site),empty=True))
            continue
        candidates=points[indices]
        ds=distances[indices]
        if pos==largest:
            utility=ds
            predicted=utility
        else:
            a,b=edirect._normalize(ds),edirect._normalize(surface.predict(candidates))
            utility=weights[pos]*a-(1-weights[pos])*b
            if score is None:
                predicted=utility
            else:
                with torch.no_grad():
                    predicted=score(torch.from_numpy(np.column_stack((a,b,np.full(len(a),weights[pos]))))).numpy().ravel()
        exact=int(np.argmax(utility))
        winner=int(np.argmax(predicted))
        proposal=candidates[winner]
        accepted=np.min(np.linalg.norm(sites-proposal,axis=1))>1e-12
        if accepted:
            chosen.append(proposal)
        top=np.sort(utility)
        records.append(dict(site=int(site),empty=False,largest=pos==largest,probe_count=len(indices),
            teacher_probe_index=int(indices[exact]),chosen_probe_index=int(indices[winner]),
            teacher_coordinate=candidates[exact].tolist(),chosen_coordinate=proposal.tolist(),
            teacher_utility=float(utility[exact]),chosen_utility=float(utility[winner]),
            teacher_utility_regret=float(utility[exact]-utility[winner]),
            teacher_top2_gap=float(top[-1]-top[-2]) if len(top)>1 else 0.,
            prediction_teacher=float(predicted[exact]),prediction_chosen=float(predicted[winner]),
            proposal_displacement=float(np.linalg.norm(candidates[exact]-proposal)),accepted=bool(accepted)))
    if not chosen:
        # Identical existing global-farthest fallback; learned nearest distances
        # already available, no hidden non-tie owner selection is substituted.
        for i in np.argsort(distances)[::-1]:
            if distances[i]>1e-12:
                chosen.append(points[i]);break
    return np.asarray(chosen,dtype=np.float64).reshape(-1,2),records


def trace_geometry(sites,losses,probes,record,state,learned,models,locked):
    tree=cKDTree(sites)
    reference_d,reference_owner=tree.query(probes)
    reference_spacing=tree.query(sites,k=2)[0][:,1]
    if learned:
        metric,margin,threshold,half=models
        captured,scan=cells.capture_nearest(sites,probes,metric,margin,threshold,state)
        p=len(probes)
        owners,distances=captured['winner'][:p],captured['winner_distance'][:p]
        spacing=captured['winner_distance'][p:]
        radius,_,_=cells.assigned_radius(owners,distances,len(sites),margin,threshold)
        h=cells.predict_half(half,spacing)
        replace,_,invalid,wrong=cells.larger_candidate(margin,threshold,radius,h)
        assert not(invalid or wrong)
        sizes=np.where(replace,h,radius)
        grouped=down._learned_group_and_represent(sizes,losses,locked)
        hull_indices=down.hull_threshold._threshold_hull(locked['skew_weight'],locked['homogeneous_weight'],
            locked['hull_theta'],locked['hull_alpha'],grouped['grouped_sizes'],grouped['grouped_losses'])
        learned_trace=smoke.locked_trace(sizes,losses,record,state,locked)
        po=np.asarray(sorted(learned_trace['po'],key=lambda i:(sizes[i],losses[i],i)),dtype=np.int64)
    else:
        owners,distances,spacing=reference_owner,reference_d,reference_spacing
        radius=np.zeros(len(sites))
        np.maximum.at(radius,owners,distances)
        h=.5*spacing
        sizes=np.maximum(radius,h)
        grouped=down.grouping_audit._teacher_trace(sizes,losses)
        hull_indices,_=down.turn._teacher_hull_and_triples(grouped['grouped_sizes'],grouped['grouped_losses'])
        po=edirect._potentially_optimal_cells(sizes,losses)
    return dict(owners=owners,distances=distances,spacing=spacing,radius=radius,half_spacing=h,sizes=sizes,
        membership=grouped['membership'],representatives=grouped['representatives'],
        hull=grouped['representatives'][hull_indices],po=po)


def exact_counts(a,b):
    count=max(len(a),len(b));common=min(len(a),len(b))
    matches=np.all(a[:common]==b[:common],axis=1) if a.ndim==2 else a[:common]==b[:common]
    return int(np.sum(matches)),count


def run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not STATIC.exists()
    candidate_path=screen.OUT/'spacefill_generator_candidate.pt'
    candidate=torch.load(candidate_path,weights_only=False)
    assert candidate['status']=='CANDIDATE'
    assert base._sha256(candidate_path)==candidate_path.with_suffix('.sha256').read_text().split()[0]
    hashes=dict(candidate['protected_hashes'])
    hashes[str(candidate_path)]=base._sha256(candidate_path)
    for module in (screen,smoke,cells,nearest,down,selection,edirect):
        path=Path(inspect.getsourcefile(module));hashes.setdefault(str(path),base._sha256(path))
    for p,h in hashes.items():assert base._sha256(Path(p))==h,p
    generator=smoke.Generator(2)
    generator.load_state_dict(candidate['state_dict'])
    metric,_=base._load_locked_distance()
    operator=torch.load(nearest.OPERATOR,weights_only=False)
    margin=nn.Linear(2,1,bias=False,dtype=torch.float64)
    margin.load_state_dict(torch.load(operator['margin_checkpoint'],weights_only=False)['state_dict'])
    threshold=nearest.threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(operator['threshold_checkpoint'],weights_only=False)['state_dict'])
    half=nn.Linear(1,1,bias=False,dtype=torch.float64)
    half.load_state_dict(torch.load(cells.OUT/'half_spacing_operator_locked.pt',weights_only=False)['state_dict'])
    score,selector=selection._load_search_models()
    models=metric,margin,threshold,half
    for m in (generator,*models,score,selector):m.eval().requires_grad_(False)
    locked=down._load_downstream()
    payload=torch.load(base.turn.RAW_TRACE_PATH,weights_only=False)
    trajectories=base.ordinary_data._trajectory_map(payload)
    # Exactly the data seed/count that produced these persisted teacher states.
    positions,_,_=selection._make_exact_labeled_split(selection.imitation.TRAIN_COUNT,selection.imitation.TRAIN_SEED)
    positions=positions.cpu().numpy()
    original_geometry=list(csv.DictReader((smoke.OUT/'spacefill_generator_geometry.csv').open()))
    rows=[];failures=[]
    for budget,state in ((1000,17105),(8600,5660),(32768,17179)):
        record=payload['states'][state];n=int(record['history_count'])
        trajectory=trajectories[int(record['trajectory_id'])]
        sites=np.asarray(trajectory['evaluated_hypotheses_normalized'][:n],dtype=np.float64)
        losses=np.asarray(trajectory['evaluated_prefix_losses'][:n],dtype=np.float64)
        position=positions[int(record['trajectory_id'])]
        check=edirect._prefix_losses_numpy(edirect._physical(sites),position)
        assert np.allclose(check,losses,rtol=2e-13,atol=1e-14),'Objective data reconstruction mismatch'
        sobol=edirect._candidate_pool(n,int(record['generator_id']),int(record['iteration']))
        learned,_=generator.generate(budget,[0.,0.],[1.,1.])
        assert np.array_equal(learned,np.load(screen.OUT/f'spacefill_probes_N{budget}.npz')['points'])
        results={}
        for arm,probes,learned_machine in (('A',sobol,False),('B',sobol,True),('C',learned,False),('D',learned,True)):
            print(f'Static N={budget} arm={arm}',flush=True)
            trace=trace_geometry(sites,losses,probes,record,state,learned_machine,models,locked)
            proposed,local=proposals(sites,losses,probes,trace['owners'],trace['distances'],trace['sizes'],trace['po'],score if learned_machine else None)
            if arm=='A':
                production=edirect._cell_proposals(sites,losses,int(record['generator_id']),int(record['iteration']))
                assert np.array_equal(proposed,production),'A differs from actual production reference'
            choice=0
            if learned_machine:
                features=selection.imitation._global_features(sites,losses,len(proposed),n,selection.NORMAL_CAP,
                    int(record['iteration']),selection.FULL_BOUNDS,'normal_edirect')
                choice,_=selection.imitation._model_choice(selector,sites,losses,proposed,'cell',features)
            # Static objective diagnostics only: evaluate the proposed coordinates
            # without updating history or recomputing any trajectory decision.
            evaluation_losses=edirect._prefix_losses_numpy(edirect._physical(proposed),position)
            trace.update(proposals=proposed,chosen=proposed[choice:choice+1],evaluation_losses=evaluation_losses,
                final_loss=evaluation_losses[choice:choice+1],local=local)
            results[arm]=trace
            source='learned' if arm in ('C','D') else 'sobol'
            coverage=next(r for r in original_geometry if int(r['N'])==budget and r['generator']==source)
            row=dict(record_type='arm',N=budget,state=state,arm=arm,site_count=n,probe_count=len(probes),
                probe_source=source,machinery='learned' if learned_machine else 'deterministic',
                estimated_covering_radius=coverage['estimated_interior_covering_radius'],
                boundary_covering_radius=coverage['boundary_face_covering_radius'],
                nearest_probe_p50=coverage['nearest_probe_distance_p50'],nearest_probe_p99=coverage['nearest_probe_distance_p99'],
                minimum_pairwise_separation=coverage['minimum_pairwise_separation'],
                cell_size_min=float(trace['sizes'].min()),cell_size_median=float(np.median(trace['sizes'])),
                cell_size_mean=float(trace['sizes'].mean()),cell_size_max=float(trace['sizes'].max()),
                groups=len(np.unique(trace['membership'])),PO_count=len(trace['po']),final_choice_index=int(choice),
                proposed_count=len(proposed),final_selected_loss=float(evaluation_losses[choice]),
                static_candidate_evaluations_only=True,trajectory_updates=0)
            for key in ('sizes','membership','representatives','hull','po','proposals','chosen','evaluation_losses'):
                row[key]=json.dumps(trace[key].tolist())
            rows.append(row)
            for r in local:
                rows.append(dict(record_type='within_cell',N=budget,state=state,arm=arm,**r))
        for left,right,kind in (('B','A','fidelity'),('D','C','fidelity'),('C','A','generator_effect'),('D','B','net_effect')):
            a,b=results[right],results[left]
            comparison=left+'_vs_'+right
            for stage in ('owners','distances','spacing','radius','half_spacing','sizes','membership','representatives','hull','po','proposals','chosen','evaluation_losses','final_loss'):
                matching,total=exact_counts(a[stage],b[stage])
                row=dict(record_type='stage_comparison',N=budget,state=state,comparison=comparison,kind=kind,
                    stage=stage,exact_matches=matching,total=total,mismatches=total-matching,exact=matching==total)
                if len(a[stage])==len(b[stage]) and a[stage].size:
                    row['max_absolute_difference']=float(np.max(np.abs(a[stage]-b[stage])))
                rows.append(row)
            added=sorted(set(b['po'])-set(a['po']));removed=sorted(set(a['po'])-set(b['po']))
            counts=np.bincount(a['owners'],minlength=n)
            boundary=lambda ids:int(np.sum(np.min(np.column_stack((sites[ids],1-sites[ids])),axis=1)<.05)) if ids else 0
            boundary_proposal=lambda x:float(np.mean(np.min(np.column_stack((x,1-x)),axis=1)<.05))
            rows.append(dict(record_type='choice_changes',N=budget,state=state,comparison=comparison,
                PO_added=json.dumps([int(i) for i in added]),PO_removed=json.dumps([int(i) for i in removed]),
                added_PO_previously_sparse_count=int(np.sum(counts[added]<=3)),sparse_definition='<=3 probes assigned in reference arm',
                added_PO_boundary_count=boundary(added),removed_PO_boundary_count=boundary(removed),boundary_band=.05,
                reference_proposal_boundary_fraction=boundary_proposal(a['proposals']),new_proposal_boundary_fraction=boundary_proposal(b['proposals']),
                final_selected_distance=float(np.linalg.norm(a['chosen']-b['chosen'])),
                reference_selected_loss=float(a['final_loss'][0]),new_selected_loss=float(b['final_loss'][0])))
            if kind=='fidelity':
                for stage in ('owners','sizes','membership','representatives','hull','po','proposals','chosen'):
                    if not np.array_equal(a[stage],b[stage]):
                        failures.append(dict(N=budget,comparison=comparison,stage=stage))
                for r in b['local']:
                    if not r.get('empty') and r['chosen_probe_index']!=r['teacher_probe_index']:
                        coordinates=np.asarray([r['teacher_coordinate'],r['chosen_coordinate']])
                        values=edirect._prefix_losses_numpy(edirect._physical(coordinates),position)
                        rows.append(dict(record_type='divergence_diagnosis',N=budget,state=state,comparison=comparison,
                            first_divergent_operator='ordinary_cell_score_network',
                            teacher_objective_loss=float(values[0]),learned_objective_loss=float(values[1]),
                            **r))
        base._write_csv(STATIC,rows)
    for p,h in hashes.items():assert base._sha256(Path(p))==h,p
    distribution_failures=[r for r in failures if r['comparison']=='D_vs_C']
    status='HARD_STOP_STATIC_DIVERGENCE' if distribution_failures else 'STATIC_PASSED_READY_FOR_BOUNDED_CLOSED_LOOP'
    summary=dict(record_type='summary',status=status,distribution_fidelity_failures=distribution_failures,
        all_fidelity_failures=failures,protected_hashes_unchanged=True,no_retraining=True,production_Sobol_unchanged=True)
    (OUT/'static_summary.json').write_text(json.dumps(dict(summary=summary,protected_hashes=hashes),indent=2)+'\n')
    base._write_csv(STATIC,[dict(record_type='summary',status=status,failures=json.dumps(failures))]+rows)
    if distribution_failures:
        base._write_csv(CLOSED,[dict(record_type='hard_stop',status='NOT_RUN',reason='Learned downstream differs from deterministic code on learned probes',
            static_failures=json.dumps(distribution_failures),trajectory_count=0,
            success_rate='NOT_MEASURED',search_efficiency='NOT_MEASURED')])
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
