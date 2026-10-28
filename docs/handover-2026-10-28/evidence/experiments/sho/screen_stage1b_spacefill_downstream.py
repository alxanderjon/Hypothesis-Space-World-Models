"""Frozen, single-state screens of recovered complete probe sets; no search."""
import csv
import inspect
import json
import shutil
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_spacefill_generator_smoke as smoke
from experiments.sho import run_stage1b_ordinary_cell_score_full as selection
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/spacefill_downstream_screen')
PRIOR=smoke.OUT
CSV=OUT/'spacefill_downstream_screen.csv'
TXT=OUT/'spacefill_downstream_summary.txt'
base,cells,nearest,downstream=smoke.base,smoke.cells,smoke.nearest,smoke.downstream


def select_proposals(sites,losses,probes,owners,distances,sizes,po,record,score,selector):
    """Existing post-PO composition, with the supplied learned geometry.

    No call to _cell_data (which would regenerate Sobol). Existing deterministic
    normalization, tie ordering, surrogate and duplicate exclusion are retained.
    Final ranker is called once; there is no true-loss evaluation or state update.
    """
    selected=np.asarray(sorted(po,key=lambda i:(sizes[i],losses[i],i)))
    surface=downstream.edirect.GaussianLocalResponse(sites,losses)
    selected_sizes=sizes[selected]
    weights=.5*downstream.edirect._normalize(selected_sizes)+.5*downstream.edirect._normalize(surface.loo_errors(selected))
    largest=int(np.argmax(selected_sizes))
    proposals=[]
    ordinary=empty=0
    for pos,site in enumerate(selected):
        mask=owners==site
        candidates=probes[mask]
        local_distance=distances[mask]
        if len(candidates)==0:
            empty+=1
            continue
        if pos==largest:
            winner=int(np.argmax(local_distance))
        else:
            ordinary+=1
            inputs=np.column_stack((downstream.edirect._normalize(local_distance),
                downstream.edirect._normalize(surface.predict(candidates)),np.full(len(candidates),weights[pos])))
            with torch.no_grad():
                scores=score(torch.from_numpy(inputs)).numpy()
            assert np.all(np.isfinite(scores))
            winner=int(np.argmax(scores))
        proposal=candidates[winner]
        if np.min(np.linalg.norm(sites-proposal,axis=1))>1e-12:
            proposals.append(proposal)
    fallback=not proposals
    if fallback:
        for i in np.argsort(distances)[::-1]:
            if distances[i]>1e-12:
                proposals.append(probes[i])
                break
    proposals=np.asarray(proposals,dtype=np.float64).reshape(-1,2)
    assert len(proposals)>0 and np.all(np.isfinite(proposals))
    features=selection.imitation._global_features(sites,losses,len(proposals),len(sites),
        selection.NORMAL_CAP,int(record['iteration']),selection.FULL_BOUNDS,'normal_edirect')
    choice,_=selection.imitation._model_choice(selector,sites,losses,proposals,'cell',features)
    return dict(proposal_count=len(proposals),ordinary_scored_cells=ordinary,selected_cells_without_probes=empty,
        proposal_fallback=fallback,final_choice_index=int(choice),final_choice=json.dumps(proposals[int(choice)].tolist()),
        proposals=json.dumps(proposals.tolist()))


def frozen_geometry(sites,losses,probes,record,state,models,locked,score,selector):
    distance,margin,threshold,half=models
    captured,scans=cells.capture_nearest(sites,probes,distance,margin,threshold,state)
    p,n=len(probes),len(sites)
    owners,distances=captured['winner'][:p],captured['winner_distance'][:p]
    radius,_,stats=cells.assigned_radius(owners,distances,n,margin,threshold)
    h=cells.predict_half(half,captured['winner_distance'][p:])
    replace,ties,invalid,wrong=cells.larger_candidate(margin,threshold,radius,h)
    assert not (invalid or wrong or stats['radius_pairwise_mistakes'] or stats['radius_invalid_logits'])
    sizes=np.where(replace,h,radius)
    trace=smoke.locked_trace(sizes,losses,record,state,locked)
    counts=np.bincount(owners,minlength=n)
    hist=np.histogramdd(probes,bins=(np.linspace(0,1,9),)*2)[0]
    row=dict(**stats,final_probe_count=p,
        pairwise_mistakes=sum(r['pairwise_mistakes'] for r in scans),
        argmin_mistakes=sum(r['final_argmin_mistakes'] for r in scans),
        nonempty_sites=int(np.sum(counts>0)),assigned_count_max=int(counts.max()),
        largest_assignment_share=float(counts.max()/p),probe_8x8_empty_bins=int(np.sum(hist==0)),
        probe_8x8_max_share=float(hist.max()/p),probe_8x8_count_cv=float(hist.std()/hist.mean()),
        cell_size_mean=float(sizes.mean()),cell_size_std=float(sizes.std()),
        group_count=len(trace['groups']),representative_count=len(trace['representatives']),
        hull_count=len(trace['hull']),PO_count=len(trace['po']),PO_fallback=trace['fallback'])
    for percentile in (0,10,25,50,75,90,95,100):
        row[f'cell_size_p{percentile}']=float(np.percentile(sizes,percentile))
    for key in ('representatives','hull','po'):
        row[key+'_sites']=json.dumps(sorted(trace[key]))
    row.update(select_proposals(sites,losses,probes,owners,distances,sizes,trace['po'],record,score,selector))
    return row,sizes,trace,counts


def run():
    torch.set_num_threads(1)
    verification=json.loads((OUT/'replay_verification.json').read_text())
    assert verification['passed'] and verification['all_metrics_exact']
    recovered_path=OUT/'spacefill_generator_recovered.pt'
    assert base._sha256(recovered_path)==recovered_path.with_suffix('.sha256').read_text().split()[0]
    recovered=torch.load(recovered_path,weights_only=False)
    assert recovered['status']=='RECOVERED_VERIFIED_PENDING_DOWNSTREAM'
    model=smoke.Generator(2)
    model.load_state_dict(recovered['state_dict'])
    model.eval().requires_grad_(False)
    hashes=json.loads((PRIOR/'summary.json').read_text())['protected_hashes']
    for path in (selection.CHECKPOINT_PATH,selection.SELECTOR_PATH,selection.SOURCE_SELECTOR_PATH,
                 Path(inspect.getsourcefile(selection)),recovered_path):
        hashes[str(path)]=base._sha256(path)
    assert base._sha256(selection.SELECTOR_PATH)==base._sha256(selection.SOURCE_SELECTOR_PATH)
    for path,digest in hashes.items():
        assert base._sha256(Path(path))==digest
    for path in (CSV,TXT):
        archive=path.with_name('before_recovery_'+path.name)
        if path.exists() and not archive.exists():
            shutil.copyfile(path,archive)
    distance,_=base._load_locked_distance()
    operator=torch.load(nearest.OPERATOR,weights_only=False)
    margin=nn.Linear(2,1,bias=False,dtype=torch.float64)
    margin.load_state_dict(torch.load(operator['margin_checkpoint'],weights_only=False)['state_dict'])
    threshold=nearest.threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(operator['threshold_checkpoint'],weights_only=False)['state_dict'])
    half=nn.Linear(1,1,bias=False,dtype=torch.float64)
    half.load_state_dict(torch.load(cells.OUT/'half_spacing_operator_locked.pt',weights_only=False)['state_dict'])
    models=(distance,margin,threshold,half)
    score,selector=selection._load_search_models()
    for module in (*models,score,selector):
        module.eval().requires_grad_(False)
    locked=downstream._load_downstream()
    payload=torch.load(base.turn.RAW_TRACE_PATH,weights_only=False)
    trajectories=base.ordinary_data._trajectory_map(payload)
    rows=[]
    comparisons=[]
    saved_geometry=list(csv.DictReader((PRIOR/'spacefill_generator_geometry.csv').open()))
    passed=True
    try:
        for budget in (1000,8600,32768):
            coord_path=OUT/f'spacefill_probes_N{budget}.npz'
            assert base._sha256(coord_path)==coord_path.with_suffix('.sha256').read_text().split()[0]
            archived=np.load(coord_path)
            learned=archived['points']
            state=int(archived['source_state_index'])
            regenerated,_=model.generate(budget,[0.,0.],[1.,1.])
            assert np.array_equal(learned,regenerated)
            record=payload['states'][state]
            n=int(record['history_count'])
            trajectory=trajectories[int(record['trajectory_id'])]
            sites=np.asarray(trajectory['evaluated_hypotheses_normalized'][:n],dtype=np.float64)
            losses=np.asarray(trajectory['evaluated_prefix_losses'][:n],dtype=np.float64)
            baseline=downstream.edirect._candidate_pool(n,int(record['generator_id']),int(record['iteration']))
            assert len(learned)==len(baseline) and min(100*n,32768)==budget
            results=[]
            for label,probes in (('sobol',baseline),('learned',learned)):
                print(f'Frozen screen N={budget} state={state} sites={n} probes={len(probes)} generator={label}',flush=True)
                metrics,sizes,trace,counts=frozen_geometry(sites,losses,probes,record,state,models,locked,score,selector)
                row=dict(record_type='pipeline',N=budget,source_state_index=state,site_count=n,generator=label,**metrics)
                rows.append(row)
                results.append((row,sizes,trace,counts))
                base._write_csv(CSV,rows)
            (br,bs,bt,bc),(lr,ls,lt,lc)=results
            delta=ls-bs
            relative=np.abs(delta)/np.maximum(bs,1e-15)
            changed=delta!=0
            material=np.abs(delta)>np.maximum(1e-12,.01*bs)
            newly_empty=(bc>0)&(lc==0)
            # Fixed diagnostic grid estimates geometry of newly empty cells;
            # it NEVER supplies owners to learned pipeline decisions.
            t=(np.arange(128)+.5)/128
            queries=np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2)
            area=np.bincount(cKDTree(sites).query(queries)[1],minlength=n)/len(queries)
            missing=newly_empty & (area*len(learned)>=5)
            po_ratio=len(lt['po'])/max(1,len(bt['po']))
            alarm=dict(nonfinite_or_nonpositive_sizes=bool(np.any(~np.isfinite(ls)) or np.any(ls<=0)),
                cell_size_max_doubled=bool(ls.max()>2*bs.max()),
                widespread_half_size_shift=bool(np.mean(relative>.5)>.2),
                PO_cardinality_collapse_or_explosion=not(.25<=po_ratio<=4),
                group_count_collapse=bool(len(lt['groups'])<.25*len(bt['groups'])),
                missing_well_supported_region=bool(np.any(missing)),
                new_empty_spatial_bins=lr['probe_8x8_empty_bins']>br['probe_8x8_empty_bins'],
                proposal_collapse=lr['proposal_count']==0)
            stable=not any(alarm.values())
            comparison=dict(record_type='comparison',N=budget,source_state_index=state,site_count=n,
                changed_d_i_count=int(changed.sum()),changed_d_i_fraction=float(changed.mean()),
                materially_changed_d_i_count=int(material.sum()),materially_changed_d_i_fraction=float(material.mean()),
                material_definition='abs(delta)>max(1e-12,0.01*Sobol_d_i)',
                d_i_delta_mean=float(delta.mean()),d_i_absolute_delta_mean=float(np.abs(delta).mean()),
                d_i_absolute_delta_max=float(np.abs(delta).max()),d_i_relative_delta_mean=float(relative.mean()),
                d_i_relative_delta_max=float(relative.max()),d_i_relative_delta_p95=float(np.percentile(relative,95)),
                newly_empty_sites=int(newly_empty.sum()),newly_occupied_sites=int(np.sum((bc==0)&(lc>0))),
                missing_well_supported_regions=int(missing.sum()),PO_cardinality_ratio=po_ratio,
                sobol_PO_count=br['PO_count'],learned_PO_count=lr['PO_count'],
                sobol_proposal_count=br['proposal_count'],learned_proposal_count=lr['proposal_count'],
                final_choice_distance=float(np.linalg.norm(np.asarray(json.loads(br['final_choice']))-np.asarray(json.loads(lr['final_choice'])))),
                interior_only_regression_percent_diagnostic=.83171031190408 if budget==1000 else 0.,
                interior_only_diagnostic_used_as_gate=False,healthy=stable,alarms=json.dumps(alarm))
            for key in ('groups','representatives','hull','po'):
                comparison[key+'_symmetric_difference']=len(bt[key]^lt[key])
                comparison[key+'_jaccard']=len(bt[key]&lt[key])/max(1,len(bt[key]|lt[key]))
            comparisons.append(comparison)
            rows.append(comparison)
            for i in range(n):
                rows.append(dict(record_type='site',N=budget,source_state_index=state,site_index=i,
                    sobol_d_i=float(bs[i]),learned_d_i=float(ls[i]),signed_change=float(delta[i]),
                    relative_absolute_change=float(relative[i]),material_change=bool(material[i]),
                    sobol_assigned_probes=int(bc[i]),learned_assigned_probes=int(lc[i]),
                    sobol_PO=i in bt['po'],learned_PO=i in lt['po']))
            base._write_csv(CSV,rows)
            print(json.dumps(comparison),flush=True)
            if not stable:
                passed=False
                break
    except Exception as exc:
        passed=False
        rows.append(dict(record_type='hard_stop',reason=repr(exc),traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
    unchanged=all(base._sha256(Path(p))==h for p,h in hashes.items())
    assert unchanged
    passed=passed and len(comparisons)==3
    summary=dict(record_type='summary',status='HEALTHY_CANDIDATE' if passed else 'STOPPED',
        replay_exact=True,verified_replay_values=verification['compared_values'],
        states_screened=len(comparisons),protected_hashes_unchanged=unchanged,
        trajectory_search=False,production_replaced=False,interior_only_diagnostic_gate=False)
    if passed:
        candidate=dict(recovered)
        candidate.update(status='CANDIDATE',screen_summary=summary,downstream_comparisons=comparisons,
            protected_hashes=hashes,probe_artifacts={str(OUT/f'spacefill_probes_N{n}.npz'):base._sha256(OUT/f'spacefill_probes_N{n}.npz') for n in (1000,8600,32768)},
            screen_implementation=str(Path(__file__).resolve()),screen_implementation_sha256=base._sha256(Path(__file__)))
        summary['candidate_checkpoint_sha256']=save_checkpoint(OUT/'spacefill_generator_candidate.pt',candidate)
    base._write_csv(CSV,[summary]+rows)
    (OUT/'downstream_manifest.json').write_text(json.dumps(dict(summary=summary,protected_hashes=hashes),indent=2)+'\n')
    text=['SPACE-FILLING DOWNSTREAM SCREEN',json.dumps(summary,indent=2),'',
        'Recovery: exactly 60 original seeded updates; all 378 saved history/geometry values reproduced exactly (max discrepancy 0). No original weight hash exists, so this verifies reproduction of the saved observations rather than byte identity to the lost weights.',
        'Recovered weights and complete probe arrays were saved/hash-checked before downstream evaluation. The downstream runner reloads them and checks regeneration exactly. No additional generator updates.',
        'Both sets use the same frozen learned distance, full candidate nearest/argmin scan, assigned-radius max, half-spacing and cell-size max, learned grouping/representative/hull and improvement operators. Exact ties retain existing deterministic scaffolding.',
        'After PO selection, both use the same saved ordinary score network and frozen final CandidateRanker, plus unchanged deterministic routing/normalization and GaussianLocalResponse surrogate. The score checkpoint has no LOCKED status metadata; it is held fixed, hash-verified and not promoted by this screen. No true losses are evaluated, no states are updated, and no trajectory loop runs.',
        'Health alarms: nonfinite/nonpositive cell sizes; doubled global maximum; >20% of sites changing >50%; PO count outside 1/4..4 times baseline; groups below 1/4 baseline; newly empty cells with >=5 expected probes on independent 128x128 geometry grid; new empty 8x8 probe bins; no proposals. These are smoke diagnostics, not a guarantee of search equivalence. Exact PO equality is not required.',
        'The 0.83171031190408% interior-only regression at N=1000 is diagnostic ONLY. Both inference inputs include all boundary-face probes. Changed cell-size estimates have no objective-quality direction by themselves.', '']
    for c in comparisons:
        n=c['N']
        records=[r for r in rows if r['record_type']=='pipeline' and r['N']==n]
        text.extend([f'N={n}',json.dumps(records,indent=2),json.dumps(c,indent=2),''])
    if comparisons:
        c=comparisons[0]
        text.append(f"N=1000 specific inspection: {c['changed_d_i_count']}/{c['site_count']} cell sizes changed; {c['materially_changed_d_i_count']} changed materially; mean absolute change={c['d_i_absolute_delta_mean']:.9g}, max={c['d_i_absolute_delta_max']:.9g}; PO count {c['sobol_PO_count']} -> {c['learned_PO_count']}; missing-region alarms={c['missing_well_supported_regions']}. {'No observed pathological downstream degradation in this matched state. The isolated effect of interior-only regression cannot be causally separated from other coordinate changes.' if c['healthy'] else 'Health alarm triggered; see comparisons.'}")
    text.extend(['', 'NEXT: A small bounded closed-loop search screen is justified; three static states cannot establish search quality or generalization. No search was run.' if passed else 'NEXT: Stop. No closed-loop screen recommendation or candidate promotion; inspect recorded failure.',
        '', 'All '+str(len(hashes))+' protected hashes unchanged. Workflow now persists pending-evaluation weights before evaluation and retains them after failed gates.'])
    TXT.write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run()
