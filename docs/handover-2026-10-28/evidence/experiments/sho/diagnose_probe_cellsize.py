"""Static sampled-reference Voronoi-radius and proposal audit; never searches."""
import argparse,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_2x2 as test
from experiments.sho import run_stage1b_spacefill_generator_smoke as smoke

OUT=Path('outputs/sho/stage1B/probe_cellsize_accuracy')
M=1048576
class Deadline(Exception):pass
def boundary(n):
    k=min(257,max(9,math.ceil(math.sqrt(n))));t=np.linspace(0.,1.,k)
    return np.unique(np.concatenate([np.column_stack((t,np.full(k,v))) for v in (0.,1.)]+[np.column_stack((np.full(k,v),t)) for v in (0.,1.)]),axis=0)
def trace(radius,spacing,losses):
    sizes=np.maximum(radius,.5*spacing)
    group=src.down.grouping_audit._teacher_trace(sizes,losses)
    hi,_=src.down.turn._teacher_hull_and_triples(group['grouped_sizes'],group['grouped_losses'])
    signatures=[None]*len(sizes)
    for g in np.unique(group['membership']):
        ix=np.flatnonzero(group['membership']==g);sig=tuple(ix.tolist())
        for j in ix:signatures[j]=sig
    return dict(sizes=sizes,groups=signatures,hull=set(group['representatives'][hi].tolist()),
                po=set(src.edirect._potentially_optimal_cells(sizes,losses).tolist()),ngroups=len(group['representatives']))
def radius_of(tree,points,n):
    distance,owner=tree.query(points);r=np.zeros(n);np.maximum.at(r,owner,distance)
    return r,distance,owner
def stat(error):
    return dict(MAE=float(np.mean(np.abs(error))),RMSE=float(np.sqrt(np.mean(error**2))),signed_bias=float(error.mean()),
                worst_underestimation=float(max(0.,-error.min())),worst_overestimation=float(max(0.,error.max())))
def run(started,deadline):
    OUT.mkdir(parents=True,exist_ok=True)
    def alarm(*args):raise Deadline('wall-clock deadline')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.time()))
    torch.set_num_threads(1);rows=[];disagreements=[];initial=[];state_results=[];error=None
    candidate=Path('outputs/sho/stage1B/spacefill_downstream_screen/spacefill_generator_candidate.pt')
    protected=json.loads(Path('outputs/sho/stage1B/grid_branch_bound/grid_branch_bound_candidate.json').read_text())['protected_hashes']
    protected[str(candidate)]=src.sha(candidate)
    model=smoke.Generator(2);model.load_state_dict(torch.load(candidate,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    saved=torch.load('outputs/sho/stage1B/sobol_vs_spacefill_quick/quick_screen_states.pt',weights_only=False)['runs']
    cache={};states=[]
    for a in saved[::2]:
        tid=a['case']['trajectory_id'];b=next(r for r in saved if r['case']['trajectory_id']==tid and r['arm']=='B')
        for history,label,target in [(a,'initial',10),(a,'medium_A',80),(a,'saturated_A',328),(b,'saturated_B',328),(a,'late_A',len(a['sites']))]:
            if label=='initial':n,it=10,0
            elif label=='late_A':n,it=len(history['sites']),history['iteration']
            else:
                first={}
                for e in history['events']:
                    if e['kind']=='cell':first.setdefault(e['iteration'],e['evaluation']-1)
                it,n=next((it,n) for it,n in first.items() if n>=target)
            states.append(dict(history=history,label=label,n=n,it=it))
    try:
        for si,s in enumerate(states):
            if deadline-time.time()<30:raise Deadline('reporting reserve')
            history=s['history'];case=history['case'];n=s['n'];sites=history['sites'][:n];losses=history['losses'][:n]
            N=min(100*n,32768);tree=cKDTree(sites);spacing=tree.query(sites,k=2)[0][:,1]
            meta=dict(state_id=f"{case['trajectory_id']}_{s['label']}",trajectory_id=case['trajectory_id'],category=case['stratum'],
                history_arm=history['arm'],stage=s['label'],site_count=n,iteration=s['it'],interior_probe_budget=N,reference_interior_count=M)
            reference_boundary=boundary(M);rng=np.random.default_rng(26092600+si)
            rb,_,boundary_owner=radius_of(tree,reference_boundary,n);ref=rb.copy();ref_counts=np.zeros(n,dtype=int);half=None;sample=None
            for chunk in range(4):
                pts=rng.random((M//4,2));rr,dd,oo=radius_of(tree,pts,n);ref=np.maximum(ref,rr);ref_counts+=np.bincount(oo,minlength=n)
                if chunk==0:sample=pts[:32768].copy()
                if chunk==1:half=ref.copy()
            rt=trace(ref,spacing,losses);ht=trace(half,spacing,losses)
            bm=np.zeros(n,bool);bm[np.unique(boundary_owner)]=True
            sparse=spacing>=np.quantile(spacing,.75)
            quartile=np.searchsorted(np.quantile(ref,[.25,.5,.75]),ref,side='right')+1
            A=src.edirect._candidate_pool(n,case['generator_id'],s['it'])
            if N not in cache:cache[N]=model.generate(N,[0.,0.],[1.,1.])[0]
            B=cache[N]
            isboundary=lambda x:np.any((x==0.)|(x==1.),axis=1)
            assert np.array_equal(A[isboundary(A)],B[isboundary(B)])
            variants={};state_row=dict(record_type='reference',**meta,reference_zero_interior_cells=int(np.sum(ref_counts==0)),
                reference_boundary_count=len(reference_boundary),reference_half_to_full_mean_radius_increase=float(np.mean(ref-half)),
                reference_half_to_full_max_radius_increase=float(np.max(ref-half)),reference_PO_change=len(rt['po']^ht['po']),
                reference_hull_change=len(rt['hull']^ht['hull']))
            rows.append(state_row)
            for name,points in [('Sobol',A),('spacefill',B)]:
                radius,dist,owner=radius_of(tree,points,n);t=trace(radius,spacing,losses);err=radius-ref
                changed=np.array([a!=b for a,b in zip(t['groups'],rt['groups'])]);under=err<0
                corrected=trace(np.maximum(radius,ref),spacing,losses)
                under_group=np.array([a!=b for a,b in zip(t['groups'],corrected['groups'])])
                variants[name]=dict(radius=radius,dist=dist,owner=owner,trace=t,error=err,points=points)
                masks=dict(all=np.ones(n,bool),boundary_cells=bm,sparse_cells=sparse,reference_hull=np.isin(np.arange(n),list(rt['hull'])),
                    reference_PO=np.isin(np.arange(n),list(rt['po'])),generator_hull=np.isin(np.arange(n),list(t['hull'])),generator_PO=np.isin(np.arange(n),list(t['po'])))
                masks.update({f'radius_quartile_{q}':quartile==q for q in range(1,5)})
                for name_mask,mask in masks.items():
                    if mask.any():rows.append(dict(record_type='radius_metrics',**meta,generator=name,subset=name_mask,cells=int(mask.sum()),**stat(err[mask])))
                rows.append(dict(record_type='selection_changes',**meta,generator=name,groups=len(set(t['groups'])),reference_groups=rt['ngroups'],
                    grouping_membership_changed_sites=int(changed.sum()),underestimated_and_group_changed_sites=int(np.sum(under&changed)),
                    group_changed_sites_after_correcting_underestimates=int(under_group.sum()),
                    hull_added=len(t['hull']-rt['hull']),hull_removed=len(rt['hull']-t['hull']),PO_added=len(t['po']-rt['po']),PO_removed=len(rt['po']-t['po']),
                    PO_changes_after_correcting_underestimates=len(t['po']^corrected['po']),hull_changes_after_correcting_underestimates=len(t['hull']^corrected['hull'])))
                for j in range(n):rows.append(dict(record_type='site',**meta,generator=name,site=j,x=sites[j,0],y=sites[j,1],reference_radius=ref[j],radius=radius[j],radius_error=err[j],
                    final_cell_size=t['sizes'][j],reference_cell_size=rt['sizes'][j],boundary_cell=bool(bm[j]),sparse_cell=bool(sparse[j]),radius_quartile=int(quartile[j]),
                    generator_hull=j in t['hull'],reference_hull=j in rt['hull'],generator_PO=j in t['po'],reference_PO=j in rt['po'],group_changed=bool(changed[j]),
                    reference_interior_hits=int(ref_counts[j])))
                interior=points[~isboundary(points)];nd=cKDTree(interior).query(interior,k=2)[0][:,1]
                cover=cKDTree(points).query(sample)[0]
                rows.append(dict(record_type='geometry',**meta,generator=name,probe_count=len(points),interior_count=len(interior),boundary_count=int(isboundary(points).sum()),
                    nn_min=float(nd.min()),nn_p01=float(np.quantile(nd,.01)),nn_p10=float(np.quantile(nd,.1)),nn_median=float(np.median(nd)),nn_p90=float(np.quantile(nd,.9)),
                    nn_max=float(nd.max()),nn_CV=float(nd.std()/nd.mean()),estimated_complete_covering_radius=float(cover.max()),nearest_reference_p99=float(np.quantile(cover,.99))))
                for bins in (2,4,8,16,32,64,128):
                    count=np.histogramdd(interior,bins=(np.linspace(0,1,bins+1),)*2)[0];cdf=count.cumsum(0).cumsum(1)/len(interior)
                    edge=np.arange(1,bins+1)/bins;expected=edge[:,None]*edge[None,:]
                    rows.append(dict(record_type='multiscale',**meta,generator=name,bins_per_axis=bins,count_mean=float(count.mean()),count_variance=float(count.var()),
                        count_fano=float(count.var()/count.mean()),count_CV=float(count.std()/count.mean()),empty_bin_fraction=float(np.mean(count==0)),max_bin_count=int(count.max()),
                        grid_anchored_discrepancy=float(np.max(np.abs(cdf-expected))),grid_anchored_RMSE=float(np.sqrt(np.mean((cdf-expected)**2)))))
            aa,bb=variants['Sobol'],variants['spacefill'];disagree=aa['trace']['po']^bb['trace']['po']
            for j in sorted(disagree):
                ae,be=aa['error'][j],bb['error'][j]
                disagreements.append(dict(**meta,site=j,x=sites[j,0],y=sites[j,1],site_loss=losses[j],reference_radius=ref[j],Sobol_radius=aa['radius'][j],Sobol_error=ae,
                    spacefill_radius=bb['radius'][j],spacefill_error=be,closer_to_reference='Sobol' if abs(ae)<abs(be) else 'spacefill' if abs(be)<abs(ae) else 'tie',
                    reference_PO=j in rt['po'],Sobol_PO=j in aa['trace']['po'],spacefill_PO=j in bb['trace']['po'],
                    reference_half_PO=j in ht['po'],reference_radius_half=half[j],reference_cell_size=rt['sizes'][j],Sobol_cell_size=aa['trace']['sizes'][j],spacefill_cell_size=bb['trace']['sizes'][j]))
            state_results.append(dict(**meta,Sobol_MAE=float(np.abs(aa['error']).mean()),spacefill_MAE=float(np.abs(bb['error']).mean()),PO_disagreements=len(disagree)))
            if s['label']=='initial':
                detail=dict(record_type='initial_divergence',**meta,reference_PO=json.dumps(sorted(rt['po'])),Sobol_PO=json.dumps(sorted(aa['trace']['po'])),spacefill_PO=json.dumps(sorted(bb['trace']['po'])))
                for name,v in variants.items():
                    for mode,sizes,po in [('own',v['trace']['sizes'],v['trace']['po']),('reference',rt['sizes'],rt['po'])]:
                        order=np.asarray(sorted(po,key=lambda i:(sizes[i],losses[i],i)))
                        pending,records=test.proposals(sites,losses,v['points'],v['owner'],v['dist'],sizes,order)
                        accepted=[x for x in records if not x.get('empty') and x.get('accepted')]
                        first=accepted[0] if accepted else {}
                        detail[f'{name}_{mode}_first_point']=json.dumps(pending[0].tolist())
                        detail[f'{name}_{mode}_first_cell']=first.get('site','fallback')
                        detail[f'{name}_{mode}_first_cell_largest']=first.get('largest','')
                        detail[f'{name}_{mode}_first_cell_teacher_gap']=first.get('teacher_top2_gap','')
                        if mode=='own':
                            h=next(r for r in saved if r['case']['trajectory_id']==case['trajectory_id'] and r['arm']==('A' if name=='Sobol' else 'B'))
                            detail[f'{name}_reproduces_archived_first_point']=bool(np.array_equal(src.edirect._physical(pending[0]),h['events'][10]['point']))
                    detail[f'{name}_first_point_unchanged_with_reference_sizes']=detail[f'{name}_own_first_point']==detail[f'{name}_reference_first_point']
                detail['different_first_points_with_shared_reference_sizes']=detail['Sobol_reference_first_point']!=detail['spacefill_reference_first_point']
                initial.append(detail);rows.append(detail)
            src.write_csv(OUT/'probe_cellsize_accuracy_audit.csv',rows)
            src.write_csv(OUT/'probe_po_disagreements.csv',disagreements or [dict(status='no_disagreements')])
            print(f"{si+1}/{len(states)} {meta['state_id']} n={n}: MAE Sobol={state_results[-1]['Sobol_MAE']:.5g} SF={state_results[-1]['spacefill_MAE']:.5g} PO disagreements={len(disagree)} elapsed={time.time()-started:.1f}s",flush=True)
    except Exception:error=traceback.format_exc();print(error,flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        unchanged=all(src.sha(p)==h for p,h in protected.items())
        site_rows=[r for r in rows if r['record_type']=='site'];aggregates=[]
        for name in ('Sobol','spacefill'):
            selected=[r for r in site_rows if r['generator']==name]
            for subset in ('all','boundary_cells','sparse_cells','reference_hull','reference_PO','radius_quartile_1','radius_quartile_2','radius_quartile_3','radius_quartile_4'):
                def include(r):
                    if subset=='all':return True
                    if subset=='boundary_cells':return r['boundary_cell']
                    if subset=='sparse_cells':return r['sparse_cell']
                    if subset.startswith('radius_quartile_'):return r['radius_quartile']==int(subset[-1])
                    return r[subset]
                values=np.array([r['radius_error'] for r in selected if include(r)])
                if len(values):aggregates.append(dict(record_type='aggregate_radius_metrics',generator=name,subset=subset,cells=len(values),**stat(values)))
        rows+=aggregates
        src.write_csv(OUT/'probe_cellsize_accuracy_audit.csv',rows or [dict(error=error)])
        src.write_csv(OUT/'probe_po_disagreements.csv',disagreements or [dict(status='no_disagreements')])
        metrics=[r for r in rows if r['record_type']=='selection_changes'];refs=[r for r in rows if r['record_type']=='reference']
        summary=dict(states_completed=len(state_results),states_planned=len(states),elapsed_seconds=time.time()-started,error=error,protected_hashes_unchanged=unchanged,
            spacefill_lower_MAE_states=sum(r['spacefill_MAE']<r['Sobol_MAE'] for r in state_results),
            PO_disagreement_entries=len(disagreements),Sobol_closer_on_PO_disagreements=sum(r['closer_to_reference']=='Sobol' for r in disagreements),
            spacefill_closer_on_PO_disagreements=sum(r['closer_to_reference']=='spacefill' for r in disagreements),
            reference_PO_stable_half_to_full_states=sum(r['reference_PO_change']==0 for r in refs),
            initial_divergence_persists_shared_reference=sum(r['different_first_points_with_shared_reference_sizes'] for r in initial))
        lines=['PROBE GENERATOR PER-CELL DIAGNOSIS',json.dumps(summary,indent=2),
            'Static matched states only; zero objective evaluations, search steps, training, or checkpoint modifications.',
            '30 planned states: six trajectories, initial/medium-A/saturated-A/saturated-B/late-A histories. Both generators compared on IDENTICAL sites/losses per state.',
            f'Reference: {M} independent IID uniform points per state (seed 26092600+state ordinal), plus production-rule endpoint boundary grid capped at 257 samples per face.',
            'Reference is a finite-sample LOWER BOUND on geometric Voronoi radius. Positive errors are relative to the sampled reference, not proof of true overestimation.',
            'Half/full-reference checks quantify remaining sampling uncertainty; reference points never enter search.',
            'Boundary cells own at least one reference face sample; very narrow boundary intersections can be missed. Sparse cells are upper-quartile nearest-site spacing.',
            'Quartiles use reference radius. Actual downstream size=max(radius,0.5*nearest-site-spacing). Teacher grouping/hull/PO logic unchanged.',
            'Grouping comparisons use sets of member sites, not arbitrary numeric group labels. Underestimate correction is a STATIC intervention replacing radius with max(radius,reference).',
            'Multiscale metrics use interior probes only; grid-anchored discrepancy is sampled on bin edges, NOT full star discrepancy. NN and covering statistics are sample estimates.',
            'Pooled radius summaries weight cell-state instances equally, not trajectories; inspect per-state and subset rows as well.','', 'AGGREGATE RADIUS ERRORS']
        lines += [json.dumps(r) for r in aggregates]
        lines += ['', 'SELECTION CHANGES RELATIVE TO SAMPLED REFERENCE']
        for name in ('Sobol','spacefill'):
            rr=[r for r in metrics if r['generator']==name]
            lines.append(json.dumps(dict(generator=name,states=len(rr),states_group_changes=sum(r['grouping_membership_changed_sites']>0 for r in rr),
                changed_group_sites=sum(r['grouping_membership_changed_sites'] for r in rr),
                states_hull_changes=sum(r['hull_added']+r['hull_removed']>0 for r in rr),hull_membership_changes=sum(r['hull_added']+r['hull_removed'] for r in rr),
                states_PO_changes=sum(r['PO_added']+r['PO_removed']>0 for r in rr),PO_membership_changes=sum(r['PO_added']+r['PO_removed'] for r in rr),
                underestimation_corrected_group_sites=sum(r['group_changed_sites_after_correcting_underestimates'] for r in rr))))
        lines += ['', 'FIRST-DIVERGENCE STATIC COUNTERFACTUALS']+[json.dumps(r) for r in initial]
        (OUT/'probe_generator_diagnosis.txt').write_text('\n'.join(lines)+'\n')
        print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--started',type=float,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args();run(a.started,a.deadline)
