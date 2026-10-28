"""Four-arm static evaluation of a persisted constrained continuation."""
import csv,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import refine_fresh_spacefill as tr
from experiments.sho import evaluate_fresh_spacefill_smoke as prior_eval
from experiments.sho import run_stage1b_spacefill_2x2 as teacher
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_generator_smoke as old
from experiments.sho.fresh_spacefill_generator import FreshGenerator

OUT=tr.OUT;PRIOR=prior_eval.OUT
def read(name):return list(csv.DictReader((PRIOR/name).open()))
def literal_ratio(a,b):return a/b if b>0 else 0. if a==0 else float('inf')
def run():
    started=time.time();torch.set_num_threads(1)
    summary=json.loads((OUT/'training_summary.json').read_text())
    assert src.sha(summary['checkpoint'])==summary['sha256']
    model=FreshGenerator(2);model.load_state_dict(torch.load(summary['checkpoint'],weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    # These three frozen arms were already evaluated on exactly this reference,
    # seed, budgets, conditions and boundary support. Reuse recorded measurements.
    geometry=read('fresh_spacefill_geometry.csv');multi=read('fresh_spacefill_multiscale.csv');fresh=read('fresh_spacefill_freshness_union.csv')
    rename={'Sobol':'Sobol','old_spacefill':'old_spacefill','new_spacefill':'fresh_v1'}
    for collection in (geometry,multi,fresh):
        for r in collection:r['generator']=rename[r['generator']]
    t=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    conditions=[(100+j,'cell') for j in range(4)]+[(100,'greedy')];coordinates={}
    for n in tr.BUDGETS:
        pools=[];current=[]
        for it,pool in conditions:
            points,interior=model.generate(n,it,pool)
            assert np.array_equal(points,model.generate(n,it,pool)[0])
            assert np.array_equal(points[np.any((points==0.)|(points==1.),axis=1)],old.faces(n,2)[0])
            pools.append((points,interior));coordinates[f'N{n}_iteration{it}_{pool}']=points
            m=dict(N=n,generator='refined',iteration=it,pool_type=pool,**prior_eval.metrics(points,interior,reference,n))
            geometry.append(m);current.append(m)
            for k in tr.SCALES:
                for shift in tr.SHIFTS:
                    count=np.histogramdd((interior+shift/k)%1,bins=(np.linspace(0,1,k+1),)*2)[0]
                    edge=np.arange(1,k+1)/k;error=count.cumsum(0).cumsum(1)/n-edge[:,None]*edge[None,:]
                    multi.append(dict(N=n,generator='refined',iteration=it,pool_type=pool,bins_per_axis=k,shift_fraction_of_bin=shift,
                        grid_anchored_discrepancy=float(np.abs(error).max()),grid_anchored_RMSE=float(np.sqrt(np.mean(error**2))),
                        count_variance=float(count.var()),count_fano=float(count.var()/count.mean()),empty_bin_fraction=float(np.mean(count==0))))
        for a,b,label in ((0,1,'cell_t_to_t1'),(1,2,'cell_t1_to_t2'),(2,3,'cell_t2_to_t3'),(0,4,'cell_to_greedy')):
            fresh.append(dict(record_type='overlap',N=n,generator='refined',comparison=label,**prior_eval.overlaps(pools[a][1],pools[b][1],n)))
        union=np.unique(np.concatenate([p[0] for p in pools[:4]]),axis=0);d=cKDTree(union).query(reference)[0]
        di=cKDTree(np.vstack([p[1] for p in pools[:4]])).query(reference)[0]
        fresh.append(dict(record_type='union',N=n,generator='refined',union_unique_count=len(union),union_complete_cover=float(d.max()),
            union_interior_only_cover=float(di.max()),union_gain_vs_mean_single=1-float(d.max())/np.mean([r['complete_cover'] for r in current[:4]]),
            union_large_hole_count=int(np.sum(d>1/math.sqrt(4*n)))))
    scale_rows=[];aggregates=[];gates=[]
    for n in tr.BUDGETS:
        sb={k:max(float(r['grid_anchored_discrepancy']) for r in multi if int(r['N'])==n and r['generator']=='Sobol' and int(r['bins_per_axis'])==k) for k in tr.SCALES}
        for name in ('Sobol','old_spacefill','fresh_v1','refined'):
            gr=[r for r in geometry if int(r['N'])==n and r['generator']==name];op=[r for r in fresh if int(r['N'])==n and r['generator']==name and r['record_type']=='overlap']
            ur=next(r for r in fresh if int(r['N'])==n and r['generator']==name and r['record_type']=='union');ratios=[];raw=[]
            for k in tr.SCALES:
                d=max(float(r['grid_anchored_discrepancy']) for r in multi if int(r['N'])==n and r['generator']==name and int(r['bins_per_axis'])==k)
                ratio=literal_ratio(d,sb[k]);operational=d/max(sb[k],1/n);ratios.append(operational);raw.append(ratio)
                scale_rows.append(dict(record_type='scale_summary',N=n,generator=name,bins_per_axis=k,discrepancy=d,Sobol_discrepancy=sb[k],
                    D_rel_literal=ratio,D_rel_one_count_floor=operational,Sobol_exact_zero=sb[k]==0))
            aggregates.append(dict(record_type='generator_summary',N=n,generator=name,complete_cover=max(float(r['complete_cover']) for r in gr),
                interior_only_cover=max(float(r['interior_only_cover']) for r in gr),boundary_cover=max(float(r['boundary_cover']) for r in gr),
                worst_D_rel_literal=max(raw),worst_D_rel_one_count_floor=max(ratios),nn_CV_mean=float(np.mean([float(r['nn_CV']) for r in gr])),
                nn_CV_min=min(float(r['nn_CV']) for r in gr),large_hole_count=max(int(r['large_hole_count']) for r in gr),
                large_hole_fraction=max(float(r['large_hole_fraction']) for r in gr),clustering_fraction=max(float(r['clustered_fraction']) for r in gr),
                exact_overlap_count=max(int(r['exact_overlap_count']) for r in op),near_overlap_fraction=max(float(r['near_overlap_fraction_005_spacing']) for r in op),
                cell_greedy_near_overlap=float(next(r['near_overlap_fraction_005_spacing'] for r in op if r['comparison']=='cell_to_greedy')),
                union_complete_cover=float(ur['union_complete_cover']),union_gain=float(ur['union_gain_vs_mean_single'])))
        a={r['generator']:r for r in aggregates if r['N']==n};ref=a['refined'];v1=a['fresh_v1'];sbm=a['Sobol']
        matched_cover_ratios=[]
        for rr in geometry:
            if int(rr['N'])==n and rr['generator']=='refined':
                ss=next(s for s in geometry if int(s['N'])==n and s['generator']=='Sobol' and int(s['iteration'])==int(rr['iteration']) and s['pool_type']==rr['pool_type'])
                matched_cover_ratios.append(float(rr['complete_cover'])/float(ss['complete_cover']))
        ref['worst_matched_cover_ratio']=max(matched_cover_ratios)
        ref['matched_cover_regressions']=sum(v>1+1e-12 for v in matched_cover_ratios)
        checks=dict(coverage=max(matched_cover_ratios)<=1+1e-12,freshness=ref['exact_overlap_count']==0 and ref['near_overlap_fraction']<=v1['near_overlap_fraction']+.005,
            no_catastrophic_scale=ref['worst_D_rel_one_count_floor']<=3.,substantial_discrepancy_improvement=ref['worst_D_rel_one_count_floor']<=.8*v1['worst_D_rel_one_count_floor'],
            not_rigid=ref['nn_CV_min']>=max(.08,.8*v1['nn_CV_min']),clustering=ref['clustering_fraction']<=.005,
            holes=ref['large_hole_fraction']<=sbm['large_hole_fraction']+.001)
        gate=dict(record_type='gate',N=n,passed=all(checks.values()),failed=json.dumps([k for k,v in checks.items() if not v]),**checks)
        gates.append(gate);print(json.dumps(dict(N=n,refined=ref,fresh_v1=v1,gate=gate)),flush=True)
    rows=aggregates+scale_rows+gates+[dict(record_type='per_condition_geometry',**r) for r in geometry]+[dict(record_type='per_condition_discrepancy',**r) for r in multi]+fresh
    src.write_csv(OUT/'refined_spacefill_geometry.csv',rows)
    np.savez_compressed(OUT/'refined_validation_probes.npz',**coordinates)
    (OUT/'refined_validation_probes.sha256').write_text(src.sha(OUT/'refined_validation_probes.npz')+'  refined_validation_probes.npz\n')
    passed=all(g['passed'] for g in gates);screen=[];placement_pass=False
    if passed:
        archive=read('fresh_spacefill_first_divergence.csv')
        for r in archive:r['generator']=rename[r['generator']]
        screen+=archive
        saved=torch.load('outputs/sho/stage1B/sobol_vs_spacefill_quick/quick_screen_states.pt',weights_only=False)['runs']
        positions=src.im._make_exact_labeled_split(1024,src.im.TEST_SEED)[0].numpy()
        for run in saved[::2]:
            case=run['case'];tid=case['trajectory_id'];sites=run['sites'][:10];losses=run['losses'][:10]
            points,_=model.generate(1000,0,'cell');tree=cKDTree(sites);d,owner=tree.query(points)
            radius=np.zeros(10);np.maximum.at(radius,owner,d);sizes=np.maximum(radius,.5*tree.query(sites,k=2)[0][:,1]);po=src.edirect._potentially_optimal_cells(sizes,losses)
            pending,details=teacher.proposals(sites,losses,points,owner,d,sizes,po);chosen=next(r for r in details if not r.get('empty') and r.get('accepted'))
            surface=src.edirect.GaussianLocalResponse(sites,losses);u=.5*src.edirect._normalize(sizes[po])+.5*src.edirect._normalize(surface.loo_errors(po));u[np.argmax(sizes[po])]=1.
            physical=src.edirect._physical(pending[0]);value=float(src.edirect._prefix_losses_numpy(physical[None,:],positions[tid])[0])
            screen.append(dict(trajectory_id=tid,category=case['stratum'],generator='refined',selected_cell=chosen['site'],ordinary_cell=not chosen['largest'],
                u=float(u[np.flatnonzero(po==chosen['site'])[0]]),normalized_coordinate=json.dumps(pending[0].tolist()),physical_coordinate=json.dumps(physical.tolist()),objective_loss=value,
                loss_source='single_static_objective_evaluation',PO=json.dumps(po.tolist())))
        placement_pass=all(float(r['objective_loss'])<=float(next(s['objective_loss'] for s in screen if int(s['trajectory_id'])==int(r['trajectory_id']) and s['generator']=='old_spacefill')) for r in screen if r['generator']=='refined')
    else:screen=[dict(status='SKIPPED_GEOMETRY_GATE_FAILED',reason='No new objective evaluations or teacher first-choice run')]
    src.write_csv(OUT/'refined_spacefill_first_choices.csv',screen)
    unchanged=all(src.sha(p)==h for p,h in summary['protected_hashes'].items()) and src.sha(summary['checkpoint'])==summary['sha256']
    promoted=passed and placement_pass and unchanged
    if promoted:
        target=OUT/'REFINED_SPACEFILL_CANDIDATE.pt';assert not target.exists();shutil.copyfile(summary['checkpoint'],target)
        target.with_suffix('.sha256').write_text(summary['sha256']+'  '+target.name+'\n')
    result=dict(status='REFINED_SPACEFILL_CANDIDATE' if promoted else 'DIAGNOSTIC_ONLY',geometry_passed=passed,first_choice_screen_run=passed,
        first_choice_gate_passed=placement_pass,protected_hashes_unchanged=unchanged,checkpoint=summary['checkpoint'],sha256=summary['sha256'],
        evaluation_seconds=time.time()-started,training_summary={k:v for k,v in summary.items() if k!='protected_hashes'},gates=gates)
    (OUT/'evaluation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['REFINED SPACEFILL: '+result['status'],json.dumps(result,indent=2),
        'Continuation starts from fresh v1 weights. Primary loss is max normalized hard-grid discrepancy; no covering/freshness rewards are mixed into it.',
        'A CIC straight-through derivative supplies gradients through exact histogram forward values. Feasibility checked separately after every proposed update; first violation stops training and rejects that update.',
        'Training conditions 8000..8003 and greedy 8000; validation conditions 100..103 and greedy 100. All sets deterministic. Boundaries unchanged.',
        'Baselines reuse archived measurements on the exact same 524288-point reference, seeds, budgets, pool conditions, and boundary support; frozen baseline checkpoints hashes verified.',
        'At zero Sobol discrepancy: literal D_rel is infinity for a positive generator discrepancy, zero when both are zero. One-count floor max(D_Sobol,1/N) is separately labeled, never substituted for literal reported ratios.',
        'Discrepancy is max over four periodic shifts per scale, then max over five pool conditions. Raw and operational ratios are reported at EVERY scale 2,4,8,16,32,64,128.',
        'Covering radii are finite-reference estimates. Training used a budget-level worst-condition coverage envelope; final validation additionally requires no worse than Sobol for EACH matched iteration/pool. No claim of a continuous-domain covering certificate.',
        'Near overlap: distance <0.05/sqrt(N), interiors only; clustering: NN distance <0.1/sqrt(N); holes: reference distance >1/sqrt(N).',
        'Hard geometry gate predeclared: worst operational discrepancy ratio <=3, at least 20% improvement vs v1 at each budget; exact freshness and near-overlap/no-lattice constraints preserved.',
        'All checkpoints saved/hashed before evaluation. No closed-loop search or automatic training extension.',
        '', 'GENERATOR SUMMARIES']+[json.dumps(r) for r in aggregates]
    (OUT/'refined_spacefill_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('gates','training_summary')},indent=2),flush=True)

if __name__=='__main__':run()
