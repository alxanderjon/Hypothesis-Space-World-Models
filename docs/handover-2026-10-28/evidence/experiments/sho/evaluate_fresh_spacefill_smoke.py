"""Independent geometry and optional six-state teacher screen; never trains."""
import json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.fresh_spacefill_generator import FreshGenerator
from experiments.sho import train_fresh_spacefill_smoke as training
from experiments.sho import run_stage1b_spacefill_2x2 as teacher
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_generator_smoke as old

OUT=training.OUT
def metrics(points,interior,reference,n):
    d=cKDTree(points).query(reference)[0];di=cKDTree(interior).query(reference)[0]
    near=cKDTree(interior).query(interior,k=2)[0][:,1]
    face=[]
    for axis in range(2):
        for side in (0.,1.):
            q=np.empty((8193,2));q[:,axis]=side;q[:,1-axis]=np.linspace(0,1,len(q));face.append(cKDTree(points).query(q)[0].max())
    return dict(complete_cover=float(d.max()),interior_only_cover=float(di.max()),boundary_cover=float(max(face)),
        nn_CV=float(near.std()/near.mean()),nn_min=float(near.min()),nn_p10=float(np.quantile(near,.1)),nn_median=float(np.median(near)),nn_p90=float(np.quantile(near,.9)),
        clustered_fraction=float(np.mean(near<.1/math.sqrt(n))),large_hole_count=int(np.sum(d>1/math.sqrt(n))),
        large_hole_fraction=float(np.mean(d>1/math.sqrt(n))),reference_count=len(reference),interior_count=len(interior),complete_count=len(points))
def overlaps(a,b,n):
    d=cKDTree(b).query(a)[0]
    return dict(exact_overlap_count=int(np.sum(d==0)),exact_overlap_fraction=float(np.mean(d==0)),
        near_overlap_fraction_005_spacing=float(np.mean(d<.05/math.sqrt(n))),near_overlap_fraction_010_spacing=float(np.mean(d<.1/math.sqrt(n))),
        median_cross_distance=float(np.median(d)))

def run():
    started=time.time();torch.set_num_threads(1)
    summary=json.loads((OUT/'training_summary.json').read_text())
    assert src.sha(summary['checkpoint'])==summary['sha256'],'trained checkpoint changed'
    if not summary['evaluation_allowed']:
        (OUT/'fresh_spacefill_summary.txt').write_text('STOP: fixed training objective did not improve; trained artifact persisted, evaluation skipped.\n');return
    # Declared before evaluating held-out geometry; these are smoke-screen
    # tolerances, not mathematical guarantees or production acceptance gates.
    criteria=dict(max_complete_cover_ratio_to_Sobol=1.10,max_union_cover_ratio_to_Sobol=1.25,
        max_multiscale_discrepancy_ratio_to_old=1.05,max_near_overlap_fraction_at_005_spacing=.10,
        min_union_gain_vs_mean_single=.10,min_nn_CV=.08,max_cluster_fraction=.005,
        maximum_large_hole_fraction_excess_over_Sobol=.001,
        first_choice_gate='new loss lower than old in >=4 of 6 states, mean loss <=old, and trajectory 280 no worse than old')
    (OUT/'evaluation_criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
    model=FreshGenerator(2);model.load_state_dict(torch.load(summary['checkpoint'],weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    prior=old.Generator(2);prior.load_state_dict(torch.load('outputs/sho/stage1B/spacefill_downstream_screen/spacefill_generator_candidate.pt',weights_only=False)['state_dict']);prior.eval().requires_grad_(False)
    t=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    conditions=[(100+j,'cell') for j in range(4)]+[(100,'greedy')]
    rows=[];pool_rows=[];multi=[];gates=[];coordinates={}
    for n in training.BUDGETS:
        ns=10 if n==1000 else 86 if n==8600 else 328
        pools={};geom={};discs={};union={}
        for name in ('Sobol','old_spacefill','new_spacefill'):
            pools[name]=[];geom[name]=[];discs[name]=[]
            for iteration,pooltype in conditions:
                if name=='Sobol':
                    points=src.edirect._candidate_pool(ns,200136,iteration+(1000000 if pooltype=='greedy' else 0))
                    interior=points[~np.any((points==0.)|(points==1.),axis=1)]
                elif name=='old_spacefill':points,interior=prior.generate(n,[0.,0.],[1.,1.])
                else:
                    points,interior=model.generate(n,iteration,pooltype)
                    assert np.array_equal(points,model.generate(n,iteration,pooltype)[0])
                    coordinates[f'N{n}_iteration{iteration}_{pooltype}']=points
                boundary=points[np.any((points==0.)|(points==1.),axis=1)]
                assert np.array_equal(boundary,old.faces(n,2)[0])
                pools[name].append((points,interior));m=metrics(points,interior,reference,n)
                row=dict(N=n,generator=name,iteration=iteration,pool_type=pooltype,**m);rows.append(row);geom[name].append(row)
                for bins in (2,4,8,16,32,64,128):
                    for shift in (0.,.25,.5,.75):
                        shifted=(interior+shift/bins)%1.
                        count=np.histogramdd(shifted,bins=(np.linspace(0,1,bins+1),)*2)[0]
                        cdf=count.cumsum(0).cumsum(1)/n;edge=np.arange(1,bins+1)/bins
                        error=cdf-edge[:,None]*edge[None,:]
                        value=float(np.abs(error).max());discs[name].append(value)
                        multi.append(dict(N=n,generator=name,iteration=iteration,pool_type=pooltype,bins_per_axis=bins,shift_fraction_of_bin=shift,
                            grid_anchored_discrepancy=value,grid_anchored_RMSE=float(np.sqrt(np.mean(error**2))),count_variance=float(count.var()),
                            count_fano=float(count.var()/count.mean()),empty_bin_fraction=float(np.mean(count==0))))
            for i,j,label in [(0,1,'cell_t_to_t1'),(1,2,'cell_t1_to_t2'),(2,3,'cell_t2_to_t3'),(0,4,'cell_to_greedy')]:
                pool_rows.append(dict(record_type='overlap',N=n,generator=name,comparison=label,**overlaps(pools[name][i][1],pools[name][j][1],n)))
            allpoints=np.unique(np.concatenate([v[0] for v in pools[name][:4]]),axis=0)
            dint=cKDTree(np.vstack([v[1] for v in pools[name][:4]])).query(reference)[0]
            dd=cKDTree(allpoints).query(reference)[0]
            union[name]=dict(record_type='union',N=n,generator=name,union_unique_count=len(allpoints),union_complete_cover=float(dd.max()),
                union_interior_only_cover=float(dint.max()),union_gain_vs_mean_single=1-float(dd.max())/np.mean([r['complete_cover'] for r in geom[name][:4]]),
                union_large_hole_count=int(np.sum(dd>1/math.sqrt(4*n))))
            pool_rows.append(union[name])
        new_overlap=[r for r in pool_rows if r['record_type']=='overlap' and r['generator']=='new_spacefill' and r['N']==n]
        checks=dict(complete_coverage=max(r['complete_cover'] for r in geom['new_spacefill'])<=1.10*max(r['complete_cover'] for r in geom['Sobol']),
            boundary_coverage=max(r['boundary_cover'] for r in geom['new_spacefill'])<=max(r['boundary_cover'] for r in geom['Sobol'])*(1+1e-12),
            multiscale_improvement=max(discs['new_spacefill'])<=1.05*max(discs['old_spacefill']),
            exact_freshness=all(r['exact_overlap_count']==0 for r in new_overlap),near_freshness=max(r['near_overlap_fraction_005_spacing'] for r in new_overlap)<.10,
            union_improvement=union['new_spacefill']['union_gain_vs_mean_single']>=.10,
            union_vs_Sobol=union['new_spacefill']['union_complete_cover']<=1.25*union['Sobol']['union_complete_cover'],
            not_rigid=min(r['nn_CV'] for r in geom['new_spacefill'])>=.08,
            clustering=max(r['clustered_fraction'] for r in geom['new_spacefill'])<=.005,
            holes=max(r['large_hole_fraction'] for r in geom['new_spacefill'])<=max(r['large_hole_fraction'] for r in geom['Sobol'])+.001)
        checks={k:bool(v) for k,v in checks.items()}
        gate=dict(N=n,passed=all(checks.values()),checks=checks,
            complete_cover_ratio_to_Sobol=max(r['complete_cover'] for r in geom['new_spacefill'])/max(r['complete_cover'] for r in geom['Sobol']),
            multiscale_worst={name:max(v) for name,v in discs.items()},union=union)
        gates.append(gate);print(json.dumps(gate),flush=True)
        src.write_csv(OUT/'fresh_spacefill_geometry.csv',rows);src.write_csv(OUT/'fresh_spacefill_multiscale.csv',multi);src.write_csv(OUT/'fresh_spacefill_freshness_union.csv',pool_rows)
    np.savez_compressed(OUT/'heldout_generated_probes.npz',**coordinates)
    (OUT/'heldout_generated_probes.sha256').write_text(src.sha(OUT/'heldout_generated_probes.npz')+'  heldout_generated_probes.npz\n')
    healthy=all(g['passed'] for g in gates);screen=[];placement=None
    if healthy:
        saved=torch.load('outputs/sho/stage1B/sobol_vs_spacefill_quick/quick_screen_states.pt',weights_only=False)['runs']
        positions=src.im._make_exact_labeled_split(1024,src.im.TEST_SEED)[0].numpy()
        for run in saved[::2]:
            case=run['case'];tid=case['trajectory_id'];sites=run['sites'][:10];losses=run['losses'][:10]
            for name in ('Sobol','old_spacefill','new_spacefill'):
                points=src.edirect._candidate_pool(10,case['generator_id'],0) if name=='Sobol' else prior.generate(1000,[0.,0.],[1.,1.])[0] if name=='old_spacefill' else model.generate(1000,0,'cell')[0]
                tree=cKDTree(sites);distance,owner=tree.query(points);radius=np.zeros(10);np.maximum.at(radius,owner,distance)
                sizes=np.maximum(radius,.5*tree.query(sites,k=2)[0][:,1]);po=src.edirect._potentially_optimal_cells(sizes,losses)
                pending,records=teacher.proposals(sites,losses,points,owner,distance,sizes,po)
                chosen=next(r for r in records if not r.get('empty') and r.get('accepted'))
                surface=src.edirect.GaussianLocalResponse(sites,losses)
                weights=.5*src.edirect._normalize(sizes[po])+.5*src.edirect._normalize(surface.loo_errors(po));weights[np.argmax(sizes[po])]=1.
                physical=src.edirect._physical(pending[0]);archived=name!='new_spacefill'
                if archived:
                    original=next(r for r in saved if r['case']['trajectory_id']==tid and r['arm']==('A' if name=='Sobol' else 'B'))['events'][10]
                    assert np.array_equal(physical,original['point']);value=original['loss']
                else:value=float(src.edirect._prefix_losses_numpy(physical[None,:],positions[tid])[0])
                screen.append(dict(trajectory_id=tid,category=case['stratum'],generator=name,selected_cell=chosen['site'],ordinary_cell=not chosen['largest'],
                    u=float(weights[np.flatnonzero(po==chosen['site'])[0]]),normalized_coordinate=json.dumps(pending[0].tolist()),physical_coordinate=json.dumps(physical.tolist()),
                    objective_loss=value,loss_source='archived' if archived else 'single_static_objective_evaluation',PO=json.dumps(po.tolist())))
        src.write_csv(OUT/'fresh_spacefill_first_divergence.csv',screen)
        pairs=[]
        for run in saved[::2]:
            tid=run['case']['trajectory_id'];r={x['generator']:x for x in screen if x['trajectory_id']==tid}
            pairs.append(dict(trajectory_id=tid,new_loss=r['new_spacefill']['objective_loss'],old_loss=r['old_spacefill']['objective_loss'],Sobol_loss=r['Sobol']['objective_loss']))
        placement=dict(pairs=pairs,new_beats_old=sum(p['new_loss']<p['old_loss'] for p in pairs),new_beats_Sobol=sum(p['new_loss']<p['Sobol_loss'] for p in pairs))
        placement['passed']=bool(placement['new_beats_old']>=4 and np.mean([p['new_loss'] for p in pairs])<=np.mean([p['old_loss'] for p in pairs]) and next(p for p in pairs if p['trajectory_id']==280)['new_loss']<=next(p for p in pairs if p['trajectory_id']==280)['old_loss'])
    else:src.write_csv(OUT/'fresh_spacefill_first_divergence.csv',[dict(status='SKIPPED_GEOMETRY_GATE_FAILED')])
    unchanged=all(src.sha(p)==h for p,h in summary['protected_hashes'].items()) and src.sha(summary['checkpoint'])==summary['sha256']
    # Passing relative-to-old smoke tolerances is not enough to claim that the
    # discrepancy problem was fixed. Keep this goal assessment separate from
    # the predeclared geometry criteria and retain all measured gate results.
    trained=torch.load(summary['checkpoint'],weights_only=False)
    monitor=[r for r in trained['training_history'] if r['record_type']=='fixed_training_monitor']
    first=min(r['step'] for r in monitor);last=max(r['step'] for r in monitor)
    initial_disc=float(np.mean([r['discrepancy'] for r in monitor if r['step']==first]))
    final_disc=float(np.mean([r['discrepancy'] for r in monitor if r['step']==last]))
    discrepancy_learning_improved=final_disc<=initial_disc
    promoted=healthy and placement['passed'] and unchanged and discrepancy_learning_improved
    if promoted:
        cp=OUT/'NEW_SPACEFILL_CANDIDATE.pt';assert not cp.exists();shutil.copyfile(summary['checkpoint'],cp)
        cp.with_suffix('.sha256').write_text(src.sha(cp)+'  '+cp.name+'\n')
    result=dict(status='NEW_SPACEFILL_CANDIDATE' if promoted else 'STOPPED_DIAGNOSTIC_ONLY',geometry_passed=healthy,geometry_gates=gates,
        discrepancy_learning_improved=discrepancy_learning_improved,initial_training_discrepancy=initial_disc,final_training_discrepancy=final_disc,
        placement=placement,protected_hashes_unchanged=unchanged,checkpoint_hash=summary['sha256'],evaluation_seconds=time.time()-started,
        no_closed_loop=True,static_objective_evaluations=6 if healthy else 0)
    (OUT/'evaluation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['FRESH MULTISCALE SPACEFILL SMOKE',json.dumps(result,indent=2),
        '60-update cap, geometry-only optimization; no downstream labels/objectives used for training. Trained weights saved and hashed BEFORE validation.',
        'Deterministic SHA256/PCG64 conditioning on N, iteration, pool type; same general boundary-face support; learned offsets remain inside partition cells.',
        'Reference: independent 512x512 midpoint grid plus 262144 IID points. Validation iterations 100..103; training iterations 5000+4*step, monitor 7000.',
        'Covering metrics are finite-reference estimates. Near overlap uses 0.05 and 0.10 times N^(-1/2). Shared boundary points excluded from freshness.',
        'Discrepancy is measured on anchored bin grids at scales 2..128 and four periodic offsets; not the full continuous star discrepancy.',
        'Geometry gate criteria were saved before evaluation. A smoke pass does not establish closed-loop reliability.',
        'Existing candidate, production Sobol, and locked components unchanged. No automatic retraining or search.']
    (OUT/'fresh_spacefill_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='geometry_gates'},indent=2),flush=True)

if __name__=='__main__':run()
