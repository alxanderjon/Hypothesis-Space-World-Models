"""Post-persistence static teacher-score and objective evaluation; never trains."""
import csv,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import train_candidate_quality_spacefill as training
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_generator_smoke as old
from experiments.sho import evaluate_fresh_spacefill_smoke as geo
from experiments.sho.fresh_spacefill_generator import FreshGenerator

OUT=training.OUT
def normalized(x,lo,hi):
    # Same constant-range convention as production _normalize, with the
    # extrema supplied by the fixed reference instead of the current pool.
    return np.full_like(x,.5) if hi-lo<=1e-15*max(1.,abs(lo),abs(hi)) else (x-lo)/(hi-lo)
def write():
    torch.set_num_threads(1);started=time.time()
    ts=json.loads((OUT/'training_summary.json').read_text());assert src.sha(ts['checkpoint'])==ts['sha256']
    data=torch.load(OUT/'offline_teacher_targets.pt',weights_only=False);states=[s for s in data['states'] if s['split']=='held_out']
    models={}
    for name,path in [('old_spacefill','outputs/sho/stage1B/spacefill_downstream_screen/spacefill_generator_candidate.pt'),('fresh_v1',str(training.INITIAL)),('candidate_quality',ts['checkpoint'])]:
        model=old.Generator(2) if name=='old_spacefill' else FreshGenerator(2)
        model.load_state_dict(torch.load(path,weights_only=False)['state_dict']);model.eval().requires_grad_(False);models[name]=model
    def generate(name,n,it,pool='cell',generator_id=200136):
        if name=='Sobol':
            sites=10 if n==1000 else 86 if n==8600 else math.ceil(n/100)
            points=src.edirect._candidate_pool(sites,generator_id,it+(1000000 if pool=='greedy' else 0))
            return points,points[~np.any((points==0.)|(points==1.),axis=1)]
        if name=='old_spacefill':return models[name].generate(n,[0.,0.],[1.,1.])
        return models[name].generate(n,it,pool)
    # Labels and model training are already complete and persisted. Only here
    # do positions/objective evaluation become available to the experiment.
    positions=src.im._make_exact_labeled_split(1024,src.im.TEST_SEED)[0].numpy()
    router=training.make_router();rows=[];objective_calls=0
    for state in states:
        sites,losses=state['sites'],state['losses'];case=state['case'];cell=state['site'];u=state['u']
        surface=src.edirect.GaussianLocalResponse(sites,losses);norm=state['reference_normalizers']
        for name in ('Sobol','old_spacefill','fresh_v1','candidate_quality'):
            points,interior=generate(name,state['N'],state['iteration'],generator_id=case['generator_id'])
            result=router.search(sites,points);owner=result['owner'][:len(points)];distance=result['distance'][:len(points)]
            ids=np.flatnonzero(owner==cell);global_target_distance,global_target_ids=cKDTree(points).query(state['targets'])
            target_hit=int(global_target_ids[np.argmin(global_target_distance)])
            row=dict(record_type='state',state_id=state['id'],trajectory_id=case['trajectory_id'],category=case['stratum'],stage=state['stage'],
                N=state['N'],site_count=len(sites),iteration=state['iteration'],generator=name,selected_cell=cell,u=u,assigned_probe_count=len(ids),
                global_nearest_target_distance=float(global_target_distance.min()),global_nearest_target_owned_by_selected_cell=bool(owner[target_hit]==cell))
            if not len(ids) or not np.any(distance[ids]>1e-12):
                row.update(status='NO_ADMISSIBLE_CANDIDATE',objective_loss='',best_native_teacher_score='',best_reference_anchored_score='',selected_cell_target_distance='')
            else:
                ds=distance[ids];pred=surface.predict(points[ids])
                native=u*src.edirect._normalize(ds)-(1-u)*src.edirect._normalize(pred)
                common=u*normalized(ds,norm['d_min'],norm['d_max'])-(1-u)*normalized(pred,norm['s_min'],norm['s_max'])
                winner=int(np.argmax(native));point=points[ids[winner]]
                if ds[winner]<=1e-12:
                    row.update(status='TEACHER_WINNER_DUPLICATE_REJECTED',objective_loss='',best_native_teacher_score=float(native[winner]),
                        best_reference_anchored_score=float(common.max()),selected_cell_target_distance='')
                else:
                    physical=src.edirect._physical(point)
                    value=float(src.edirect._prefix_losses_numpy(physical[None,:],positions[case['trajectory_id']])[0]);objective_calls+=1
                    near=cKDTree(points[ids]).query(state['targets'])[0]
                    row.update(status='OK',best_native_teacher_score=float(native[winner]),best_reference_anchored_score=float(common.max()),
                        selected_reference_anchored_score=float(common[winner]),selected_normalized_coordinate=json.dumps(point.tolist()),
                        selected_physical_coordinate=json.dumps(physical.tolist()),selected_cell_target_distance=float(near.min()),
                        objective_loss=value,objective_usage='POST_HOC_ONLY; never used in labels/training/checkpoint selection')
            rows.append(row)
        src.write_csv(OUT/'candidate_quality_offline_evaluation.csv',rows)
        print('offline '+state['id']+' '+json.dumps({r['generator']:r.get('objective_loss') for r in rows[-4:]}),flush=True)
    # Independent, static geometry evaluation at three representative budgets.
    prior=Path('outputs/sho/stage1B/fresh_spacefill_smoke')
    def read(name):return list(csv.DictReader((prior/name).open()))
    geometry=read('fresh_spacefill_geometry.csv');fresh=read('fresh_spacefill_freshness_union.csv')
    for collection in (geometry,fresh):
        for r in collection:
            if r['generator']=='new_spacefill':r['generator']='fresh_v1'
    grid=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    conditions=[(100+j,'cell') for j in range(4)]+[(100,'greedy')];gates=[];savedpoints={}
    for n in (1000,8600,32768):
        pools=[];local=[]
        for it,pool in conditions:
            points,interior=generate('candidate_quality',n,it,pool)
            assert np.array_equal(points,generate('candidate_quality',n,it,pool)[0])
            assert np.array_equal(points[np.any((points==0.)|(points==1.),axis=1)],old.faces(n,2)[0])
            pools.append((points,interior));savedpoints[f'N{n}_{it}_{pool}']=points
            r=dict(N=n,generator='candidate_quality',iteration=it,pool_type=pool,**geo.metrics(points,interior,reference,n));geometry.append(r);local.append(r)
        overlap=[]
        for a,b,label in ((0,1,'cell_t_to_t1'),(1,2,'cell_t1_to_t2'),(2,3,'cell_t2_to_t3'),(0,4,'cell_to_greedy')):
            r=dict(record_type='overlap',N=n,generator='candidate_quality',comparison=label,**geo.overlaps(pools[a][1],pools[b][1],n));overlap.append(r);fresh.append(r)
        union=np.unique(np.concatenate([v[0] for v in pools[:4]]),axis=0);d=cKDTree(union).query(reference)[0]
        fresh.append(dict(record_type='union',N=n,generator='candidate_quality',union_complete_cover=float(d.max()),union_unique_count=len(union),
            union_gain_vs_mean_single=1-float(d.max())/np.mean([r['complete_cover'] for r in local[:4]])))
        baseline=[r for r in geometry if int(r['N'])==n and r['generator']=='Sobol'];v1=[r for r in fresh if int(r['N'])==n and r['generator']=='fresh_v1' and r['record_type']=='overlap']
        ratio=max(r['complete_cover'] for r in local)/max(float(r['complete_cover']) for r in baseline)
        paired=[]
        for r in local:
            b=next(b for b in baseline if int(b['iteration'])==r['iteration'] and b['pool_type']==r['pool_type'])
            paired.append(r['complete_cover']/float(b['complete_cover']))
        checks=dict(rough_cover=ratio<=1.10,freshness=all(r['exact_overlap_count']==0 for r in overlap) and max(r['near_overlap_fraction_005_spacing'] for r in overlap)<=max(float(r['near_overlap_fraction_005_spacing']) for r in v1)+.01,
            no_clustering=max(r['clustered_fraction'] for r in local)<=.005,no_large_holes=max(r['large_hole_fraction'] for r in local)<=max(float(r['large_hole_fraction']) for r in baseline)+.001,
            not_rigid=min(r['nn_CV'] for r in local)>=.08,boundary_unchanged=True)
        gates.append(dict(record_type='geometry_gate',N=n,passed=all(checks.values()),complete_cover_ratio=ratio,worst_matched_cover_ratio=max(paired),
            strict_matched_cover_failures=sum(x>1+1e-12 for x in paired),failed=json.dumps([k for k,v in checks.items() if not v]),**checks))
    src.write_csv(OUT/'candidate_quality_geometry.csv',[dict(record_type='geometry',**r) for r in geometry]+fresh+gates)
    np.savez_compressed(OUT/'candidate_quality_validation_probes.npz',**savedpoints)
    pairs=[]
    for state in states:
        group={r['generator']:r for r in rows if r['state_id']==state['id']}
        sb=group['Sobol'];base=float(sb['objective_loss']) if sb['status']=='OK' else float('inf')
        for name,r in group.items():
            valid=r['status']=='OK';loss=float(r['objective_loss']) if valid else float('inf')
            r.update(beats_Sobol=valid and loss<base,loss_difference_vs_Sobol=loss-base if np.isfinite(base) else '',
                relative_loss_difference_vs_Sobol=(loss-base)/max(abs(base),1e-300) if np.isfinite(base) else '')
        q,v=group['candidate_quality'],group['fresh_v1'];ql=float(q['objective_loss']) if q['status']=='OK' else float('inf');vl=float(v['objective_loss']) if v['status']=='OK' else float('inf')
        qscore=float(q['best_reference_anchored_score']) if q.get('best_reference_anchored_score')!='' else -float('inf')
        vscore=float(v['best_reference_anchored_score']) if v.get('best_reference_anchored_score')!='' else -float('inf')
        pairs.append(dict(state_id=state['id'],stage=state['stage'],trajectory_id=state['case']['trajectory_id'],new_valid=q['status']=='OK',
            new_beats_v1=ql<vl,new_beats_Sobol=ql<base,new_vs_v1_loss_delta=ql-vl if np.isfinite(ql) or np.isfinite(vl) else '',
            reference_best_score_gain=qscore-vscore if np.isfinite(qscore) or np.isfinite(vscore) else '',
            target_distance_change=q['global_nearest_target_distance']-v['global_nearest_target_distance']))
    aggregate=[]
    for category in ('all','interior','near_boundary','very_near_boundary'):
        for name in ('Sobol','old_spacefill','fresh_v1','candidate_quality'):
            rr=[r for r in rows if r['generator']==name and (category=='all' or r['category']==category)]
            delta=[float(r['loss_difference_vs_Sobol']) for r in rr if r['loss_difference_vs_Sobol']!='']
            finite=[x for x in delta if np.isfinite(x)]
            aggregate.append(dict(record_type='aggregate',category=category,generator=name,states=len(rr),valid_candidates=sum(r['status']=='OK' for r in rr),
                beats_Sobol=sum(r['beats_Sobol'] for r in rr),fraction_beating_Sobol=sum(r['beats_Sobol'] for r in rr)/len(rr),
                median_loss_difference_vs_Sobol=float(np.median(delta)) if delta else '',worst_loss_difference_vs_Sobol=max(delta) if delta else '',
                valid_only_median_loss_difference_vs_Sobol=float(np.median(finite)) if finite else '',valid_only_worst_loss_difference_vs_Sobol=max(finite) if finite else ''))
    src.write_csv(OUT/'candidate_quality_offline_evaluation.csv',rows+aggregate)
    src.write_csv(OUT/'candidate_quality_first_divergence.csv',[r for r in rows if r['stage']=='small'])
    src.write_csv(OUT/'candidate_quality_paired.csv',pairs)
    all_valid=all(r['new_valid'] for r in pairs);score_gains=[r['reference_best_score_gain'] for r in pairs if r['reference_best_score_gain']!='']
    target_improved=np.median([r['target_distance_change'] for r in pairs])<0
    geometry_pass=all(r['passed'] for r in gates)
    initial_rows=[r for r in rows if r['stage']=='small'];small_better_old=[]
    for tid in sorted(set(r['trajectory_id'] for r in initial_rows)):
        rr={r['generator']:r for r in initial_rows if r['trajectory_id']==tid}
        small_better_old.append((tid,float(rr['candidate_quality']['objective_loss'])<=float(rr['old_spacefill']['objective_loss'])))
    quality_pass=bool(all_valid and target_improved and score_gains and np.median(score_gains)>=0 and sum(r['new_beats_v1'] for r in pairs)>=math.ceil(2*len(pairs)/3)
        and sum(r['new_beats_Sobol'] for r in pairs)>=math.ceil(len(pairs)/2) and sum(v for _,v in small_better_old)>=4 and dict(small_better_old)[280])
    unchanged=all(src.sha(p)==h for p,h in ts['protected_hashes'].items()) and src.sha(ts['checkpoint'])==ts['sha256']
    promoted=geometry_pass and quality_pass and unchanged
    if promoted:
        target=OUT/'CANDIDATE_QUALITY_SPACEFILL.pt';assert not target.exists();shutil.copyfile(ts['checkpoint'],target);target.with_suffix('.sha256').write_text(ts['sha256']+'  '+target.name+'\n')
    result=dict(status='CANDIDATE_QUALITY_SPACEFILL' if promoted else 'DIAGNOSTIC_ONLY',geometry_passed=geometry_pass,quality_passed=quality_pass,
        held_out_states=len(states),new_valid_candidates=sum(r['new_valid'] for r in pairs),new_beats_fresh_v1=sum(r['new_beats_v1'] for r in pairs),new_beats_Sobol=sum(r['new_beats_Sobol'] for r in pairs),
        median_target_distance_change=float(np.median([r['target_distance_change'] for r in pairs])),median_reference_best_score_gain=float(np.median(score_gains)) if score_gains else None,
        static_posthoc_objective_evaluations=objective_calls,protected_hashes_unchanged=unchanged,checkpoint=ts['checkpoint'],sha256=ts['sha256'],evaluation_seconds=time.time()-started,gates=gates,
        training={k:v for k,v in ts.items() if k!='protected_hashes'})
    (OUT/'evaluation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['CANDIDATE-QUALITY SPACEFILL: '+result['status'],json.dumps(result,indent=2),
        'No closed-loop search. All actual prefix losses are POST-HOC only after the trained checkpoint was saved and hashed. No true parameter inputs or labels.',
        'Archived observed losses reconstruct the fixed teacher surrogate, PO and u only; the generator sees N, iteration and pool type, never the state or objective values.',
        'Trajectory-disjoint training and held-out states. Training includes some previously named test trajectories; the NEW held-out set is the six quick-screen trajectories, fully disjoint.',
        'Each state keeps its first admissible Sobol teacher-selected ordinary cell and its teacher u fixed across generators; largest/farthest cells excluded.',
        'Label pool: independent 131072 or 8*N global random points, dense endpoint boundary support, and 65536 independent local geometric points around the selected cell. Every point uses the locked distance/margin/threshold branch-and-bound assignment.',
        'Top eight labels use only structured teacher score, with normalization on the dense selected-cell reference. No reference points enter search.',
        'Native pool normalization chooses each generator candidate exactly as the teacher would within the FIXED ordinary cell. Native best scores are not directly comparable across pools.',
        'Reference-anchored scores provide a separate common comparison scale; reference extrema are held fixed and not clipped. They do not replace native teacher selection.',
        'Empty cells or rejected duplicate winners are retained as failures, never silently dropped. Missing candidates have infinite loss differences in all-state summaries; finite-only summaries are separately labeled.',
        'Training constraint and validation covering gate use worst-condition cover per budget <=1.10 Sobol; per-condition strict regressions are reported separately.',
        'Baseline geometry reuses unchanged archived measurements with the identical reference sample and conditions. No old artifact was overwritten.',
        '', 'ACTUAL PREFIX-LOSS COMPARISONS BY CATEGORY']+[json.dumps(r) for r in aggregate]
    (OUT/'candidate_quality_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('gates','training')},indent=2),flush=True)

if __name__=='__main__':write()
