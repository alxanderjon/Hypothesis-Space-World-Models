"""Offline score-target supervision; no objective calls before persistence."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_generator_smoke as old
from experiments.sho import refine_fresh_spacefill as geometry
from experiments.sho.fresh_spacefill_generator import FreshGenerator
from experiments.sho.grid_branch_bound import GridBranchBound
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/candidate_quality_spacefill')
SEED=27095001
INITIAL=Path('outputs/sho/stage1B/fresh_spacefill_smoke/fresh_spacefill_trained_pending_evaluation.pt')

def make_router():
    metric,_=src.cells.base._load_locked_distance();op=torch.load(src.nearest.OPERATOR,weights_only=False)
    margin=nn.Linear(2,1,bias=False,dtype=torch.float64)
    margin.load_state_dict(torch.load(op['margin_checkpoint'],weights_only=False)['state_dict'])
    threshold=src.nearest.threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(op['threshold_checkpoint'],weights_only=False)['state_dict'])
    for model in (metric,margin,threshold):model.eval().requires_grad_(False)
    return GridBranchBound(metric,margin,threshold)

def states_from(run,split):
    states=[];first={}
    for e in run['events']:
        if e['kind']=='cell':first.setdefault(e['iteration'],e['evaluation']-1)
    for stage,target in [('small',10),('medium',80),('saturated',328)]:
        if target==10:it,n=0,10
        else:it,n=next((it,n) for it,n in first.items() if n>=target)
        case=run['case'];states.append(dict(id=f"{split}_{case['split']}_{case['trajectory_id']}_{stage}",split=split,stage=stage,
            case=case,N=min(100*n,32768),iteration=it,sites=run['sites'][:n].copy(),losses=run['losses'][:n].copy()))
    return states

def label(state,router,ordinal):
    sites,losses=state['sites'],state['losses'];n=len(sites);case=state['case']
    probes=src.edirect._candidate_pool(n,case['generator_id'],state['iteration'])
    routed=router.search(sites,probes);p=len(probes);owner=routed['owner'][:p];d=routed['distance'][:p]
    radius=np.zeros(n);np.maximum.at(radius,owner,d);sizes=np.maximum(radius,.5*routed['distance'][p:])
    po=src.edirect._potentially_optimal_cells(sizes,losses);surface=src.edirect.GaussianLocalResponse(sites,losses)
    weights=.5*src.edirect._normalize(sizes[po])+.5*src.edirect._normalize(surface.loo_errors(po))
    largest=int(np.argmax(sizes[po]));weights[largest]=1.
    chosen=None
    for j,site in enumerate(po):
        mask=owner==site
        if j==largest or not mask.any():continue
        utility=weights[j]*src.edirect._normalize(d[mask])-(1-weights[j])*src.edirect._normalize(surface.predict(probes[mask]))
        point=probes[mask][int(np.argmax(utility))]
        if np.linalg.norm(sites-point,axis=1).min()>1e-12:chosen=(int(site),float(weights[j]));break
    if chosen is None:raise RuntimeError(f'no teacher-selected ordinary cell: {state["id"]}')
    site,u=chosen;count=max(131072,8*state['N']);seed=SEED+ordinal
    rng=np.random.default_rng(seed)
    # Tiny selected cells can receive zero global random samples even with a
    # large pool. Add independent geometry-only local coverage for EVERY state.
    # The cell identity stays fixed, and every local point is reassigned by the
    # locked router; local samples are never fed to an actual search.
    reach=2*float(sizes[site]);lo=np.maximum(0,sites[site]-reach);hi=np.minimum(1,sites[site]+reach)
    global_reference=rng.random((count,2));local_reference=lo+(hi-lo)*rng.random((65536,2))
    refs=np.unique(np.vstack((global_reference,local_reference,old.faces(count,2)[0])),axis=0)
    ref=router.search(sites,refs);inside=ref['owner'][:len(refs)]==site
    points=refs[inside];distance=ref['distance'][:len(refs)][inside]
    if not len(points):raise RuntimeError(f'dense reference missed selected cell: {state["id"]}')
    pred=surface.predict(points);score=u*src.edirect._normalize(distance)-(1-u)*src.edirect._normalize(pred)
    # Near-site duplicates are excluded under the unchanged proposal rule.
    admissible=distance>1e-12;ids=np.flatnonzero(admissible)
    top=ids[np.argsort(-score[ids],kind='stable')[:8]]
    if not len(top):raise RuntimeError('no admissible top-set label')
    state.update(site=site,u=u,targets=points[top],target_scores=score[top],reference_normalizers=dict(d_min=float(distance.min()),d_max=float(distance.max()),s_min=float(pred.min()),s_max=float(pred.max())),
        reference_count=len(refs),reference_seed=seed,reference_cell_count=len(points),reference_local_count=65536,
        reference_local_bounds=np.stack((lo,hi)).tolist(),baseline_PO=po,baseline_sizes=sizes)
    return dict(state_id=state['id'],split=state['split'],trajectory_id=case['trajectory_id'],category=case['stratum'],stage=state['stage'],
        site_count=n,N=state['N'],iteration=state['iteration'],selected_cell=site,u=u,reference_count=len(refs),reference_cell_count=len(points),
        target_count=len(top),top_score=float(score[top[0]]),targets=json.dumps(points[top].tolist()),target_scores=json.dumps(score[top].tolist()),
        reference_local_count=65536,reference_local_bounds=json.dumps(np.stack((lo,hi)).tolist()),
        label_source='structured_teacher_score_only; no new objective evaluations')

def set_loss(model,state):
    interior=model.interior(state['N'],state['iteration'],'cell')
    boundary=torch.from_numpy(old.faces(state['N'],2)[0]);points=torch.cat((interior,boundary))
    target=torch.from_numpy(state['targets'])
    # Multiple generated neighbors and multiple teacher targets; no forced
    # one-to-one coordinate imitation. Discrete neighbor IDs are stop-gradient.
    ix=cKDTree(points.detach().numpy()).query(state['targets'],k=8)[1]
    squared=(points[torch.from_numpy(ix)]-target[:,None,:]).square().sum(-1)*state['N']
    tau=.02
    return -tau*(torch.logsumexp(-squared.flatten()/tau,0)-math.log(squared.numel())),float(squared.detach().min())

def run():
    started=time.time();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=True);checkpoint=OUT/'candidate_quality_trained_diagnostic.pt'
    if checkpoint.exists():raise FileExistsError('No automatic repeated smoke training')
    protected=json.loads(Path('outputs/sho/stage1B/fresh_spacefill_smoke/training_summary.json').read_text())['protected_hashes']
    # Protect every previously persisted spacefill artifact, not just weights.
    for directory in ('spacefill_downstream_screen','fresh_spacefill_smoke','refined_spacefill'):
        for path in (Path('outputs/sho/stage1B')/directory).rglob('*'):
            if path.is_file():protected[str(path)]=src.sha(path)
    evalruns=torch.load('outputs/sho/stage1B/sobol_vs_spacefill_quick/quick_screen_states.pt',weights_only=False)['runs'][::2]
    held_ids={(r['case']['split'],r['case']['trajectory_id']) for r in evalruns}
    candidates=[]
    for path in sorted(Path('outputs/sho/stage1B/fifo_vs_ranker_closedloop').glob('*_A.partial.pt')):
        r=torch.load(path,weights_only=False)
        if (r['case']['split'],r['case']['trajectory_id']) not in held_ids and len(r['sites'])>=400:candidates.append(r)
    selected=[]
    for category in ('interior','near_boundary','very_near_boundary'):
        group=[r for r in candidates if r['case']['stratum']==category];assert len(group)>=2
        selected+=group[:2]
    states=[s for run in selected for s in states_from(run,'train')]+[s for run in evalruns for s in states_from(run,'held_out')]
    assert not ({(s['case']['split'],s['case']['trajectory_id']) for s in states if s['split']=='train'} & held_ids)
    router=make_router();labels=[]
    for i,state in enumerate(states):
        labels.append(label(state,router,i));print(f'labels {i+1}/{len(states)} {state["id"]}: cell {state["site"]}, u={state["u"]:.4g}, top={len(state["targets"])}',flush=True)
    torch.save(dict(states=states,reference_seed_rule='27095001+state_ordinal',no_objective_evaluations=True),OUT/'offline_teacher_targets.pt')
    src.write_csv(OUT/'candidate_quality_teacher_targets.csv',labels)
    train=[s for s in states if s['split']=='train'];model=FreshGenerator(2);model.load_state_dict(torch.load(INITIAL,weights_only=False)['state_dict'])
    model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    conditions=[(9000+j,'cell') for j in range(4)]+[(9000,'greedy')]
    refs={n:np.random.default_rng(SEED+100000+n).random((65536,2)) for n in (1000,8600,32768)}
    limits={};constraint_rows=[]
    def constraints(step,initial=False):
        failures=[]
        for n in refs:
            st=geometry.sample_stats([model.generate(n,*c) for c in conditions],refs[n],n)
            if initial:
                sb=geometry.sample_stats([geometry.sobol(n,*c) for c in conditions],refs[n],n)
                limits[n]=dict(cover=1.10*sb['cover'],near=st['near']+.01,clustering=.005,holes=sb['holes']+.001,min_CV=.08)
            bad=[k for k in ('cover','near','clustering','holes') if st[k]>limits[n][k]]
            if st['min_CV']<limits[n]['min_CV']:bad.append('min_CV')
            if st['exact']:bad.append('exact_overlap')
            constraint_rows.append(dict(step=step,N=n,**st,failed=json.dumps(bad)));failures += [f'{n}:{b}' for b in bad]
        return failures
    failures=constraints(0,True);history=[]
    def quality(step):
        with torch.no_grad():values=[set_loss(model,s) for s in train]
        mean=float(np.mean([x[0].item() for x in values]));nearest=float(np.mean([x[1] for x in values]))
        history.append(dict(record_type='training_monitor',step=step,mean_softmin=mean,mean_nearest_squared_scaled=nearest))
        return mean
    initial=quality(0);best=initial;best_state=copy.deepcopy(model.state_dict());best_step=0;last_improved=0
    accepted=0;attempted=0;stop='60_update_cap';train_started=time.time()
    criteria=dict(objective='mean differentiable soft minimum over top-8 target coordinates and their eight nearest generated probes; squared distances scaled by N',
        softmin_temperature=.02,learning_rate=.002,updates_cap=60,training_geometry_limits=limits,
        trajectory_disjoint=True,no_true_parameters=True,no_new_true_objective_evaluations_during_training=True,
        archived_losses_usage='Only to reconstruct existing teacher response surfaces, PO selection, and u; not generator inputs or optimization labels.',
        evaluation_gate='geometry feasible; held-out distance-to-target improves; median reference-anchored best-score gain >=0; new wins vs fresh-v1 actual loss in >=2/3 states and vs Sobol in >=1/2; no missing candidates; initial six preserve old improvement in >=4 and trajectory 280 no worse than old')
    (OUT/'criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
    if failures:stop='initial_geometry_failure:'+','.join(failures)
    else:
        for step in range(1,61):
            optimizer.zero_grad();losses=[set_loss(model,s)[0] for s in train];loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):stop='nonfinite_quality_loss';break
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();attempted=step
            if step%5==0:
                value=quality(step);failures=constraints(step)
                print(f'quality update={step}/60 train_softmin={value:.6g} initial={initial:.6g} geometry_bad={failures}',flush=True)
                if failures:stop='geometry_failure:'+','.join(failures);break
                accepted=step
                if value<best*.999:
                    best=value;best_state=copy.deepcopy(model.state_dict());best_step=step;last_improved=step
                if step>=15 and best>=initial*.99:stop='no_clear_quality_improvement';break
                if step-last_improved>=15:stop='quality_plateau';break
            if time.time()-train_started>180:stop='three_minute_training_cap';break
    model.load_state_dict(best_state);model.eval().requires_grad_(False)
    digest=save_checkpoint(checkpoint,dict(status='TRAINED_DIAGNOSTIC_PENDING_EVALUATION',state_dict=model.state_dict(),dimension=2,source_checkpoint=str(INITIAL),source_sha256=src.sha(INITIAL),
        attempted_updates=attempted,feasible_updates=accepted,selected_update=best_step,stop_reason=stop,criteria=criteria,
        teacher_targets_sha256=src.sha(OUT/'offline_teacher_targets.pt'),training_state_ids=[s['id'] for s in train],held_out_state_ids=[s['id'] for s in states if s['split']=='held_out'],
        seed=SEED,initial_quality_loss=initial,selected_quality_loss=best,implementation='experiments/sho/fresh_spacefill_generator.py',implementation_sha256=src.sha('experiments/sho/fresh_spacefill_generator.py')))
    src.write_csv(OUT/'candidate_quality_training.csv',history);src.write_csv(OUT/'candidate_quality_training_constraints.csv',constraint_rows)
    result=dict(checkpoint=str(checkpoint),sha256=digest,initial_quality_loss=initial,selected_quality_loss=best,selected_update=best_step,
        attempted_updates=attempted,feasible_updates=accepted,stop_reason=stop,training_seconds=time.time()-train_started,preparation_and_training_seconds=time.time()-started,
        training_states=len(train),held_out_states=len(states)-len(train),protected_hashes=protected,protected_unchanged=all(src.sha(p)==h for p,h in protected.items()))
    (OUT/'training_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='protected_hashes'},indent=2),flush=True)

if __name__=='__main__':run()
