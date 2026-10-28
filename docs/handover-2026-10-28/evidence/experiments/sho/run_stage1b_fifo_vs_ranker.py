"""Small paired Sobol-only FIFO versus frozen CandidateRanker recovery screen."""
import csv
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sho import run_stage1b_learned_global_search_imitation as im
from experiments.sho import run_stage1b_ordinary_cell_score_full as ordinary
from experiments.sho import validate_stage1b_learned_cell_size as cells
from experiments.sho import run_stage1b_learned_powell_2d as local
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho import run_stage1b_n3_rollout_precision as rollout

edirect = im.edirect
nearest = cells.nearest
down = nearest.downstream
OUT = Path('outputs/sho/stage1B/fifo_vs_ranker_closedloop')
COMPARISON = OUT/'fifo_vs_ranker_closedloop.csv'
DIVERGENCES = OUT/'fifo_vs_ranker_divergences.csv'
GATE = 1e-4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def data_and_plan():
    disagreements = Counter()
    with Path('outputs/sho/stage1B/final_selector_audit/final_selector_distribution.csv').open() as f:
        for r in csv.DictReader(f):
            if r['record_type']=='decision' and r['mode']=='normal_edirect' and r['ranker_error']=='True':
                disagreements[int(r['trajectory_id'])] += 1
    datasets = {}
    for split, count, seed in [('train',256,im.TRAIN_SEED),('test',1024,im.TEST_SEED)]:
        p, _, c = im._make_exact_labeled_split(count, seed)
        rng = torch.Generator().manual_seed(seed)
        torch.rand(count,generator=rng,dtype=torch.float64); torch.randperm(count,generator=rng)
        initial = 2*torch.rand(count,2,generator=rng,dtype=torch.float64)-1
        omega = torch.acos(c/2)/float(rollout.DELTA_T)
        times = torch.arange(rollout.P,dtype=torch.float64)[None,:].expand(count,-1)*float(rollout.DELTA_T)
        assert torch.equal(rollout._state_at_time(initial,omega,times)[0],p)
        edge = np.minimum(np.minimum(c.numpy()-edirect.LOWER,edirect.UPPER-c.numpy()), .99)
        datasets[split] = dict(p=p.numpy(), c=c.numpy(), initial=initial, omega=omega, edge=edge, seed=seed)
    strata = {}
    for name in ['interior','near_boundary','very_near_boundary']:
        split = 'test' if name=='very_near_boundary' else 'train'
        d = datasets[split]; e=d['edge']
        mask = e>=.1 if name=='interior' else (e>=.01)&(e<.1) if name=='near_boundary' else (e>=0)&(e<.01)
        ids = [int(i) for i in np.flatnonzero(mask) if split=='test' or disagreements[int(i)]>0]
        ids.sort(key=lambda i:d['c'][i])
        assert len(ids)>=6
        chosen = [ids[int(j)] for j in np.linspace(0,len(ids)-1,6).round().astype(int)]
        strata[name] = [dict(split=split, trajectory_id=i, stratum=name,
                            boundary_distance=float(e[i]), generator_id=im.SPLIT_ID_OFFSET[split]+i,
                            archived_normal_disagreements=disagreements[i] if split=='train' else None)
                        for i in chosen]
    cases = [strata[s][j] for j in range(6) for s in strata]
    return datasets, cases


class FrozenProposals:
    """Same assembled frozen components in both arms; production Sobol only."""
    def __init__(self):
        self.distance,_ = cells.base._load_locked_distance()
        op = torch.load(nearest.OPERATOR,weights_only=False)
        self.margin = nn.Linear(2,1,bias=False,dtype=torch.float64)
        self.margin.load_state_dict(torch.load(op['margin_checkpoint'],weights_only=False)['state_dict'])
        self.threshold = nearest.threshold.NonzeroThreshold()
        self.threshold.load_state_dict(torch.load(op['threshold_checkpoint'],weights_only=False)['state_dict'])
        self.half = nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.half.load_state_dict(torch.load(cells.OUT/'half_spacing_operator_locked.pt',weights_only=False)['state_dict'])
        self.score,self.ranker = ordinary._load_search_models()
        self.locked=down._load_downstream()
        for m in [self.distance,self.margin,self.threshold,self.half,self.score,self.ranker]:
            m.eval().requires_grad_(False)
        self.cache={}

    def __call__(self, sites, losses, case, iteration):
        key=(case['generator_id'],iteration,array_hash(sites),array_hash(losses))
        if key in self.cache:
            proposal,meta=self.cache[key]
            return proposal.copy(),meta
        probes=edirect._candidate_pool(len(sites),case['generator_id'],iteration)
        captured,_=cells.capture_nearest(sites,probes,self.distance,self.margin,self.threshold,iteration)
        p=len(probes); owners=captured['winner'][:p]; distances=captured['winner_distance'][:p]
        radius,_,stats=cells.assigned_radius(owners,distances,len(sites),self.margin,self.threshold)
        half=cells.predict_half(self.half,captured['winner_distance'][p:])
        replace,_,invalid,wrong=cells.larger_candidate(self.margin,self.threshold,radius,half)
        assert not(invalid or wrong or stats['radius_pairwise_mistakes'] or stats['radius_invalid_logits'])
        sizes=np.where(replace,half,radius)
        locked=self.locked
        grouped=down._learned_group_and_represent(sizes,losses,locked)
        hull=down.hull_threshold._threshold_hull(locked['skew_weight'],locked['homogeneous_weight'],
             locked['hull_theta'],locked['hull_alpha'],grouped['grouped_sizes'],grouped['grouped_losses'])
        pool=down.improvement_and._build_pool([dict(source_state_index=iteration,
             trajectory_id=case['trajectory_id'],iteration=iteration,state_bin=cells.base.turn._state_bin(len(sizes)),
             sizes=grouped['grouped_sizes'],losses=grouped['grouped_losses'],teacher_hull=hull)],'fifo_ranker_frozen')
        with torch.no_grad(): passed=locked['and_model'](torch.from_numpy(pool['inputs'])).numpy().ravel()>=0
        selected=hull[pool['hull_position'][passed]]
        if not len(selected):selected=np.asarray([np.argmax(grouped['grouped_sizes'])])
        po=sorted(grouped['representatives'][selected].tolist(),key=lambda i:(sizes[i],losses[i],i))
        surface=edirect.GaussianLocalResponse(sites,losses)
        weights=.5*edirect._normalize(sizes[po])+.5*edirect._normalize(surface.loo_errors(np.asarray(po)))
        largest=int(np.argmax(sizes[po])); proposals=[]
        for j,site in enumerate(po):
            ix=np.flatnonzero(owners==site)
            if not len(ix):continue
            if j==largest:
                winner=int(np.argmax(distances[ix]))
            else:
                x=np.column_stack((edirect._normalize(distances[ix]),
                                  edirect._normalize(surface.predict(probes[ix])),np.full(len(ix),weights[j])))
                with torch.no_grad():winner=int(np.argmax(self.score(torch.from_numpy(x)).numpy()))
            point=probes[ix[winner]]
            if np.min(np.linalg.norm(sites-point,axis=1))>1e-12:proposals.append(point)
        if not proposals:
            for i in np.argsort(distances)[::-1]:
                if distances[i]>1e-12:proposals.append(probes[i]);break
        proposals=np.asarray(proposals,dtype=np.float64).reshape(-1,2)
        assert len(proposals)>0
        meta=dict(probe_sha256=array_hash(probes),probe_count=len(probes),PO_cells=len(po))
        self.cache[key]=(proposals.copy(),meta)
        return proposals,meta


def run_search(case, position, arm, frozen):
    started=time.monotonic(); last_notice=started
    bounds=ordinary.FULL_BOUNDS
    partial=OUT/f"{case['split']}_{case['trajectory_id']}_{arm}.partial.pt"
    elapsed_before=0.
    if partial.exists():
        saved=torch.load(partial,weights_only=False)
        assert saved['case']==case and saved['arm']==arm
        sites,losses,events,iteration=saved['sites'],saved['losses'],saved['events'],saved['iteration']
        assert np.array_equal(sites[:10],edirect._initial_design())
        elapsed_before=saved.get('elapsed_seconds',0.)
        print(f"Resuming {case['split']}:{case['trajectory_id']} {arm} from saved batch boundary, eval={len(sites)} iteration={iteration}",flush=True)
    else:
        sites=edirect._initial_design().copy()
        losses=edirect._prefix_losses_numpy(edirect._physical(sites),position)
        events=[]
        for i,(point,loss) in enumerate(zip(sites,losses)):
            events.append(dict(mode='normal',evaluation=i+1,iteration=-1,kind='initial_design',
                               point=edirect._physical(point),loss=float(loss),best_loss=float(losses[:i+1].min())))
        iteration=0
    last_snapshot=len(sites)
    while len(sites)<im.NORMAL_CAP and losses.min()>GATE:
        pending,meta=frozen(sites,losses,case,iteration)
        while len(pending) and len(sites)<im.NORMAL_CAP and losses.min()>GATE:
            global_=im._global_features(sites,losses,len(pending),len(sites),im.NORMAL_CAP,iteration,bounds,'normal_edirect')
            if arm=='B':choice,scores=im._model_choice(frozen.ranker,sites,losses,pending,'cell',global_)
            else:choice,scores=0,None
            point=pending[choice].copy()
            loss=float(edirect._prefix_losses_numpy(edirect._physical(point[None,:]),position)[0])
            queue=pending.copy()
            sites=np.vstack((sites,point));losses=np.r_[losses,loss]
            pending=np.delete(pending,choice,axis=0)
            order=np.lexsort((sites[:,1],sites[:,0]))
            events.append(dict(mode='normal',evaluation=len(sites),iteration=iteration,kind='cell',
                point=edirect._physical(point),loss=loss,best_loss=float(losses.min()),choice=choice,
                queue=queue,scores=scores,history_order_hash=array_hash(np.column_stack((sites,losses))),
                evaluated_set_hash=array_hash(np.column_stack((sites[order],losses[order]))),
                remaining_queue_hash=array_hash(pending),**meta))
        if len(sites)<im.NORMAL_CAP and losses.min()>GATE:
            greedy=edirect._greedy_proposal(sites,losses,case['generator_id'],iteration)
            if greedy is not None:
                value=float(edirect._prefix_losses_numpy(edirect._physical(greedy[None,:]),position)[0])
                sites=np.vstack((sites,greedy));losses=np.r_[losses,value]
                order=np.lexsort((sites[:,1],sites[:,0]))
                events.append(dict(mode='normal',evaluation=len(sites),iteration=iteration,kind='greedy',
                    point=edirect._physical(greedy),loss=value,best_loss=float(losses.min()),
                    history_order_hash=array_hash(np.column_stack((sites,losses))),
                    evaluated_set_hash=array_hash(np.column_stack((sites[order],losses[order]))),
                    remaining_queue_hash=array_hash(np.empty((0,2)))))
        iteration+=1
        if len(sites)-last_snapshot>=100:
            temp=partial.with_suffix('.tmp')
            torch.save(dict(case=case,arm=arm,status='RUNNING',sites=sites,losses=losses,
                            events=events,iteration=iteration,
                            elapsed_seconds=elapsed_before+time.monotonic()-started),temp)
            temp.replace(partial)
            last_snapshot=len(sites)
        if time.monotonic()-last_notice>20:
            print(f"{case['split']}:{case['trajectory_id']} {arm} eval={len(sites)} iteration={iteration} best={losses.min():.4g}",flush=True)
            last_notice=time.monotonic()
    normal=im.ModeRun('normal_edirect',bounds.copy(),sites,losses,len(losses),bool(losses.min()<=GATE))
    modes=[normal]
    if not normal.success:
        print(f"{case['split']}:{case['trajectory_id']} {arm}: common deterministic boundary rescue",flush=True)
        rescued,_=im._run_boundary_rescue(split=case['split'],trajectory_id=case['trajectory_id'],
             generator_id=case['generator_id'],position=position,normal=normal,collect_trace=False)
        modes.append(rescued)
        if not rescued.success:
            print(f"{case['split']}:{case['trajectory_id']} {arm}: common nested rescue",flush=True)
            modes.append(im._run_nested_rescue(split=case['split'],trajectory_id=case['trajectory_id'],
                         position=position,boundary_run=rescued,collect_trace=False))
    best=normal.best_loss
    for mode in modes[1:]:
        start=len(mode.losses)-mode.new_evaluations
        for p,value in zip(im._physical(mode.points[start:],mode.bounds),mode.losses[start:]):
            best=min(best,float(value))
            events.append(dict(mode=mode.mode,evaluation=len(events)+1,iteration='',kind='common_rescue',
                               point=p,loss=float(value),best_loss=best))
    final=min(modes,key=lambda m:m.best_loss)
    return dict(case=case,arm=arm,events=events,modes=modes,final=final,
                normal_success=normal.success,success=any(m.success for m in modes),
                total_evaluations=len(events),elapsed_seconds=elapsed_before+time.monotonic()-started)


def finish_recovery(run, dataset):
    case=run['case'];tid=case['trajectory_id'];final=run['final'];position=dataset['p'][tid]
    if not run['success']:return dict(polish_status='NOT_RUN_GLOBAL_HANDOFF_FAILED')
    if len(run['modes'])>1: lower,upper=run['modes'][1].bounds[:,0],run['modes'][1].bounds[:,1]
    else:
        spacing=im.DOMAIN_WIDTH/math.sqrt(run['modes'][0].new_evaluations)
        lower=np.maximum(edirect.LOWER,final.best_physical-spacing)
        upper=np.minimum(edirect.UPPER,final.best_physical+spacing)
    # Existing 500-evaluation policy plus its existing 1500 continuation budget;
    # same line seeds and stopping rules in both arms. Snapshots retain 500 result.
    n2=local._run_trajectory((case['generator_id'],position,lower,upper,final.best_physical,
                             final.best_loss,'matched_fifo_ranker_handoff'),max_evaluations=1500,
                             snapshot_budgets=(500,750,1000,1500))
    endpoint=polish.N2Endpoint(tid,n2.final_theta,n2.final_loss,'frozen_learned_powell',case['stratum'],False)
    n3=polish._polish(endpoint,torch.from_numpy(position))
    theta=n3.theta_history[-1]; loss=n3.loss_history[-1]
    reference=np.asarray([-1.,dataset['c'][tid]])
    result=dict(polish_status='COMPLETED',local_evaluations=n2.evaluations,local_prefix_loss=n2.final_loss,
        local_a=float(n2.final_theta[0]),local_b=float(n2.final_theta[1]),
        local_lower=json.dumps(lower.tolist()),local_upper=json.dumps(upper.tolist()),
        newton_steps=n3.accepted_steps,newton_objective_evaluations=n3.actual_loss_evaluations,
        newton_stop_reason=n3.stop_reason,final_a=float(theta[0]),final_b=float(theta[1]),
        final_parameter_max_error=float(np.max(np.abs(theta-reference))),
        final_parameter_l2_error=float(np.linalg.norm(theta-reference)),final_prefix_loss=loss,
        reaches_1e_minus29=loss<=1e-29,reaches_1e_minus28=loss<=1e-28,reaches_1e_minus24=loss<=1e-24,
        reference_prefix_loss=float(polish._scalar_loss(torch.from_numpy(reference),torch.from_numpy(position))))
    c=dataset['c'][tid]
    for periods in rollout.PERIODS:
        steps=int(math.ceil(periods*math.tau/math.acos(c/2)))
        times=torch.arange(rollout.P,rollout.P+steps,dtype=torch.float64)*float(rollout.DELTA_T)
        omega=dataset['omega'][tid];ic=dataset['initial'][tid]
        truth=ic[0]*torch.cos(omega*times)+(ic[1]/omega)*torch.sin(omega*times)
        prediction=rollout._recurrence_rollout(torch.tensor([theta[0]]),torch.tensor([theta[1]]),
                    torch.tensor([position[-2]]),torch.tensor([position[-1]]),steps)[0]
        error=prediction-truth
        result[f'rollout_{periods}_period_mse']=float(torch.mean(error**2))
        result[f'rollout_{periods}_period_max_abs']=float(torch.max(torch.abs(error)))
    return result


def compare_runs(a,b):
    ea,eb=a['events'],b['events'];pair=[];first=None; reconverged=[];later_same=0
    for i in range(min(len(ea),len(eb))):
        x,y=ea[i],eb[i]
        equal=np.array_equal(x['point'],y['point'])
        if first is None and not equal:
            first=i
            assert 'queue' in x and 'queue' in y
            assert np.array_equal(x['queue'],y['queue']), 'Predivergence inputs differ'
            for arm,event in [('A',x),('B',y)]:
                for j,p in enumerate(event['queue']):
                    row=dict(record_type='first_divergence_queue',**a['case'],arm=arm,evaluation=i+1,
                        iteration=event['iteration'],queue_index=j,normalized_x=float(p[0]),normalized_y=float(p[1]),
                        physical_point=json.dumps(edirect._physical(p).tolist()),chosen=j==event['choice'],
                        actual_evaluated_loss=event['loss'] if j==event['choice'] else '',
                        ranker_logit=float(event['scores'][j]) if event.get('scores') is not None else '',
                        probe_sha256=event['probe_sha256'])
                    pair.append(row)
        if first is not None and i>first:
            later_same+=equal
            if x.get('evaluated_set_hash') and x.get('evaluated_set_hash')==y.get('evaluated_set_hash'):
                reconverged.append(i+1)
    info=dict(first_divergence_evaluation=first+1 if first is not None else '',
              first_divergence_iteration=ea[first]['iteration'] if first is not None else '',
              first_divergence_A_loss=ea[first]['loss'] if first is not None else '',
              first_divergence_B_loss=eb[first]['loss'] if first is not None else '',
              first_divergence_A_point=json.dumps(ea[first]['point'].tolist()) if first is not None else '',
              first_divergence_B_point=json.dumps(eb[first]['point'].tolist()) if first is not None else '',
              later_identical_next_points=later_same,evaluated_set_reconvergence_count=len(reconverged),
              first_evaluated_set_reconvergence=reconverged[0] if reconverged else '',
              full_ordered_history_reconverged=False if first is not None else True,
              final_global_point_exact=np.array_equal(a['final'].best_physical,b['final'].best_physical),
              final_evaluated_sets_equal=sorted(tuple(r['point']) for r in ea)==sorted(tuple(r['point']) for r in eb))
    pair.append(dict(record_type='trajectory_divergence_summary',**a['case'],**info))
    return info,pair


def run_pair(job):
    case,dataset=job
    try:
        torch.set_num_threads(1)
        frozen=FrozenProposals()
        position=dataset['p'][case['trajectory_id']]
        results=[]
        print(f"Started matched case {case['split']}:{case['trajectory_id']} {case['stratum']}",flush=True)
        for arm in ['A','B']:
            path=OUT/f"{case['split']}_{case['trajectory_id']}_{arm}.pt"
            if path.exists():
                result=torch.load(path,weights_only=False)
                assert result['case']==case and result['arm']==arm
            else:
                result=run_search(case,position,arm,frozen)
                print(f"  {case['split']}:{case['trajectory_id']} {arm}: handoff={result['success']} evals={result['total_evaluations']} best={result['final'].best_loss:.4g}; local/polish",flush=True)
                result['recovery']=finish_recovery(result,dataset)
                torch.save(result,path)
            results.append(result)
        return case,results,None
    except Exception:
        return case,None,traceback.format_exc()


def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    datasets,cases=data_and_plan()
    protected=json.loads(Path('outputs/sho/stage1B/final_selector_audit/audit_verification.json').read_text())['protected_hashes']
    # Do not load any space-filling generator. Retain prior hashes as protection only.
    for path in [local.LINE_CHECKPOINT,ordinary.CHECKPOINT_PATH,ordinary.SELECTOR_PATH]+[
        Path(inspect.getsourcefile(m)) for m in (im,edirect,local,polish,rollout,ordinary,cells,nearest,down)]:
        protected.setdefault(str(path),sha(path))
    assert all(sha(p)==h for p,h in protected.items())
    workers=int(os.environ.get('FIFO_RANKER_WORKERS','4'))
    plan=dict(cases=cases,planned_trajectories=18,normal_cap=im.NORMAL_CAP,rescue_cap=im.RESCUE_CAP,
        nested_rescue='unchanged existing finite grid/refinement schedule',gate=GATE,
        local_cap=1500,local_snapshots=[500,750,1000,1500],newton_max_steps=polish.MAX_ACCEPTED_STEPS,
        arms={'A':'FIFO','B':'old frozen CandidateRanker'},
        upstream='same frozen distance, argmin, cell size, grouping, representatives, hull, improvement, old ordinary scorer; existing deterministic largest-cell/greedy scaffolding',
        candidate_ranker_scope='normal cell queues only; identical initial evaluated sites; common deterministic rescue unchanged',
        probe_source='unchanged production Sobol+boundary recipe; same generator id, N and iteration imply identical coordinates',
        local_randomness='same frozen line controller, LINE_SEED+100003*generator_id+line_serial in each arm',
        stop='complete a matched pair; stop remaining screen immediately if A succeeds and B fails handoff or A reaches existing 1e-24 polish milestone and B does not',
        floor_calibration='existing N3 baseline worst 1.663267547402652e-26 at parameter errors <=4.440892098500626e-16; report 1e-29/1e-28 but do not use those as universal reliability gates',
        execution=f'{workers} isolated worker processes; each matched pair shares its own frozen upstream cache',
        preliminary_execution='serial first arm terminated before any handoff or paired result, then replayed with identical settings for parallel execution',
        reference_usage='stratification and posthoc metrics only',protected_hashes=protected)
    if (OUT/'plan.json').exists():
        previous=json.loads((OUT/'plan.json').read_text())
        for key in ('cases','normal_cap','rescue_cap','local_cap','gate','upstream','protected_hashes'):
            assert previous[key]==plan[key],f'Resume would change {key}'
    assert not (OUT/'summary.json').exists(),'Completed screen must not be restarted'
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    summaries=[];divergences=[];curves=[];reason=None
    pool=multiprocessing.get_context('spawn').Pool(workers)
    completed_keys=set()
    jobs=[(case,datasets[case['split']]) for case in cases]
    for case,results,error in pool.imap_unordered(run_pair,jobs,chunksize=1):
        if error:
            reason='FROZEN_PIPELINE_OR_HARNESS_ERROR'
            (OUT/'execution_error.txt').write_text(json.dumps(case)+'\n'+error)
            print(error,flush=True)
            break
        a,b=results
        completed_keys.add((case['split'],case['trajectory_id']))
        info,pair=compare_runs(a,b);divergences+=pair
        for result in results:
            events=result['events'];hits=[e['evaluation'] for e in events if e['best_loss']<=GATE]
            row=dict(record_type='trajectory',**case,arm=result['arm'],global_handoff_success=result['success'],
                normal_handoff_success=result['normal_success'],evaluations_to_handoff=hits[0] if hits else '',
                search_evaluations=result['total_evaluations'],search_best_loss=result['final'].best_loss,
                best_loss_at_budget_or_earlier_handoff=result['final'].best_loss,
                unused_search_budget_after_handoff=result['success'],
                rescue_invoked=len(result['modes'])>1,nested_rescue_invoked=len(result['modes'])>2,
                global_seconds=result['elapsed_seconds'],
                ranker_nonFIFO_choices=sum(e.get('choice',0)!=0 for e in events),
                stalled_last100=len(events)>=100 and events[-1]['best_loss']==events[-100]['best_loss'],
                **info,**result['recovery'])
            summaries.append(row)
            for e in events:
                curves.append(dict(record_type='global_evaluation',**case,arm=result['arm'],
                    evaluation=e['evaluation'],mode=e['mode'],iteration=e['iteration'],kind=e['kind'],
                    evaluated_a=float(e['point'][0]),evaluated_b=float(e['point'][1]),
                    observed_loss=e['loss'],best_loss=e['best_loss'],queue_choice=e.get('choice','')))
        print('PAIR '+json.dumps([{k:r.get(k) for k in ['arm','global_handoff_success','evaluations_to_handoff','final_prefix_loss','final_parameter_max_error']} for r in summaries[-2:]])+f" divergence={info['first_divergence_evaluation']}",flush=True)
        write_csv(COMPARISON,summaries+curves);write_csv(DIVERGENCES,divergences)
        assert all(sha(p)==h for p,h in protected.items())
        if a['success'] and not b['success']:
            reason='RANKER_GLOBAL_RECOVERY_LOSS';break
        if a['recovery'].get('reaches_1e_minus24') and not b['recovery'].get('reaches_1e_minus24'):
            reason='RANKER_FINAL_NUMERICAL_FLOOR_LOSS';break
    if reason:pool.terminate()
    else:pool.close()
    pool.join()
    completed=len(summaries)//2
    for case in cases:
        if (case['split'],case['trajectory_id']) in completed_keys:continue
        for arm in ['A','B']:
            full=OUT/f"{case['split']}_{case['trajectory_id']}_{arm}.pt"
            partial=OUT/f"{case['split']}_{case['trajectory_id']}_{arm}.partial.pt"
            if full.exists():
                r=torch.load(full,weights_only=False)
                summaries.append(dict(record_type='completed_arm_pair_interrupted',**case,arm=arm,reason=reason,
                    global_handoff_success=r['success'],search_evaluations=r['total_evaluations'],
                    search_best_loss=r['final'].best_loss,**r['recovery']))
            elif partial.exists():
                r=torch.load(partial,weights_only=False)
                summaries.append(dict(record_type='interrupted_partial_arm',**case,arm=arm,reason=reason,
                    search_evaluations=len(r['losses']),search_best_loss=float(r['losses'].min()),
                    partial_artifact=str(partial),status='CENSORED_NOT_A_RECOVERY_FAILURE'))
            else:summaries.append(dict(record_type='not_run_hard_stop',**case,arm=arm,reason=reason))
    write_csv(COMPARISON,summaries+curves);write_csv(DIVERGENCES,divergences)
    result=dict(completed_matched_trajectories=completed,planned_trajectories=18,stop_reason=reason,
                protected_hashes_unchanged=all(sha(p)==h for p,h in protected.items()))
    for arm in ['A','B']:
        rows=[r for r in summaries if r['record_type']=='trajectory' and r['arm']==arm]
        hits=[r['evaluations_to_handoff'] for r in rows if r['global_handoff_success']]
        result[arm]=dict(successes=sum(r['global_handoff_success'] for r in rows),
            failures=[f"{r['split']}:{r['trajectory_id']}" for r in rows if not r['global_handoff_success']],
            median_evaluations_to_handoff=float(np.median(hits)) if hits else None,
            worst_evaluations_to_handoff=max(hits) if hits else None,
            floor_1e_minus28=sum(r.get('reaches_1e_minus28',False) for r in rows),
            worst_final_loss=max((r.get('final_prefix_loss',float('inf')) for r in rows),default=None))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
