"""Actual-set geometry curriculum; existing inference/architecture untouched.

Mirrored weight mutations optimize measured FINAL hard-set occupancy, avoiding
the expected-occupancy gradient surrogate from the preceding experiment.
Sobol/fresh-v1 measurements are loaded only after stage checkpoints are saved.
"""
import copy
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator, boundary_faces
from experiments.sho.train_progressive_spacefill_smoke import sha, write_csv
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics, overlap

N = 8600
STAGES = (8, 16, 32, 64, 128)
OUT = Path('outputs/sho/stage1B/progressive_scale_curriculum')
PARENT = Path('outputs/sho/stage1B/progressive_spacefill_smoke')
INITIAL = PARENT/'progressive_best_before_evaluation.pt'
MONITOR = [(8400,'cell'),(8401,'cell'),(8400,'greedy')]
HELDOUT = [(100+j,'cell') for j in range(4)]+[(100,'greedy')]


def counts(points, scales):
    result = {}
    for k in scales:
        fano, anchored, rms = [], [], []
        for shift in (0., .25, .5, .75):
            idx = (((points+shift/k)%1)*k).astype(int).clip(0,k-1)
            hist = np.bincount(idx[:,0]*k+idx[:,1], minlength=k*k).reshape(k,k)
            residual = hist-N/k**2
            fano.append(float(np.sum(residual**2)/N))
            rms.append(float(np.sqrt(np.mean(residual**2))))
            edge = np.arange(1,k+1)/k
            anchored.append(float(np.abs(hist.cumsum(0).cumsum(1)/N-edge[:,None]*edge[None,:]).max()))
        result[k] = dict(count_fano_mean=float(np.mean(fano)), count_RMSE=float(np.sqrt(np.mean(np.square(rms)))),
                         count_fano_worst=max(fano), anchored_discrepancy_worst=max(anchored))
    return result


def sample(model, conditions, scales, reference=None):
    values, pools, geometry = [], [], []
    with torch.no_grad():
        for iteration,pool in conditions:
            complete, interior = model.generate(N,iteration,pool)
            values.append(counts(interior,scales)); pools.append(interior)
            if reference is not None:
                distance=cKDTree(complete).query(reference)[0]
                near=cKDTree(interior).query(interior,k=2)[0][:,1]
                geometry.append(dict(cover=float(distance.max()), holes=int(np.sum(distance>1/math.sqrt(N))),
                    hole_fraction=float(np.mean(distance>1/math.sqrt(N))), nn_CV=float(near.std()/near.mean())))
    scale_stats={k:{field:float(np.mean([v[k][field] for v in values])) if field in ('count_fano_mean','count_RMSE')
                       else max(v[k][field] for v in values) for field in values[0][k]} for k in scales}
    result=dict(scales=scale_stats)
    if reference is not None:
        result.update(cover=max(v['cover'] for v in geometry), holes=max(v['holes'] for v in geometry),
            hole_fraction=max(v['hole_fraction'] for v in geometry), nn_CV=min(v['nn_CV'] for v in geometry),
            overlaps=[overlap(pools[0],v,N) for v in pools[1:]])
    return result


def primary(value, base, active):
    ratios=[value['scales'][k]['count_fano_mean']/base['scales'][k]['count_fano_mean'] for k in active]
    return ratios[-1]+(.5*np.mean(ratios[:-1]) if len(ratios)>1 else 0.)


def retains(value, base, active):
    return all(value['scales'][k]['count_fano_mean'] <= 1.10*base['scales'][k]['count_fano_mean']
               and value['scales'][k]['anchored_discrepancy_worst'] <= 1.10*base['scales'][k]['anchored_discrepancy_worst']+1e-15
               for k in active[:-1])


def safeguards(value, base):
    return (value['cover'] <= 1.05*base['cover'] and value['hole_fraction'] <= base['hole_fraction']+.0005
            and value['nn_CV'] >= .08
            and all(o['exact_overlap_count']==0 and o['near_overlap_fraction_005_spacing']<=.03 for o in value['overlaps']))


def freeze(state):
    return {k:v.detach().clone() for k,v in state.items()}


def save_stage(model, stage, active, parent, best_round, history, config):
    cp=OUT/f'stage_{stage}_{active[-1]}x{active[-1]}_best.pt'
    digest=save_checkpoint(cp,dict(state_dict=freeze(model.state_dict()),stage=stage,active_loss_scales=active,
        N=N,parent_checkpoint=str(parent),parent_sha256=sha(parent),best_round=best_round,
        config=config,training_history=history,status='STAGE_TRAINED_PENDING_HELDOUT_GATE'))
    return cp,digest


def heldout(model, stage, active, reference, detail_rows):
    vals=[]; geom=[]; pools=[]
    for iteration,pool in HELDOUT:
        with torch.no_grad(): complete,interior=model.generate(N,iteration,pool)
        boundary=complete[np.any((complete==0)|(complete==1),axis=1)]
        assert np.array_equal(boundary,boundary_faces(N,2))
        m=metrics(complete,interior,reference,N); geom.append(m); pools.append(interior)
        c=counts(interior,active); vals.append(c)
        for k in active:
            detail_rows.append(dict(record_type='heldout_set',stage=stage,current_scale=active[-1],scale=k,N=N,
                iteration=iteration,pool_type=pool,**c[k],**m))
    ov=[overlap(pools[i],pools[j],N) for i,j in [(0,1),(1,2),(2,3),(0,4)]]
    result=dict(scales={k:{field:float(np.mean([v[k][field] for v in vals])) if field in ('count_fano_mean','count_RMSE') else max(v[k][field] for v in vals) for field in vals[0][k]} for k in active},
        cover=max(v['complete_cover'] for v in geom), interior_cover=max(v['interior_only_cover'] for v in geom),
        boundary_cover=max(v['boundary_cover'] for v in geom), holes=max(v['large_hole_count'] for v in geom),
        hole_fraction=max(v['large_hole_fraction'] for v in geom), nn_CV=min(v['nn_CV'] for v in geom),
        nn_CV_mean=float(np.mean([v['nn_CV'] for v in geom])), overlaps=ov,
        exact_overlap_max=max(v['exact_overlap_fraction'] for v in ov),
        near_overlap_max=max(v['near_overlap_fraction_005_spacing'] for v in ov),
        near_overlap_010_max=max(v['near_overlap_fraction_010_spacing'] for v in ov),
        cell_greedy_near_overlap=ov[-1]['near_overlap_fraction_005_spacing'])
    np.savez_compressed(OUT/f'stage_{stage}_heldout_interiors.npz',**{f'iteration{i}_{p}':x for (i,p),x in zip(HELDOUT,pools)})
    return result


def baseline(k):
    # Diagnostics only: never referenced by the optimizer, monitor or selection.
    rows=list(csv.DictReader((PARENT/'progressive_spacefill_multiscale.csv').open()))
    answer={}
    for name in ('Sobol','fresh_v1','progressive'):
        values=[v for v in rows if int(float(v['N']))==N and v['generator']==name and int(float(v['bins_per_axis']))==k]
        answer[name]=dict(count_fano_mean=float(np.mean([float(v['count_fano']) for v in values])),
                         anchored_discrepancy_worst=max(float(v['grid_anchored_discrepancy']) for v in values))
    return answer


def run():
    start=time.monotonic(); torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=False)
    parent_summary=json.loads((PARENT/'training_summary.json').read_text())
    protected=dict(parent_summary['protected_hashes'])
    protected.update({str(p):sha(p) for p in PARENT.iterdir() if p.is_file()})
    protected['experiments/sho/progressive_spacefill_generator.py']=sha('experiments/sho/progressive_spacefill_generator.py')
    assert sha(INITIAL)==parent_summary['sha256']
    config=dict(N=N,stages=STAGES,initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),
        architecture='unchanged float64 34->32 tanh->1 and exact existing generate() method',
        warm_start='existing progressive weights, previously trained across scales; THIS continuation introduces loss scales 8,16,32,64,128 sequentially',
        optimizer='mirrored (1+8) weight-mutation search; actual final hard-set occupancy; common context seeds',
        optimizer_seed=27112026,mutation_sigmas=[.006,.012,.024,.048],max_rounds_per_stage=40,
        plateau_rounds=10,monitor_target_improvement=.20,heldout_required_improvement=.15,
        loss='mean count Fano across four shifts and contexts at current scale / entry value; +0.5 mean normalized previous-scale losses',
        previous_scale_max_regression=.10,coverage_max_regression=.05,hole_fraction_max_increase=.0005,
        near_overlap_max=.03,near_overlap_max_increase=.005,min_nn_CV=.08,
        input_features='all original features retained; only active scale terms contribute to loss',
        checkpoint='best saved immediately after stage training, before heldout gate; next stage MUST reload this saved checkpoint',
        stop='plateau without reaching monitor target, training cap, heldout gate failure, or mastered-scale damage; no automatic continuation',
        train_conditions='8100..8107 cell/greedy, rotating two common-random contexts per round',
        monitor_conditions=MONITOR,heldout_conditions=HELDOUT,
        no_Sobol_supervision=True,no_search_labels_or_objective=True)
    (OUT/'curriculum_config.json').write_text(json.dumps(config,indent=2)+'\n')
    model=ProgressiveGenerator().requires_grad_(False)
    model.load_state_dict(torch.load(INITIAL,weights_only=False)['state_dict'])
    parent=INITIAL; rng=np.random.default_rng(config['optimizer_seed'])
    monitor_reference=np.random.default_rng(27112027).random((32768,2))
    t=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    all_history=[]; summary_rows=[]; details=[]; stage_summaries=[]
    for stage,k in enumerate(STAGES,1):
        stage_start=time.monotonic(); active=STAGES[:stage]
        # Reload exact persisted parent. Never randomly reinitialize stages.
        model.load_state_dict(torch.load(parent,weights_only=False)['state_dict'])
        entry_state=freeze(model.state_dict()); best_state=freeze(model.state_dict())
        entry_monitor=sample(model,MONITOR,active,monitor_reference)
        entry_heldout=heldout(model,f'{stage}_entry',active,reference,details)
        best_monitor=entry_monitor; best_score=primary(best_monitor,entry_monitor,active)
        best_round=0; stale=0; train_cache={}; history=[]
        print(json.dumps(dict(event='stage_start',stage=stage,active=active,parent_sha256=sha(parent),entry_monitor=entry_monitor)),flush=True)
        stop_reason='training cap reached without monitor target'
        for rnd in range(1,config['max_rounds_per_stage']+1):
            pair=[(8100+(rnd-1)%8,'cell'),(8100+(rnd+2)%8,'greedy')]
            key=tuple(pair)
            if key not in train_cache:
                model.load_state_dict(entry_state); train_cache[key]=sample(model,pair,active)
            train_base=train_cache[key]
            model.load_state_dict(best_state); centre=sample(model,pair,active)
            winner_state=best_state; winner_train=primary(centre,train_base,active)
            sigma=config['mutation_sigmas'][(rnd-1)%len(config['mutation_sigmas'])]
            for direction in range(4):
                noise={name:torch.from_numpy(rng.normal(size=tuple(v.shape))).to(dtype=v.dtype) for name,v in best_state.items()}
                for sign in (-1,1):
                    candidate={name:v+sign*sigma*noise[name] for name,v in best_state.items()}
                    model.load_state_dict(candidate); measure=sample(model,pair,active)
                    loss=primary(measure,train_base,active)
                    if loss < winner_train and retains(measure,train_base,active):
                        winner_state=candidate; winner_train=loss
            improved=False; monitor_score=best_score; feasible=None
            if winner_state is not best_state:
                model.load_state_dict(winner_state)
                measured=sample(model,MONITOR,active,monitor_reference)
                monitor_score=primary(measured,entry_monitor,active)
                feasible=retains(measured,entry_monitor,active) and safeguards(measured,entry_monitor)
                if feasible and monitor_score < best_score*(1-1e-4):
                    best_state=freeze(winner_state); best_monitor=measured; best_score=monitor_score
                    best_round=rnd; improved=True
            stale=0 if improved else stale+1
            improvement=1-best_monitor['scales'][k]['count_fano_mean']/entry_monitor['scales'][k]['count_fano_mean']
            row=dict(stage=stage,current_scale=k,round=rnd,sigma=sigma,training_score=winner_train,
                trial_monitor_score=monitor_score,best_monitor_score=best_score,monitor_improvement=improvement,
                accepted=improved,feasible=feasible,stale_rounds=stale,elapsed_seconds=time.monotonic()-stage_start)
            history.append(row);all_history.append(row)
            print(json.dumps(dict(event='round',**row)),flush=True)
            if improvement>=config['monitor_target_improvement']:
                stop_reason='monitor target reached';break
            if stale>=config['plateau_rounds']:
                stop_reason='realized-discrepancy plateau';break
        model.load_state_dict(best_state)
        # Save BEFORE baseline comparison and held-out evaluation.
        cp,digest=save_stage(model,stage,active,parent,best_round,history,config)
        print(json.dumps(dict(event='stage_saved',stage=stage,checkpoint=str(cp),sha256=digest,stop_reason=stop_reason)),flush=True)
        result=heldout(model,stage,active,reference,details)
        new_gain=1-result['scales'][k]['count_fano_mean']/entry_heldout['scales'][k]['count_fano_mean']
        gates=dict(training_target=stop_reason=='monitor target reached',meaningful_new_scale=new_gain>=.15,
            previous_scales=retains(result,entry_heldout,active),
            current_anchored_discrepancy=result['scales'][k]['anchored_discrepancy_worst']<=1.10*entry_heldout['scales'][k]['anchored_discrepancy_worst']+1e-15,
            geometry=safeguards(result,entry_heldout),freshness=result['near_overlap_max']<=min(.03,entry_heldout['near_overlap_max']+.005),
            boundary=result['boundary_cover']==entry_heldout['boundary_cover'])
        passed=all(gates.values())
        for scale in active:
            baselines=baseline(scale)
            summary_rows.append(dict(stage=stage,N=N,current_scale=k,reported_scale=scale,passed=passed,
                stop_reason=stop_reason,scale_role='new' if scale==k else 'retained',
                entry_count_fano=entry_heldout['scales'][scale]['count_fano_mean'],**result['scales'][scale],
                discrepancy_improvement_fraction=1-result['scales'][scale]['count_fano_mean']/entry_heldout['scales'][scale]['count_fano_mean'],
                **{f'{name}_{metric}':value for name,values in baselines.items() for metric,value in values.items()},
                complete_cover=result['cover'],interior_cover=result['interior_cover'],boundary_cover=result['boundary_cover'],
                entry_complete_cover=entry_heldout['cover'],large_holes=result['holes'],entry_large_holes=entry_heldout['holes'],
                nn_CV=result['nn_CV_mean'],exact_overlap_max=result['exact_overlap_max'],near_overlap_005_max=result['near_overlap_max'],
                near_overlap_010_max=result['near_overlap_010_max'],cell_greedy_near_overlap=result['cell_greedy_near_overlap'],
                best_round=best_round,rounds_attempted=len(history),checkpoint=str(cp),sha256=digest,
                gates=json.dumps(gates),seconds=time.monotonic()-stage_start))
        stage_summaries.append(dict(stage=stage,active_scales=active,passed=passed,stop_reason=stop_reason,gates=gates,
            checkpoint=str(cp),sha256=digest,parent_sha256=sha(parent),best_round=best_round,
            monitor_improvement=1-best_monitor['scales'][k]['count_fano_mean']/entry_monitor['scales'][k]['count_fano_mean'],
            heldout_improvement=new_gain,heldout=result))
        write_csv(OUT/'scale_curriculum_stages.csv',summary_rows)
        write_csv(OUT/'scale_curriculum_training_history.csv',all_history)
        write_csv(OUT/'scale_curriculum_set_details.csv',details)
        (OUT/f'stage_{stage}_gate.json').write_text(json.dumps(stage_summaries[-1],indent=2)+'\n')
        print(json.dumps(dict(event='stage_gate',stage=stage,passed=passed,gates=gates,heldout_improvement=new_gain)),flush=True)
        if not passed:
            for future_stage,future_scale in enumerate(STAGES[stage:],stage+1):
                summary_rows.append(dict(stage=future_stage,N=N,current_scale=future_scale,
                    status='NOT_RUN',stop_reason=f'blocked by Stage {stage} gate failure'))
            write_csv(OUT/'scale_curriculum_stages.csv',summary_rows)
            break
        # New scale introduced ONLY after successful gate and checkpoint reload.
        assert sha(cp)==digest
        parent=cp
    unchanged=all(sha(p)==h for p,h in protected.items())
    assert unchanged,'protected artifact modified'
    promoted=len(stage_summaries)==5 and all(s['passed'] for s in stage_summaries)
    if promoted:
        import shutil
        dest=OUT/'SCALE_CURRICULUM_SPACEFILL_CANDIDATE.pt'
        shutil.copyfile(parent,dest); dest.with_suffix('.sha256').write_text(sha(dest)+'  '+dest.name+'\n')
    final=dict(status='SCALE_CURRICULUM_SPACEFILL_CANDIDATE' if promoted else 'STOPPED_DIAGNOSTIC_ONLY',
        stages=stage_summaries,elapsed_seconds=time.monotonic()-start,protected_hashes_unchanged=unchanged,
        protected_hashes=protected,N=N,source_sha256=sha('experiments/sho/progressive_spacefill_generator.py'),
        no_search=True,no_Sobol_supervision=True,no_objective_or_search_labels=True)
    (OUT/'scale_curriculum_results.json').write_text(json.dumps(final,indent=2)+'\n')
    lines=['PROGRESSIVE SCALE CURRICULUM — '+final['status'],f'N=8600 only. Total run: {final["elapsed_seconds"]:.1f} seconds.',
        'Existing architecture, block generation, boundary support, and minimum separation unchanged. Warm start from existing progressive checkpoint; not a fresh-from-random curriculum comparison.',
        'Direct mirrored weight-mutation optimization of ACTUAL final-set count Fano across four shifted grids. No expected-occupancy objective or soft-selection gradient.',
        'Stage1 loss8 only. Successive stages add16,32,64,128 while retaining previous losses. Earlier input features remain unchanged.',
        'Eight weight perturbations per round; common deterministic iteration/pool seeds; best accepted by separate monitor and coverage/freshness constraints.',
        'Predeclared monitor target20% reduction; heldout gate15%; retained scale count/anchored metrics may regress at most10%. Coverage may regress at most5%, hole fraction+.0005, near overlap <=3% and increase <=.5 percentage point.',
        'All training/selection metrics come from generated geometry. Baseline comparison uses cached measurements at matching scale and heldout conditions, only after stage checkpoint persistence.',
        'Primary discrepancy is normalized occupancy squared error (Fano); raw count RMSE and sampled anchored-CDF discrepancy also reported, without conflating them.',
        'Heldout reference is524288 independent audit points. Near overlap thresholds .05/sqrt(N) and .10/sqrt(N); large holes >1/sqrt(N). Boundaries excluded from freshness.',
        'Each completed stage checkpoint saved and hashed before heldout evaluation; no next stage without passing gate and loading the saved predecessor.',
        'No N1000/N32768 training, Sobol coordinates, SHO loss/parameters, downstream labels, or closed-loop search. Existing artifacts verified unchanged.','']
    for s in stage_summaries:
        lines += [f'Stage{s["stage"]}, scales{s["active_scales"]}: {"PASS" if s["passed"] else "STOP"}. {s["stop_reason"]}.',
                  f'Monitor improvement {s["monitor_improvement"]:.2%}; heldout improvement {s["heldout_improvement"]:.2%}; best round{s["best_round"]}.',
                  'Gates: '+json.dumps(s['gates']),f'Checkpoint: {s["checkpoint"]}',f'SHA256: {s["sha256"]}','']
    lines += ['Per-stage and matching Sobol/fresh-v1/previous-progressive metrics: scale_curriculum_stages.csv. Full generated-set diagnostics: scale_curriculum_set_details.csv.']
    (OUT/'scale_curriculum_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in final.items() if k not in ('stages','protected_hashes')}),flush=True)


if __name__=='__main__':run()
