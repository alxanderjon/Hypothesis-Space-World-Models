"""Short coverage/hole refinement of the saved 65-parameter placement only."""
import copy
import csv
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.sho import conditioned_leaf_placement_screen as base

OUT = base.ROOT/'conditioned_leaf_placement_refinement'
OLD = base.OUT
CHECKPOINT = OLD/'placement_h8_diagnostic.pt'


def dump(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def write(name, rows):
    if not rows:
        return
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def references(c, i):
    rng = np.random.default_rng(base.SEED+10000+i)
    refs = [rng.random((4096, c['d']))]
    for axis in range(c['d']):
        for side in (0., 1.):
            x = rng.random((256, c['d'])); x[:, axis] = side; refs.append(x)
    return np.concatenate(refs)


def cases_load():
    result = []
    for r in json.loads((OLD/'case_splits.json').read_text()):
        path = (OLD if r['split']=='fresh' else base.FROZEN)/(r['name']+'_leaves.json' if r['split']=='fresh' else r['name']+'.json')
        result.append(base.case_from_leaves(r['name'], json.loads(path.read_text()), r['split']))
    return result


def checkpoint(model, step, metrics):
    path = OUT/'best_refined_diagnostic.pt'
    torch.save(dict(state_dict=model.state_dict(), hidden=8, dtype='float64', parameters=65,
                    resumed_from=str(CHECKPOINT), resumed_sha256=base.sha(CHECKPOINT),
                    additional_updates=step, status='DIAGNOSTIC_PENDING_GATES', metrics=metrics), path)
    digest = base.sha(path)
    path.with_suffix('.sha256').write_text(digest+'\n')
    return str(path), digest


def key(row):
    return row['case'], int(row['iteration']), int(row['pool'])


def assessment(rows, before):
    rows = [r for r in rows if r['split']!='sobol_reference']
    violation = sum(max(0., r['cover_ratio']-1.10)+max(0., r['added_hole_fraction']-.001)
                    +max(0.,r['discrepancy_ratio']-1.05)+max(0.,r['cross_condition_near_fraction']-.05) for r in rows)
    regressions = sum(before[key(r)]['passed'] and not r['passed'] for r in rows)
    # Track per-property regressions, including cases previously failing another property.
    property_regressions = sum(len(set(r['failed_gates'].split(','))-{''}-set(before[key(r)]['failed_gates'].split(','))) for r in rows)
    return dict(conditions=len(rows), failures=sum(not r['passed'] for r in rows),
                previously_passing_regressions=regressions, per_property_regressions=property_regressions,
                violation=float(violation), worst_cover=max(r['cover_ratio'] for r in rows),
                max_added_holes=max(r['added_hole_fraction'] for r in rows))


def new_condition_evaluation(model, cases):
    # Same metrics/gates, genuinely new random references/history/context; no source file edits.
    source = inspect.getsource(base.evaluate)
    source = source.replace('SEED+10000+ci', 'SEED+910000+ci').replace('SEED+20000+ci', 'SEED+920000+ci')
    old = "t = 32 if c['split'] == 'heldout' else 128 if c['split'] == 'fresh' else 4"
    assert source.count(old)==1
    source = source.replace(old, 't = 512')
    scope = dict(base.__dict__)
    exec(compile(source, 'independent_placement_condition_evaluation', 'exec'), scope)
    return scope['evaluate'](model, cases, 8)


def integration(model, c, saved_leaves):
    """One cached frozen-partition -> placement -> unchanged nearest/cell check."""
    from experiments.sho import validate_stage1b_learned_cell_size as cells
    from experiments.sho.fast_nearest_argmin import FastNearestArgmin
    nn = torch.nn
    distance, _ = cells.base._load_locked_distance()
    meta = torch.load(cells.nearest.OPERATOR, weights_only=False)
    margin = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    margin.load_state_dict(torch.load(meta['margin_checkpoint'], weights_only=False)['state_dict'])
    threshold = cells.nearest.threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(meta['threshold_checkpoint'], weights_only=False)['state_dict'])
    half = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    half.load_state_dict(torch.load(cells.OUT/'half_spacing_operator_locked.pt', weights_only=False)['state_dict'])
    for x in (distance, margin, threshold, half):
        x.eval().requires_grad_(False)
    fast = FastNearestArgmin(distance, margin, threshold, half)
    history = base.history_for(c, 512, 'evaluation')
    with torch.no_grad():
        probes = base.placed(model, c, 512, 0, history).numpy()
    pool = np.unique(np.concatenate((probes, base.edges(c['n'], c['d']))), axis=0)
    start = time.monotonic(); out = fast.cells(history, pool); elapsed = time.monotonic()-start
    # Independent pre-existing scan/max assembly, same inputs and checkpoints.
    captured, _ = cells.capture_nearest(history, pool, distance, margin, threshold, 0)
    owners = captured['winner'][:len(pool)]; dist = captured['winner_distance'][:len(pool)]
    spacing = captured['winner_distance'][len(pool):]
    radius, witness, _ = cells.assigned_radius(owners, dist, len(history), margin, threshold)
    half_ref = cells.predict_half(half, spacing)
    replace, _, _, _ = cells.larger_candidate(margin, threshold, radius, half_ref)
    size = np.where(replace, half_ref, radius)
    comparisons = [('probe_owner', owners), ('probe_distance', dist), ('spacing', spacing),
                   ('radius', radius), ('radius_witness', witness), ('half', half_ref), ('sizes', size)]
    mismatches = {name:int(np.sum(out[name]!=v)) for name,v in comparisons}
    valid = bool(np.isfinite(out['sizes']).all() and (out['sizes']>0).all()
                 and (out['probe_owner']>=0).all() and (out['probe_owner']<len(history)).all())
    np.savez_compressed(OUT/'integration_arrays.npz', probes=pool, sites=history,
                        **{name:out[name] for name,_ in comparisons})
    return dict(passed=valid and not any(mismatches.values()), sites=len(history), interior_probes=len(probes),
                total_probes=len(pool), mismatches=mismatches, valid_outputs=valid, runtime_seconds=elapsed,
                partition_leaves=saved_leaves, scope='One small d2 frozen partition; no active-size N1 pool or A/B validation')


def main():
    started = time.monotonic(); torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    protect = json.loads((OLD/'protected_hashes.json').read_text())
    protect.update({str(p):base.sha(p) for p in OLD.iterdir() if p.is_file()})
    protect[str(Path(base.__file__))] = base.sha(base.__file__)
    assert all(base.sha(p)==h for p,h in protect.items())
    dump('protected_hashes.json', protect)
    dump('plan.json', dict(start=str(CHECKPOINT),start_sha256=base.sha(CHECKPOINT),architecture_unchanged=True,
                           parameters=65,dtype='float64',maximum_updates=400,learning_rate=.002,
                           gates=base.GATES, optimizer_state='Not in diagnostic checkpoint; new Adam, restored model weights',
                           training='All39 former train/heldout/fresh conditions are now refinement/replay, not independent holdouts',
                           targeting='Four saved failing conditions get4x constraint loss; strong replay on all35 passing conditions',
                           objective='Full original reference sets: max coverage hinge and largest retained distances under exact allowed hole-count budget; coordinate distillation; separation/history/condition retention',
                           monitor='Original unchanged rolling-history evaluation every25 steps; first all-pass stops refinement',
                           independent='New conditions512/513 and references on existing geometries plus new frozen d2/3/4 N12 geometry; evaluated once after checkpoint selection',
                           scope='No A/B, objective calls, N2/N3, locked changes. One downstream check only if all placement gates pass.'))
    cases = cases_load()
    cp = torch.load(CHECKPOINT, weights_only=False)
    model = base.Placement(8); model.load_state_dict(cp['state_dict'])
    assert sum(p.numel() for p in model.parameters())==65
    oldrows, _, _ = base.evaluate(model, cases, 8)
    before = {key(r):r for r in oldrows if r['split']!='sobol_reference'}
    write('before_metrics.csv', oldrows)
    archive = np.load(OLD/'screen_coordinates.npz')
    replay = []
    for ci,c in enumerate(cases):
        ref = references(c,ci); center = (c['lo']+c['hi'])/2; h=c['n']**(-1/c['d'])
        baseline = cKDTree(center).query(ref)[0]
        allowed = int(np.sum(baseline>h))+math.floor(.001*len(ref))
        t = 32 if c['split']=='heldout' else 128 if c['split']=='fresh' else 4
        for iteration,pool in ((t,0),(t+1,0),(t,1)):
            hist = archive[f'8_{ci}_{iteration}_{pool}_history']
            original = archive[f'8_{ci}_{iteration}_{pool}_points']
            with torch.no_grad():
                assert np.array_equal(base.placed(model,c,iteration,pool,hist).numpy(),original)
            replay.append(dict(c=c,ref=torch.from_numpy(ref),h=h,cover=float(baseline.max()),
                               keep=len(ref)-allowed,history=torch.from_numpy(hist),
                               x=base.features(c,iteration,pool,hist),old=torch.from_numpy(original),
                               first=torch.from_numpy(archive[f'8_{ci}_{t}_0_points']),
                               weight=4. if not before[(c['name'],iteration,pool)]['passed'] else 1.,
                               cross=not(iteration==t and pool==0)))
    # Reserve new geometries; frozen execution only, no geometry/model modification.
    selector = base.load_partition_runtime(); rng=np.random.default_rng(2026092908)
    fresh=[]
    for d in (2,3,4):
        root=dict(d=d,n=12,lower=[0.]*d,upper=np.exp(rng.uniform(-.4,.4,d)).tolist())
        name=f'new_refinement_holdout_d{d}_n12'
        leaves=base.partition(selector,root,name); dump(name+'_leaves.json',leaves)
        fresh.append(base.case_from_leaves(name,leaves,'fresh'))
        print(json.dumps(dict(event='reserved_fresh_geometry',name=name)),flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    best=None; beststate=None; beststep=0; history=[]
    for step in range(401):
        if step:
            opt.zero_grad();total=0.
            for r in replay:
                c=r['c'];p=model(r['x'],torch.from_numpy(c['lo']),torch.from_numpy(c['hi']))
                distance=torch.cdist(r['ref'],p).min(1).values
                cov=torch.relu(distance.max()/r['cover']-1.085).square()
                retained=torch.sort(distance).values[:r['keep']]
                # Move the worst permitted non-hole below h with a small margin;
                # do not average away a handful of large-hole violations.
                holes=torch.relu(retained[-min(64,len(retained)):]/r['h']-.998).square().mean()
                stay=((p-r['old'])/r['h']).square().mean()
                hd=torch.cdist(p,r['history']).min(1).values
                valid=torch.relu(.001-hd/r['h']).square().mean()
                freshloss=0.
                if r['cross']:
                    cross=torch.cdist(p,r['first']).min(1).values
                    freshloss=torch.relu(.012-cross/r['h']).square().mean()
                loss=r['weight']*(2500*cov+2500*holes)+2*stay+100*(valid+freshloss)
                (loss/len(replay)).backward();total+=float(loss.detach())/len(replay)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
        else: total=0.
        if step%25==0:
            rows, _, _=base.evaluate(model,cases,8)
            m=assessment(rows,before);rank=(m['failures'],m['per_property_regressions'],m['violation'])
            if best is None or rank<best:
                best=rank;beststate=copy.deepcopy(model.state_dict());beststep=step
                checkpoint(model,step,m)
            history.append(dict(update=step,objective=total,**m));write('refinement_history.csv',history)
            print(json.dumps(dict(event='refinement_monitor',**history[-1])),flush=True)
            if step and m['failures']==0 and m['per_property_regressions']==0:
                break
    model.load_state_dict(beststate);model.eval().requires_grad_(False)
    rows,_,coords=base.evaluate(model,cases,8);m=assessment(rows,before)
    write('after_replay_metrics.csv',rows);np.savez_compressed(OUT/'after_replay_coordinates.npz',**coords)
    result=dict(status='STOPPED_REPLAY_GATE_FAILURE',updates_run=step,best_update=beststep,
                replay=m,architecture_parameters=65,architecture_changed=False,source_checkpoint=str(CHECKPOINT),
                placement_promoted=False,integration_run=False,integration_promoted=False,n1_ab_run=False)
    if m['failures']==0 and m['per_property_regressions']==0:
        # New references/conditions on seen geometries and independent new geometries.
        vr,vfail,vc=new_condition_evaluation(model,cases+fresh)
        write('independent_metrics.csv',vr);np.savez_compressed(OUT/'independent_coordinates.npz',**vc)
        result['independent']=dict(conditions=sum(r['split']!='sobol_reference' for r in vr),failures=len(vfail),
                                   new_geometry_conditions=sum(r['case'].startswith('new_refinement_') and r['split']!='sobol_reference' for r in vr))
        if not vfail:
            cp_path=OUT/'leaf_to_probe_placement.pt'
            torch.save(dict(state_dict=model.state_dict(),hidden=8,parameters=65,dtype='float64',
                            source_checkpoint=str(CHECKPOINT),additional_updates=beststep,status='LOCKED_CANDIDATE'),cp_path)
            manifest=dict(operator='leaf_to_probe_placement',status='LOCKED_CANDIDATE',checkpoint=str(cp_path),sha256=base.sha(cp_path),
                          architecture='Unchanged shared6->8 tanh->1 sigmoid',parameters=65,gates=base.GATES,
                          validation=dict(replay=m,independent=result['independent']),
                          scope='Small N8/12/16 leaf pools; not active N1 budget or end-to-end validation')
            dump('leaf_to_probe_placement_locked_candidate.json',manifest)
            result.update(placement_promoted=True,status='PLACEMENT_PASSED')
            integ=integration(model,fresh[0],str(OUT/(fresh[0]['name']+'_leaves.json')))
            result.update(integration_run=True,integration=integ)
            if integ['passed']:
                dump('integrated_learned_spacefill_pool_validated_candidate.json',dict(operator='integrated_learned_spacefill_pool',
                     status='VALIDATED_CANDIDATE',placement=manifest,validation=integ,
                     boundary='Unchanged existing 2D support',n1_ab_run=False))
                result.update(integration_promoted=True,status='VALIDATED_CANDIDATE')
            else:result['status']='STOPPED_INTEGRATION_GATE_FAILURE'
        else:result['status']='STOPPED_INDEPENDENT_GATE_FAILURE'
    assert all(base.sha(p)==h for p,h in protect.items())
    result.update(protected_files=len(protect),protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,
                  checkpoint=str(OUT/'best_refined_diagnostic.pt'),checkpoint_sha256=base.sha(OUT/'best_refined_diagnostic.pt'))
    dump('results.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
