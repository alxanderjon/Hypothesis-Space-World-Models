"""Deterministic fixed-quota CDF exchange screen. No fitting or input changes.

Uses the historical cumulative-count fields at the historical four evaluation
shifts. A sweep reassigns each saved selected slot within its incidence class.
The exact mean-square CDF change is calculated with power-of-two integer
scaling. Evaluation still uses the unchanged historical float64 metric.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.sho.audit_online_quota_reservoir import A, META, N, capacity, ids
from experiments.sho.progressive_spacefill_generator import SCALES, boundary_faces, discrepancy_rows
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics, overlap
from experiments.sho.validate_joint_quota_assignment_teacher import bounds, metrics as count_metrics
from experiments.sho.train_dyadic_spacefill_curriculum import health

ROOT = Path('outputs/sho/stage1B')
SOURCE = ROOT / 'complete_deterministic_probe_generator'
OUT = ROOT / 'shift_aware_quota_realizer'
SHIFTS = (0., .25, .5, .75)
MAX_SWEEPS = 3  # Predeclared short screen; never a relaxed acceptance condition.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def cumulative(array, reverse=False):
    result = array
    for axis in range(array.ndim):
        result = np.flip(np.flip(result, axis).cumsum(axis=axis), axis) if reverse else result.cumsum(axis=axis)
    return result


class CDFState:
    """Generic dimension and class IDs; no axis-specific routing decisions."""
    def __init__(self, candidates, selected, scales=SCALES, shifts=SHIFTS):
        self.points = np.asarray(candidates, dtype=np.float64)
        self.n, self.dimension = len(selected), self.points.shape[1]
        self.scales, self.shifts = tuple(scales), tuple(shifts)
        self.denominator = int(max(scales) ** (2 * self.dimension))
        # Bound every intermediate and the sum of all grid score terms using
        # Python integers before entering the exact int64 implementation.
        bound=sum((4*int(self.n)*int(k)**(2*self.dimension)+2*int(k)**(2*self.dimension))
                  *(self.denominator//int(k)**(2*self.dimension)) for k in scales for _ in shifts)
        if bound > np.iinfo(np.int64).max:
            raise OverflowError('Exact CDF-score int64 bound exceeded; no silent overflow')
        self.fields = []
        for k in scales:
            cells = int(k ** self.dimension)
            shape = (k,) * self.dimension
            anchor_volume_numerator = np.ones(shape, dtype=np.int64)
            for axis in range(self.dimension):
                reshape = [1] * self.dimension
                reshape[axis] = k
                anchor_volume_numerator *= np.arange(1, k + 1).reshape(reshape)
            for shift in shifts:
                # Preserve the historical periodic shift and bin ownership.
                index = np.minimum(((self.points + shift/k) % 1 * k).astype(int), k-1)
                flat = np.ravel_multi_index(index.T, shape)
                counts = np.bincount(flat[selected], minlength=cells).reshape(shape)
                error = cells * cumulative(counts) - self.n * anchor_volume_numerator
                self.fields.append(dict(k=k, cells=cells, index=index, flat=flat,
                                        error=error, factor=self.denominator//(cells*cells),
                                        volume_numerator=anchor_volume_numerator))

    def effects(self, old, alternatives):
        """Exact scaled change in sum of historical per-grid CDF mean squares."""
        total = np.zeros(len(alternatives), dtype=np.int64)
        for f in self.fields:
            k, cells = f['k'], f['cells']
            old_index = f['index'][old]
            new_index = f['index'][alternatives]
            suffix = cumulative(f['error'], reverse=True)
            linear = 2 * (suffix[tuple(new_index.T)] - suffix[tuple(old_index)])
            effect_squared = (np.prod(k-new_index, axis=1) + np.prod(k-old_index)
                              - 2*np.prod(k-np.maximum(new_index, old_index), axis=1))
            total += (linear + cells * effect_squared) * f['factor']
        return total

    def swap(self, old, new):
        if old == new:
            return
        for f in self.fields:
            f['error'][tuple(slice(int(j), None) for j in f['index'][old])] -= f['cells']
            f['error'][tuple(slice(int(j), None) for j in f['index'][new])] += f['cells']

    def fingerprint(self):
        digest = hashlib.sha256()
        for f in self.fields:
            digest.update(f['error'].tobytes())
        return digest.hexdigest()

    def loss(self):
        return sum(float(np.mean((f['error'].astype(float)/f['cells'])**2)) for f in self.fields)


def realize(candidates, classes, selected, trace_path, max_sweeps=MAX_SWEEPS, expected=None):
    """Fixed slot order, best one-exchange CDF descent; retain incumbent on ties."""
    start = time.monotonic()
    selected = np.asarray(selected, dtype=np.int64).copy()
    used = np.zeros(len(candidates), bool)
    used[selected] = True
    groups = {int(c): np.flatnonzero(classes == c) for c in np.unique(classes)}
    state = CDFState(candidates, selected)
    counts = np.bincount(classes[selected], minlength=int(classes.max())+1)
    events, sweep_rows = 0, []
    replay_stream = gzip.open(expected, 'rt') if expected else None
    if replay_stream:
        first = json.loads(next(replay_stream))
        assert first['selected'] == selected.tolist() and first['deficit_hash'] == state.fingerprint()
    with gzip.open(trace_path, 'wt') as trace:
        trace.write(json.dumps(dict(type='initial', selected=selected.tolist(), quota=counts.tolist(),
            deficit_hash=state.fingerprint(), scales=list(SCALES), shifts=list(SHIFTS),
            candidates_ref='input_reservoirs.npz', classes_ref='state_arrays.npz',
            state='E=k**d*C-N*product(anchor_integer_indices)',
            candidate_effect='same-class remove/add orthants; reconstruct from stored candidate bin indices',
            score_denominator=state.denominator,
            tie='KEEP incumbent unless strict improvement; then first ascending reservoir ID among minimum scores',
            max_sweeps=max_sweeps))+'\n')
        for sweep in range(max_sweeps):
            before, swaps = state.loss(), 0
            for slot in range(len(selected)):
                old = int(selected[slot])
                cls = int(classes[old])
                group = groups[cls]
                alternatives = group[(~used[group]) | (group == old)]
                costs = state.effects(old, alternatives)
                assert int(costs[np.flatnonzero(alternatives == old)[0]]) == 0
                best = int(np.argmin(costs))
                new = int(alternatives[best]) if costs[best] < 0 else old
                event = dict(input_state_ref=events, sweep=sweep, slot=slot, quota_class=cls,
                    removed_candidate=old, compatible_candidates=alternatives.tolist(),
                    candidate_effect_scores=costs.tolist(), selected_candidate=new,
                    deficit_effect='-orthant(old)+orthant(new)',
                    output_state_ref=events+1)
                if replay_stream:
                    assert json.loads(next(replay_stream)) == event, 'decision/deficit-effect replay mismatch'
                trace.write(json.dumps(event, separators=(',', ':'))+'\n')
                state.swap(old, new)
                used[old] = False
                used[new] = True
                selected[slot] = new
                swaps += int(old != new)
                events += 1
            after = state.loss()
            assert after <= before
            assert np.array_equal(np.bincount(classes[selected], minlength=len(counts)), counts)
            fresh_state = CDFState(candidates, selected)
            assert fresh_state.fingerprint() == state.fingerprint(), 'incremental CDF lost precision'
            row = dict(sweep=sweep+1, swaps=swaps, before=before, after=after,
                       deficit_hash=state.fingerprint(), seconds=time.monotonic()-start)
            sweep_rows.append(row)
            print(json.dumps(dict(event='sweep', trace=Path(trace_path).name, **row)), flush=True)
            if not swaps:
                break
    if replay_stream:
        assert replay_stream.read() == ''
        replay_stream.close()
    return selected, dict(events=events, sweeps=sweep_rows, converged=sweep_rows[-1]['swaps']==0,
                          runtime_seconds=time.monotonic()-start)


def formula_checks():
    """Independent brute CDF recomputation in d=1,2,3; integer score equality."""
    checks = []
    for d in (1, 2, 3):
        rng = np.random.default_rng(292610+d)
        p = rng.random((43, d)); chosen = np.arange(17)
        state = CDFState(p, chosen, scales=(2,4,8))
        old = 5; alternatives = np.r_[old, np.arange(17, 43)]
        values = state.effects(old, alternatives)
        baseline = state.loss()
        errors = []
        for j, score in zip(alternatives, values):
            trial = chosen.copy(); trial[old] = j
            rebuilt = CDFState(p, trial, scales=(2,4,8))
            errors.append(abs((rebuilt.loss()-baseline)*state.denominator-int(score)))
        assert max(errors) == 0
        checks.append(dict(dimension=d, comparisons=len(values), scaled_score_errors=0))
    return checks


def evaluate(pools, previous, source_result):
    grid = (np.arange(512)+.5)/512
    reference = np.vstack((np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2),
                           np.random.default_rng(27093001).random((262144,2))))
    geometry, multi, count_rows = [], [], []
    for name, p in pools.items():
        complete = np.unique(np.concatenate((p,boundary_faces(N,p.shape[1]))), axis=0)
        assert np.array_equal(complete[np.any((complete==0)|(complete==1),axis=1)],boundary_faces(N,p.shape[1]))
        it = int(name.removeprefix('iteration').split('_')[0]); pool = name.rsplit('_',1)[-1]
        geometry.append(dict(condition=name, iteration=it, pool_type=pool, **metrics(complete,p,reference,N)))
        for row in discrepancy_rows(p):
            multi.append(dict(condition=name, cohort='archived' if it<10000 else 'fresh',**row))
        for family, value in count_metrics(A@capacity(p), META, N).items():
            count_rows.append(dict(condition=name, family=family, **value))
    cohorts = [('archived',[f'iteration{100+i}_cell' for i in range(4)]+['iteration100_greedy'],[(0,1),(1,2),(2,3),(0,4)]),
               ('fresh',['iteration12000_cell','iteration12001_cell','iteration12000_greedy'],[(0,1),(0,2)])]
    health_rows, freshness = [], []
    for cohort, names, pairs in cohorts:
        g = [r for r in geometry if r['condition'] in names]
        ov = [dict(cohort=cohort,left=names[i],right=names[j],**overlap(pools[names[i]],pools[names[j]],N)) for i,j in pairs]
        freshness.extend(ov)
        value = dict(cover=max(r['complete_cover'] for r in g),hole_fraction=max(r['large_hole_fraction'] for r in g),
            min_nn_CV=min(r['nn_CV'] for r in g),clustering=max(r['clustered_fraction'] for r in g),
            nn_p01=min(float(np.quantile(cKDTree(pools[name]).query(pools[name],k=2)[0][:,1],.01)) for name in names),
            exact_overlap=max(r['exact_overlap_fraction'] for r in ov),near_overlap=max(r['near_overlap_fraction_005_spacing'] for r in ov))
        entry = next(r['entry'] for r in source_result['cohort_health'] if r['cohort']==cohort)
        health_rows.append(dict(cohort=cohort, value=value, entry=entry, passed=bool(health(value,entry))))
    baseline = list(csv.DictReader((ROOT/'progressive_spacefill_smoke/progressive_spacefill_multiscale.csv').open()))
    baseline = [r for r in baseline if int(float(r['N'])) == N]
    scale_rows = []
    scales = {gen:{k:max(float(r['grid_anchored_discrepancy']) for r in baseline if r['generator']==gen and int(float(r['bins_per_axis']))==k) for k in SCALES} for gen in ('Sobol','old_spacefill','fresh_v1')}
    for cohort, names, _ in cohorts:
        scales[cohort] = {}
        for k in SCALES:
            rr = [r for r in multi if r['cohort']==cohort and r['bins_per_axis']==k]
            prior = [r for r in source_result['multiscale'] if r['bins_per_axis']==k and (r['iteration']<10000)==(cohort=='archived')]
            scales[cohort][k]=max(r['grid_anchored_discrepancy'] for r in rr)
            scale_rows.append(dict(cohort=cohort,scale=k,before_max=max(r['grid_anchored_discrepancy'] for r in prior),
                after_max=scales[cohort][k],before_rms=max(r['grid_anchored_RMSE'] for r in prior),
                after_rms=max(r['grid_anchored_RMSE'] for r in rr),count_fano=max(r['count_fano'] for r in rr),
                empty_fraction=max(r['empty_bin_fraction'] for r in rr)))
    ratio=lambda gen:max(scales[gen][k]/max(scales['Sobol'][k],1/N) for k in SCALES)
    gate = dict(discrepancy_materially_better=all(ratio('archived')<=.8*ratio(g) for g in ('old_spacefill','fresh_v1')),
        no_scale_regression=all(scales['archived'][k]<=1.1*scales['fresh_v1'][k] for k in SCALES))
    # Existing full progressive geometric gate on the original archived cohort.
    base_g=list(csv.DictReader((ROOT/'progressive_spacefill_smoke/progressive_spacefill_geometry.csv').open()))
    sb=[r for r in base_g if int(float(r['N']))==N and r['generator']=='Sobol']
    ar=[r for r in geometry if r['iteration']<10000]
    paired=[r['complete_cover']/float(next(s['complete_cover'] for s in sb if int(float(s['iteration']))==r['iteration'] and s['pool_type']==r['pool_type'])) for r in ar]
    union=np.unique(np.concatenate([np.concatenate((pools[f'iteration{100+i}_cell'],boundary_faces(N,2))) for i in range(4)]),axis=0)
    uc=float(cKDTree(union).query(reference)[0].max())
    bu=list(csv.DictReader((ROOT/'progressive_spacefill_smoke/progressive_spacefill_freshness_union.csv').open()))
    su=float(next(r['union_complete_cover'] for r in bu if int(float(r['N']))==N and r['generator']=='Sobol' and r['record_type']=='union'))
    gate.update(coverage=max(r['complete_cover'] for r in ar)<=1.1*max(float(r['complete_cover']) for r in sb) and max(paired)<=1.15,
        boundary=max(r['boundary_cover'] for r in ar)==max(float(r['boundary_cover']) for r in sb),
        freshness=max(r['exact_overlap_fraction'] for r in freshness if r['cohort']=='archived')==0 and max(r['near_overlap_fraction_005_spacing'] for r in freshness if r['cohort']=='archived')<=.03,
        union=uc<=1.25*su,holes=max(r['large_hole_count'] for r in ar)<=max(float(r['large_hole_count']) for r in sb)+.001*len(reference),
        not_rigid_or_clustered=float(np.mean([r['nn_CV'] for r in ar]))>=.08 and max(r['clustered_fraction'] for r in ar)<=.005)
    gate={k:bool(v) for k,v in gate.items()}
    for filename,rows in [('geometry.csv',geometry),('multiscale.csv',multi),('counts.csv',count_rows),('freshness.csv',freshness),('scale_comparison.csv',scale_rows)]:
        write_csv(OUT/filename,rows)
    return dict(historical_gate=gate,cohort_health=health_rows,scale_comparison=scale_rows,
                archived_one_count_ratio={g:ratio(g) for g in ('Sobol','old_spacefill','fresh_v1','archived')},
                fresh_comparison_note='Fresh before/after uses the same three saved fresh reservoirs; ratio baselines above are archived and are not relabeled fresh.',
                geometry=geometry,union_cover=uc,passed=all(gate.values()) and all(r['passed'] for r in health_rows))


def fresh_baseline_check():
    """Same historical discrepancy gate with matching saved-fresh contexts."""
    import torch
    from experiments.sho.fresh_spacefill_generator import FreshGenerator
    from experiments.sho.run_stage1b_spacefill_generator_smoke import Generator
    from experiments.sho.run_stage1b_edirect_vs_direct import _candidate_pool
    torch.set_num_threads(1)
    begin=time.monotonic()
    v1_info=json.loads((ROOT/'fresh_spacefill_smoke/training_summary.json').read_text())
    v1=FreshGenerator(2)
    assert sha(v1_info['checkpoint'])==v1_info['sha256']
    v1.load_state_dict(torch.load(v1_info['checkpoint'],weights_only=False)['state_dict'])
    old_path=ROOT/'spacefill_downstream_screen/spacefill_generator_candidate.pt'
    old=Generator(2);old.load_state_dict(torch.load(old_path,weights_only=False)['state_dict'])
    v1.eval().requires_grad_(False);old.eval().requires_grad_(False)
    realized=np.load(OUT/'realized_pools.npz');prior=np.load(SOURCE/'interior_pools.npz')
    rows=[]
    for it,pool in [(12000,'cell'),(12001,'cell'),(12000,'greedy')]:
        name=f'iteration{it}_{pool}'
        sb=_candidate_pool(86,200136,it+(1000000 if pool=='greedy' else 0))
        sb=sb[np.all((sb>0)&(sb<1),axis=1)]
        pp={'Sobol':sb,'fresh_v1':v1.generate(N,it,pool)[1],
            'old_spacefill':old.generate(N,[0.,0.],[1.,1.])[1],
            'composed_before':prior[name],'shift_aware':realized[name]}
        for gen,p in pp.items():
            assert len(p)==N
            for r in discrepancy_rows(p):rows.append(dict(condition=name,generator=gen,**r))
    scales={g:{k:max(r['grid_anchored_discrepancy'] for r in rows if r['generator']==g and r['bins_per_axis']==k) for k in SCALES} for g in pp}
    ratio=lambda g:max(scales[g][k]/max(scales['Sobol'][k],1/N) for k in SCALES)
    gate=dict(discrepancy_materially_better=all(ratio('shift_aware')<=.8*ratio(g) for g in ('old_spacefill','fresh_v1')),
              no_scale_regression=all(scales['shift_aware'][k]<=1.1*scales['fresh_v1'][k] for k in SCALES))
    write_csv(OUT/'fresh_baseline_multiscale.csv',rows)
    result=dict(gates=gate,passed=all(gate.values()),scale_maxima=scales,
                worst_one_count_ratios={g:ratio(g) for g in pp},
                realized_pool_sha256=sha(OUT/'realized_pools.npz'),
                checkpoint_sources={v1_info['checkpoint']:sha(v1_info['checkpoint']),str(old_path):sha(old_path)},
                runtime_seconds=time.monotonic()-begin)
    dump(OUT/'fresh_baseline_results.json',result)
    return result


def finalize():
    """Read-only trace replay, matched fresh baselines, and failure diagnosis."""
    begin=time.monotonic()
    cached=OUT/'fresh_baseline_results.json'
    fresh=json.loads(cached.read_text()) if cached.exists() else fresh_baseline_check()
    if fresh.get('realized_pool_sha256')!=sha(OUT/'realized_pools.npz'):
        fresh=fresh_baseline_check()
    result=json.loads((OUT/'results.json').read_text())
    inputs=np.load(ROOT/'geometry_preserving_quota_realizer/input_reservoirs.npz')
    initial=np.load(SOURCE/'interior_pools.npz')
    arrays=np.load(OUT/'state_arrays.npz');final=np.load(OUT/'realized_pools.npz')
    source=json.loads((SOURCE/'results.json').read_text())
    grid=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    traces=[];witnesses=[]
    for name in final.files:
        candidates=inputs[name];classes=arrays[name+'_classes'];selected=arrays[name+'_initial_selected'].copy()
        used=np.zeros(len(candidates),bool);used[selected]=True
        quota=arrays[name+'_quotas'];groups={int(c):np.flatnonzero(classes==c) for c in np.unique(classes)}
        removals={};steps=0;swaps=0
        with gzip.open(OUT/(name+'.jsonl.gz'),'rt') as stream:
            header=json.loads(next(stream));assert selected.tolist()==header['selected']
            for line in stream:
                e=json.loads(line);old=int(e['removed_candidate']);new=int(e['selected_candidate']);cl=int(e['quota_class'])
                assert e['input_state_ref']==steps and int(selected[e['slot']])==old
                group=groups[cl];eligible=group[(~used[group])|(group==old)]
                assert eligible.tolist()==e['compatible_candidates']
                effects=np.asarray(e['candidate_effect_scores'],dtype=np.int64)
                assert effects[np.flatnonzero(eligible==old)[0]]==0
                best=int(np.argmin(effects));expected=int(eligible[best]) if effects[best]<0 else old
                assert expected==new and classes[old]==classes[new]
                if old!=new:
                    removals[old]=dict(event=steps,replacement=new,scaled_cdf_change=int(effects[best]))
                used[old]=False;used[new]=True;selected[e['slot']]=new
                steps+=1;swaps+=int(old!=new)
        assert np.array_equal(candidates[selected],final[name])
        assert np.isfinite(final[name]).all() and np.all((final[name]>0)&(final[name]<1))
        assert np.array_equal(np.bincount(classes[selected],minlength=len(quota)),quota)
        state=CDFState(candidates,selected)
        pool_result=next(r for r in result['rows'] if r['condition']==name)
        assert state.fingerprint()==pool_result['sweeps'][-1]['deficit_hash']
        traces.append(dict(condition=name,events=steps,swaps=swaps,action_errors=0,quota_errors=0,final_deficit_exact=True))
        # Evaluation-only original reference witnesses, never used to select.
        support=boundary_faces(N,candidates.shape[1])
        after_complete=np.unique(np.concatenate((final[name],support)),axis=0)
        before_complete=np.unique(np.concatenate((initial[name],support)),axis=0)
        after_distance=cKDTree(after_complete).query(reference)[0]
        worst=np.argsort(-after_distance,kind='stable')[:20]
        before_distance=cKDTree(before_complete).query(reference[worst])[0]
        removed=np.setdiff1d(arrays[name+'_initial_selected'],selected)
        removed_distance,nearest_removed=cKDTree(candidates[removed]).query(reference[worst])
        for rank,(ref_id,bd,rd,rid) in enumerate(zip(worst,before_distance,removed_distance,nearest_removed)):
            old=int(removed[rid]);event=removals.get(old,{})
            witnesses.append(dict(condition=name,rank=rank+1,reference_id=int(ref_id),
                reference_coordinate=reference[ref_id].tolist(),coverage_before=float(bd),coverage_after=float(after_distance[ref_id]),
                removed_candidate_id=old,removed_coordinate=candidates[old].tolist(),distance_to_removed=float(rd),
                quota_class=int(classes[old]),**event))
    write_csv(OUT/'trace_validation.csv',traces);write_csv(OUT/'geometry_failure_witnesses.csv',witnesses)
    result['fresh_discrepancy']=fresh
    result['all_trace_validation']=traces
    result['all_trace_events']=sum(r['events'] for r in traces)
    result['all_swaps']=sum(r['swaps'] for r in traces)
    result['passed']=bool(result['passed'] and fresh['passed'])
    result['status']='VALIDATED_DETERMINISTIC_TEACHER' if result['passed'] else 'STOPPED_HISTORICAL_GATE_FAILURE'
    result['first_remaining_mechanism']='Geometry-aware admissibility of same-class CDF-improving exchanges; CDF descent alone does not retain coverage and holes.'
    for row in result['cohort_health']:
        v,e=row['value'],row['entry']
        row['checks']=dict(coverage=v['cover']<=1.05*e['cover'],holes=v['hole_fraction']<=e['hole_fraction']+.0005,
            nn_CV=v['min_nn_CV']>=.08,nn_p01=v['nn_p01']>=.95*e['nn_p01'],clustering=v['clustering']<=.005,
            exact_overlap=v['exact_overlap']==0,near_overlap=v['near_overlap']<=min(.03,e['near_overlap']+.005))
        assert row['passed']==all(row['checks'].values())
    result['postvalidation_seconds']=time.monotonic()-begin
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    assert all(sha(p)==h for p,h in protected.items())
    dump(OUT/'results.json',result)
    doc=[
        '# Shift-aware fixed-quota realization — deterministic teacher screen', '',
        '**'+result['status']+'**. No learned model was trained; no input pool, quota plan, checkpoint, proposal generator, or downstream operator was modified. No N1/N2/N3 run occurred.', '',
        '## Exact state and teacher rule', '',
        '`ProgressiveGenerator.make_context` uses scales 2,4,8,16,32,64,128 and shifts 0,0.5. The historical hard evaluation uses shifts 0,0.25,0.5,0.75. This deterministic teacher uses all four existing evaluation shifts, with their original periodic shift/bin ownership.', '',
        'For dimension d, grid size k, shift s, and anchor index a: bin(p)=min(int(((p+s/k)%1)*k),k-1); C(a) is the repeated cumulative histogram; V(a)=product((a_j+1)/k); R(a)=C(a)-N*V(a). The historical context field is the reverse cumulative sum of R divided by k**d and then sqrt(1+m). The original context also supplies cell occupancy and nearest-prefix-distance fields.', '',
        'The new rule starts with the saved geometry-valid full pool, not an empty prefix. Visit selected slots in saved order for three predeclared sweeps. Compare the incumbent and every currently unused candidate of the same fixed incidence class. Choose the candidate minimizing the exact change in the sum of the historical per-grid CDF mean squares. KEEP on no strict decrease; equal improving candidates choose smallest reservoir ID. Both the old and replacement points have identical incidence, so every quota and the total budget are unchanged at every exchange.', '',
        'This is a new deterministic full-set exchange adaptation of the historical `losses.cdf_change`, not a claim to reproduce the original soft proposal policy, occupancy loss, logsumexp aggregation, or geometric secondary gradient. No Sobol coordinates, coverage reference points, or evaluation losses enter selection.', '',
        'For replacement o→j, H_i(a)=1[bin_i<=a coordinatewise], D=H_j-H_o. The exact change per grid is mean(2*R*D+D**2). Using suffix sums F of R, this equals (2*(F(bin_j)-F(bin_o))+product(k-bin_j)+product(k-bin_o)-2*product(k-max(bin_j,bin_o)))/k**d.', '',
        'Implementation stores E=k**d*C-N*product(a+1), an exact integer residual. The common power-of-two denominator is max(k)**(2*d); integer-scaled scores remove cancellation from tie decisions. This is an exact representation of the same mean-square CDF quantities, not a new discrepancy metric. Independent brute recomputation gives 0 errors on 81 candidate changes across d=1,2,3. Every pool/sweep rebuilds the CDF from selected coordinates and checks the full residual hash. Higher-dimensional full-pool performance is not claimed.', '',
        '## Fixed inputs and scope', '',
        'Five archived and three previously saved fresh contexts, each N=8,600, use the complete-composition quota vectors in `complete_deterministic_probe_generator/quota_assignment_trace.json`. These are not substituted with an earlier quota plan. Reservoirs are the immutable full guarded buffers from `geometry_preserving_quota_realizer/input_reservoirs.npz`. Initial selected points are the saved complete-composition pools. No allocation or quota solve is rerun.', '',
        'There is no newly collected independent reservoir holdout in this run. The three fresh contexts were previously used for component validation. Matched Sobol and fresh-v1 evaluation baselines are generated read-only at those same fresh contexts.', '',
        '## Discrepancy by scale', '',
        '| Cohort | Scale | Before max | After max | Before RMS | After RMS | After Fano |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in result['scale_comparison']:
        doc.append(f"| {r['cohort']} | {r['scale']} | {r['before_max']:.9g} | {r['after_max']:.9g} | {r['before_rms']:.9g} | {r['after_rms']:.9g} | {r['count_fano']:.9g} |")
    doc+=['','Archived historical discrepancy gates: '+json.dumps({k:result['historical_gate'][k] for k in ('discrepancy_materially_better','no_scale_regression')})+'.',
          'Matched fresh historical discrepancy gates: '+json.dumps(fresh['gates'])+'.',
          'Archived worst one-count-floor ratios: '+json.dumps(result['archived_one_count_ratio'])+'.',
          'Matched fresh worst one-count-floor ratios: '+json.dumps(fresh['worst_one_count_ratios'])+'.','',
          '## Geometry, freshness, and invariants','',
          '| Cohort | Coverage | Hole fraction | NN-p01 | Min NN CV | Exact overlap | Near overlap | Historical health |',
          '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in result['cohort_health']:
        v=r['value'];doc.append(f"| {r['cohort']} | {v['cover']:.10g} | {v['hole_fraction']:.10g} | {v['nn_p01']:.10g} | {v['min_nn_CV']:.10g} | {v['exact_overlap']:.6g} | {v['near_overlap']:.6g} | {r['passed']} |")
    doc+=['','Original full progressive gate: '+json.dumps(result['historical_gate'])+'.',
          'Every pool has zero quota, budget, coarse-count, duplicate, nonfinite, and boundary-support errors. All selected points are a subset of the same mutually guarded reservoir, preserving its 0.4/sqrt(N) separation floor. This does not guarantee retention of the stronger historical NN-p01 safeguard.', '',
          f"Exact reexecution checked {result['replayed_decisions']:,} decisions and all alternative scores for the first pool. All {result['all_trace_events']:,} saved events additionally replay with zero action/membership/quota errors and exact final residual states. There were {result['all_swaps']:,} actual same-class exchanges.", '',
          '## Failure diagnosis and stop', '',
          'The scale-8+ discrepancy failure is removed on this screen, but CDF-only descent gives up the coverage and hole safeguards. The first missing mechanism is geometry-aware feasibility of quota-compatible CDF moves. Existing reservoir separation is insufficient to protect coverage/holes. The NN-p01 and NN-CV thresholds still pass, along with clustering and freshness. Evaluation-only worst-hole witnesses identify removed covering points and the CDF-improving exchanges that discarded them.', '',
          'This is evidence against the tested CDF-only rule, not proof that fixed quotas, reservoir capacity, or the joint CDF/geometry problem are infeasible. Three sweeps were a predeclared screen, not a convergence claim; some improving exchanges remain. No extension, retuning, or second selection rule was tried after validation failure. No promotion or downstream N1 replay.', '',
          '## Complexity, runtime, and supervision', '',
          'Let G=sum over scales/shifts of k**d, C be reservoir size, N selected slots, and J the largest class size. Each sweep costs O(N*(d*G+number_of_grids*d*J)); storage is O(G+number_of_grids*C*d), excluding traces. This is a dimension-agnostic rule with exponential grid storage in dimension; the actual N8600 screen is 2D. Initial state/coordinates plus sparse remove/add effects reconstruct all deficit states losslessly.', '',
          f"Main execution and first exact replay: {result['runtime_seconds']:.2f} s. Matched-fresh baselines, all-trace validation and failure diagnostics: {result['postvalidation_seconds']:.2f} s. Protected files unchanged: {result['protected_files']:,}.", '',
          'Trace schema: initial selected IDs, fixed quotas, residual hash, scales/shifts; each event stores current-state reference, removed candidate, quota class, all compatible IDs, exact deficit-change scores, selected candidate and next-state reference. Candidate coordinates and class arrays plus the orthant formula encode every full candidate effect, avoiding repeated dense CDF tensors. `trace_sources.json` resolves source paths/hashes. `geometry_failure_witnesses.csv` is evaluation-only and never supervision for this run.', '',
          '**The deterministic Sobol-like hypothesis space is not yet validated.**']
    (OUT/'report.md').write_text('\n'.join(doc)+'\n')
    dump(OUT/'trace_sources.json',dict(candidates=dict(path=str(ROOT/'geometry_preserving_quota_realizer/input_reservoirs.npz'),sha256=sha(ROOT/'geometry_preserving_quota_realizer/input_reservoirs.npz')),
        initial_pool=dict(path=str(SOURCE/'interior_pools.npz'),sha256=sha(SOURCE/'interior_pools.npz')),
        quotas=dict(path=str(SOURCE/'quota_assignment_trace.json'),sha256=sha(SOURCE/'quota_assignment_trace.json')),
        state_arrays=dict(path=str(OUT/'state_arrays.npz'),sha256=sha(OUT/'state_arrays.npz'))))
    dump(OUT/'manifest.json',dict(shift_aware_quota_realizer=result['status'],promoted=result['passed'],trained=False,
        implementation=str(Path(__file__)),implementation_sha256=sha(__file__),results_sha256=sha(OUT/'results.json'),
        report_sha256=sha(OUT/'report.md'),pool_sha256=sha(OUT/'realized_pools.npz')))
    print(json.dumps(dict(status=result['status'],fresh_discrepancy=fresh['gates'],all_trace_events=result['all_trace_events'],postvalidation_seconds=result['postvalidation_seconds']),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--screen-one',action='store_true')
    parser.add_argument('--fresh-baselines-only',action='store_true')
    parser.add_argument('--finalize-only',action='store_true');args=parser.parse_args()
    if args.finalize_only:
        finalize()
        return
    if args.fresh_baselines_only:
        print(json.dumps(fresh_baseline_check(),indent=2),flush=True)
        return
    OUT.mkdir(exist_ok=True);start=time.monotonic()
    protected_paths=list(ROOT.rglob('*.pt'))+list(Path('experiments').rglob('*.py'))
    protected_paths=[p for p in protected_paths if p.resolve()!=Path(__file__).resolve()]
    for folder in (SOURCE,ROOT/'geometry_preserving_quota_realizer',ROOT/'joint_quota_assignment_teacher',ROOT/'online_quota_reservoir_audit'):
        protected_paths.extend(p for p in folder.iterdir() if p.is_file())
    protected={str(p):sha(p) for p in protected_paths};dump(OUT/'protected_hashes.json',protected)
    checks=formula_checks()
    source_result=json.loads((SOURCE/'results.json').read_text())
    inputs=np.load(ROOT/'geometry_preserving_quota_realizer/input_reservoirs.npz')
    initial=np.load(SOURCE/'interior_pools.npz')
    plans={r['condition']:np.array(r['fixed_quota'],dtype=np.int64) for r in source_result['quota_traces']}
    names=list(initial.files);names=names[:1] if args.screen_one else names
    dump(OUT/'plan.json',dict(rule='cyclic exact same-class exchange minimizing sum of existing per-grid mean-square CDF residuals',
        shifts=SHIFTS,scales=SCALES,max_sweeps=MAX_SWEEPS,gate_source='evaluate_progressive_spacefill_smoke.run + train_dyadic_spacefill_curriculum.health',
        preserve_quotas=True,training=False,geometry_reference_used_by_selection=False,scope='5 archived + 3 saved fresh N8600 pools',
        score_source='ProgressiveGenerator.make_context residual field and losses.cdf_change; deterministic full-set exchange adaptation, not original soft policy'))
    pools={};selected_store={};arrays={};rows=[]
    for name in names:
        candidates=inputs[name];classes=ids(candidates)
        distance,selected=cKDTree(candidates).query(initial[name]);assert np.all(distance==0)
        assert np.array_equal(np.bincount(classes[selected],minlength=len(plans[name])),plans[name])
        assert cKDTree(candidates).query(candidates,k=2)[0][:,1].min()>=.4/np.sqrt(N)
        arrays[name+'_initial_selected']=selected;arrays[name+'_classes']=classes;arrays[name+'_quotas']=plans[name]
        path=OUT/(name+'.jsonl.gz')
        chosen,stats=realize(candidates,classes,selected,path)
        p=candidates[chosen];pools[name]=p;selected_store[name]=chosen
        assert np.array_equal(capacity(p),plans[name]) and len(np.unique(p,axis=0))==N and np.isfinite(p).all()
        lo,hi=bounds(META,N);counts=A@capacity(p);assert np.all(counts>=lo) and np.all(counts<=hi)
        row=dict(condition=name,**stats,quota_errors=0,budget_errors=0,duplicate_points=0,nonfinite_points=0)
        rows.append(row);dump(OUT/'progress.json',rows)
        np.savez_compressed(OUT/'state_arrays.npz',**arrays)
        np.savez_compressed(OUT/'realized_pools.npz',**pools)
        print(json.dumps(dict(event='pool_complete',**row)),flush=True)
    np.savez_compressed(OUT/'selected_candidate_ids.npz',**selected_store)
    if args.screen_one:
        print(json.dumps(discrepancy_rows(pools[names[0]])),flush=True)
        return
    first=names[0]
    replay_ids,replay_stats=realize(inputs[first],arrays[first+'_classes'],arrays[first+'_initial_selected'],
                                 OUT/'determinism_replay.jsonl.gz',expected=OUT/(first+'.jsonl.gz'))
    assert np.array_equal(replay_ids,selected_store[first])
    evaluation=evaluate(pools,initial,source_result)
    assert all(sha(p)==h for p,h in protected.items())
    result=dict(status='VALIDATED_DETERMINISTIC_TEACHER' if evaluation['passed'] else 'STOPPED_HISTORICAL_GATE_FAILURE',
        trained=False,rows=rows,formula_checks=checks,deterministic_replay=True,replayed_decisions=replay_stats['events'],
        protected_unchanged=True,protected_files=len(protected),runtime_seconds=time.monotonic()-start,**evaluation)
    dump(OUT/'results.json',result)
    dump(OUT/'manifest.json',dict(shift_aware_quota_realizer=result['status'],promoted=evaluation['passed'],trained=False,
        implementation=str(Path(__file__)),implementation_sha256=sha(__file__),results_sha256=sha(OUT/'results.json'),
        pool_sha256=sha(OUT/'realized_pools.npz')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','geometry','scale_comparison')},indent=2),flush=True)


if __name__=='__main__':
    main()
