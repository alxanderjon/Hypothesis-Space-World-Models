"""Frozen geometry screen only. Cached baselines use the identical references."""
import csv
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator, discrepancy_rows, boundary_faces, SCALES
from experiments.sho.train_progressive_spacefill_smoke import OUT, BUDGETS, sha, write_csv

BASE = Path('outputs/sho/stage1B/fresh_spacefill_smoke')


def read_baseline(name):
    rows = []
    for r in csv.DictReader((BASE/name).open()):
        for k,v in list(r.items()):
            if v == '': continue
            try: r[k] = float(v)
            except ValueError: pass
        if r.get('generator') == 'new_spacefill': r['generator'] = 'fresh_v1'
        rows.append(r)
    return rows


def metrics(points, interior, reference, n):
    tree = cKDTree(points)
    d = tree.query(reference)[0]; di = cKDTree(interior).query(reference)[0]
    near = cKDTree(interior).query(interior, k=2)[0][:,1]
    face = []
    # Generic codimension-one reference support; this screen has dimension 2.
    for axis in range(interior.shape[1]):
        for side in (0.,1.):
            q = np.empty((8193, 2)); q[:,axis] = side
            q[:,1-axis] = np.linspace(0,1,len(q)); face.append(tree.query(q)[0].max())
    return dict(complete_cover=float(d.max()), interior_only_cover=float(di.max()), boundary_cover=float(max(face)),
        nn_CV=float(near.std()/near.mean()), nn_min=float(near.min()), nn_p10=float(np.quantile(near,.1)),
        nn_median=float(np.median(near)), nn_p90=float(np.quantile(near,.9)),
        clustered_fraction=float(np.mean(near<.1/math.sqrt(n))), large_hole_count=int(np.sum(d>1/math.sqrt(n))),
        large_hole_fraction=float(np.mean(d>1/math.sqrt(n))), reference_count=len(reference),
        interior_count=len(interior), complete_count=len(points))


def overlap(a, b, n):
    d = cKDTree(b).query(a)[0]
    return dict(exact_overlap_count=int(np.sum(d==0)), exact_overlap_fraction=float(np.mean(d==0)),
        near_overlap_fraction_005_spacing=float(np.mean(d<.05/math.sqrt(n))),
        near_overlap_fraction_010_spacing=float(np.mean(d<.1/math.sqrt(n))), median_cross_distance=float(np.median(d)))


def run():
    started = time.monotonic(); torch.set_num_threads(1)
    training = json.loads((OUT/'training_summary.json').read_text())
    assert sha(training['checkpoint']) == training['sha256']
    assert sha('experiments/sho/progressive_spacefill_generator.py') == training['source_sha256']
    model = ProgressiveGenerator()
    model.load_state_dict(torch.load(training['checkpoint'], weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    # These thresholds are declared before evaluation, not tuned afterwards.
    criteria = dict(discrepancy='>=20% smaller worst one-count-floored Sobol ratio than BOTH prior learned generators at every N; no scale >1.10 fresh-v1',
        max_complete_cover_ratio_to_Sobol=1.10, max_paired_complete_cover_ratio_to_Sobol=1.15,
        max_union_cover_ratio_to_Sobol=1.25, max_near_overlap_fraction_005=.03,
        max_large_hole_fraction_excess=.001, min_nn_CV=.08, max_clustered_fraction=.005)
    (OUT/'evaluation_criteria.json').write_text(json.dumps(criteria, indent=2)+'\n')
    rows = read_baseline('fresh_spacefill_geometry.csv')
    multi = read_baseline('fresh_spacefill_multiscale.csv')
    freshness = read_baseline('fresh_spacefill_freshness_union.csv')
    # EXACT same independent reference as all three cached baselines. The
    # midpoint grid is evaluation-only: never candidate coordinates/training.
    t = (np.arange(512)+.5)/512
    reference = np.vstack((np.stack(np.meshgrid(t,t), axis=-1).reshape(-1,2),
                           np.random.default_rng(27093001).random((262144,2))))
    conditions = [(100+j, 'cell') for j in range(4)]+[(100,'greedy')]
    coordinates = {}; checks = []
    for n in BUDGETS:
        pools = []; group = []
        for iteration, pool in conditions:
            begin = time.monotonic()
            with torch.no_grad(): points, interior = model.generate(n, iteration, pool)
            generation_seconds = time.monotonic()-begin
            if iteration == 100 and pool == 'cell':
                with torch.no_grad(): replay = model.generate(n, iteration, pool)[0]
                assert np.array_equal(points,replay), 'generator is not reproducible'
            boundary = points[np.any((points==0)|(points==1), axis=1)]
            assert np.array_equal(boundary, boundary_faces(n,2))
            assert len(interior) == n
            coordinates[f'N{n}_iteration{iteration}_{pool}'] = interior
            row = dict(N=n, generator='progressive', iteration=iteration, pool_type=pool,
                       **metrics(points, interior, reference, n), generation_seconds=generation_seconds)
            rows.append(row); group.append(row); pools.append((points, interior))
            for d in discrepancy_rows(interior):
                multi.append(dict(N=n, generator='progressive', iteration=iteration, pool_type=pool, **d))
        for i,j,label in [(0,1,'cell_t_to_t1'),(1,2,'cell_t1_to_t2'),(2,3,'cell_t2_to_t3'),(0,4,'cell_to_greedy')]:
            freshness.append(dict(record_type='overlap', N=n, generator='progressive', comparison=label,
                                  **overlap(pools[i][1],pools[j][1],n)))
        union = np.unique(np.concatenate([v[0] for v in pools[:4]]),axis=0)
        d = cKDTree(union).query(reference)[0]
        di = cKDTree(np.concatenate([v[1] for v in pools[:4]])).query(reference)[0]
        freshness.append(dict(record_type='union', N=n, generator='progressive', union_unique_count=len(union),
            union_complete_cover=float(d.max()), union_interior_only_cover=float(di.max()),
            union_gain_vs_mean_single=1-float(d.max())/np.mean([r['complete_cover'] for r in group[:4]]),
            union_large_hole_count=int(np.sum(d>1/math.sqrt(4*n)))))
        print(json.dumps(dict(event='heldout_budget_complete', N=n,
              complete_cover_max=max(r['complete_cover'] for r in group),
              nn_CV_mean=float(np.mean([r['nn_CV'] for r in group])))), flush=True)
    ratios = []; tables = []
    names = ('Sobol','old_spacefill','fresh_v1','progressive')
    for n in BUDGETS:
        scales = {}
        for name in names:
            scales[name] = {}
            for k in SCALES:
                v = [r['grid_anchored_discrepancy'] for r in multi if r['N']==n and r['generator']==name and r['bins_per_axis']==k]
                scales[name][k] = max(v)
        for name in names:
            for k in SCALES:
                a,b = scales[name][k], scales['Sobol'][k]
                ratios.append(dict(N=n, generator=name, bins_per_axis=k, worst_anchored_discrepancy=a,
                    Sobol_discrepancy=b, ratio_vs_Sobol=a/b if b else (0. if a==0 else float('inf')),
                    ratio_with_one_count_denominator_floor=a/max(b,1/n)))
            g = [r for r in rows if r['N']==n and r['generator']==name]
            f = [r for r in freshness if r['N']==n and r['generator']==name and r['record_type']=='overlap']
            u = next(r for r in freshness if r['N']==n and r['generator']==name and r['record_type']=='union')
            tables.append(dict(N=n, generator=name, complete_cover_max=max(r['complete_cover'] for r in g),
                interior_only_cover_max=max(r['interior_only_cover'] for r in g), boundary_cover_max=max(r['boundary_cover'] for r in g),
                nn_CV_mean=float(np.mean([r['nn_CV'] for r in g])), large_hole_count_max=max(r['large_hole_count'] for r in g),
                clustered_fraction_max=max(r['clustered_fraction'] for r in g),
                exact_overlap_max=max(r['exact_overlap_fraction'] for r in f),
                near_overlap_005_max=max(r['near_overlap_fraction_005_spacing'] for r in f),
                cell_vs_greedy_near_overlap_005=next(r['near_overlap_fraction_005_spacing'] for r in f if r['comparison']=='cell_to_greedy'),
                union_complete_cover=u['union_complete_cover'], union_gain=u['union_gain_vs_mean_single'],
                worst_discrepancy_ratio=max(r['ratio_vs_Sobol'] for r in ratios if r['N']==n and r['generator']==name),
                worst_discrepancy_ratio_one_count_floor=max(r['ratio_with_one_count_denominator_floor'] for r in ratios if r['N']==n and r['generator']==name)))
        tab = {r['generator']:r for r in tables if r['N']==n}; new=tab['progressive']; sobol=tab['Sobol']
        paired = [r['complete_cover']/next(s['complete_cover'] for s in rows if s['N']==n and s['generator']=='Sobol' and s['iteration']==r['iteration'] and s['pool_type']==r['pool_type'])
                  for r in rows if r['N']==n and r['generator']=='progressive']
        gate = dict(N=n, discrepancy_materially_better=all(new['worst_discrepancy_ratio_one_count_floor']<=.8*tab[name]['worst_discrepancy_ratio_one_count_floor'] for name in ('old_spacefill','fresh_v1')),
            no_scale_regression=all(scales['progressive'][k]<=1.1*scales['fresh_v1'][k] for k in SCALES),
            coverage=new['complete_cover_max']<=1.10*sobol['complete_cover_max'] and max(paired)<=1.15,
            boundary=new['boundary_cover_max']==sobol['boundary_cover_max'],
            freshness=new['exact_overlap_max']==0 and new['near_overlap_005_max']<=.03,
            union=new['union_complete_cover']<=1.25*sobol['union_complete_cover'],
            holes=new['large_hole_count_max']<=sobol['large_hole_count_max']+.001*len(reference),
            not_rigid_or_clustered=new['nn_CV_mean']>=.08 and new['clustered_fraction_max']<=.005,
            paired_cover_ratio_max=max(paired))
        gate['passed'] = all(v for k,v in gate.items() if k not in ('N','paired_cover_ratio_max'))
        checks.append(gate)
    unchanged = all(sha(p)==h for p,h in training['protected_hashes'].items())
    assert unchanged and sha(training['checkpoint'])==training['sha256']
    promoted = all(c['passed'] for c in checks) and training['best_cycle']>0
    write_csv(OUT/'progressive_spacefill_geometry.csv', rows)
    write_csv(OUT/'progressive_spacefill_multiscale.csv', multi)
    write_csv(OUT/'progressive_spacefill_scale_ratios.csv', ratios)
    write_csv(OUT/'progressive_spacefill_freshness_union.csv', freshness)
    write_csv(OUT/'progressive_spacefill_comparison.csv', tables)
    np.savez_compressed(OUT/'progressive_heldout_interiors.npz', **coordinates)
    coords_hash = sha(OUT/'progressive_heldout_interiors.npz')
    (OUT/'progressive_heldout_interiors.sha256').write_text(coords_hash+'  progressive_heldout_interiors.npz\n')
    if promoted:
        cp = OUT/'PROGRESSIVE_SPACEFILL_CANDIDATE.pt'
        assert not cp.exists(); shutil.copyfile(training['checkpoint'],cp)
        cp.with_suffix('.sha256').write_text(sha(cp)+'  '+cp.name+'\n')
    summary = dict(status='PROGRESSIVE_SPACEFILL_CANDIDATE' if promoted else 'DIAGNOSTIC_ONLY',
        training_seconds=training['elapsed_seconds'], evaluation_seconds=time.monotonic()-started,
        gates=checks, checkpoint=training['checkpoint'], sha256=training['sha256'],
        coordinates_sha256=coords_hash, protected_hashes_unchanged=unchanged,
        baseline_metrics='cached identical 524288-point reference and same iteration/pool conditions; no baseline coordinates used for training',
        baseline_hashes={p.name:sha(p) for p in [BASE/'fresh_spacefill_geometry.csv',BASE/'fresh_spacefill_multiscale.csv',BASE/'fresh_spacefill_freshness_union.csv']},
        no_closed_loop=True, no_objective_or_parameters_or_downstream_labels=True)
    (OUT/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['PROGRESSIVE GEOMETRY-ONLY GENERATOR SMOKE', json.dumps(summary,indent=2),
        'Architecture: random-initialized float64 34->32 tanh->1 placement policy. Blocks 32/128/256; IID proposals oversampled 8x.',
        'Policy sees seven occupancy scales and anchored residual fields plus nearest-prefix distances, N/iteration/pool. Updates prefix after each block.',
        'Training is a detached-prefix differentiable expected-block surrogate, not differentiation through hard sampling or a full end-to-end unroll.',
        'Primary uses count/CDF discrepancy; projected secondary hole gradient cannot oppose primary to first order and is <=10% its norm.',
        'IID proposals and Gumbel selection use deterministic SHA256/PCG64 conditioning. A .4/sqrt(N) geometric exclusion prevents collapse; no lattice/digital-net interior coordinates.',
        'Boundary scaffolding is generic codimension-one face enumeration; fixed boundaries excluded from freshness statistics.',
        'All pretrained baseline generators are evaluation-only. No Sobol coordinates, search objectives/parameters, archived targets, or downstream labels entered training.',
        'Best weights persisted and hashed BEFORE this evaluation. Initial random policy can have positive feedback/clustering; its large training improvement is not evidence of beating prior generators.',
        'Held out: iterations100..103 cell and100 greedy; training5000+, selection monitor7000. All three budgets are seen during training.',
        'Reference: 262144 IID + 512x512 midpoints, solely evaluation. Covering radii are finite-sample estimates.',
        'Discrepancy: anchored CDF on resolutions2..128 and four periodic shifts; also count variance/Fano/empty fractions in multiscale CSV. Not exact continuous star discrepancy.',
        'Ratios are scale matched. Literal ratio is infinite when Sobol discrepancy is zero and generator discrepancy nonzero; one-count denominator floor reported separately.',
        'Freshness near threshold=.05/sqrt(N), with .10/sqrt(N) also in detail CSV. Large holes: reference distance>1/sqrt(N).',
        'Evaluation criteria are smoke heuristics, not mathematical guarantees. No automatic extension, downstream screen, or closed-loop search.']
    lines += ['\nCOMPARISON:']+[json.dumps(r) for r in tables]
    lines += ['\nFAILED GATES:']+[f"N={r['N']}: "+', '.join(k for k,v in r.items() if v is False) for r in checks]
    (OUT/'progressive_spacefill_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': run()
