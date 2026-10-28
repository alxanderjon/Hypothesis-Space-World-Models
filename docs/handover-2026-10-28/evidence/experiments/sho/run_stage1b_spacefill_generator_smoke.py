"""Geometry-only shared point-placement model; bounded, frozen-pipeline smoke."""
from __future__ import annotations

import csv
import inspect
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from experiments.sho import validate_stage1b_learned_cell_size as cells
from experiments.sho.smoke_artifacts import save_checkpoint

base, nearest, downstream = cells.base, cells.nearest, cells.nearest.downstream
OUT = Path('outputs/sho/stage1B/spacefill_generator_smoke')
SEED = 9262026


def partitions(n, dimension):
    """N equal-volume boxes, longest-axis bisection with count-proportional cut."""
    leaves = []
    def split(lo, hi, count):
        if count == 1:
            leaves.append((lo, hi))
            return
        axis = int(np.argmax(hi-lo))
        left = count//2
        cut = lo[axis]+(hi[axis]-lo[axis])*left/count
        midhi, midlo = hi.copy(), lo.copy()
        midhi[axis] = midlo[axis] = cut
        split(lo, midhi, left)
        split(midlo, hi, count-left)
    split(np.zeros(dimension), np.ones(dimension), n)
    lo, hi = np.asarray([v[0] for v in leaves]), np.asarray([v[1] for v in leaves])
    return (lo+hi)/2, hi-lo


def faces(n, dimension):
    """2d codimension-one faces; endpoints inclusive; exact sorted dedup."""
    k = max(2, math.ceil(n**(1/dimension)))
    grid = np.linspace(0., 1., k)
    blocks = []
    for axis in range(dimension):
        free = [j for j in range(dimension) if j != axis]
        mesh = np.asarray(list(itertools.product(grid, repeat=dimension-1)))
        for side in (0., 1.):
            block = np.empty((len(mesh), dimension))
            block[:, axis] = side
            if free:
                block[:, free] = mesh
            blocks.append(block)
    return np.unique(np.concatenate(blocks), axis=0), k


class Generator(nn.Module):
    """Shared weights for arbitrary N; d is an architecture parameter.

    The partition and faces are deterministic scaffolding. Only within-cell
    offsets are learned. Bounds map normalized outputs by an affine transform.
    """
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.net = nn.Sequential(nn.Linear(2*dimension, 24), nn.Tanh(), nn.Linear(24, dimension)).double()

    def forward(self, centers, widths):
        features = torch.cat((2*centers-1, widths*len(centers)**(1/self.dimension)-1), dim=1)
        return centers + 0.35*widths*torch.tanh(self.net(features))

    def generate(self, n, lower, upper):
        lower, upper = np.asarray(lower), np.asarray(upper)
        assert len(lower) == self.dimension and np.all(upper > lower) and n > 0
        center, width = partitions(n, self.dimension)
        with torch.no_grad():
            interior = self(torch.from_numpy(center), torch.from_numpy(width)).numpy()
        boundary, _ = faces(n, self.dimension)
        points = np.unique(np.concatenate((interior, boundary)), axis=0)
        assert len(points) == n+len(boundary)
        return lower+(upper-lower)*points, lower+(upper-lower)*interior


def train(budgets):
    torch.manual_seed(SEED)
    model = Generator(2)
    opt = torch.optim.Adam(model.parameters(), lr=.006)
    data = {}
    for n in budgets:
        center, width = partitions(n, 2)
        boundary, _ = faces(n, 2)
        # Training references are independent of all validation references.
        reference = np.random.default_rng(SEED+n).random((max(32768, 4*n), 2))
        data[n] = tuple(torch.from_numpy(a) for a in (center, width, boundary, reference))
    history = []
    started = time.monotonic()
    for step in range(60):
        n = budgets[step % len(budgets)]
        center, width, boundary, reference = data[n]
        points = model(center, width)
        combined = torch.cat((points, boundary))
        # Discrete nearest-neighbor identities are recomputed each step; the
        # exact distance to that selected point remains differentiable.
        ids = cKDTree(combined.detach().numpy()).query(reference.numpy())[1]
        distance = torch.linalg.vector_norm(reference-combined[ids], dim=1)*math.sqrt(n)
        neighbors = cKDTree(points.detach().numpy()).query(points.detach().numpy(), k=2)[1][:, 1]
        separation = torch.linalg.vector_norm(points-points[neighbors], dim=1)*math.sqrt(n)
        covering = distance.max()
        anti_cluster = torch.relu(.65-separation).square().mean()
        loss = covering + .02*distance.square().mean() + .02*anti_cluster
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(dict(record_type='training', step=step+1, N=n,
            estimated_training_cover=float(covering.detach())/math.sqrt(n),
            normalized_loss=float(loss.detach()), anti_cluster=float(anti_cluster.detach())))
        if (step+1)%15 == 0:
            print(f'geometry-only training {step+1}/60 elapsed={time.monotonic()-started:.1f}s', flush=True)
    model.eval().requires_grad_(False)
    return model, history


def metrics(points, interior, reference, n, label, state):
    tree = cKDTree(points)
    distance = tree.query(reference)[0]
    interior_only = cKDTree(interior).query(reference)[0]
    neighbor = tree.query(points, k=2)[0][:, 1]
    interior_neighbor = cKDTree(interior).query(interior, k=2)[0][:, 1]
    boundary_radii = []
    for axis in range(2):
        for side in (0., 1.):
            query = np.empty((16385, 2))
            query[:, axis] = side
            query[:, 1-axis] = np.linspace(0, 1, len(query))
            boundary_radii.append(float(tree.query(query)[0].max()))
    boundary_count = int(np.any((points == 0)|(points == 1), axis=1).sum())
    row = dict(record_type='geometry', source_state_index=state, generator=label, N=n,
        total_probe_count=len(points), interior_probe_count=len(interior), boundary_probe_count=boundary_count,
        reference_count=len(reference), estimated_interior_covering_radius=float(distance.max()),
        interior_only_covering_radius=float(interior_only.max()),
        boundary_face_covering_radius=max(boundary_radii), face_radii=json.dumps(boundary_radii),
        minimum_pairwise_separation=float(neighbor.min()), minimum_interior_separation=float(interior_neighbor.min()),
        close_pair_point_fraction=float(np.mean(neighbor < .1/math.sqrt(n))),
        interior_close_pair_point_fraction=float(np.mean(interior_neighbor < .1/math.sqrt(n))),
        large_hole_reference_fraction=float(np.mean(distance > 1/math.sqrt(n))),
        covering_to_spacing_ratio=float(distance.max()*math.sqrt(n)))
    for percentile in (0, 50, 90, 95, 99, 99.9, 100):
        row[f'nearest_probe_distance_p{percentile}'] = float(np.percentile(distance, percentile))
    return row, distance


def locked_trace(sizes, losses, record, state, locked):
    grouped = downstream._learned_group_and_represent(sizes, losses, locked)
    hull = downstream.hull_threshold._threshold_hull(locked['skew_weight'], locked['homogeneous_weight'],
        locked['hull_theta'], locked['hull_alpha'], grouped['grouped_sizes'], grouped['grouped_losses'])
    pool = downstream.improvement_and._build_pool([dict(source_state_index=state,
        trajectory_id=int(record['trajectory_id']), iteration=int(record['iteration']),
        state_bin=base.turn._state_bin(len(sizes)), sizes=grouped['grouped_sizes'],
        losses=grouped['grouped_losses'], teacher_hull=hull)], 'spacefill_frozen_screen')
    with torch.no_grad():
        passed = locked['and_model'](torch.from_numpy(pool['inputs'])).numpy().ravel() >= 0
    selected = hull[pool['hull_position'][passed]]
    fallback = len(selected) == 0
    if fallback:
        selected = np.asarray([np.argmax(grouped['grouped_sizes'])])
    return dict(groups={tuple(np.flatnonzero(grouped['membership'] == i)) for i in range(len(grouped['grouped_sizes']))},
        representatives=set(grouped['representatives'].tolist()), hull=set(grouped['representatives'][hull].tolist()),
        po=set(grouped['representatives'][selected].tolist()), fallback=fallback)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    geo_path, down_path = OUT/'spacefill_generator_geometry.csv', OUT/'spacefill_generator_downstream.csv'
    assert not geo_path.exists() and not down_path.exists(), 'Refusing to overwrite a smoke run'
    protected = [Path(r['checkpoint_path']) for r in csv.DictReader(cells.HASHES.open()) if r['checkpoint_path']]
    protected += [Path(inspect.getsourcefile(m)) for m in (cells, nearest, downstream, downstream.edirect)]
    hashes = {str(p): base._sha256(p) for p in protected}
    rows = [r for r in csv.DictReader(Path('outputs/sho/stage1B/probe_generation_audit/probe_generation_counts.csv').open())
            if r['record_type']=='state' and r['split']=='held_out']
    chosen = [min(rows, key=lambda r:(abs(int(r['requested_sobol'])-target),int(r['evaluated_sites']))) for target in (1000,8000,32768)]
    budgets = [int(r['requested_sobol']) for r in chosen]
    model, history = train(budgets)
    # Training artifacts survive every later gate, exception, and rejection.
    # Persistence does not confer CANDIDATE or LOCKED status.
    save_checkpoint(OUT/'spacefill_trained_pending_evaluation.pt',dict(
        status='TRAINED_PENDING_EVALUATION',state_dict=model.state_dict(),dimension=2,
        seed=SEED,training_budgets=budgets,training_updates=60,optimizer='Adam',lr=.006,
        training_history=history,implementation=str(Path(__file__).resolve()),
        implementation_sha256=base._sha256(Path(__file__))))
    base._write_csv(OUT/'training_history.csv', history)
    # Independent million-point cell-centered grid plus random points. This is
    # a covering-radius ESTIMATE, not a continuous-domain certificate.
    t = (np.arange(1000)+.5)/1000
    reference = np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),
        np.random.default_rng(SEED+1000000).random((262144,2))))
    geo, pools, passed = [], [], True
    for row, n in zip(chosen, budgets):
        state = int(row['source_state_index'])
        teacher = downstream.edirect._candidate_pool(int(row['evaluated_sites']),int(row['generator_id']),int(row['iteration']))
        teacher_interior = teacher[~np.any((teacher==0)|(teacher==1),axis=1)]
        learned, interior = model.generate(n, [0.,0.], [1.,1.])
        coordinate_path=OUT/f'spacefill_probes_N{n}.npz'
        np.savez_compressed(coordinate_path,points=learned,interior=interior,source_state_index=state)
        coordinate_path.with_suffix('.sha256').write_text(base._sha256(coordinate_path)+'  '+coordinate_path.name+'\n')
        assert len(teacher) == len(learned) and len(teacher_interior) == n
        teacher_metrics, td = metrics(teacher, teacher_interior, reference, n, 'sobol', state)
        learned_metrics, ld = metrics(learned, interior, reference, n, 'learned', state)
        tol = 1+1e-12  # Numerical comparison only, not a coverage allowance.
        checks = {
            'complete_pool_interior_coverage': learned_metrics['estimated_interior_covering_radius'] <= teacher_metrics['estimated_interior_covering_radius']*tol,
            # Conservative supplemental check: remove boundary support. This
            # is stricter than the complete generator's actual coverage.
            'supplemental_interior_only_coverage': learned_metrics['interior_only_covering_radius'] <= teacher_metrics['interior_only_covering_radius']*tol,
            'boundary_coverage': learned_metrics['boundary_face_covering_radius'] <= teacher_metrics['boundary_face_covering_radius']*tol,
            'large_holes': learned_metrics['large_hole_reference_fraction'] <= teacher_metrics['large_hole_reference_fraction'],
            'clustering': learned_metrics['close_pair_point_fraction'] <= teacher_metrics['close_pair_point_fraction']+.001}
        gate = all(checks.values())
        learned_metrics.update(geometry_gate_passed=gate,
            failed_checks=json.dumps([key for key,value in checks.items() if not value]),
            cover_ratio_to_sobol=learned_metrics['estimated_interior_covering_radius']/teacher_metrics['estimated_interior_covering_radius'],
            references_beyond_sobol_max=int(np.sum(ld > td.max()*tol)))
        geo += [teacher_metrics, learned_metrics]
        pools.append((row,teacher,learned))
        passed &= gate
        print(f'N={n} cover Sobol={td.max():.8g} learned={ld.max():.8g} ratio={ld.max()/td.max():.4f} geometry_pass={gate}',flush=True)
    base._write_csv(geo_path, geo)
    out_rows = []
    if passed:
        distance, _ = base._load_locked_distance()
        operator = torch.load(nearest.OPERATOR,weights_only=False)
        margin = nn.Linear(2,1,bias=False,dtype=torch.float64)
        margin.load_state_dict(torch.load(operator['margin_checkpoint'],weights_only=False)['state_dict'])
        threshold = nearest.threshold.NonzeroThreshold()
        threshold.load_state_dict(torch.load(operator['threshold_checkpoint'],weights_only=False)['state_dict'])
        half = nn.Linear(1,1,bias=False,dtype=torch.float64)
        half.load_state_dict(torch.load(cells.OUT/'half_spacing_operator_locked.pt',weights_only=False)['state_dict'])
        for m in (distance,margin,threshold,half):
            m.eval().requires_grad_(False)
        locked = downstream._load_downstream()
        payload = torch.load(base.turn.RAW_TRACE_PATH,weights_only=False)
        trajectories = base.ordinary_data._trajectory_map(payload)
        for row, teacher, learned in pools:
            state = int(row['source_state_index'])
            record = payload['states'][state]
            n = int(record['history_count'])
            trajectory = trajectories[int(record['trajectory_id'])]
            sites = np.asarray(trajectory['evaluated_hypotheses_normalized'][:n],dtype=np.float64)
            losses = np.asarray(trajectory['evaluated_prefix_losses'][:n],dtype=np.float64)
            results = []
            for label, probes in [('sobol',teacher),('learned',learned)]:
                print(f'frozen full candidate scan state={state} sites={n} probes={len(probes)} generator={label}',flush=True)
                captured, scans = cells.capture_nearest(sites,probes,distance,margin,threshold,state)
                p = len(probes)
                radius, _, stats = cells.assigned_radius(captured['winner'][:p],captured['winner_distance'][:p],n,margin,threshold)
                h = cells.predict_half(half,captured['winner_distance'][p:])
                replace, ties, invalid, errors = cells.larger_candidate(margin,threshold,radius,h)
                assert not (errors or invalid or stats['radius_pairwise_mistakes'] or stats['radius_invalid_logits'])
                sizes = np.where(replace,h,radius)
                trace = locked_trace(sizes,losses,record,state,locked)
                results.append((sizes,trace))
                out_rows.append(dict(record_type='pipeline', source_state_index=state,generator=label,site_count=n,
                    N=int(row['requested_sobol']), total_probe_count=p, **stats,
                    pairwise_mistakes=sum(r['pairwise_mistakes'] for r in scans),
                    final_argmin_mistakes=sum(r['final_argmin_mistakes'] for r in scans),
                    cell_size_min=float(sizes.min()),cell_size_p50=float(np.median(sizes)),
                    cell_size_p90=float(np.percentile(sizes,90)),cell_size_max=float(sizes.max()),
                    cell_size_mean=float(sizes.mean()),group_count=len(trace['groups']),representative_count=len(trace['representatives']),
                    hull_count=len(trace['hull']),PO_count=len(trace['po']),PO_fallback=trace['fallback'],
                    representatives=json.dumps(sorted(trace['representatives'])),hull=json.dumps(sorted(trace['hull'])),
                    PO=json.dumps(sorted(trace['po']))))
            (ts,tt),(ls,lt) = results
            relative = np.abs(ls-ts)/np.maximum(ts,1e-15)
            ratio = len(lt['po'])/max(1,len(tt['po']))
            # Predeclared smoke alarms, not claims of trajectory equivalence:
            # finite positive sizes, <2x global max, <=20% sites >50% shift,
            # nonempty PO and no >4x / <1/4x cardinality jump.
            stable = bool(np.all(np.isfinite(ls)) and np.all(ls>0) and ls.max()<=2*ts.max()
                and np.mean(relative>.5)<=.2 and .25<=ratio<=4 and len(lt['po'])>0)
            compare = dict(record_type='comparison',source_state_index=state,N=int(row['requested_sobol']),
                material_definition='abs(delta)>max(1e-12,0.01*teacher_d_i)',
                materially_changed_sites=int(np.sum(np.abs(ls-ts)>np.maximum(1e-12,.01*ts))),
                sites_relative_change_over_50_percent=int(np.sum(relative>.5)),max_relative_cell_size_change=float(relative.max()),
                mean_relative_cell_size_change=float(relative.mean()),PO_cardinality_ratio=ratio,downstream_gate_passed=stable)
            for key in ('groups','representatives','hull','po'):
                compare[key+'_symmetric_difference'] = len(tt[key]^lt[key])
                compare[key+'_jaccard'] = len(tt[key]&lt[key])/max(1,len(tt[key]|lt[key]))
            out_rows.append(compare)
            print(json.dumps(compare),flush=True)
            base._write_csv(down_path,out_rows)
            if not stable:
                passed = False
                break
    else:
        out_rows.append(dict(record_type='hard_stop',reason='geometry_gate_failed',downstream_run=False,
            failed_cases=json.dumps([dict(N=r['N'],failed_checks=r['failed_checks']) for r in geo
                if r['generator']=='learned' and not r['geometry_gate_passed']]),
            note='Interior-only coverage is a supplemental stricter check; complete-pool coverage includes boundary support.'))
    assert all(base._sha256(Path(p))==h for p,h in hashes.items()), 'Protected input changed'
    status = 'CANDIDATE' if passed else 'REJECTED'
    summary = dict(record_type='summary',status=status,protected_hashes_unchanged=True,
        states_screened=sum(r['record_type']=='comparison' for r in out_rows),trajectory_search=False,production_replaced=False)
    out_rows.insert(0,summary)
    base._write_csv(down_path,out_rows)
    if passed:
        path = OUT/'spacefill_generator_candidate.pt'
        torch.save(dict(status='CANDIDATE',state_dict=model.state_dict(),dimension=2,training_budgets=budgets,
            training_steps=60,seed=SEED,geometry_only=True,implementation=str(Path(__file__).resolve()),
            implementation_sha256=base._sha256(Path(__file__)),protected_hashes=hashes,
            scaffold='equal-volume box partition + endpoint-inclusive 2d faces; learned shared bounded offsets',
            generality='arbitrary N with shared 2D weights; dimension-parametric architecture needs separate geometry training for other d',
            normalization='affine unit-box map; anisotropic physical-distance quality untested',summary=summary),path)
        summary['checkpoint_sha256'] = base._sha256(path)
        path.with_suffix('.sha256').write_text(summary['checkpoint_sha256']+'  '+path.name+'\n')
        base._write_csv(down_path,out_rows)
    (OUT/'summary.json').write_text(json.dumps(dict(**summary,protected_hashes=hashes,
        geometry_reference='1,000,000 grid centers + 262,144 independent uniform points; 16,385 per boundary face',
        training_objective='max nearest-reference distance + .02 mean squared distance + .02 short-separation penalty, in N^-1/d units',
        limitations='Three 2D held-out states only; no continuous-domain coverage certificate; deterministic partition scaffold; no unseen-N or dimension generalization claim.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    run()
