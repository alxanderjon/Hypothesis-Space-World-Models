"""Small, gated placement-only screen on immutable continuous-partition leaves.

No frozen file is written. Inference/training never uses Sobol target coordinates.
Full N1 integration is allowed only after the isolated gate in results.json passes.
"""
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc
from torch import nn

ROOT = Path('outputs/sho/stage1B')
FROZEN = ROOT / 'learned_brent_completion'
OUT = ROOT / 'conditioned_leaf_placement_screen'
SEED = 2026092807
GATES = dict(cover_ratio=1.10, added_hole_fraction=.001,
             separation_ratio=.5, discrepancy_ratio=1.05,
             cross_condition_near_fraction=.05, near_radius_in_spacing=.01,
             history_exclusion_radius=1e-12)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csvwrite(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_partition_runtime():
    """Reuse the emitted locked evaluation assembly without invoking a run().

    Its existing reference side is retained; only the learned returned child
    boxes are used. This does not change candidate generation or search logic.
    """
    from experiments.geometry import validate_branch_boundary_baseline as base
    from experiments.geometry.load_completed_brent import load_completed_brent
    from experiments.geometry.finish_learned_brent import Run
    from experiments.geometry.audit_exact_branch_brent import Audit
    manifest = json.loads((FROZEN / 'learned_continuous_recursive_selector_locked.json').read_text())
    assert manifest['status'] == 'LOCKED'
    hashes = json.loads((FROZEN / 'artifact_hashes.json').read_text())
    path = FROZEN / 'integrated_selector.py'
    assert sha(path) == hashes[str(path)]
    _, solver = load_completed_brent()
    callback_owner = Run.__new__(Run)
    callback_owner.audit = Audit()
    env = dict(base.__dict__)
    exec(compile(path.read_text(), str(path), 'exec'), env)
    cls = env['CompletedSelector']

    def locate_branch(self, record, piece, samples):
        request = dict(parent=record, axis=piece['axis'], count=piece['k'])
        cut, _, _, _ = solver(callback_owner.callback(request), float(samples[0]), float(samples[1]), -1)
        return cut

    cls.locate_branch = locate_branch
    return cls()


def partition(selector, root, name):
    stack = [('r', root)]
    leaves = []
    while stack:
        path, p = stack.pop()
        if p['n'] == 1:
            leaves.append(dict(path=path, parent=p))
            continue
        _, got, _ = selector.select(p, name + ':' + path)
        assert sum(got['counts']) == p['n'] and min(got['counts']) > 0
        for j in (1, 0):
            child = dict(d=p['d'], n=got['counts'][j], lower=got['child_lower'][j], upper=got['child_upper'][j])
            stack.append((path + str(j), child))
    assert len(leaves) == root['n']
    return leaves


def case_from_leaves(name, leaves, split):
    rawlo = np.array([x['parent']['lower'] for x in leaves], dtype=np.float64)
    rawhi = np.array([x['parent']['upper'] for x in leaves], dtype=np.float64)
    origin, extent = rawlo.min(0), rawhi.max(0) - rawlo.min(0)
    # Coordinate normalization is bookkeeping; original coordinates are saved.
    lo, hi = (rawlo-origin)/extent, (rawhi-origin)/extent
    return dict(name=name, split=split, lo=lo, hi=hi, origin=origin, extent=extent,
                n=len(lo), d=lo.shape[1])


def context_noise(name, condition, pool, shape):
    key = f'{SEED}:{name}:{condition}:{pool}'.encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    return np.random.default_rng(seed).uniform(-1., 1., shape)


def history_for(c, condition, prefix='train'):
    seed = int.from_bytes(hashlib.sha256(f'{prefix}:{c["name"]}:{condition}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    center = (c['lo'] + c['hi']) / 2
    # Center/history collisions explicitly exercise displacement from old sites.
    hist = [center[rng.permutation(c['n'])[:max(2, c['n']//2)]], rng.random((10, c['d']))]
    if c['d'] == 2:
        i = np.arange(10)
        hist.append(np.column_stack(((i+.5)/10, ((3*i)%10+.5)/10)))
    return np.concatenate(hist)


def features(c, iteration, pool, history):
    center = (c['lo']+c['hi'])/2
    width = c['hi']-c['lo']
    h = c['n']**(-1/c['d'])
    dist, ids = cKDTree(history).query(center)
    noise = context_noise(c['name'], iteration, pool, center.shape)
    x = np.stack((noise, 2*center-1, np.log(width/h),
                  np.clip((center-history[ids])/width, -2, 2),
                  np.broadcast_to(np.minimum(dist/h, 2)[:, None], center.shape),
                  np.full_like(center, c['d']/4)), axis=-1)
    return torch.from_numpy(x)


class Placement(nn.Module):
    """Shared coordinate map: six local features -> bounded leaf fraction."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.net = (nn.Linear(6, 1) if not hidden else
                    nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, 1))).double()

    def forward(self, x, lo, hi):
        z = torch.sigmoid(self.net(x).squeeze(-1))
        return lo + z*(hi-lo)


def placed(model, c, t, p, history):
    return model(features(c, t, p, history), torch.from_numpy(c['lo']), torch.from_numpy(c['hi']))


def anchor_groups(d, seed, count):
    rng = np.random.default_rng(seed)
    a = rng.random((count, d))
    for i in range(count):
        active = 1+i%d
        a[i, rng.choice(d, d-active, replace=False)] = 1.
    return a


def discrepancy(p, anchors):
    err = np.all(p[None] <= anchors[:, None], axis=2).mean(1)-anchors.prod(1)
    return float(np.sqrt(np.mean(err*err)))


def training_material(c, index):
    rng = np.random.default_rng(SEED+index)
    ref = np.concatenate((rng.random((384, c['d'])), np.array(list(__import__('itertools').product((0., 1.), repeat=c['d'])))))
    center = (c['lo']+c['hi'])/2
    near = cKDTree(center).query(ref)[0]
    sep = cKDTree(center).query(center, k=2)[0][:, 1]
    a = anchor_groups(c['d'], SEED+200+index, 96)
    return dict(ref=torch.from_numpy(ref), cover=near.max(), sep=float(np.quantile(sep, .01)),
                anchors=torch.from_numpy(a), center=torch.from_numpy(center))


def objective(model, c, material, t):
    hist = history_for(c, t)
    points = [placed(model, c, t, 0, hist), placed(model, c, t+1, 0, hist), placed(model, c, t, 1, hist)]
    h = c['n']**(-1/c['d'])
    anchors = material['anchors']
    tau = .08*float(np.median(c['hi']-c['lo']))
    def soft(p):
        member = torch.sigmoid((anchors[:, None]-p[None])/tau)
        member = torch.where(anchors[:, None] == 1, torch.ones_like(member), member)
        return (member.prod(2).mean(1)-anchors.prod(1)).square().mean()
    base = soft(material['center']).detach().clamp_min(1e-7)
    values = []
    for p in points:
        distance = torch.cdist(material['ref'], p).min(1).values
        pair = torch.cdist(p, p).masked_fill(torch.eye(c['n'], dtype=torch.bool), torch.inf).min(1).values
        hd = torch.cdist(p, torch.from_numpy(hist)).min(1).values
        values.append(soft(p)/base + 100*torch.relu(distance.max()/material['cover']-1.03).square()
                      + 15*torch.relu(.7-pair/material['sep']).square().mean()
                      + 5*torch.relu(1-hd/(.035*h)).square().mean())
    freshness = sum(torch.relu(1-torch.linalg.vector_norm(points[0]-p, dim=1)/(.035*h)).square().mean() for p in points[1:])
    return torch.stack(values).mean()+5*freshness


def train(hidden, cases):
    torch.manual_seed(SEED+hidden)
    model = Placement(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    material = [training_material(c, i) for i, c in enumerate(cases)]
    rng = np.random.default_rng(SEED+hidden)
    best, weights, step_best = math.inf, None, 0
    records = []
    for step in range(1, 601):
        i = int(rng.integers(len(cases)))
        loss = objective(model, cases[i], material[i], int(rng.integers(16)))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                value = float(np.mean([float(objective(model, c, m, 3)) for c, m in zip(cases, material)]))
            records.append(dict(hidden=hidden, step=step, training_objective=value))
            if value < best:
                best, weights, step_best = value, copy.deepcopy(model.state_dict()), step
            print(json.dumps(dict(event='placement_training', **records[-1])), flush=True)
    model.load_state_dict(weights)
    path = OUT/f'placement_h{hidden}_diagnostic.pt'
    torch.save(dict(state_dict=weights, hidden=hidden, seed=SEED+hidden, dtype='float64',
                    status='SAVED_BEFORE_EVALUATION', selected_update=step_best,
                    gates=GATES, training_objective=best), path)
    model.eval().requires_grad_(False)
    return model, records, dict(hidden=hidden, parameters=sum(p.numel() for p in model.parameters()),
                                checkpoint=str(path), sha256=sha(path), selected_update=step_best)


def edges(n, d):
    if d != 2:
        return np.empty((0, d))  # Only 2D has a current N1 boundary-support contract.
    count = min(257, max(9, int(math.ceil(math.sqrt(n)))))
    v = np.linspace(0, 1, count)
    return np.unique(np.concatenate([np.column_stack((v, np.zeros(count))), np.column_stack((v, np.ones(count))),
                                    np.column_stack((np.zeros(count), v)), np.column_stack((np.ones(count), v))]), axis=0)


def evaluate(model, cases, hidden):
    rows, failures, examples = [], [], {}
    for ci, c in enumerate(cases):
        rng = np.random.default_rng(SEED+10000+ci)
        center = (c['lo']+c['hi'])/2
        ref = [rng.random((4096, c['d']))]
        for axis in range(c['d']):
            for side in (0., 1.):
                z = rng.random((256, c['d'])); z[:, axis] = side; ref.append(z)
        ref = np.concatenate(ref)
        h = c['n']**(-1/c['d'])
        base_dist = cKDTree(center).query(ref)[0]
        base_sep = np.quantile(cKDTree(center).query(center, k=2)[0][:, 1], .01)
        anchors = anchor_groups(c['d'], SEED+20000+ci, 4096)
        base_disc = discrepancy(center, anchors)
        t = 32 if c['split'] == 'heldout' else 128 if c['split'] == 'fresh' else 4
        hist = history_for(c, t, 'evaluation')
        prev = None
        for condition, pool in ((t, 0), (t+1, 0), (t, 1)):
            with torch.no_grad():
                p = placed(model, c, condition, pool, hist).numpy()
                duplicate_call = placed(model, c, condition, pool, hist).numpy()
            finite = bool(np.isfinite(p).all())
            outside = int(np.sum(np.any((p<c['lo']) | (p>c['hi']), axis=1)))
            duplicates = len(p)-len(np.unique(p, axis=0))
            hd = cKDTree(hist).query(p)[0]
            dist = cKDTree(p).query(ref)[0]
            nn = cKDTree(p).query(p, k=2)[0][:, 1]
            fresh = cKDTree(prev).query(p)[0] if prev is not None else None
            face = edges(c['n'], c['d'])
            merged = np.unique(np.concatenate((p, face)), axis=0)
            actual_face = merged[np.any((merged == 0) | (merged == 1), axis=1)] if c['d'] == 2 else face
            row = dict(hidden=hidden, case=c['name'], split=c['split'], d=c['d'], n=c['n'], iteration=condition, pool=pool,
                       finite=finite, outside=outside, interior_duplicates=duplicates,
                       count_ok=len(p)==c['n'] and len(merged)==len(p)+len(face),
                       boundary_exact=bool(np.array_equal(actual_face, face)), repeat_exact=bool(np.array_equal(p, duplicate_call)),
                       stale_history=int(np.sum(hd<=1e-12)), min_history_distance=float(hd.min()),
                       cover=float(dist.max()), cover_ratio=float(dist.max()/base_dist.max()),
                       added_hole_fraction=float(np.mean(dist>h)-np.mean(base_dist>h)),
                       min_separation=float(nn.min()), separation_ratio=float(np.quantile(nn, .01)/base_sep),
                       anchored_rms=discrepancy(p, anchors), discrepancy_ratio=discrepancy(p, anchors)/base_disc,
                       cross_condition_exact=int(np.sum(fresh==0)) if fresh is not None else 0,
                       cross_condition_near_fraction=float(np.mean(fresh<=.01*h)) if fresh is not None else 0.)
            gate = dict(containment=outside==0, finite=finite, count=row['count_ok'], duplicates=duplicates==0,
                        boundary=row['boundary_exact'], history=row['stale_history']==0,
                        repeat=row['repeat_exact'], coverage=row['cover_ratio']<=GATES['cover_ratio'],
                        holes=row['added_hole_fraction']<=GATES['added_hole_fraction'],
                        separation=row['separation_ratio']>=GATES['separation_ratio'],
                        discrepancy=row['discrepancy_ratio']<=GATES['discrepancy_ratio'],
                        freshness=row['cross_condition_exact']==0 and row['cross_condition_near_fraction']<=GATES['cross_condition_near_fraction'])
            row['failed_gates'] = ','.join(k for k, v in gate.items() if not v)
            row['passed'] = all(gate.values())
            rows.append(row)
            if not row['passed']:
                failures.append(row)
            examples[f'{hidden}_{ci}_{condition}_{pool}_points'] = p
            examples[f'{hidden}_{ci}_{condition}_{pool}_history'] = hist.copy()
            # Simulated evaluated history contains a subset of the preceding pool.
            # All preceding points are used for cross-condition freshness metrics.
            if prev is None:
                prev = p.copy()
            hist = np.concatenate((hist, p[::max(1, c['n']//4)]))
        # Independent Sobol geometry only; never used as a training target.
        sobol = qmc.Sobol(c['d'], scramble=True, seed=SEED+30000+ci).random_base2(math.ceil(math.log2(c['n'])))[:c['n']]
        sd = cKDTree(sobol).query(ref)[0]
        rows.append(dict(hidden=hidden, case=c['name'], split='sobol_reference', d=c['d'], n=c['n'],
                         cover=float(sd.max()), anchored_rms=discrepancy(sobol, anchors),
                         min_separation=float(cKDTree(sobol).query(sobol, k=2)[0][:, 1].min())))
    return rows, failures, examples


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p): sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('locked', 'manifest', 'validated'))})
    protected.update(json.loads((FROZEN/'artifact_hashes.json').read_text()))
    assert all(sha(p)==v for p, v in protected.items())
    dump('protected_hashes.json', protected)
    dump('plan.json', dict(seed=SEED, gates=GATES, architectures=[dict(hidden=0, parameters=7), dict(hidden=8, parameters=65)],
                           updates_per_architecture=600, scope='small placement-only architecture screen',
                           selection='training objective only; checkpoint before heldout/fresh evaluation',
                           teacher_target='No unique coordinate target; geometric losses and history/condition separation, no Sobol labels',
                           history_semantics='interior proposal distance to evaluated sites >1e-12; mandated boundary points may repeat and retain unchanged downstream filtering',
                           bookkeeping='hash-seeded context noise; nearest-history feature; exact leaf reconstruction; deterministic boundary grid and unique',
                           comparison_gate='No N1 integration or A/B unless isolated placement passes; no extension or tuning after failure',
                           geometry_gates='Retain coverage/hole/separation safeguards and <=5% anchored RMS degradation; do not claim the old 5% improvement gate passed'))
    cases = []
    paths = sorted(FROZEN.glob('*_leaves.json'))
    held = {'recursive_d2_n16_leaves', 'recursive_d3_n16_leaves', 'recursive_d4_n16_leaves'}
    for path in paths:
        cases.append(case_from_leaves(path.stem, json.loads(path.read_text()), 'heldout' if path.stem in held else 'train'))
    assert len(cases) == 10
    print(json.dumps(dict(event='loaded_locked_leaves', cases=len(cases))), flush=True)
    # New geometry is generated by unchanged learned selector, not old menu leaves.
    selector = load_partition_runtime()
    rng = np.random.default_rng(SEED+777)
    for d, n in ((2, 12), (3, 12), (4, 12)):
        extent = np.exp(rng.uniform(-.35, .35, d))
        root = dict(d=d, n=n, lower=np.zeros(d).tolist(), upper=extent.tolist())
        name = f'fresh_continuous_d{d}_n{n}'
        leaves = partition(selector, root, name)
        dump(name+'_leaves.json', leaves)
        cases.append(case_from_leaves(name, leaves, 'fresh'))
        print(json.dumps(dict(event='fresh_frozen_partition', name=name, elapsed=time.monotonic()-start)), flush=True)
    dump('case_splits.json', [{k:c[k] for k in ('name', 'split', 'n', 'd')} for c in cases])
    traincases = [c for c in cases if c['split']=='train']
    metrics, training, summaries, coordinates = [], [], [], {}
    for hidden in (0, 8):
        model, history, summary = train(hidden, traincases)
        rows, failures, points = evaluate(model, cases, hidden)
        metrics.extend(rows); training.extend(history); coordinates.update(points)
        validation_failures = [r for r in failures if r['split'] != 'train']
        summary.update(passed=not validation_failures, failed_validation_conditions=len(validation_failures),
                       failures_by_gate={k:sum(k in r['failed_gates'].split(',') for r in validation_failures)
                                         for k in ('containment','finite','count','duplicates','boundary','history','repeat','coverage','holes','separation','discrepancy','freshness')})
        summaries.append(summary)
        print(json.dumps(dict(event='architecture_evaluated', **summary)), flush=True)
    csvwrite('placement_metrics.csv', metrics)
    csvwrite('training.csv', training)
    csvwrite('gate_failures.csv', [r for r in metrics if r.get('failed_gates')])
    np.savez_compressed(OUT/'screen_coordinates.npz', **coordinates)
    unchanged = all(sha(p)==v for p, v in protected.items())
    assert unchanged
    result = dict(status='PLACEMENT_SCREEN_PASSED_PENDING_DOWNSTREAM' if any(s['passed'] for s in summaries) else 'STOPPED_PLACEMENT_GATE_FAILURE',
                  architectures=summaries, protected_files=len(protected), protected_hashes_unchanged=unchanged,
                  elapsed_seconds=time.monotonic()-start, n1_ab_run=False, n2_n3_run=False,
                  checkpoint_promoted=False, integrated_pool_promoted=False)
    dump('results.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
