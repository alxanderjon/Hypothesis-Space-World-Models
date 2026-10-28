"""Two generic sampling policies; original deterministic E-DIRECT cell/greedy logic.

SciPy Sobol is the production oracle already recovered bit-exactly by the
locked Sobol assembly. No fitting or checkpoint writes occur here.
"""
from __future__ import annotations
import math
import types
import numpy as np
from scipy.special import ndtri
from scipy.spatial import cKDTree
from scipy.stats import qmc
from experiments.sho import run_stage1b_edirect_vs_direct as teacher
from experiments.sho.run_full_k_transfer import initial_design


def directions(d):
    if d < 1:
        raise ValueError('positive dimension required')
    if d == 1:
        return np.array([[1.], [-1.]])
    if d == 2:
        angles = np.linspace(0., 2.*math.pi, 24, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    # A rotationally symmetric Gaussian normalized to S^(d-1). Scrambled
    # Sobol gives a deterministic low-discrepancy source; explicit antipodes
    # cancel directional bias. Linear count 12*d, preserving 24 at d=2.
    count = 6*d
    u = qmc.Sobol(d, scramble=True, seed=teacher.SEARCH_SEED).random_base2(
        int(math.ceil(math.log2(count))))[:count]
    z = ndtri(np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.)))
    z /= np.linalg.norm(z, axis=1)[:, None]
    return np.stack((z, -z), axis=1).reshape(-1, d)


def boundary(d, requested, seed, lower=None, upper=None):
    lower = np.zeros(d) if lower is None else np.asarray(lower, float)
    upper = np.ones(d) if upper is None else np.asarray(upper, float)
    if lower.shape != (d,) or upper.shape != (d,) or not np.all(upper > lower):
        raise ValueError('invalid bounds')
    count = min(257, max(9, int(math.ceil(math.sqrt(requested)))))
    if d == 1:
        unit = np.array([[0.], [1.]])
        ids = np.array([[0, 0], [0, 1]])
    else:
        # Historical 2D raw order: bottom, top, left, right. Generic reverse
        # axis order, lower then upper, recovers this without axis identities.
        free = (np.linspace(0., 1., count)[:, None] if d == 2 else
                qmc.Sobol(d-1, scramble=True, seed=seed).random_base2(
                    int(math.ceil(math.log2(count))))[:count])
        blocks = []; labels = []
        for axis in reversed(range(d)):
            for side in (0, 1):
                block = np.empty((count, d)); block[:, axis] = side
                block[:, np.arange(d) != axis] = free
                blocks.append(block); labels.extend([(axis, side)]*count)
        unit = np.concatenate(blocks); ids = np.asarray(labels)
    return lower + (upper-lower)*unit, ids


def pool(d, sample_count, trajectory_id, iteration):
    # Original budget and seed: no dimension-dependent pool inflation.
    n = min(teacher.MC_FACTOR_PER_SITE*sample_count, teacher.MC_CANDIDATE_CAP)
    seed = teacher.SEARCH_SEED+104729*trajectory_id+1009*iteration
    interior = qmc.Sobol(d, scramble=True, seed=seed).random_base2(
        int(math.ceil(math.log2(max(2, n)))))[:n]
    faces, _ = boundary(d, n, seed)
    return np.unique(np.concatenate((interior, faces)), axis=0)


def local_greedy(points, losses):
    nearest = (cKDTree(points).query(points, k=2)[0][:, 1] if len(points)>1
               else np.ones(len(points)))
    dirs = directions(points.shape[1]); candidates = []
    for index in np.argsort(losses)[:min(12, len(points))]:
        radius = max(float(nearest[index]), 1e-5)
        for fraction in (.5, .25, .125, .0625):
            candidates.append(np.clip(points[index][None, :]+fraction*radius*dirs, 0., 1.))
    return np.concatenate(candidates) if candidates else np.empty((0, points.shape[1]))


def composition(d):
    # Private function globals: no monkey-patching or modifying teacher sources.
    ns = dict(teacher.__dict__)
    ns['_candidate_pool'] = lambda m, case, iteration: pool(d, m, case, iteration)
    ns['_local_greedy_candidates'] = local_greedy
    for name in ('_cell_data', '_cell_proposals', '_greedy_proposal'):
        f = getattr(teacher, name)
        ns[name] = types.FunctionType(f.__code__, ns, name, f.__defaults__, f.__closure__)
    return ns


def search(objective, lower, upper, case_id, cap=5000, gate=1e-4):
    lower = np.asarray(lower, float); upper = np.asarray(upper, float); d = len(lower)
    api = composition(d); points = []; losses = []; events = []; rounds = 0
    def evaluate(u, source):
        value = float(objective(lower+(upper-lower)*u))
        if not np.isfinite(value):
            raise ValueError('nonfinite objective')
        points.append(u.copy()); losses.append(value)
        events.append(dict(evaluation=len(losses),iteration=rounds,source=source,
                           point=u.tolist(),loss=value,boundary=bool(np.any((u==0)|(u==1)))))
    # Reuse existing full-K dimensional interface, preserving original 2D design.
    design = teacher._initial_design() if d==2 else initial_design(d)
    for u in design:
        if len(losses)>=cap: break
        evaluate(u, 'initial')
        if min(losses)<=gate: break
    while len(losses)<cap and min(losses)>gate:
        before = len(losses)
        proposals = api['_cell_proposals'](np.asarray(points), np.asarray(losses), case_id, rounds)
        for u in proposals:
            if len(losses)>=cap: break
            evaluate(u, 'cell')
            if min(losses)<=gate: break
        if len(losses)<cap and min(losses)>gate:
            u = api['_greedy_proposal'](np.asarray(points), np.asarray(losses), case_id, rounds)
            if u is not None: evaluate(u, 'greedy')
        rounds += 1
        if len(losses)==before: break
    best = int(np.argmin(losses)); theta = lower+(upper-lower)*points[best]
    # Existing dimension-general full-K interface, not a new N2 box policy.
    radius = (upper-lower)/len(losses)**(1/d)
    return dict(theta=theta,loss=losses[best],evaluations=len(losses),iterations=rounds,
                points=np.asarray(points),values=np.asarray(losses),events=events,
                lower=np.maximum(lower,theta-radius),upper=np.minimum(upper,theta+radius),
                success=bool(losses[best]<=gate))
