"""Separate geometry-only block policy. No sequence/partition coordinates.

Uniform random proposals are selected by a learned policy using prefix occupancy,
anchored-count residual fields and nearest-prefix distances. Grid cells describe
occupancy only: their centers/corners are never interior point proposals.
"""
import hashlib
import itertools
import math
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

SCALES = (2, 4, 8, 16, 32, 64, 128)
SHIFTS = (0., .5)
SEED = 27102026


def boundary_faces(n, dimension):
    # Same endpoint-inclusive generic face rule as the existing generators.
    k = max(2, math.ceil(n ** (1 / dimension)))
    mesh = np.array(list(itertools.product(np.linspace(0., 1., k), repeat=dimension-1)))
    result = []
    for axis in range(dimension):
        free = [j for j in range(dimension) if j != axis]
        for side in (0., 1.):
            block = np.empty((len(mesh), dimension))
            block[:, axis] = side
            block[:, free] = mesh
            result.append(block)
    return np.unique(np.concatenate(result), axis=0)


def discrepancy_rows(points):
    n = len(points)
    rows = []
    for k in SCALES:
        edge = np.arange(1, k+1)/k
        for shift in (0., .25, .5, .75):
            idx = np.minimum(((points + shift/k) % 1 * k).astype(int), k-1)
            count = np.bincount(idx[:, 0]*k+idx[:, 1], minlength=k*k).reshape(k, k)
            residual = count.cumsum(0).cumsum(1)/n-edge[:, None]*edge[None, :]
            rows.append(dict(bins_per_axis=k, shift_fraction_of_bin=shift,
                grid_anchored_discrepancy=float(np.abs(residual).max()),
                grid_anchored_RMSE=float(np.sqrt(np.mean(residual**2))),
                count_variance=float(count.var()), count_fano=float(count.var()/count.mean()),
                empty_bin_fraction=float(np.mean(count == 0))))
    return rows


class ProgressiveGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 fields x 2 shifts x 7 scales, distance, progress, N/iteration/pool.
        self.net = nn.Sequential(nn.Linear(34, 32), nn.Tanh(), nn.Linear(32, 1)).double()

    def make_context(self, points, candidates, n, iteration, pool):
        m = len(points)
        fields, grids = [], []
        for k in SCALES:
            edge = np.arange(1, k+1)/k
            area = edge[:, None]*edge[None, :]
            for shift in SHIFTS:
                idx = np.minimum(((candidates+shift/k) % 1*k).astype(int), k-1)
                flat = idx[:, 0]*k+idx[:, 1]
                if m:
                    pi = np.minimum(((points+shift/k) % 1*k).astype(int), k-1)
                    count = np.bincount(pi[:, 0]*k+pi[:, 1], minlength=k*k).reshape(k, k)
                else:
                    count = np.zeros((k, k))
                residual = count.cumsum(0).cumsum(1)-m*area
                # Integrated discrepancy gradient, evaluated at proposals.
                field = residual[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]/(k*k)
                fields.extend(((count.ravel()[flat]-m/(k*k))/math.sqrt(1+m/(k*k)),
                               field.ravel()[flat]/math.sqrt(1+m)))
                grids.append((torch.from_numpy(flat), torch.from_numpy(count.copy()).double(),
                              torch.from_numpy(area), k))
        nearest = cKDTree(points).query(candidates)[0] if m else np.full(len(candidates), 2/math.sqrt(n))
        fields.extend((np.minimum(nearest*math.sqrt(n), 4), np.full(len(candidates), m/n),
                       np.full(len(candidates), math.log(n)/12),
                       np.full(len(candidates), math.sin(iteration)),
                       np.full(len(candidates), math.cos(iteration)),
                       np.full(len(candidates), float(pool == 'greedy'))))
        return torch.from_numpy(np.stack(fields, axis=1)), grids, nearest

    def losses(self, logits, grids, nearest, n, m, b):
        # Soft block expectation. Each spatial scale contributes equally; the
        # smooth maximum prevents a coarse scale hiding a poor fine scale.
        mass = b*torch.softmax(logits/.7, dim=0)
        terms = []
        for flat, count, area, k in grids:
            added = torch.zeros(k*k, dtype=torch.float64).scatter_add(0, flat, mass).reshape(k, k)
            residual = count+added-(m+b)/(k*k)
            prior = count-m/(k*k)
            count_change = ((residual**2).sum()-(prior**2).sum())/b
            cdf = (count+added).cumsum(0).cumsum(1)-(m+b)*area
            old_cdf = count.cumsum(0).cumsum(1)-m*area
            cdf_change = ((cdf**2).mean()-(old_cdf**2).mean())/b
            terms.append(count_change+.5*cdf_change)
        primary = torch.logsumexp(torch.stack(terms)*2, 0)/2
        # Hole-filling proxy, secondary gradient norm strictly bounded by 10%
        # of primary and projected to not oppose its descent direction.
        secondary = -(mass*torch.from_numpy(np.minimum(nearest*math.sqrt(n), 4)**2)).sum()/b
        return primary, secondary

    def generate(self, n, iteration, pool_type, optimizer=None, history=None):
        if n <= 0 or iteration < 0 or pool_type not in ('cell', 'greedy'):
            raise ValueError('invalid generator context')
        key = f'{SEED}:2:{n}:{iteration}:{pool_type}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        block_size = 32 if n <= 1000 else 128 if n <= 8600 else 256
        points = np.empty((0, 2))
        while len(points) < n:
            m = len(points); b = min(block_size, n-m)
            candidates = rng.random((max(256, 8*b), 2))
            features, grids, nearest = self.make_context(points, candidates, n, iteration, pool_type)
            logits = self.net(features).flatten().clamp(-8, 8)
            valid = nearest >= .4/math.sqrt(n)
            logits = logits.masked_fill(torch.from_numpy(~valid), -100)
            if optimizer is not None:
                primary, secondary = self.losses(logits, grids, nearest, n, m, b)
                parameters = list(self.parameters())
                gp = torch.autograd.grad(primary, parameters, retain_graph=True)
                gs = torch.autograd.grad(secondary, parameters)
                dot = sum((a*c).sum() for a, c in zip(gp, gs))
                norm = sum((a*a).sum() for a in gp)+1e-30
                if dot < 0:
                    gs = tuple(c-dot/norm*a for a, c in zip(gp, gs))
                secnorm = sum((c*c).sum() for c in gs)+1e-30
                factor = min(1., float(.1*torch.sqrt(norm/secnorm)))
                optimizer.zero_grad()
                for p, a, c in zip(parameters, gp, gs):
                    p.grad = a+factor*c
                torch.nn.utils.clip_grad_norm_(parameters, 5.)
                optimizer.step()
                if history is not None:
                    history.append(dict(N=n, iteration=iteration, pool_type=pool_type,
                        prefix_count=m, block_count=b, primary=float(primary.detach()),
                        secondary=float(secondary.detach()), secondary_gradient_factor=factor))
            # Fresh deterministic stochastic policy. No bin-center coordinates;
            # a small geometric exclusion prevents within-block collapse.
            score = logits.detach().numpy()+.7*rng.gumbel(size=len(candidates))
            accepted = []
            for j in np.argsort(-score, kind='stable'):
                if not valid[j]:
                    continue
                if accepted and np.min(np.sum((candidates[accepted]-candidates[j])**2, axis=1)) < .16/n:
                    continue
                accepted.append(j)
                if len(accepted) == b:
                    break
            if not accepted:
                raise RuntimeError('no admissible random block proposals')
            points = np.vstack((points, candidates[accepted]))
        boundary = boundary_faces(n, 2)
        complete = np.unique(np.concatenate((points, boundary)), axis=0)
        assert len(complete) == n+len(boundary)
        return complete, points
