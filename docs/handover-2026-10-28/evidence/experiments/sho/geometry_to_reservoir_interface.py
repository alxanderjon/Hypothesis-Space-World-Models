"""Region-aware incidence adapter. No fitting or changes to proposal coordinates.

Region counts are hard equalities alongside the existing family bounds. Classes
are intersections of ALL memberships, so unchanged same-class exchanges preserve
the region counts as well as projection/dyadic counts.
"""
from dataclasses import dataclass
import itertools
import math

import numpy as np
from scipy.spatial import cKDTree

from experiments.sho import validate_joint_quota_assignment_teacher as fixed


@dataclass
class ReservoirState:
    points: np.ndarray
    region_ids: np.ndarray
    class_ids: np.ndarray
    incidence: np.ndarray
    capacities: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    regional_counts: np.ndarray
    metadata: list
    context: dict

    @property
    def budget(self):
        return int(self.regional_counts.sum())

    def solve(self, learned_priority):
        from experiments.sho.train_joint_quota_assignment_operator import operator
        return operator(learned_priority, self.incidence, self.capacities,
                        np.zeros(len(self.lower), dtype=np.int64),
                        self.lower, self.upper, self.budget)

    def verify(self, assignment):
        z = np.asarray(assignment)
        if z.dtype.kind not in 'iu' or z.shape != self.capacities.shape:
            raise ValueError('Assignment must be an integer vector over joint classes')
        counts = self.incidence @ z
        return dict(budget_error=int(z.sum()) - self.budget,
                    capacity_errors=int(np.count_nonzero((z < 0) | (z > self.capacities))),
                    regional_errors=int(np.count_nonzero(counts[:len(self.regional_counts)] != self.regional_counts)),
                    constraint_errors=int(np.count_nonzero((counts < self.lower) | (counts > self.upper))))

    def shortages(self):
        available = self.incidence @ self.capacities
        return [dict(row=int(i), **self.metadata[i], available=int(available[i]),
                     required=int(self.lower[i]), shortfall=int(self.lower[i]-available[i]))
                for i in np.flatnonzero(available < self.lower)]


def region_membership(points, lower, upper):
    """Half-open ownership, with the outer unit-cube upper face included.

    No nearest-region reassignment or coordinate clipping is permitted.
    """
    points, lower, upper = map(lambda a: np.asarray(a, dtype=np.float64), (points, lower, upper))
    if points.ndim != 2 or lower.ndim != 2 or upper.shape != lower.shape or points.shape[1] != lower.shape[1]:
        raise ValueError('Inconsistent dimension')
    if not all(np.isfinite(a).all() for a in (points, lower, upper)):
        raise ValueError('Nonfinite coordinates')
    if np.any(lower >= upper) or np.any(lower < 0) or np.any(upper > 1):
        raise ValueError('Regions must be positive boxes in normalized root coordinates')
    owner = np.full(len(points), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        member = np.all((points >= lo) & ((points < hi) | ((hi == 1) & (points == hi))), axis=1)
        if np.any(owner[member] != -1):
            raise ValueError('Overlapping region ownership')
        owner[member] = i
    if np.any(owner < 0):
        raise ValueError('Candidate outside the supplied recursive frontier')
    return owner


def connect(points, region_lower, region_upper, regional_counts, families, context):
    points = np.asarray(points, dtype=np.float64)
    counts = np.asarray(regional_counts)
    if counts.dtype.kind not in 'iu' or len(counts) != len(region_lower) or np.any(counts < 0):
        raise ValueError('Regional counts must be nonnegative integers, one per region')
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Duplicate candidates; input ordering/dedup policy must be preserved upstream')
    owner = region_membership(points, region_lower, region_upper)
    # Historical incidence arithmetic and bin ownership remain authoritative.
    family_I, meta, _ = fixed.incidence(points, families)
    family_lower, family_upper = fixed.bounds(meta, int(counts.sum()))
    regions = (np.arange(len(counts))[:, None] == owner).astype(np.int64)
    # Canonical bit-pattern ordering is the existing teacher's aggregation rule.
    incidence, capacities, class_ids = fixed.aggregate(np.vstack((regions, family_I)))
    metadata = [dict(kind='region', region_id=i, target=int(n)) for i, n in enumerate(counts)]
    metadata += [dict(kind='projection_dyadic', **m) for m in meta]
    return ReservoirState(points, owner, class_ids, incidence, capacities,
                          np.r_[counts, family_lower], np.r_[counts, family_upper],
                          counts.astype(np.int64), metadata, dict(context))


def capture_proposals(model, budget, iteration, pool):
    """Observe the existing online generator without altering any decisions."""
    if not hasattr(model, 'make_context'):
        raise TypeError('Proposal generator does not expose its context interface')
    original = model.make_context
    blocks = []
    def record(points, candidates, n, it, kind):
        blocks.append(candidates.copy())
        return original(points, candidates, n, it, kind)
    had_override = 'make_context' in model.__dict__
    model.make_context = record
    try:
        complete, base = model.generate(budget, iteration, pool)
    finally:
        if had_override:
            model.make_context = original
        else:
            del model.make_context
    return complete, base, np.concatenate(blocks)


def guarded_buffer(base, raw, budget):
    """Historical buffer algorithm with N supplied explicitly, unchanged order.

    This historical proposal/spacing policy is 2D. The incidence adapter above
    is dimension-agnostic; it does not pretend a 2D policy is validated in d>2.
    """
    if base.shape[1] != 2 or raw.shape[1] != 2:
        raise ValueError('Historical conditioned proposal/guard path supports dimension 2 only')
    radius = .4 / math.sqrt(budget)
    squared = .16 / budget
    candidates = raw[cKDTree(base).query(raw)[0] >= radius]
    grid, extra = {}, []
    offsets = list(itertools.product((-1, 0, 1), repeat=base.shape[1]))
    for q in candidates:
        key = tuple(np.floor(q / radius).astype(int))
        okay = True
        for delta in offsets:
            near = grid.get(tuple(k+d for k, d in zip(key, delta)), [])
            if near and min(np.sum((q-v)**2) for v in near) < squared:
                okay = False
                break
        if okay:
            extra.append(q)
            grid.setdefault(key, []).append(q)
    return np.vstack((base, np.asarray(extra).reshape(-1, base.shape[1])))
