"""Read-only capacity checks on saved proposals; no generator/partition replay.

Internal-frontier checks are conditional examples, not selection of a new
frontier policy or whole-generator validation.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments.sho import validate_joint_quota_assignment_teacher as fixed
from experiments.sho.geometry_to_reservoir_interface import connect
from experiments.sho.train_joint_quota_assignment_operator import CanonicalPriority, operator
from experiments.sho.train_geometry_allocation_curriculum import GeometryAllocator, record, feature_tensor
from experiments.sho.hierarchical_allocation_poc import conserve
from experiments.sho.train_dyadic_spacefill_curriculum import PHASES

ROOT = Path('outputs/sho/stage1B')
SOURCE = ROOT/'geometry_to_reservoir_interface'
OUT = ROOT/'regional_reservoir_semantics_audit'


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    protected = list(ROOT.rglob('*.pt')) + list(Path('experiments').rglob('*.py'))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes = {str(p): sha(p) for p in protected}
    data = np.load(SOURCE/'N1000_iteration100_cell_state.npz')
    points = data['points']; n = int(data['regional_counts'].sum()); d = points.shape[1]
    splits = json.loads((SOURCE/'N1000_splits.json').read_text())
    priority = CanonicalPriority()
    priority.load_state_dict(torch.load(ROOT/'joint_quota_assignment_operator/joint_quota_assignment_operator.pt', weights_only=False)['state_dict'])
    priority.eval().requires_grad_(False)
    allocation = GeometryAllocator()
    allocation.load_state_dict(torch.load(ROOT/'geometry_allocation_curriculum/stage_4_best.pt', weights_only=False)['state_dict'])
    allocation.eval().requires_grad_(False)
    families = [(i, tuple(f)) for i, (_, _, fs) in enumerate(PHASES) for f in fs]
    I, meta, _ = fixed.incidence(points, families)
    lower, upper = fixed.bounds(meta, n)
    J, cap, ids = fixed.aggregate(I)
    z = operator(priority, J, cap, np.zeros(len(lower), int), lower, upper, n)
    rows = [dict(name='established_projection_dyadic_families', feasible=z is not None,
                 families=len(families), classes=len(cap), budget=n,
                 count_errors=None if z is None else int(np.count_nonzero((J@z<lower)|(J@z>upper))))]
    witnesses = {}
    if z is not None:
        witnesses[rows[-1]['name']] = dict(assignment=z.tolist(), realized_counts=(J@z).tolist())

    # Historical depth-1 / depth-2 strip quota semantics. Each family is checked
    # independently just as in the POC; it is already included in PHASES.
    for axis in range(d):
        for depth in (1, 2):
            regions = [(np.zeros(d), np.ones(d), n)]
            for _ in range(depth):
                following = []
                for lo, hi, budget in regions:
                    cut = (lo[axis]+hi[axis])/2
                    ahi=hi.copy(); ahi[axis]=cut
                    blo=lo.copy(); blo[axis]=cut
                    boxes=[(lo,ahi),(blo,hi)]
                    unit=[((a-lo)/(hi-lo),(b-lo)/(hi-lo)) for a,b in boxes]
                    r=record(unit,budget,'historical_strip',d,'audit')
                    f,mask,_=feature_tensor([r])
                    with torch.no_grad(): fractions=allocation(f,mask).numpy()[0,:2]
                    counts=conserve(budget,fractions)
                    following.extend((a,b,int(c)) for (a,b),c in zip(boxes,counts))
                regions=following
            state=connect(points,[a for a,b,c in regions],[b for a,b,c in regions],
                          np.array([c for a,b,c in regions]),families,dict(iteration=100,pool='cell'))
            q=state.solve(priority)
            item=dict(name=f'historical_strip_axis{axis}_depth{depth}',semantic_scope='historically defined stratum family',
                      regions=len(regions),targets=state.regional_counts.tolist(),feasible=q is not None,
                      shortages=state.shortages(),checks=None if q is None else state.verify(q))
            rows.append(item)
            if q is not None: witnesses[item['name']]=dict(assignment=q.tolist(), realized_counts=(state.incidence@q).tolist())

    # Inspect actual saved internal nodes only. Do not regenerate the recursive
    # tree, change its stopping rule, or designate a passing depth as policy.
    for depth in (1,2):
        nodes=sorted([x for x in splits if len(x['path'])-1==depth],key=lambda x:x['path'])
        parents=[x['parent'] for x in nodes]
        state=connect(points,[p['lower'] for p in parents],[p['upper'] for p in parents],
                      np.array([p['n'] for p in parents]),families,dict(iteration=100,pool='cell'))
        q=state.solve(priority)
        item=dict(name=f'actual_saved_internal_frontier_depth{depth}',semantic_scope='conditional diagnostic; frontier policy not established',
                  paths=[x['path'] for x in nodes],targets=state.regional_counts.tolist(),feasible=q is not None,
                  shortages=state.shortages(),checks=None if q is None else state.verify(q))
        rows.append(item)
        if q is not None: witnesses[item['name']]=dict(assignment=q.tolist(), realized_counts=(state.incidence@q).tolist())
    changes=[p for p,h in hashes.items() if sha(p)!=h]
    assert not changes
    result=dict(status='SEMANTICS_AUDIT_ONLY',budget=n,dimension=d,reservoir_size=len(points),rows=rows,
                no_training=True,no_replenishment=True,no_generator_validation=True,
                correct_arbitrary_recursive_frontier_not_specified_by_history=True,
                protected_files=len(hashes),protected_changes=changes,runtime_seconds=time.monotonic()-start)
    (OUT/'capacity_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'feasible_witnesses.json').write_text(json.dumps(witnesses,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
