"""Deterministic, feasibility-constrained geometric frontier; no learned changes.

Only existing tree children can be activated. Family cell widths supply the
resolution request, not a fixed depth. Count-one nodes are never activated.
"""
import json
from pathlib import Path
import numpy as np
from experiments.sho.geometry_to_reservoir_interface import connect
from experiments.sho.validate_joint_quota_assignment_teacher import solve


def read_tree(splits, leaves):
    nodes = {r['path']: r['parent'] for r in splits + leaves}
    for row in splits:
        path, parent, selected = row['path'], row['parent'], row['selected']
        children = [nodes[path+str(j)] for j in (0, 1)]
        assert sum(p['n'] for p in children) == parent['n']
        for j, child in enumerate(children):
            assert child['n'] == selected['counts'][j]
            assert child['lower'] == selected['child_lower'][j]
            assert child['upper'] == selected['child_upper'][j]
    return nodes


def select_frontier(nodes, points, families, priority, context, trace_path):
    root = nodes['r']; budget = root['n']; d = root['d']
    origin = np.asarray(root['lower']); extent = np.asarray(root['upper']) - origin
    assert points.shape[1] == d and budget >= 2
    active = ['r']; frozen_counts = {}; trace = []; levels = []

    def build(paths):
        region = [nodes[p] for p in paths]
        lo = (np.array([p['lower'] for p in region])-origin)/extent
        hi = (np.array([p['upper'] for p in region])-origin)/extent
        state = connect(points, lo, hi, np.array([p['n'] for p in region]), families, context)
        offset = len(paths)
        for i, target in frozen_counts.items():
            state.lower[offset+i] = state.upper[offset+i] = target
        return state

    state = build(active); plan = state.solve(priority)
    if plan is None:
        raise ValueError('Historical quota constraints are infeasible even at root frontier')
    family_meta = state.metadata[1:]
    prior_rows = {}
    for tier in sorted(set(t for t, _ in families)):
        phase_families = [tuple(f) for t, f in families if t == tier]
        for family in phase_families:
            f = np.asarray(family)
            assert len(f) == d
            cursor = 0
            while cursor < len(active):
                path = active[cursor]; node = nodes[path]
                span = (np.asarray(node['upper'])-node['lower'])/extent
                needs_refinement = bool(np.any(span*f > 1))
                event = dict(event=len(trace), tier=int(tier), family=list(family), node=path,
                             node_count=int(node['n']), span=span.tolist(), active_before=active.copy(),
                             frozen_family_counts={str(i):int(v) for i,v in frozen_counts.items()})
                if not needs_refinement:
                    event['decision'] = 'KEEP_FAMILY_RESOLUTION'
                elif path+'0' not in nodes or path+'1' not in nodes:
                    event['decision'] = 'KEEP_NO_CHILDREN'
                else:
                    child_paths = [path+'0', path+'1']
                    counts = [int(nodes[p]['n']) for p in child_paths]
                    event['proposed_children'] = child_paths; event['child_counts'] = counts
                    assert sum(counts) == node['n']
                    if min(counts) < 2:
                        event['decision'] = 'KEEP_MULTIPROBE_PARENT'
                    else:
                        trial = sorted(active[:cursor] + child_paths + active[cursor+1:])
                        candidate = build(trial)
                        z = solve(np.zeros(len(candidate.capacities)),
                                  np.vstack((np.ones(len(candidate.capacities), dtype=np.int64), candidate.incidence)),
                                  np.r_[budget, candidate.lower], np.r_[budget, candidate.upper],
                                  np.zeros(len(candidate.capacities), dtype=np.int64), candidate.capacities)
                        event['joint_feasible'] = z is not None
                        event['shortages'] = candidate.shortages()
                        if z is None:
                            event['decision'] = 'KEEP_INCOMPATIBLE_PARENT'
                        else:
                            checks = candidate.verify(z); assert not any(checks.values())
                            event.update(decision='ACTIVATE_CHILDREN', assignment=z.tolist(), checks=checks)
                            active, state, plan = trial, candidate, z
                            # Visit the new left child next; no depth cutoff.
                            trace.append(event)
                            continue
                trace.append(event); cursor += 1
        plan = state.solve(priority)
        assert plan is not None
        realized = state.incidence @ plan
        family_counts = realized[len(active):]
        retention_errors = sum(int(family_counts[i] != value) for i, value in prior_rows.items())
        assert retention_errors == 0
        for i, meta in enumerate(family_meta):
            if meta['tier'] <= tier:
                frozen_counts[i] = int(family_counts[i])
        prior_rows = frozen_counts.copy()
        levels.append(dict(tier=int(tier), families=[list(f) for f in phase_families],
                           active_regions=len(active), active_paths=active.copy(),
                           counts=[int(nodes[p]['n']) for p in active],
                           budget=int(sum(nodes[p]['n'] for p in active)),
                           min_count=int(min(nodes[p]['n'] for p in active)),
                           coarse_retention_errors=retention_errors,
                           committed_family_rows=len(frozen_counts)))
    # Reconstruct final fixed rows, without changing the plan.
    state = build(active)
    assert not any(state.verify(plan).values())
    header = dict(type='initial',context=context,tree=nodes,families=families,
                  rule='family widths -> compatible child refinement; min child count two; ordered family/tier then binary path',
                  candidate_state_ref=Path(trace_path).with_suffix('.npz').name)
    with Path(trace_path).open('w') as stream:
        stream.write(json.dumps(header)+'\n')
        for e in trace: stream.write(json.dumps(e)+'\n')
        stream.write(json.dumps(dict(type='final',active=active,levels=levels,assignment=plan.tolist()))+'\n')
    np.savez_compressed(Path(trace_path).with_suffix('.npz'),points=points,class_ids=state.class_ids,
                        incidence=state.incidence,capacities=state.capacities,lower=state.lower,upper=state.upper)
    return state, plan, active, levels, trace
