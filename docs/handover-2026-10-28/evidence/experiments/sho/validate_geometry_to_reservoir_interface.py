"""Stop-at-first-failure integration screen; all learned components are read-only."""
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.sho.geometry_to_reservoir_interface import connect, capture_proposals, guarded_buffer
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator
from experiments.sho.train_joint_quota_assignment_operator import CanonicalPriority
from experiments.sho.train_dyadic_spacefill_curriculum import PHASES
from experiments.sho.train_progressive_spacefill_smoke import BUDGETS

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'geometry_to_reservoir_interface'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def full_partition(selector, budget):
    """Execute every split through the locked selector; retain its child counts."""
    root = dict(d=2, n=budget, lower=[0., 0.], upper=[1., 1.])
    stack, leaves, splits = [('r', root)], [], []
    while stack:
        path, p = stack.pop()
        if p['n'] == 1:
            leaves.append(dict(path=path, parent=p))
            continue
        _, got, _ = selector.select(p, f'reservoir_interface_N{budget}:{path}')
        assert sum(got['counts']) == p['n'] and min(got['counts']) > 0
        splits.append(dict(path=path, parent=p, selected=got))
        if len(splits) % 100 == 0:
            print('partition', budget, 'splits', len(splits), 'leaves', len(leaves), flush=True)
        for j in (1, 0):
            child = dict(d=p['d'], n=got['counts'][j], lower=got['child_lower'][j], upper=got['child_upper'][j])
            stack.append((path + str(j), child))
    assert len(leaves) == budget
    return leaves, splits


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    protected_paths = list(ROOT.rglob('*.pt')) + list(Path('experiments').rglob('*.py'))
    protected = {str(p): sha(p) for p in protected_paths}
    result = dict(status='RUNNING', requested_budgets=list(BUDGETS), rows=[],
                  no_training=True, downstream_n1_run=False,
                  dimensions_of_historical_online_generator=[2],
                  generic_interface_unit_test_dimensions=[1, 2, 3, 4])
    try:
        from experiments.sho.conditioned_leaf_placement_screen import load_partition_runtime
        selector = load_partition_runtime()
        proposal = ProgressiveGenerator()
        proposal.load_state_dict(torch.load(ROOT/'dyadic_spacefill_curriculum/converged_continuation/phase_1_halves_best.pt', weights_only=False)['state_dict'])
        proposal.eval().requires_grad_(False)
        priority = CanonicalPriority()
        priority.load_state_dict(torch.load(ROOT/'joint_quota_assignment_operator/joint_quota_assignment_operator.pt', weights_only=False)['state_dict'])
        priority.eval().requires_grad_(False)
        families = [(i, tuple(f)) for i, (_, _, fs) in enumerate(PHASES) for f in fs]
        for budget in BUDGETS:
            begin = time.monotonic()
            leaves, splits = full_partition(selector, budget)
            dump(OUT/f'N{budget}_leaves.json', leaves)
            dump(OUT/f'N{budget}_splits.json', splits)
            lo = np.array([p['parent']['lower'] for p in leaves])
            hi = np.array([p['parent']['upper'] for p in leaves])
            counts = np.array([p['parent']['n'] for p in leaves], dtype=np.int64)
            for iteration, pool in ((100, 'cell'), (12000, 'cell'), (12000, 'greedy')):
                context = dict(budget=budget, iteration=iteration, pool=pool)
                print('reservoir', context, flush=True)
                with torch.no_grad():
                    complete, base, raw = capture_proposals(proposal, budget, iteration, pool)
                    replay_complete, replay_base, replay_raw = capture_proposals(proposal, budget, iteration, pool)
                assert np.array_equal(base, replay_base) and np.array_equal(raw, replay_raw)
                assert np.array_equal(complete, replay_complete)
                buffer = guarded_buffer(base, raw, budget)
                assert np.array_equal(buffer, guarded_buffer(replay_base, replay_raw, budget))
                state = connect(buffer, lo, hi, counts, families, context)
                replay = connect(buffer, lo, hi, counts, families, context)
                assert np.array_equal(state.incidence, replay.incidence)
                assert np.array_equal(state.class_ids, replay.class_ids)
                tag = f'N{budget}_iteration{iteration}_{pool}'
                np.savez_compressed(OUT/f'{tag}_state.npz', points=buffer, raw=raw, base=base,
                                    region_lower=lo, region_upper=hi, regional_counts=counts,
                                    region_ids=state.region_ids, class_ids=state.class_ids,
                                    incidence=state.incidence, capacities=state.capacities,
                                    lower=state.lower, upper=state.upper)
                dump(OUT/f'{tag}_constraints.json', state.metadata)
                z = state.solve(priority)
                row = dict(**context, recursive_splits=len(splits), leaves=len(leaves),
                           raw_proposals=len(raw), reservoir_size=len(buffer),
                           joint_classes=len(state.capacities), constraint_rows=len(state.lower),
                           regional_budget_error=int(counts.sum())-budget,
                           finite=bool(np.isfinite(buffer).all()), duplicates=len(buffer)-len(np.unique(buffer, axis=0)),
                           deterministic_replay=True, feasible=z is not None,
                           shortages=state.shortages(), seconds=time.monotonic()-begin)
                if z is None:
                    row['failure'] = 'fixed reservoir cannot satisfy joint regional and projection/dyadic counts'
                    result['rows'].append(row)
                    result['status'] = 'STOPPED_INTERFACE_CAPACITY_OR_JOINT_FEASIBILITY_FAILURE'
                    result['first_failed_stage'] = 'regional_joint_quota_feasibility'
                    break
                row['assignment_checks'] = state.verify(z)
                assert not any(row['assignment_checks'].values())
                np.save(OUT/f'{tag}_assignment.npy', z)
                result['rows'].append(row)
                # Full downstream path is intentionally gated on interface validation.
            if result['status'].startswith('STOPPED'):
                break
        if result['status'] == 'RUNNING':
            result['status'] = 'INTERFACE_PASSED_PENDING_UNCHANGED_REALIZER_REPLAY'
    except Exception as exc:
        result.update(status='STOPPED_RUNTIME_DEPENDENCY_FAILURE', exception=repr(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        changed = [p for p, h in protected.items() if sha(p) != h]
        result.update(protected_files=len(protected), protected_changes=changed,
                      runtime_seconds=time.monotonic()-started)
        dump(OUT/'results.json', result)
        dump(OUT/'protected_hashes.json', protected)
        print(json.dumps(result, indent=2), flush=True)
        assert not changed


if __name__ == '__main__':
    main()
