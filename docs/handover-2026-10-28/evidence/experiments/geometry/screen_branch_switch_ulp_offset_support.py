"""Mandatory new-geometry offset-support screen before any model fitting."""
import json
import time
from collections import Counter
from pathlib import Path

import mpmath as mp
import numpy as np
import torch

from experiments.geometry import train_branch_location_residual as source
from experiments.geometry.audit_brent_branch_offsets import bits, read
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.sho.audit_fixed_allocation_intervals import Audit

OUT = ROOT / 'branch_switch_ulp_offset_training'
# Fixed before observing labels. Each batch generates nine unrelated parents.
SEEDS = (928202601, 928202602, 928202603)


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    mp.mp.dps = 80
    OUT.mkdir(exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p): sha(p) for p in ROOT.rglob('*.json')
                      if any(s in p.name for s in ('manifest', 'locked', 'validated'))})
    archive = read(ROOT / 'brent_branch_switch_offset_audit/brent_offsets_unique.csv')
    support = sorted({int(r['signed_k']) for r in archive})
    # Conservative support check: use the union of ALL known cases, before a
    # narrower train/validation split. Any failure also fails every subset.
    json_write(OUT / 'support_screen_plan.json', dict(
        seeds=SEEDS, known_unique_cases=len(archive), supported_offsets=support,
        support_policy='All 774 known geometry states; fresh labels never added to support',
        fresh_dimensions=[2, 3, 4], fresh_budgets=[7, 15, 23],
        maximum_fresh_parents=27, stop_on_first_batch_with_unsupported_offsets=True,
        teacher='Existing 129-interior-sample convention, locked availability, local Brent labels',
        no_training_before_support_gate=True, no_recursion=True))
    rows, parents, audits = [], [], []
    a = Audit.__new__(Audit)
    for batch, seed in enumerate(SEEDS):
        # Only the data-generation seed is changed in process. The imported
        # training entry point is never called and source files stay untouched.
        old_seed = source.SEED
        try:
            source.SEED = seed
            fresh, new_parents, audit = source.fresh_data()
        finally:
            source.SEED = old_seed
        parents.extend(new_parents)
        audits.append(audit)
        for r in fresh:
            r = dict(r, batch=batch, seed=seed)
            left, right = r['c_left'], r['c_right']
            b = float(np.clip(r['boundary'], left, right))
            offset = bits(r['teacher_cut']) - bits(b)
            a.setup(r['parent'], r['axis'])
            f = a.formula(np.array([left, right]), r['k'])
            r.update(projected_boundary=b, signed_k=offset,
                     unsupported=offset not in support,
                     projected_boundary_hex=b.hex(), teacher_cut_hex=float(r['teacher_cut']).hex(),
                     a=a.a, B=a.B, h=a.h, omin=a.omin, omax=a.omax,
                     competing_q=f[6].tolist(), competing_log_aspect=f[7].tolist(),
                     teacher_sample_derivatives=f[1].tolist(),
                     q_identities=f[3].tolist(), aspect_identities=f[4].tolist())
            rows.append(r)
        print(json.dumps(dict(batch=batch, parents=len(parents), positives=len(rows),
                              unsupported_offsets=sorted({r['signed_k'] for r in rows if r['unsupported']}))), flush=True)
        if audit['availability_errors'] or any(r['unsupported'] for r in rows):
            break
    failed = [r for r in rows if r['unsupported']]
    json_write(OUT / 'fresh_parents.json', parents)
    json_write(OUT / 'fresh_candidate_states.json', rows)
    json_write(OUT / 'unsupported_offset_cases.json', failed)
    json_write(OUT / 'availability_audits.json', audits)
    flat = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
            for r in rows]
    write_csv(OUT / 'fresh_offset_support.csv', flat)
    if failed:
        write_csv(OUT / 'unsupported_offset_cases.csv', [r for r in flat if r['unsupported']])
    breakdown = []
    for typ in ('all', 'q_only', 'aspect_only', 'simultaneous'):
        rr = rows if typ == 'all' else [r for r in rows if r['switch_type'] == typ]
        breakdown.append(dict(group=typ, cases=len(rr), unsupported_cases=sum(r['unsupported'] for r in rr),
                              observed_offsets=sorted({r['signed_k'] for r in rr})))
    assert all(sha(Path(p)) == digest for p, digest in protected.items())
    availability_errors = sum(len(x['availability_errors']) for x in audits)
    result = dict(status='STOP_UNSUPPORTED_FRESH_OFFSETS' if failed else
                  'STOP_AVAILABILITY_ERROR' if availability_errors else 'SUPPORT_SCREEN_PASSED',
                  known_cases=len(archive), known_offset_classes=len(support), supported_offsets=support,
                  fresh_parents=len(parents), fresh_cases=len(rows),
                  fresh_unique_inputs=len({json.dumps([r['parent'],r['axis'],r['k'],r['side'],r['c_left'],r['c_right']],sort_keys=True) for r in rows}),
                  new_unsupported_offsets=sorted({r['signed_k'] for r in failed}),
                  unsupported_case_count=len(failed),
                  unsupported_offset_counts=dict(sorted(Counter(r['signed_k'] for r in failed).items())),
                  breakdown=breakdown, availability_errors=availability_errors,
                  training_updates=0, checkpoint_created=False, downstream_replay_run=False,
                  existing_protected_files=len(protected), protected_hashes_unchanged=True,
                  elapsed_seconds=time.monotonic()-start)
    json_write(OUT / 'support_screen_result.json', result)
    json_write(OUT / 'protected_hashes.json', protected)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
