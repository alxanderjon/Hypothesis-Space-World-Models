"""Read-only saved-state audit of Brent step application; no solver execution."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.geometry import train_division_operator as source
from experiments.geometry.brent_interpolation_acceptance import BrentInterpolationAcceptance
from experiments.geometry.validate_locked_brent_linear import read, write, dump, bits

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'brent_step_application_isolation_audit'


def teacher(xcur, scur, sbis, delta):
    # Exact source order, including the strict sign test rather than copysign.
    raw = abs(scur) > delta
    applied = scur if raw else (delta if sbis > 0 else -delta)
    return raw, applied, xcur + applied


def run():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    protected = {str(p): source.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p): source.sha(p) for p in ROOT.rglob('*.json')
                      if any(t in p.name for t in ('locked', 'manifest', 'validated'))})
    protected.update({str(p): source.sha(p) for p in Path('experiments/geometry').glob('*.py')})
    input_paths = [
        ROOT/'exact_branch_switch_brent_audit/brent_iterations.csv',
        ROOT/'brent_interpolation_eligibility_training/solve_splits.csv',
        ROOT/'brent_acceptance_isolation_audit/acceptance_decisions.csv',
        ROOT/'brent_fallback_isolation_audit/fallback_events.csv',
        ROOT/'brent_inverse_quadratic_assembly_validation/solve_replay.csv',
    ]
    protected.update({str(p): source.sha(p) for p in input_paths})
    dump(OUT/'protected_hashes.json', protected)
    model = BrentInterpolationAcceptance()
    splits = {int(r['case_id']): r['group'] for r in read(input_paths[1])}
    events = []
    for r in read(input_paths[0]):
        if r['stop'] == 'True':
            continue
        state = json.loads(r['after_proposal'])
        method = r['effective_method']
        row = dict(solve_id=int(r['case_id']), iteration=int(r['iteration']),
                   split=splits[int(r['case_id'])], xcur=state['xcur'], scur=state['scur'],
                   sbis=float(r['sbis']), delta=float(r['delta']), xblk=state['xblk'],
                   proposal_source='FALLBACK' if method == 'BISECTION' else
                   'LINEAR_ACCEPTED' if method == 'LINEAR_INTERPOLATION' else 'IQ_ACCEPTED',
                   query_provenance='persisted complete teacher trace',
                   persisted_next_x=float(r['query']))
        raw, step, nxt = teacher(*(row[k] for k in ('xcur', 'scur', 'sbis', 'delta')))
        assert bits(nxt) == bits(row['persisted_next_x'])
        assert (not raw) == (r['minimum_step'] == 'True')
        assert bits(state['scur']) == bits(float(r['selected_step']))
        events.append(row)
    # Saved additional-fresh states: materialize labels without running Brent.
    for r in read(input_paths[2]):
        if r['split'] != 'additional_fresh':
            continue
        accepted = r['label'] == 'ACCEPT'
        events.append(dict(solve_id=int(r['solve_id']), iteration=int(r['iteration']),
            split=r['split'], xcur=float(r['xcur']), xblk=float(r['xblk']),
            scur=float(r['stry'] if accepted else r['sbis']), sbis=float(r['sbis']), delta=float(r['delta']),
            proposal_source=('LINEAR_ACCEPTED' if r['interpolation_branch'] == 'LINEAR' else 'IQ_ACCEPTED') if accepted else 'FALLBACK',
            query_provenance='saved fresh proposal state; native audited expression label', persisted_next_x=None))
    for r in read(input_paths[3]):
        if r['split'] != 'additional_fresh' or r['reason'] != 'NOT_ELIGIBLE':
            continue
        events.append(dict(solve_id=int(r['case_id']), iteration=int(r['iteration']), split=r['split'],
            **{k: float(r[k]) for k in ('xcur', 'xblk', 'sbis', 'delta')}, scur=float(r['selected_step']),
            proposal_source='FALLBACK', query_provenance='saved recovered fresh fallback state; native audited expression label',
            persisted_next_x=None))
    events.sort(key=lambda r: (r['solve_id'], r['iteration']))
    counts = Counter(r['solve_id'] for r in events)
    replay = read(input_paths[4])
    assert len(events) == 29062 and len(counts) == 875
    assert len({(r['solve_id'], r['iteration']) for r in events}) == len(events)
    for r in replay:
        assert counts[int(r['case_id'])] == int(r['iteration_queries'])

    # Broad boundary controls are algebraic unit tests, not claimed reachable
    # solver states. In particular sbis=0 normally terminates before application.
    deltas = np.unique(np.array([r['delta'] for r in events]))
    selected = deltas[np.linspace(0, len(deltas)-1, min(64, len(deltas))).astype(int)]
    selected = np.unique(np.r_[selected, np.ldexp(1., [-1074, -1022, -100, -52, 0, 100, 600])])
    controls = []
    for d in selected:
        for mag in (np.nextafter(d, 0.), d, np.nextafter(d, np.inf)):
            for sg in (1., -1.):
                for sb in (float(d), -float(d), 0., -0.):
                    for xc in (0., -0., .25, -.25):
                        controls.append(dict(xcur=xc, scur=float(sg*mag), sbis=sb, delta=float(d),
                                             split='boundary_control', proposal_source='CONTROL'))
    for d in (0., -0.):
        for sc in (0., -0., np.nextafter(0., 1.), np.nextafter(0., -1.)):
            for sb in (0., -0., 1., -1.):
                for xc in (0., -0.):
                    controls.append(dict(xcur=xc, scur=float(sc), sbis=sb, delta=d,
                                         split='zero_control', proposal_source='CONTROL'))

    metrics = []
    def screen(rows, scope, context=False):
        a = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in ('xcur', 'scur', 'sbis', 'delta')}
        native = [teacher(*(float(r[k]) for k in ('xcur', 'scur', 'sbis', 'delta'))) for r in rows]
        raw = np.array([n[0] for n in native]); applied = np.array([n[1] for n in native]); nxt = np.array([n[2] for n in native])
        magnitude = abs(a['scur'])
        margin = model.subtract(magnitude, a['delta'])
        with torch.no_grad():
            decision = model.threshold.decision(torch.from_numpy(margin)).numpy()
            sign = model.threshold.decision(torch.from_numpy(a['sbis'])).numpy()
        learned_raw = decision > 0
        positive = sign > 0
        # This mux is audited deterministic scaffold, NOT a newly learned head.
        neg_delta = -a['delta']
        minimum = np.where(positive, a['delta'], neg_delta)
        step = np.where(learned_raw, a['scur'], minimum)
        query = model.add(a['xcur'], step)
        stages = dict(abs_scur=(magnitude, magnitude), comparison_margin=(magnitude-a['delta'], margin),
            negative_delta=(-a['delta'], neg_delta), minimum_step=(np.where(a['sbis']>0, a['delta'], -a['delta']), minimum),
            applied_step=(applied, step), next_x=(nxt, query))
        if context:
            ctx = model.context(a['xcur'], np.array([r['xblk'] for r in rows]))
            stages.update(delta=(a['delta'], ctx['delta']), sbis=(a['sbis'], ctx['sbis']))
        for name, (truth, prediction) in stages.items():
            metric = source.stats(truth, prediction, scope+'/'+name)
            metrics.append(metric)
            if metric['errors']:
                index = int(np.flatnonzero(truth.view('u8') != prediction.view('u8'))[0])
                dump(OUT/'first_arithmetic_failure.json', dict(scope=scope, stage=name, input=rows[index],
                     teacher=float(truth[index]).hex(), learned=float(prediction[index]).hex()))
                raise RuntimeError('First arithmetic mismatch at '+name)
        assert np.array_equal(raw, learned_raw) and np.array_equal(positive, a['sbis']>0)
        for i, r in enumerate(rows):
            r.update(abs_scur=float(magnitude[i]), comparison_margin=float(margin[i]),
                comparison_state='GT' if magnitude[i]>a['delta'][i] else 'EQ' if magnitude[i]==a['delta'][i] else 'LT',
                sbis_positive=bool(positive[i]), teacher_use_raw=bool(raw[i]), learned_use_raw=bool(learned_raw[i]),
                negative_delta=float(neg_delta[i]), minimum_step=float(minimum[i]), teacher_applied_step=float(applied[i]),
                teacher_next_x=float(nxt[i]), assembled_applied_step=float(step[i]), assembled_next_x=float(query[i]),
                action='USE_SCUR' if raw[i] else 'USE_POSITIVE_DELTA' if positive[i] else 'USE_NEGATIVE_DELTA')
            r['hex_values'] = json.dumps({k: v.hex() for k, v in r.items() if isinstance(v, float)})
        return dict(cases=len(rows), raw_step=int(raw.sum()), minimum_step=int((~raw).sum()),
            equality=int((magnitude==a['delta']).sum()), positive_minimum=int((~raw & positive).sum()),
            negative_minimum=int((~raw & ~positive).sum()), comparison_errors=0, sign_test_errors=0)

    actual = screen(events, 'saved_events', True)
    boundary = screen(controls, 'boundary_controls')
    write(OUT/'step_application_events.csv', events)
    write(OUT/'boundary_controls.csv', controls)
    write(OUT/'arithmetic_validation.csv', metrics)
    breakdown = []
    for field in ('split', 'proposal_source'):
        for name in sorted({r[field] for r in events}):
            rr = [r for r in events if r[field] == name]
            breakdown.append(dict(by=field, group=name, events=len(rr), raw_step=sum(r['teacher_use_raw'] for r in rr),
                minimum_step=sum(not r['teacher_use_raw'] for r in rr), equality=sum(r['comparison_state']=='EQ' for r in rr),
                positive_minimum=sum(r['action']=='USE_POSITIVE_DELTA' for r in rr),
                negative_minimum=sum(r['action']=='USE_NEGATIVE_DELTA' for r in rr),
                next_x_errors=int(sum(bits(r['teacher_next_x'])!=bits(r['assembled_next_x']) for r in rr))))
    write(OUT/'event_counts.csv', breakdown)
    table = [dict(step_exceeds_delta=b, sbis_positive=p,
                  action='USE_SCUR' if b else 'USE_POSITIVE_DELTA' if p else 'USE_NEGATIVE_DELTA')
             for b in (False, True) for p in (False, True)]
    write(OUT/'minimal_routing_truth_table.csv', table)
    domains = {k: dict(min=min(r[k] for r in events), max=max(r[k] for r in events),
                       zero_count=sum(r[k]==0 for r in events)) for k in ('xcur', 'scur', 'sbis', 'delta')}
    unchanged = all(Path(p).exists() and source.sha(Path(p)) == h for p, h in protected.items())
    assert unchanged
    result = dict(status='AUDIT_ONLY', actual=actual, boundary_controls=boundary, solves=len(counts),
        persisted_query_checks=27049, saved_fresh_state_expression_checks=2013,
        all_arithmetic_bit_exact=True, all_comparisons_exact=True, protected_files=len(protected), protected_unchanged=unchanged,
        source='experiments/geometry/audit_exact_branch_brent.py:83',
        numerical_domains=domains, counts=breakdown, training=False, solver_executed=False,
        bracket_update_executed=False, promotion=False,
        first_unlearned_scaffold='Selection mux: USE_SCUR / USE_POSITIVE_DELTA / USE_NEGATIVE_DELTA; comparisons already reusable',
        smallest_new_router_if_requested='Two categorical booleans, four states, three actions; no raw numerical inputs needed')
    dump(OUT/'results.json', result)
    report = f'''# Brent step-application / minimum-step isolation

Authority: `experiments/geometry/audit_exact_branch_brent.py`, lines 83–85. No solver, training, bracket update or checkpoint mutation was executed. {len(protected)} protected artifacts retained identical hashes.

## Exact rule and order

```python
abs_scur = abs(scur)
use_raw = abs_scur > delta
if use_raw:
    applied_step = scur
else:
    applied_step = delta if sbis > 0 else -delta
next_x = xcur + applied_step
```

Each operation uses float64. The final addition rounds once; there is no algebraic rearrangement, FMA, nextafter, clipping, or sign lookup from scur. Equality takes the minimum step. For minimum steps, positive sbis selects +delta; negative sbis, +0 and -0 select -delta. `copysign(delta,sbis)` is therefore WRONG at +0. Unary negation flips delta's sign bit, including zero. The actual applied increment is saved separately from `next_x-xcur`, which can differ due to rounding.

Upstream delta remains `(xtol + rtol*abs(xcur))/2` with xtol=5e-324 and rtol=1e-15, and sbis remains `(xblk-xcur)/2`. Both were rechecked using existing arithmetic on every saved event. Prior stopping excludes sbis=0 with positive delta; zero/signed-zero unit controls are explicitly algebraic controls, not claims of reachable Brent states. Later callback and state/bracket updates are outside this audit.

## Coverage and exactness

{len(events):,} applications from {len(counts)} complete saved solves: 27,049 have persisted teacher query coordinates, checked bit-for-bit; 2,013 additional-fresh events have saved proposal state and native labels materialized from the exact audited expression. Those 2,013 are expression checks, not a newly executed fresh solve replay. The additional-fresh pool is the previously generated independent 57-solve pool; no new solves were generated. Coverage agrees with every saved per-solve query count.

{actual['raw_step']:,} raw steps; {actual['minimum_step']:,} minimum steps; {actual['equality']} exact-equality events; {actual['positive_minimum']} positive and {actual['negative_minimum']} negative minimum steps.

All stages exact: abs(scur), frozen subtraction comparison margin, unary negative delta, selected minimum increment, applied step, frozen-addition next_x, upstream delta and sbis. Zero one-ULP/larger errors, signed-zero mismatches, nonfinite mismatches, comparison errors and sign-test errors. Detailed split/source counts are in `event_counts.csv`; per-stage counts are in `arithmetic_validation.csv`.

{len(controls):,} independent algebraic controls include nextafter(delta,0), delta, nextafter(delta,+inf), both scur directions, both sbis directions, both signed zeros, deliberate scur/sbis sign disagreement, positive subnormal delta, broad exponents, signed-zero xcur and zero delta. All arithmetic and decisions match exactly. No nonfinite inputs/outputs were introduced.

## Dependencies and smallest remaining operation

| Operation | Classification | Exact interface |
|---|---|---|
| abs(scur) | Deterministic existing scaffold | scalar -> nonnegative float64 |
| abs(scur)-delta | Locked subtraction | two float64 -> exact native difference |
| strict comparison >0 | Existing frozen magnitude threshold | margin -> negative / tie / positive; choose positive |
| sbis>0 | Same existing frozen threshold | sbis -> sign category; tie is not positive |
| -delta | Deterministic unary sign-bit negation | float64 -> exact signed float64 |
| Choose applied increment | Deterministic selection scaffold; no dedicated matching learned mux identified | (use_raw,sbis_positive) -> one of three stored operands |
| xcur+applied_step | Frozen addition [1,1] | two float64 -> native-rounded sum |

The first strict comparison has exactly the same semantics as eligibility's `abs(spre)>delta` with scur substituted for spre. Its scalar subtraction-plus-frozen-threshold composition was reused and checked directly, including equality/subnormal boundaries. No unresolved arithmetic or numerical predicate remains on this tested domain.

Existing AND head has Boolean conjunction semantics, not three-way increment selection. Existing bracket-action router compares a query sign to opposite endpoint signs and includes DEFER; those are different state semantics and were not forced into this task. This audit uses an explicit deterministic mux and does NOT claim it is learned.

If the remaining actuation scaffold must be learned next, the smallest isolated routing target is two booleans `(abs(scur)>delta, sbis>0)` and three categorical outputs. Four-state table: `(0,0)->USE_NEGATIVE_DELTA`, `(0,1)->USE_POSITIVE_DELTA`, `(1,0)->USE_SCUR`, `(1,1)->USE_SCUR`. Alternatively the first stage alone is the already-supported strict predicate: LT/EQ -> USE_MINIMUM_STEP, GT -> USE_SCUR, leaving its sign selection deterministic. No new model was trained or promoted.

## Split and proposal counts

```json
{json.dumps(breakdown, indent=2)}
```

## Required scalar domains

```json
{json.dumps(domains, indent=2)}
```

Stop point: only step application has been isolated. No Brent state/bracket update, stopping or final-return implementation was changed or learned.
'''
    (OUT/'step_application_audit.md').write_text(report)
    dump(OUT/'artifact_hashes.json', {str(p): source.sha(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
