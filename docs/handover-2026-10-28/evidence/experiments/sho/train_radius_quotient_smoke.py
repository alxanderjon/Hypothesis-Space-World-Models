"""Two learned log-quotient weights; frozen upstream composition, no search.

Teacher aspect penalties are held fixed only for ranking attribution. Invalid
zero-count candidates retain +inf and never enter a logarithm.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import (
    CharacteristicSpacing, generate, inputs, metrics, sha, write_csv)
from experiments.sho.train_child_normalized_diameter_smoke import ChildNormalizedDiameter, target
from experiments.sho.validate_child_qmax_assembly import predict_q, reduce_max, bucket
from experiments.sho.fast_nearest_argmin import FastNearestArgmin
from experiments.sho.run_stage1b_nearest_threshold import NonzeroThreshold
from experiments.sho.audit_split_geometry_score import reconstruct

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT/'radius_quotient_smoke'
SEED = 27222026
PRECISION = 256*np.finfo(np.float64).eps


class RadiusQuotient(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False, dtype=torch.float64)

    def log_forward(self, log_inputs):
        return self.linear(log_inputs).squeeze(-1)

    def forward(self, log_inputs):
        return torch.exp(self.log_forward(log_inputs))


def evaluate(model, q, h):
    x = torch.from_numpy(np.column_stack((np.log(q), np.log(h))))
    with torch.no_grad():
        log = model.log_forward(x).numpy()
        radius = model(x).numpy()
    return log, radius


def log_metrics(truth, prediction):
    e = prediction-truth
    return dict(log_rmse=float(np.sqrt(np.mean(e*e))),
                log_mae=float(np.mean(abs(e))), max_log_error=float(abs(e).max()))


def rank_audit(a, b, source, parent, component, cuts, axes, changes):
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    ids = np.flatnonzero(np.isfinite(a)); a = a[ids]; b = b[ids]
    tol = 1e-12*max(1., float(a.min()))
    eq_a = a <= a.min()+tol
    eq_b = b <= b.min()+1e-12*max(1., float(b.min()))
    oa = np.argsort(a, kind='stable'); ob = np.argsort(b, kind='stable')
    i, j = np.triu_indices(len(a), 1)
    sa = np.sign(a[i]-a[j]); sb = np.sign(b[i]-b[j])
    changed = sa != sb; reversal = sa*sb < 0
    collapse = (sa != 0) & (sb == 0); broken = (sa == 0) & (sb != 0)
    outside = changed & (abs(a[i]-a[j]) > tol)
    for t in np.flatnonzero(changed):
        ci, cj = ids[i[t]], ids[j[t]]
        changes.append(dict(source=source, parent=parent, component=component,
            candidate_i=int(ci), candidate_j=int(cj), axis_i=int(axes[ci]), axis_j=int(axes[cj]),
            cut_i=float(cuts[ci]), cut_j=float(cuts[cj]), teacher_i=float(a[i[t]]),
            teacher_j=float(a[j[t]]), learned_i=float(b[i[t]]), learned_j=float(b[j[t]]),
            teacher_gap=float(abs(a[i[t]]-a[j[t]])), teacher_best_tolerance=tol,
            strict_reversal=bool(reversal[t]), collapse=bool(collapse[t]),
            exact_tie_broken=bool(broken[t]), outside_teacher_equivalence=bool(outside[t])))
    return dict(source=source, parent=parent, component=component,
        raw_order_changed=int(np.any(oa != ob)), raw_best_changed=int(oa[0] != ob[0]),
        canonical_best_changed=int(np.flatnonzero(eq_a)[0] != np.flatnonzero(eq_b)[0]),
        equivalence_mask_changed=int(np.any(eq_a != eq_b)),
        learned_best_outside_teacher_equivalence=int(not eq_a[ob[0]]),
        strict_reversals=int(reversal.sum()), strict_order_collapses=int(collapse.sum()),
        exact_ties_broken=int(broken.sum()), pair_changes_outside_equivalence=int(outside.sum()))


def run():
    started = time.monotonic()
    torch.set_num_threads(1); torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    qpath = ROOT/'child_normalized_diameter_smoke/child_normalized_diameter_before_evaluation.pt'
    hpath = ROOT/'characteristic_spacing_smoke/characteristic_spacing_before_evaluation.pt'
    opath = ROOT/'nearest_argmin_assembled/nearest_argmin_operator_locked.pt'
    op = torch.load(opath, weights_only=False)
    assert op['status'] == 'LOCKED'
    for name, path in [('child_normalized_diameter_smoke', qpath), ('characteristic_spacing_smoke', hpath)]:
        assert sha(path) == json.loads((ROOT/name/'candidate_manifest.json').read_text())['sha256']
    for name in ('margin', 'threshold'):
        assert sha(op[name+'_checkpoint']) == op[name+'_sha256']
    paths = sorted((ROOT/'recursive_region_splitter_poc').glob('*.npz')) + sorted((ROOT/'split_quality_smoke').glob('*.npz'))
    frozen_paths = [qpath, hpath, opath, Path(op['margin_checkpoint']), Path(op['threshold_checkpoint']),
        ROOT/'child_qmax_assembly/results.json', Path('experiments/sho/validate_child_qmax_assembly.py'),
        Path('experiments/sho/fast_nearest_argmin.py'), Path('experiments/sho/train_characteristic_spacing_smoke.py'),
        Path('experiments/sho/train_child_normalized_diameter_smoke.py')]
    protected = {str(p): sha(p) for p in paths+frozen_paths}
    qmodel = ChildNormalizedDiameter(); qmodel.load_state_dict(torch.load(qpath, weights_only=False)['state_dict'])
    hmodel = CharacteristicSpacing(); hmodel.load_state_dict(torch.load(hpath, weights_only=False)['state_dict'])
    margin = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    margin.load_state_dict(torch.load(op['margin_checkpoint'], weights_only=False)['state_dict'])
    threshold = NonzeroThreshold(); threshold.load_state_dict(torch.load(op['threshold_checkpoint'], weights_only=False)['state_dict'])
    for m in (qmodel, hmodel, margin, threshold): m.eval().requires_grad_(False)
    execution = FastNearestArgmin(None, margin, threshold, None)

    # Broad independent positive inputs identify both coefficients without any
    # target coefficients, upstream weights, or ranking labels in training.
    rng = np.random.default_rng(SEED)
    qtrain, htrain = np.exp(rng.uniform(-40., 32., (2, 8192)))
    x = np.column_stack((np.log(qtrain), np.log(htrain)))
    y = np.log(qtrain)-np.log(htrain)
    np.savez_compressed(OUT/'training_pairs.npz', qmax=qtrain, h=htrain, log_radius=y)
    model = RadiusQuotient(); initial = model.linear.weight.detach().numpy().ravel().copy()
    history = []; best = [np.inf, None]
    def residual(w):
        # Same no-bias float64 two-input linear forward as inference.
        with torch.no_grad(): z = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w[None, :])).numpy().ravel()
        e = z-y; loss = float(np.mean(e*e))
        history.append(dict(evaluation=len(history)+1, log_rmse=float(np.sqrt(loss)), w_q=float(w[0]), w_h=float(w[1])))
        if loss < best[0]: best[:] = [loss, w.copy()]
        return e
    opt = least_squares(residual, initial, jac=lambda w: x, method='trf',
                        max_nfev=30, ftol=1e-14, xtol=1e-14, gtol=1e-14)
    with torch.no_grad(): model.linear.weight.copy_(torch.from_numpy(best[1][None, :]))
    config = dict(seed=SEED, dtype='float64', bias=False, train_examples=len(y),
        training_log_input_range=[-40., 32.], model='exp(w_q*log(qmax)+w_h*log(h))',
        optimizer='TRF continuous least squares, exact linear Jacobian, max_nfev=30',
        precision_gate=PRECISION, teacher_equivalence='1e-12 * max(1, best)', protected_hashes=protected)
    checkpoint = OUT/'radius_quotient_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
        initial_weights=initial.tolist(), objective_calls=len(history), optimizer_message=opt.message,
        status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'training_history.csv', history)
    print(json.dumps(dict(event='trained_and_saved', weights=best[1].tolist(), initial=initial.tolist(),
        calls=len(history), sha256=digest)), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict']); model.eval().requires_grad_(False)

    rows = []; geometry_rows = []; comparison_mistakes = 0
    qrng = np.random.default_rng(SEED+1)
    qtest, htest = np.exp(qrng.uniform(-46., 40., (2, 16384)))
    log, predicted = evaluate(model, qtest, htest)
    standalone = dict(**metrics(qtest/htest, predicted), **log_metrics(np.log(qtest)-np.log(htest), log))
    rows.append(dict(section='independent_positive_heldout', dimension='all', budget_bucket='all', **standalone))
    # Independent synthetic parent/child boxes spanning all five dimensions.
    # These tests exercise frozen upstreams too; no geometry enters training.
    groups = ('combinations_and_budget_interpolation', 'aspect_interpolation', 'aspect_extrapolation',
              'budget_extrapolation', 'scale_extrapolation')
    for d in range(1, 6):
        for gi, group in enumerate(groups):
            boxes = generate(d, 128, SEED+100+d*10+gi, group)
            for r in boxes: r['n'] = max(2, r['n'])
            lengths = np.array([r['sides'] for r in boxes]); budgets = np.array([r['n'] for r in boxes])
            cuts = qrng.choice([.137, .23, .5, .71, .863], len(boxes))
            counts1 = np.clip(np.rint(budgets*cuts).astype(np.int64), 1, budgets-1)
            counts = np.column_stack((counts1, budgets-counts1))
            child = np.repeat(lengths[:, None, :], 2, axis=1)
            child[:, 0, 0] *= cuts; child[:, 1, 0] *= 1-cuts
            qt = np.array([[target(ell, int(n), d) for ell, n in zip(ls, ns)] for ls, ns in zip(child, counts)]).max(1)
            ht = np.array([np.exp((np.log(np.prod(r['sides']))-np.log(r['n']))/d) for r in boxes])
            qhat, _, st = reduce_max(execution, predict_q(qmodel, child, counts, d))
            comparison_mistakes += st['comparison_mistakes']
            with torch.no_grad(): hhat = hmodel(*inputs(boxes)).numpy()
            lp, rp = evaluate(model, qhat, hhat)
            il, ir = evaluate(model, qt, ht)
            for i, r in enumerate(boxes):
                geometry_rows.append(dict(dimension=d, upstream_dimension_heldout=d in (2, 4), group=group,
                    budget=r['n'], budget_bucket=bucket(r['n']), qmax_teacher=float(qt[i]), h_teacher=float(ht[i]),
                    qmax_learned=float(qhat[i]), h_learned=float(hhat[i]), radius_teacher=float(qt[i]/ht[i]),
                    radius_learned=float(rp[i]), isolated_quotient_radius=float(ir[i]),
                    teacher_log=float(np.log(qt[i])-np.log(ht[i])), learned_log=float(lp[i]), isolated_log=float(il[i])))
    write_csv(OUT/'radius_quotient_dimension_cases.csv', geometry_rows)
    for dim in ('all', 1, 2, 3, 4, 5):
        for b in ('all', '2-5', '6-10', '11-20', '>20'):
            subset = [r for r in geometry_rows if (dim == 'all' or r['dimension'] == dim) and (b == 'all' or r['budget_bucket'] == b)]
            if not subset: continue
            t = np.array([r['radius_teacher'] for r in subset]); lt = np.array([r['teacher_log'] for r in subset])
            for label, rk, lk in [('synthetic_frozen_chain', 'radius_learned', 'learned_log'),
                                  ('synthetic_quotient_isolated', 'isolated_quotient_radius', 'isolated_log')]:
                rows.append(dict(section=label, dimension=dim, budget_bucket=b,
                    **metrics(t, np.array([r[rk] for r in subset])), **log_metrics(lt, np.array([r[lk] for r in subset]))))
    for group in groups:
        subset = [r for r in geometry_rows if r['group'] == group]
        rows.append(dict(section='synthetic_frozen_chain_by_holdout', group=group, dimension='all', budget_bucket='all',
            **metrics(np.array([r['radius_teacher'] for r in subset]), np.array([r['radius_learned'] for r in subset]))))

    cases = []; rankings = []; changes = []; invalid = 0; total = 0
    for path in paths:
        a = np.load(path); source = path.parent.name+'/'+path.stem
        if 'scores' in a.files:
            k = 14; pi = np.repeat(np.arange(len(a['budget'])), k)
            axes = np.tile(np.repeat(np.arange(2), 7), len(a['budget']))
            cuts = np.tile(np.array([.2, .3, .4, .5, .6, .7, .8]*2), len(a['budget']))
            saved = a['scores'].reshape(-1); counts = a['counts'].reshape(-1, 2)
        else:
            pi = a['parent_index']; axes = a['axis'][pi]; cuts = a['cut']; saved = a['y']; counts = a['counts']
            k = len(pi)//len(a['budget'])
        score, valid, values, _, lengths = reconstruct(a['lower'], a['upper'], a['budget'], pi, axes, cuts, counts)
        assert np.array_equal(score.view(np.uint64), saved.view(np.uint64)), source
        valid_ids = np.flatnonzero(valid)
        qhat, _, st = reduce_max(execution, predict_q(qmodel, lengths[valid], counts[valid], 2))
        comparison_mistakes += st['comparison_mistakes']
        parent_records = [dict(sides=hi-lo, n=int(n), d=2) for lo, hi, n in zip(a['lower'], a['upper'], a['budget'])]
        with torch.no_grad(): hp = hmodel(*inputs(parent_records)).numpy()
        hhat = hp[pi[valid]]
        qt = values['worst_child_diameter_over_sqrt_count']; ht = values['parent_spacing']
        rt = values['radius_term']; lp, rp = evaluate(model, qhat, hhat)
        il, ir = evaluate(model, qt, ht)
        oldr = np.full(len(score), np.inf); newr = oldr.copy(); isolatedr = oldr.copy()
        oldr[valid] = rt; newr[valid] = rp; isolatedr[valid] = ir
        newscore = score.copy(); isolatedscore = score.copy()
        newscore[valid] = rp+values['weighted_aspect_penalty']
        isolatedscore[valid] = ir+values['weighted_aspect_penalty']
        for j, cid in enumerate(valid_ids):
            n = int(a['budget'][pi[cid]])
            cases.append(dict(source=source, candidate=int(cid), parent=int(pi[cid]), dimension=2, budget=n,
                budget_bucket=bucket(n), axis=int(axes[cid]), cut=float(cuts[cid]),
                qmax_teacher=float(qt[j]), qmax_learned=float(qhat[j]), h_teacher=float(ht[j]), h_learned=float(hhat[j]),
                radius_teacher=float(rt[j]), radius_learned=float(rp[j]), relative_error=float((rp[j]-rt[j])/rt[j]),
                isolated_quotient_radius=float(ir[j]), upstream_exact_division=float(qhat[j]/hhat[j])))
        for parent in range(len(a['budget'])):
            s = slice(parent*k, (parent+1)*k)
            for label, old, new in [('assembled_radius', oldr, newr), ('assembled_radius_in_teacher_score', score, newscore),
                                    ('isolated_quotient_radius', oldr, isolatedr), ('isolated_quotient_in_teacher_score', score, isolatedscore)]:
                rankings.append(rank_audit(old[s], new[s], source, parent, label, cuts[s], axes[s], changes))
        total += len(score); invalid += int((~valid).sum())
        print(json.dumps(dict(event='archived_pool_validated', source=source, candidates=len(score))), flush=True)
    assert total == 117511 and len(cases) == 113837
    write_csv(OUT/'radius_quotient_assembled_cases.csv', cases)
    write_csv(OUT/'radius_quotient_parent_rankings.csv', rankings)
    write_csv(OUT/'radius_quotient_ranking_changes.csv', changes)
    for b in ('all', '2-5', '6-10', '11-20', '>20'):
        subset = [r for r in cases if b == 'all' or r['budget_bucket'] == b]
        if not subset: continue
        for label, key in [('archive_frozen_chain', 'radius_learned'), ('archive_quotient_isolated', 'isolated_quotient_radius'),
                           ('archive_upstream_exact_division', 'upstream_exact_division')]:
            rows.append(dict(section=label, dimension=2, budget_bucket=b,
                **metrics(np.array([r['radius_teacher'] for r in subset]), np.array([r[key] for r in subset]))))
    write_csv(OUT/'radius_quotient_smoke.csv', rows)
    aggregate = {}
    keys = [k for k in rankings[0] if k not in ('source', 'parent', 'component')]
    for component in sorted(set(r['component'] for r in rankings)):
        rs = [r for r in rankings if r['component'] == component]
        aggregate[component] = dict(parents=len(rs), **{k:sum(r[k] for r in rs) for k in keys})
    unchanged = all(sha(p) == h for p, h in protected.items())
    numeric = all(r['max_relative_error'] <= PRECISION for r in rows)
    ordering = all(r['pair_changes_outside_equivalence'] == 0 and r['canonical_best_changed'] == 0
                   and r['learned_best_outside_teacher_equivalence'] == 0 for r in rankings)
    passed = numeric and ordering and unchanged and comparison_mistakes == 0
    summary = dict(radius_quotient_operator='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY', passed=passed,
        checkpoint=str(checkpoint), sha256=digest, initial_weights=initial.tolist(), learned_weights=best[1].tolist(),
        optimizer_evaluations=len(history), independent_heldout=standalone,
        assembled=[r for r in rows if r['section']=='archive_frozen_chain' and r['budget_bucket']=='all'][0],
        synthetic_chain=[r for r in rows if r['section']=='synthetic_frozen_chain' and r['dimension']=='all' and r['budget_bucket']=='all'][0],
        archived_candidates=total, valid_candidates=len(cases), invalid_zero_count_candidates=invalid,
        ranking=aggregate, protected_unchanged=unchanged, comparator_mistakes=comparison_mistakes,
        numerical_precision_gate=PRECISION, teacher_equivalence='1e-12 * max(1, best)',
        elapsed_seconds=time.monotonic()-started,
        qmax_status_note='Prior qmax strict raw-order gate remains NOT_VALIDATED; reused unchanged under requested equivalence semantics.',
        attribution='All archived candidates are 2D. Full frozen chain also tested on synthetic 1D-5D splits. Exact aspect penalties used solely for radius attribution, not learned.')
    (OUT/'results.json').write_text(json.dumps(summary, indent=2))
    if passed: (OUT/'candidate_manifest.json').write_text(json.dumps(summary, indent=2))
    text = ['Radius quotient: '+summary['radius_quotient_operator'],
        'Only two no-bias float64 log-space weights trained; no rounding or coefficient constraints.',
        f'Initial weights: {initial.tolist()}; learned: {best[1].tolist()}',
        f'Checkpoint persisted and reloaded before evaluation: {checkpoint}; SHA256 {digest}',
        f'Archived candidates: {total}; valid: {len(cases)}; invalid +inf retained: {invalid}',
        'Teacher scores reconstructed bit-exactly before comparisons.',
        'Canonical winner: first original candidate within 1e-12*max(1,best) of minimum.',
        'Outside-equivalence pair changes use the same parent-best tolerance.',
        json.dumps(summary['independent_heldout'], indent=2), json.dumps(summary['assembled'], indent=2),
        json.dumps(summary['synthetic_chain'], indent=2), json.dumps(aggregate, indent=2),
        summary['qmax_status_note'], summary['attribution'],
        f'All protected inputs and components unchanged: {unchanged}. Elapsed: {summary["elapsed_seconds"]:.3f} seconds.',
        'No aspect/final-score training, recursion, or search. Stop after this gate.']
    (OUT/'radius_quotient_summary.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': run()
