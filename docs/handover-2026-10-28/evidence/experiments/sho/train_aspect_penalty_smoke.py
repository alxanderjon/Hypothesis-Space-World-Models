"""Dimension-general aspect branch only; immutable comparators, no final-score training."""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv
from experiments.sho.fast_nearest_argmin import FastNearestArgmin
from experiments.sho.run_stage1b_nearest_threshold import NonzeroThreshold
from experiments.sho.validate_child_qmax_assembly import reduce_max, bucket
from experiments.sho.train_radius_quotient_smoke import rank_audit
from experiments.sho.audit_split_geometry_score import reconstruct

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT/'aspect_penalty_smoke'
SEED = 27232026
PRECISION = 256*np.finfo(np.float64).eps


class LogAspect(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(.2*torch.randn(2, dtype=torch.float64))

    def forward(self, log_sides):
        return self.weights[0]*log_sides[:, 0]+self.weights[1]*log_sides[:, 1]


class PenaltyScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(.2*torch.randn(1, dtype=torch.float64))

    def forward(self, maximum):
        return self.weight[0]*maximum


def metrics(y, p):
    y = np.asarray(y).reshape(-1); p = np.asarray(p).reshape(-1); e = p-y
    normalized = abs(e)/np.maximum(1., abs(y))
    return dict(examples=len(y), rmse=float(np.sqrt(np.mean(e*e))), mae=float(np.mean(abs(e))),
        max_error=float(abs(e).max()), max_scaled_absolute_error=float(normalized.max()),
        teacher_min=float(y.min()), teacher_max=float(y.max()), exact_values=int(np.sum(y == p)))


def side_reduce(execution, lengths, maximum):
    current = lengths[:, 0].copy(); owner = np.zeros(len(lengths), dtype=int)
    for j in range(1, lengths.shape[1]):
        incoming = lengths[:, j]
        if maximum:
            replace = execution.larger(current, incoming)
        else:
            active = current != incoming; replace = np.zeros(len(current), dtype=bool)
            # Same locked comparison and sign interpretation as nearest().
            replace[active] = execution.logits(current[active], incoming[active]) > 0
        current = np.where(replace, incoming, current); owner = np.where(replace, j, owner)
    return current, owner


def side_inputs(execution, lengths):
    d = lengths.shape[-1]; flat = lengths.reshape(-1, d)
    hi, hi_owner = side_reduce(execution, flat, True)
    lo, lo_owner = side_reduce(execution, flat, False)
    x = torch.from_numpy(np.column_stack((np.log(hi), np.log(lo))))
    return x, hi, lo, hi_owner, lo_owner


def teacher(lengths):
    # Preserve teacher execution order: extrema -> division -> log -> max -> scale.
    hi = np.max(lengths, axis=-1); lo = np.min(lengths, axis=-1)
    aspect = hi/lo; logs = np.log(aspect); worst = np.max(logs, axis=1)
    return hi, lo, aspect, logs, worst, .05*worst


def forward(execution, log_model, scale_model, lengths):
    x, hi, lo, hi_owner, lo_owner = side_inputs(execution, lengths)
    with torch.no_grad(): logs = log_model(x).numpy().reshape(lengths.shape[:2])
    # No clamping or sign repair: the locked comparator is used on its valid
    # nonnegative domain, including exact-zero pairs handled as exact ties.
    if np.any(logs < 0) or not np.isfinite(logs).all():
        raise ValueError('Learned log-aspects leave the nonnegative comparator domain; stop without a clamp.')
    worst, owner, stat = reduce_max(execution, logs)
    with torch.no_grad(): penalty = scale_model(torch.from_numpy(worst)).numpy()
    return dict(logs=logs, maximum=worst, owner=owner, penalty=penalty,
        hi=hi, lo=lo, hi_owner=hi_owner, lo_owner=lo_owner, stats=stat)


def generate(training):
    rng = np.random.default_rng(SEED+(0 if training else 1000)); result = []
    dims = (1, 3, 5) if training else range(1, 6)
    for d in dims:
        for k in ((2, 4, 8) if training else (2, 3, 8)):
            for j in range(192 if training else 128):
                child = []
                for _ in range(k):
                    while True:
                        ratio = np.exp(rng.uniform(0, np.log(1e4 if training else 1e6)))
                        if not training or not (3 < ratio < 4 or 12 < ratio < 18): break
                    scale = 10**rng.uniform(-4 if training else -12, 4 if training else 12)
                    if d == 1: side = np.array([scale])
                    else:
                        z = rng.uniform(0, 1, d); z[0] = 0; z[1] = 1; rng.shuffle(z)
                        side = scale*np.exp((z-.5)*np.log(ratio))
                    if j % 16 == 0: side[:] = scale  # Exact isotropic supervision/tests.
                    child.append(side)
                result.append(dict(d=d, lengths=np.array(child), source='synthetic_train' if training else 'synthetic_heldout'))
    if not training:
        for d in range(1, 6):
            for scale in (1e-12, .137, 1., 3., 1e12):
                for steps in (0, 1, 2, 8, 64):
                    for base_ratio in (1., 2.):
                        a = np.full(d, scale); a[-1] *= base_ratio
                        b = a.copy()
                        for _ in range(steps): b[-1] = np.nextafter(b[-1], np.inf)
                        for reverse in (False, True):
                            result.append(dict(d=d, lengths=np.array([b, a] if reverse else [a, b]),
                                source='exact_and_ulp_stress', steps=steps, reverse=reverse))
    return result


def batches(records):
    grouped = defaultdict(list)
    for i, r in enumerate(records): grouped[(r['d'], len(r['lengths']), r['source'])].append((i, r))
    for (d, k, source), entries in grouped.items():
        yield d, k, source, [i for i, _ in entries], np.array([r['lengths'] for _, r in entries])


def fit(model, x, y):
    param = next(model.parameters()); initial = param.detach().numpy().copy()
    history = []; best = [np.inf, None]
    def residual(w):
        with torch.no_grad():
            param.copy_(torch.from_numpy(w)); pred = model(x).numpy()
        e = pred-y; loss = float(np.mean(e*e))
        history.append(dict(evaluation=len(history)+1, rmse=float(np.sqrt(loss)), weights=json.dumps(w.tolist())))
        if loss < best[0]: best[:] = [loss, w.copy()]
        return e
    jac = x.numpy() if x.ndim == 2 else x.numpy()[:, None]
    opt = least_squares(residual, initial, jac=lambda _: jac, method='trf', max_nfev=30,
                        ftol=1e-14, xtol=1e-14, gtol=1e-14)
    with torch.no_grad(): param.copy_(torch.from_numpy(best[1]))
    return dict(initial=initial.tolist(), learned=best[1].tolist(), calls=len(history),
                message=opt.message, training_rmse=float(np.sqrt(best[0]))), history


def write_summary(result, rows):
    lines = [
        'Aspect-penalty branch: '+result['aspect_penalty_operator'],
        'Only log-aspect and the scalar penalty multiplier were trained, in float64 without bias.',
        'Teacher arithmetic: side max/min -> max/min division -> log -> child max -> .05 scale.',
        'Learned log-space subtraction is compared against division-then-log, not substituted into teacher labels.',
        'Absolute errors are reported because one-dimensional/isotropic log-aspects are exactly zero.',
        'Log-aspect fit: '+json.dumps(result['log_fit']),
        'Scale fit: '+json.dumps(result['scale_fit']),
        f"Archived candidates: {result['archived_candidates']}; finite: {result['valid_candidates']}; invalid +inf retained: {result['invalid_candidates_preserved']}",
        'All archived teacher scores independently reconstructed bit-exactly before attribution.',
        'Side min/max and permutation checks: '+json.dumps(result['side_and_comparator_checks']),
        '', 'scope | component | RMSE | MAE | max absolute error']
    for row in result['metrics']:
        lines.append(f"{row['scope']} | {row['component']} | {row['rmse']:.8e} | {row['mae']:.8e} | {row['max_error']:.8e}")
    lines += ['', 'Held-out synthetic child log-aspect by dimension and aspect range:',
              'dimension | aspect range | examples | RMSE | maximum absolute error']
    for r in rows:
        if r['scope']=='synthetic' and r['component']=='child_log_aspect' and (str(r['dimension'])=='all' or r['aspect_range']=='all'):
            lines.append(f"{r['dimension']} | {r['aspect_range']} | {r['examples']} | {float(r['rmse']):.8e} | {float(r['max_error']):.8e}")
    lines += ['', 'Max across children: '+str(result['child_winner_changes'])+' raw witness changes; '+
        str(result['child_changes_outside_equivalence'])+' outside teacher equivalence.',
        'Archived score ranking with exact teacher radius held fixed: '+json.dumps(result['ranking']),
        'Canonical winner is first original candidate within 1e-12*max(1,best) of the minimum.',
        'This addition is attribution only; the final score sum was not trained.',
        'Checkpoint saved before evaluation: '+result['checkpoint'], 'SHA256: '+result['sha256'],
        'All protected checkpoints unchanged: '+str(result['protected_unchanged']),
        f"Training and validation elapsed: {result['elapsed_seconds']:.3f} seconds.",
        'No clamping, retraining of comparators, recursion, or search. Stop after this branch.']
    (OUT/'aspect_penalty_summary.txt').write_text('\n'.join(lines)+'\n')


def run():
    started = time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    paths = sorted((ROOT/'recursive_region_splitter_poc').glob('*.npz')) + sorted((ROOT/'split_quality_smoke').glob('*.npz'))
    opath = ROOT/'nearest_argmin_assembled/nearest_argmin_operator_locked.pt'
    op = torch.load(opath, weights_only=False); assert op['status'] == 'LOCKED'
    protected = {str(p):sha(p) for p in paths+list(ROOT.glob('*/*.pt'))}
    for file in ('fast_nearest_argmin.py', 'validate_child_qmax_assembly.py', 'train_radius_quotient_smoke.py'):
        p = Path('experiments/sho')/file; protected[str(p)] = sha(p)
    margin = nn.Linear(2, 1, bias=False, dtype=torch.float64); threshold = NonzeroThreshold()
    for name, model in [('margin', margin), ('threshold', threshold)]:
        assert sha(op[name+'_checkpoint']) == op[name+'_sha256']
        model.load_state_dict(torch.load(op[name+'_checkpoint'], weights_only=False)['state_dict'])
        model.eval().requires_grad_(False)
    execution = FastNearestArgmin(None, margin, threshold, None)
    config = dict(seed=SEED, dtype='float64', bias=False, training_dimensions=[1, 3, 5], heldout_dimensions=[2, 4],
        teacher_order='side max/min -> division -> log -> child max -> multiply by .05',
        learned_branch='locked side max/min -> learned two-log sum -> locked max -> learned scalar multiply',
        precision_gate=PRECISION, near_tie_rule='1e-12 * max(1, teacher best)', protected_hashes=protected,
        initialization='random normal * .2; no coefficient constraints, snapping, or clipping',
        objective='continuous squared regression residual; TRF least_squares, max_nfev=30',
        final_score_attribution='Teacher radius held fixed; no sum or radius training')
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    train = generate(True)
    (OUT/'training_geometry.json').write_text(json.dumps([dict(r, lengths=r['lengths'].tolist()) for r in train]))
    xs = []; ys = []
    for d, k, source, ids, lengths in batches(train):
        x, hi, lo, _, _ = side_inputs(execution, lengths)
        th, tl, _, logs, _, _ = teacher(lengths)
        assert np.array_equal(hi, th.ravel()) and np.array_equal(lo, tl.ravel())
        xs.append(x); ys.append(logs.ravel())
    log_model = LogAspect()
    log_fit, log_history = fit(log_model, torch.cat(xs), np.concatenate(ys))
    lp = OUT/'log_aspect_before_evaluation.pt'
    lhash = save_checkpoint(lp, dict(state_dict=log_model.state_dict(), config=config, fit=log_fit, status='SAVED_BEFORE_EVALUATION'))
    log_model.load_state_dict(torch.load(lp, weights_only=False)['state_dict']); log_model.eval().requires_grad_(False)
    write_csv(OUT/'log_aspect_training.csv', log_history)
    sx = []; sy = []
    for d, k, source, ids, lengths in batches(train):
        x, _, _, _, _ = side_inputs(execution, lengths)
        with torch.no_grad(): logs = log_model(x).numpy().reshape(-1, k)
        if np.any(logs < 0): raise ValueError('Training log-aspect negative: stop, no comparator-domain alteration.')
        maximum, _, _ = reduce_max(execution, logs)
        sx.append(maximum); sy.append(teacher(lengths)[-1])
    scale_model = PenaltyScale()
    scale_fit, scale_history = fit(scale_model, torch.from_numpy(np.concatenate(sx)), np.concatenate(sy))
    combined = nn.ModuleDict(dict(log_aspect=log_model, scale=scale_model))
    cp = OUT/'aspect_penalty_before_evaluation.pt'
    digest = save_checkpoint(cp, dict(state_dict=combined.state_dict(), config=config,
        log_fit=log_fit, scale_fit=scale_fit, log_checkpoint_sha256=lhash, status='SAVED_BEFORE_EVALUATION'))
    combined.load_state_dict(torch.load(cp, weights_only=False)['state_dict']); combined.eval().requires_grad_(False)
    write_csv(OUT/'scale_training.csv', scale_history)
    print(json.dumps(dict(event='trained_saved', log_fit=log_fit, scale_fit=scale_fit, sha256=digest)), flush=True)

    child_rows = []; set_rows = []; metric_rows = []; rank_rows = []; changes = []
    statistics = defaultdict(int); permutation_max = 0.; perm_rng = np.random.default_rng(SEED+2000)
    def audit_batch(lengths, source, ids, budgets=None):
        nonlocal permutation_max
        d = lengths.shape[2]; k = lengths.shape[1]
        th, tl, aspect, logs, worst, penalty = teacher(lengths)
        got = forward(execution, log_model, scale_model, lengths)
        statistics['comparisons'] += got['stats']['comparisons']
        statistics['comparator_mistakes'] += got['stats']['comparison_mistakes']
        flat = lengths.reshape(-1, d)
        for key, ref, owners in [('max', th.ravel(), flat.argmax(1)), ('min', tl.ravel(), flat.argmin(1))]:
            val = got['hi' if key == 'max' else 'lo']; idx = got['hi_owner' if key == 'max' else 'lo_owner']
            statistics['side_'+key+'_value_changes'] += int(np.sum(val != ref))
            statistics['side_'+key+'_winner_changes'] += int(np.sum(idx != owners))
            statistics['side_'+key+'_meaningful_changes'] += int(np.sum(abs(val-ref)/ref > PRECISION))
        # Independent coordinate permutation per child; no teacher min/max in inference.
        shuffled = np.take_along_axis(lengths, np.argsort(perm_rng.random(lengths.shape), axis=-1), axis=-1)
        pg = forward(execution, log_model, scale_model, shuffled)
        permutation_max = max(permutation_max, float(abs(pg['logs']-got['logs']).max()), float(abs(pg['penalty']-got['penalty']).max()))
        statistics['permutation_output_changes'] += int(np.sum(pg['logs'] != got['logs']))
        true_owner = logs.argmax(1); chosen_truth = logs[np.arange(len(lengths)), got['owner']]
        tol = 1e-12*np.maximum(1., worst)
        eq = logs >= worst[:, None]-tol[:, None]
        pred_eq = got['logs'] >= got['maximum'][:, None]-1e-12*np.maximum(1., got['maximum'])[:, None]
        sorted_logs = np.sort(logs, axis=1); gap = sorted_logs[:, -1]-sorted_logs[:, -2]
        ulp = sorted_logs[:, -1].view(np.uint64)-sorted_logs[:, -2].view(np.uint64)
        for i, cid in enumerate(ids):
            set_rows.append(dict(source=source, case=cid, dimension=d, children=k, budget=int(budgets[i]) if budgets is not None else '',
                teacher_max_log=float(worst[i]), learned_max_log=float(got['maximum'][i]),
                max_log_error=float(got['maximum'][i]-worst[i]), teacher_penalty=float(penalty[i]), learned_penalty=float(got['penalty'][i]),
                penalty_error=float(got['penalty'][i]-penalty[i]), teacher_owner=int(true_owner[i]), learned_owner=int(got['owner'][i]),
                raw_owner_changed=bool(true_owner[i] != got['owner'][i]),
                outside_teacher_equivalence=bool(worst[i]-chosen_truth[i] > tol[i]),
                canonical_max_child_changed=bool(np.argmax(eq[i]) != np.argmax(pred_eq[i])),
                exact_max_tie=bool(gap[i] == 0), one_ulp_gap=bool(ulp[i] == 1), near_tie=bool(0 < gap[i] <= tol[i])))
            for c in range(k):
                ratio = aspect[i, c]
                band = '1' if ratio == 1 else '(1,2)' if ratio < 2 else '[2,10)' if ratio < 10 else '[10,1000)' if ratio < 1000 else '>=1000'
                child_rows.append(dict(source=source, case=cid, child=c, dimension=d, aspect_range=band,
                    aspect=float(ratio), teacher_log_aspect=float(logs[i, c]), learned_log_aspect=float(got['logs'][i, c]),
                    error=float(got['logs'][i, c]-logs[i, c])))
        return got, penalty

    heldout = generate(False)
    (OUT/'heldout_geometry.json').write_text(json.dumps([dict(r, lengths=r['lengths'].tolist()) for r in heldout]))
    for d, k, source, ids, lengths in batches(heldout): audit_batch(lengths, source, ids)
    total = 0; invalid = 0
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
        assert np.array_equal(score.view(np.uint64), saved.view(np.uint64))
        got, tp = audit_batch(lengths[valid], source, np.flatnonzero(valid).tolist(), a['budget'][pi[valid]])
        assert np.array_equal(tp, values['weighted_aspect_penalty'])
        substituted = score.copy(); substituted[valid] = values['radius_term']+got['penalty']
        for parent in range(len(a['budget'])):
            sl = slice(parent*k, (parent+1)*k)
            rank_rows.append(rank_audit(score[sl], substituted[sl], source, parent,
                'teacher_radius_learned_aspect', cuts[sl], axes[sl], changes))
        total += len(score); invalid += int((~valid).sum())
        print(json.dumps(dict(event='archive_validated', source=source, candidates=len(score))), flush=True)
    assert total == 117511 and invalid == 3674
    write_csv(OUT/'aspect_penalty_child_cases.csv', child_rows)
    write_csv(OUT/'aspect_penalty_assembled_cases.csv', set_rows)
    write_csv(OUT/'aspect_penalty_parent_rankings.csv', rank_rows)
    if changes: write_csv(OUT/'aspect_penalty_ranking_changes.csv', changes)
    else: (OUT/'aspect_penalty_ranking_changes.csv').write_text('source,parent,component\n')
    for scope in ('synthetic', 'archived', 'all'):
        cr = [r for r in child_rows if scope == 'all' or (('/' in r['source']) == (scope == 'archived'))]
        sr = [r for r in set_rows if scope == 'all' or (('/' in r['source']) == (scope == 'archived'))]
        for d in ('all', 1, 2, 3, 4, 5):
            for band in ('all', '1', '(1,2)', '[2,10)', '[10,1000)', '>=1000'):
                rs = [r for r in cr if (d == 'all' or r['dimension'] == d) and (band == 'all' or r['aspect_range'] == band)]
                if rs: metric_rows.append(dict(scope=scope, component='child_log_aspect', dimension=d, aspect_range=band,
                    **metrics([r['teacher_log_aspect'] for r in rs], [r['learned_log_aspect'] for r in rs])))
            rs = [r for r in sr if d == 'all' or r['dimension'] == d]
            if not rs: continue
            for label, tk, pk in [('max_log_aspect', 'teacher_max_log', 'learned_max_log'), ('penalty', 'teacher_penalty', 'learned_penalty')]:
                metric_rows.append(dict(scope=scope, component=label, dimension=d, aspect_range='all',
                    **metrics([r[tk] for r in rs], [r[pk] for r in rs]),
                    winning_child_changes=sum(r['raw_owner_changed'] for r in rs),
                    changes_outside_equivalence=sum(r['outside_teacher_equivalence'] for r in rs),
                    canonical_child_changes=sum(r['canonical_max_child_changed'] for r in rs)))
    for label, test in [('exact_ties', lambda r:r['exact_max_tie']), ('one_ulp', lambda r:r['one_ulp_gap']),
                        ('near_ties', lambda r:r['near_tie'])]:
        rs = [r for r in set_rows if test(r)]
        if rs: metric_rows.append(dict(scope='stress_'+label, component='max_log_aspect', dimension='all', aspect_range='all',
            **metrics([r['teacher_max_log'] for r in rs], [r['learned_max_log'] for r in rs]),
            winning_child_changes=sum(r['raw_owner_changed'] for r in rs),
            changes_outside_equivalence=sum(r['outside_teacher_equivalence'] for r in rs)))
    write_csv(OUT/'aspect_penalty_smoke.csv', metric_rows)
    rank_keys = [key for key in rank_rows[0] if key not in ('source', 'parent', 'component')]
    ranks = dict(parents=len(rank_rows), **{key:sum(r[key] for r in rank_rows) for key in rank_keys})
    unchanged = all(sha(p) == h for p, h in protected.items())
    numeric = all(r['max_scaled_absolute_error'] <= PRECISION for r in metric_rows)
    side_ok = statistics['side_max_meaningful_changes'] == statistics['side_min_meaningful_changes'] == 0
    winner_ok = all(not r['outside_teacher_equivalence'] and not r['canonical_max_child_changed'] for r in set_rows)
    rank_ok = ranks['pair_changes_outside_equivalence'] == ranks['canonical_best_changed'] == ranks['learned_best_outside_teacher_equivalence'] == 0
    passed = bool(numeric and side_ok and winner_ok and rank_ok and unchanged and permutation_max == 0 and statistics['comparator_mistakes'] == 0)
    selected = [r for r in metric_rows if r['aspect_range'] == 'all' and r['dimension'] == 'all' and r['scope'] in ('synthetic', 'archived')]
    result = dict(aspect_penalty_operator='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY', passed=passed,
        checkpoint=str(cp), sha256=digest, log_checkpoint=str(lp), log_sha256=lhash,
        log_fit=log_fit, scale_fit=scale_fit, archived_candidates=total, valid_candidates=total-invalid,
        invalid_candidates_preserved=invalid, training_sets=len(train), synthetic_test_sets=len(heldout),
        metrics=selected, side_and_comparator_checks=dict(statistics), permutation_max_error=permutation_max,
        child_winner_changes=sum(r['raw_owner_changed'] for r in set_rows),
        child_changes_outside_equivalence=sum(r['outside_teacher_equivalence'] for r in set_rows),
        ranking=ranks, protected_unchanged=unchanged, elapsed_seconds=time.monotonic()-started,
        precision_gate=PRECISION, error_convention='absolute and abs(error)/max(1,abs(teacher)); relative error is unsuitable at zero log-aspect',
        notes=['Teacher retains extrema/division/log/max/scale execution order.',
            'Learned log-linear branch approximates division-then-log without silently rewriting teacher labels.',
            'Dimensions 2 and 4 entirely held out of training; archived candidates are 2D.',
            'Exact comparator ties keep first child/coordinate; equivalence used only in audit.',
            'No clamping, comparator training, final-sum training, recursion, or search.'])
    (OUT/'results.json').write_text(json.dumps(result, indent=2))
    if passed: (OUT/'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    write_summary(result, metric_rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': run()
