"""Train only r,p -> finite score. Frozen upstream chain; no validity model."""
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import CharacteristicSpacing, inputs, metrics, sha, write_csv
from experiments.sho.train_child_normalized_diameter_smoke import ChildNormalizedDiameter
from experiments.sho.train_radius_quotient_smoke import RadiusQuotient, evaluate as quotient, rank_audit
from experiments.sho.train_aspect_penalty_smoke import LogAspect, PenaltyScale, forward as aspect_forward
from experiments.sho.validate_child_qmax_assembly import predict_q, reduce_max, bucket
from experiments.sho.fast_nearest_argmin import FastNearestArgmin
from experiments.sho.run_stage1b_nearest_threshold import NonzeroThreshold
from experiments.sho.audit_split_geometry_score import reconstruct

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT/'finite_split_score_smoke'
SEED = 27242026
PRECISION = 256*np.finfo(np.float64).eps


class FiniteSplitScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(.2*torch.randn(2, dtype=torch.float64))

    def forward(self, rp):
        return self.weights[0]*rp[:, 0]+self.weights[1]*rp[:, 1]


def load_frozen():
    models = dict(q=ChildNormalizedDiameter(), h=CharacteristicSpacing(), radius=RadiusQuotient(),
        aspect=nn.ModuleDict(dict(log_aspect=LogAspect(), scale=PenaltyScale())))
    folders = dict(q='child_normalized_diameter_smoke', h='characteristic_spacing_smoke',
                   radius='radius_quotient_smoke', aspect='aspect_penalty_smoke')
    manifests = {}
    for key, folder in folders.items():
        m = json.loads((ROOT/folder/'candidate_manifest.json').read_text())
        assert m['passed'] and sha(m['checkpoint']) == m['sha256']
        models[key].load_state_dict(torch.load(m['checkpoint'], weights_only=False)['state_dict'])
        manifests[key] = m
    op = torch.load(ROOT/'nearest_argmin_assembled/nearest_argmin_operator_locked.pt', weights_only=False)
    assert op['status'] == 'LOCKED'
    models['margin'] = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    models['threshold'] = NonzeroThreshold()
    for name in ('margin', 'threshold'):
        assert sha(op[name+'_checkpoint']) == op[name+'_sha256']
        models[name].load_state_dict(torch.load(op[name+'_checkpoint'], weights_only=False)['state_dict'])
    for model in models.values(): model.eval().requires_grad_(False)
    execution = FastNearestArgmin(None, models['margin'], models['threshold'], None)
    return models, execution, manifests


def upstream(models, execution, parent_sides, budgets, children, counts):
    # All rows supplied by the caller are already finite/valid audit fixtures.
    assert np.all(counts >= 1) and np.all(children > 0)
    d = children.shape[-1]
    q = predict_q(models['q'], children, counts, d)
    qm, _, qs = reduce_max(execution, q)
    records = [dict(sides=ell, n=int(n), d=d) for ell, n in zip(parent_sides, budgets)]
    with torch.no_grad(): h = models['h'](*inputs(records)).numpy()
    _, radius = quotient(models['radius'], qm, h)
    aspect = aspect_forward(execution, models['aspect']['log_aspect'], models['aspect']['scale'], children)
    assert qs['comparison_mistakes'] == aspect['stats']['comparison_mistakes'] == 0
    return np.column_stack((radius, aspect['penalty']))


def synthetic_teacher(parents, budgets, children, counts):
    d = children.shape[-1]
    volume = np.multiply.reduce(parents, axis=1)
    if d == 2:
        # Exact audited 2D evaluation order, not the algebraic log rewrite.
        h = np.array([math.sqrt(float(v)/int(n)) for v, n in zip(volume, budgets)])
        roots = np.sqrt(counts)
    else:
        h = np.exp((np.log(volume)-np.log(budgets))/d)
        roots = np.power(counts.astype(np.float64), np.float64(1)/np.float64(d))
    diagonal = np.sqrt(np.add.reduce(children*children, axis=2))
    q = diagonal/roots
    radius = np.maximum.reduce(q, axis=1)/h
    ratio = np.maximum.reduce(children, axis=2)/np.minimum.reduce(children, axis=2)
    logs = np.log(ratio)
    penalty = .05*np.maximum.reduce(logs, axis=1)
    return radius, penalty, radius+penalty, ratio.max(1)


def generate(training):
    rng = np.random.default_rng(SEED+(0 if training else 1000)); rows = []
    dimensions = (1, 3, 5) if training else range(1, 6)
    budgets = [2, 4, 8, 12, 20, 32, 128, 512, 2048] if training else [3, 5, 6, 7, 9, 10, 11, 13, 17, 19, 64, 256, 8600, 32768, 1000000]
    for d in dimensions:
        for i in range(1024 if training else 512):
            n = int(budgets[i % len(budgets)])
            k = int(rng.choice([x for x in ((2, 4, 8) if training else (2, 3, 5, 8)) if x <= n]))
            aspect = np.exp(rng.uniform(0, np.log(64 if training else 1e4)))
            if d == 1: sides = np.ones(1)
            else:
                z = rng.random(d); z[0] = 0; z[1] = 1; rng.shuffle(z)
                sides = np.exp((z-.5)*np.log(aspect))
            sides *= 10**rng.uniform(-3 if training else -12, 3 if training else 12)
            fractions = rng.dirichlet(np.full(k, 2.)); axis = int(rng.integers(d))
            edges = np.r_[0., np.cumsum(fractions)]; edges[-1] = 1.
            children = np.tile(sides, (k, 1)); children[:, axis] = np.diff(edges*sides[axis])
            counts = 1+rng.multinomial(n-k, fractions)
            assert counts.sum() == n and np.all(children > 0)
            rows.append(dict(d=d, n=n, sides=sides, children=children, counts=counts, axis=axis))
    return rows


def aspect_bucket(a):
    return '1' if a == 1 else '(1,2)' if a < 2 else '[2,10)' if a < 10 else '[10,100)' if a < 100 else '[100,1000)' if a < 1000 else '>=1000'


def synthetic_pairs(records, models, execution):
    grouped = defaultdict(list)
    for i, r in enumerate(records): grouped[(r['d'], len(r['counts']))].append((i, r))
    features = np.empty((len(records), 2)); truth = np.empty(len(records)); metadata = [None]*len(records)
    for (d, k), pairs in grouped.items():
        ids = [i for i, _ in pairs]; rs = [r for _, r in pairs]
        parents = np.array([r['sides'] for r in rs]); budgets = np.array([r['n'] for r in rs])
        children = np.array([r['children'] for r in rs]); counts = np.array([r['counts'] for r in rs])
        features[ids] = upstream(models, execution, parents, budgets, children, counts)
        rt, pt, st, aspect = synthetic_teacher(parents, budgets, children, counts)
        truth[ids] = st
        for j, i in enumerate(ids):
            metadata[i] = dict(dimension=d, dimension_heldout=d in (2, 4), budget=int(budgets[j]),
                budget_bucket=bucket(int(budgets[j])), children=k, aspect=float(aspect[j]), aspect_range=aspect_bucket(aspect[j]),
                teacher_radius=float(rt[j]), teacher_penalty=float(pt[j]))
    return features, truth, metadata


def persist_geometry(path, records):
    rows = [{k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in r.items()} for r in records]
    path.write_text(json.dumps(rows))


def predict(model, features):
    with torch.no_grad(): return model(torch.from_numpy(features)).numpy()


def write_summary(result, rows):
    lines = ['Finite split-score sum: '+result['finite_split_score_operator'],
        'Only score = w_r * learned_radius + w_p * learned_penalty was trained.',
        'Bias absent; float64; random initialization; no coefficient constraints or snapping.',
        'Training fit: '+json.dumps(result['fit']),
        'Training uses new synthetic geometry only; every archived candidate is held out from this sum training.',
        'Dimensions 2 and 4 held out entirely; held-out budgets, scales, and geometry are also tested.',
        f"Archived candidates accounted for: {result['archived_candidates']}; finite tested: {result['finite_candidates']}; invalid excluded: {result['invalid_candidates_excluded']}",
        'Teacher scores independently reconstructed bit-exactly in original order.',
        'All learned inputs recomputed from frozen models; no cached teacher radius/penalty used as learned inputs.',
        '', 'scope | breakdown | value | examples | RMSE | MAE | max error | max relative error']
    for r in rows:
        lines.append(f"{r['scope']} | {r['breakdown']} | {r['value']} | {r['examples']} | {r['rmse']:.8e} | {r['mae']:.8e} | {r['max_error']:.8e} | {r['max_relative_error']:.8e}")
    lines += ['', 'Ranking: '+json.dumps(result['ranking']),
        'Canonical winner: first original finite candidate within 1e-12*max(1,best) of minimum.',
        'Raw changes include near-tie reversals, collapsed strict orders, and broken exact ties; report separately from equivalence.',
        'Protected components unchanged: '+str(result['protected_unchanged']),
        'Checkpoint persisted/reloaded before evaluation: '+result['checkpoint'], 'SHA256: '+result['sha256'],
        f"Training and validation elapsed: {result['elapsed_seconds']:.3f} seconds.",
        'Existing q_max raw-order audit status unchanged; this test uses requested teacher-equivalence semantics.',
        'No validity gate training, recursion, or search. Stop here.']
    (OUT/'finite_split_score_summary.txt').write_text('\n'.join(lines)+'\n')


def run():
    started = time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    paths = sorted((ROOT/'recursive_region_splitter_poc').glob('*.npz'))+sorted((ROOT/'split_quality_smoke').glob('*.npz'))
    protected = {str(p):sha(p) for p in paths+list(ROOT.glob('*/*.pt'))}
    for name in ('train_characteristic_spacing_smoke.py', 'train_child_normalized_diameter_smoke.py',
                 'validate_child_qmax_assembly.py', 'train_radius_quotient_smoke.py', 'train_aspect_penalty_smoke.py', 'fast_nearest_argmin.py'):
        p = Path('experiments/sho')/name; protected[str(p)] = sha(p)
    models, execution, manifests = load_frozen()
    train = generate(True); persist_geometry(OUT/'training_geometry.json', train)
    x, y, _ = synthetic_pairs(train, models, execution)
    np.savez_compressed(OUT/'training_pairs.npz', learned_inputs=x, teacher_score=y)
    model = FiniteSplitScore(); initial = model.weights.detach().numpy().copy(); history = []; best = [np.inf, None]
    def residual(w):
        with torch.no_grad(): model.weights.copy_(torch.from_numpy(w))
        e = predict(model, x)-y; loss = float(np.mean(e*e))
        history.append(dict(evaluation=len(history)+1, rmse=float(np.sqrt(loss)), w_r=float(w[0]), w_p=float(w[1])))
        if loss < best[0]: best[:] = [loss, w.copy()]
        return e
    opt = least_squares(residual, initial, jac=lambda _:x, method='trf', max_nfev=30, ftol=1e-14, xtol=1e-14, gtol=1e-14)
    with torch.no_grad(): model.weights.copy_(torch.from_numpy(best[1]))
    fit = dict(initial_weights=initial.tolist(), learned_weights=best[1].tolist(), calls=len(history),
        training_rmse=float(np.sqrt(best[0])), optimizer_message=opt.message, bias=0, bias_parameter_present=False)
    config = dict(seed=SEED, dtype='float64', bias=False, train_dimensions=[1,3,5], heldout_dimensions=[2,4],
        objective='Continuous unweighted squared finite teacher-score residual on frozen learned inputs',
        optimizer='TRF least_squares, exact linear Jacobian, max_nfev=30', numerical_gate=PRECISION,
        teacher_equivalence='1e-12 * max(1, best)', protected_hashes=protected,
        inputs='Actual frozen learned radius and aspect penalty, recomputed from synthetic geometry',
        teacher='Original radius arithmetic, original aspect divide/log/max/scale arithmetic, then radius + penalty',
        upstream={k:dict(checkpoint=m['checkpoint'],sha256=m['sha256']) for k,m in manifests.items()})
    cp = OUT/'finite_split_score_before_evaluation.pt'
    digest = save_checkpoint(cp, dict(state_dict=model.state_dict(), config=config, fit=fit, status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'training_history.csv', history)
    model.load_state_dict(torch.load(cp, weights_only=False)['state_dict']); model.eval().requires_grad_(False)
    print(json.dumps(dict(event='trained_saved', fit=fit, sha256=digest)), flush=True)

    heldout = generate(False); persist_geometry(OUT/'heldout_geometry.json', heldout)
    hx, hy, meta = synthetic_pairs(heldout, models, execution); hp = predict(model, hx)
    cases = [dict(scope='synthetic', source='heldout_geometry', candidate=i, **r,
        learned_radius=float(hx[i,0]), learned_penalty=float(hx[i,1]), teacher_score=float(hy[i]),
        learned_score=float(hp[i]), absolute_error=float(hp[i]-hy[i]), relative_error=float((hp[i]-hy[i])/hy[i])) for i,r in enumerate(meta)]
    rank_rows = []; changes = []; total = 0; invalid = 0
    for path in paths:
        a = np.load(path); source = path.parent.name+'/'+path.stem
        if 'scores' in a.files:
            k=14; pi=np.repeat(np.arange(len(a['budget'])), k)
            axes=np.tile(np.repeat(np.arange(2),7),len(a['budget']))
            cuts=np.tile(np.array([.2,.3,.4,.5,.6,.7,.8]*2),len(a['budget']))
            saved=a['scores'].reshape(-1); counts=a['counts'].reshape(-1,2)
        else:
            pi=a['parent_index']; axes=a['axis'][pi]; cuts=a['cut']; saved=a['y']; counts=a['counts']; k=len(pi)//len(a['budget'])
        score, valid, values, _, lengths = reconstruct(a['lower'],a['upper'],a['budget'],pi,axes,cuts,counts)
        assert np.array_equal(score.view(np.uint64), saved.view(np.uint64))
        ids = np.flatnonzero(valid)
        features = upstream(models, execution, values['parent_side'], a['budget'][pi[valid]], lengths[valid], counts[valid])
        pred = predict(model, features)
        # Existing invalid placeholders are retained solely to preserve original
        # candidate indices in ranking diagnostics. No learned validity decision.
        learned = score.copy(); learned[valid] = pred
        aspect = values['child_aspect'].max(1)
        for j, cid in enumerate(ids):
            n=int(a['budget'][pi[cid]])
            cases.append(dict(scope='archived', source=source, candidate=int(cid), parent=int(pi[cid]),
                dimension=2, dimension_heldout=True, budget=n, budget_bucket=bucket(n), children=2,
                aspect=float(aspect[j]), aspect_range=aspect_bucket(aspect[j]), axis=int(axes[cid]), cut=float(cuts[cid]),
                teacher_radius=float(values['radius_term'][j]), teacher_penalty=float(values['weighted_aspect_penalty'][j]),
                learned_radius=float(features[j,0]), learned_penalty=float(features[j,1]), teacher_score=float(score[cid]),
                learned_score=float(pred[j]), absolute_error=float(pred[j]-score[cid]), relative_error=float((pred[j]-score[cid])/score[cid])))
        for parent in range(len(a['budget'])):
            sl=slice(parent*k,(parent+1)*k)
            row=rank_audit(score[sl],learned[sl],source,parent,'complete_finite_score',cuts[sl],axes[sl],changes)
            finite=np.isfinite(score[sl]); aa=score[sl][finite]; bb=learned[sl][finite]
            i,j=np.triu_indices(len(aa),1)
            rev=np.sign(aa[i]-aa[j])*np.sign(bb[i]-bb[j])<0
            row['strict_reversals_outside_equivalence']=int(np.sum(rev & (abs(aa[i]-aa[j])>1e-12*max(1,float(aa.min())))))
            rank_rows.append(row)
        total+=len(score); invalid+=int((~valid).sum())
        print(json.dumps(dict(event='archive_validated', source=source, candidates=len(score))),flush=True)
    assert total==117511 and invalid==3674
    write_csv(OUT/'finite_split_score_cases.csv',cases)
    write_csv(OUT/'finite_split_score_parent_rankings.csv',rank_rows)
    if changes: write_csv(OUT/'finite_split_score_ranking_changes.csv',changes)
    else: (OUT/'finite_split_score_ranking_changes.csv').write_text('source,parent,component\n')
    rows=[]
    for scope in ('synthetic','archived'):
        rs=[r for r in cases if r['scope']==scope]
        for key in ('all','dimension','budget_bucket','aspect_range'):
            for v in (['all'] if key=='all' else sorted(set(r[key] for r in rs),key=str)):
                subset=rs if key=='all' else [r for r in rs if r[key]==v]
                rows.append(dict(scope=scope,breakdown=key,value=v,
                    **metrics(np.array([r['teacher_score'] for r in subset]),np.array([r['learned_score'] for r in subset]))))
    write_csv(OUT/'finite_split_score_smoke.csv',rows)
    keys=[k for k in rank_rows[0] if k not in ('source','parent','component')]
    ranks=dict(parents=len(rank_rows),**{k:sum(r[k] for r in rank_rows) for k in keys})
    unchanged=all(sha(p)==h for p,h in protected.items())
    numeric=all(r['max_relative_error']<=PRECISION for r in rows)
    passed=bool(numeric and unchanged and ranks['strict_reversals_outside_equivalence']==0
                and ranks['canonical_best_changed']==0 and ranks['pair_changes_outside_equivalence']==0)
    result=dict(finite_split_score_operator='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',passed=passed,
        checkpoint=str(cp),sha256=digest,fit=fit,training_examples=len(train),heldout_synthetic_examples=len(heldout),
        archived_candidates=total,finite_candidates=total-invalid,invalid_candidates_excluded=invalid,
        overall=[r for r in rows if r['breakdown']=='all'],ranking=ranks,protected_unchanged=unchanged,
        numerical_relative_error_gate=PRECISION,teacher_equivalence='1e-12 * max(1, best)',
        elapsed_seconds=time.monotonic()-started,validity_gate_trained=False,
        qmax_audit_status='Prior raw-order audit status unchanged; reused under requested equivalence semantics.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if passed: (OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2))
    write_summary(result,rows)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': run()
