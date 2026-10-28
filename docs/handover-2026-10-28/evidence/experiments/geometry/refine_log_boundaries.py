"""Continue saved log model with teacher boundary/consistency supervision.

Architecture and inference evaluation order are unchanged. Float64 teacher ties
are allowed by explicit user clarification; downward steps remain forbidden.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.geometry.train_log_operator import ScalarLogPolynomial, predict, metrics, EPS
from experiments.geometry.train_orientation_transform import OrientationTransform, inputs, teacher as orientation_teacher
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic, metrics as arithmetic_metrics
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv

ROOT = Path('outputs/sho/stage1B')
SOURCE = ROOT / 'log_operator_smoke'
OUT = ROOT / 'log_operator_boundary_refine'
SEED = 27402026


def centers():
    k = np.arange(-1073, 1024)
    powers = np.ldexp(np.ones(len(k)), k)
    actual = np.ldexp(np.full(len(k), .75), k)
    return powers, actual


def adjacent_cloud(center, width):
    left, right = center.copy(), center.copy()
    columns_left, columns_right = [], []
    for _ in range(width):
        left = np.nextafter(left, -np.inf)
        right = np.nextafter(right, np.inf)
        columns_left.append(left.copy())
        columns_right.append(right.copy())
    return np.unique(np.concatenate(columns_left[::-1] + [center] + columns_right))


def pair_report(x, p):
    y = np.log(x)
    pd, td = np.diff(p), np.diff(y)
    return dict(inputs=len(x), pairs=len(x) - 1,
                reversals=int(np.sum(pd < 0)),
                collapsed_teacher_strict_pairs=int(np.sum((td > 0) & (pd == 0))),
                permitted_teacher_ties=int(np.sum((td == 0) & (pd == 0))),
                largest_drop=float(-pd[pd < 0].min()) if np.any(pd < 0) else 0.)


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    source = SOURCE / 'log_before_evaluation.pt'
    model = ScalarLogPolynomial()
    model.load_state_dict(torch.load(source, weights_only=False)['state_dict'])
    original_w = model.weights.detach().numpy().copy()
    old_manifest = json.loads((SOURCE / 'candidate_manifest.json').read_text())
    original_training = np.load(SOURCE / 'training_data.npz')
    broad_v, broad_y = original_training['v'], original_training['targets']
    powers, actual = centers()
    boundary_v = adjacent_cloud(np.concatenate((powers, actual)), 1)
    boundary_v = boundary_v[(boundary_v > 0) & np.isfinite(boundary_v)]
    boundary_y = np.log(boundary_v)
    # Ordered neighboring inputs on each side of both boundary families.
    pair_left, pair_right = boundary_v[:-1], boundary_v[1:]
    pair_gap = np.diff(boundary_y)
    log2_teacher = float(np.log(np.float64(2.)))
    consistency_v = np.ldexp(rng.uniform(.5, 1., 4096), rng.integers(-1000, 1000, 4096))
    assert np.all(consistency_v > 0) and np.isfinite(2. * consistency_v).all()
    with torch.no_grad():
        feature = lambda v: model.features(torch.from_numpy(v)).numpy()
        fb, fe = feature(broad_v), feature(boundary_v)
        fp = fe[1:] - fe[:-1]
        fc = feature(2. * consistency_v) - feature(consistency_v)
    broad_div = abs(broad_y).copy()
    broad_div[broad_div == 0] = 1.
    boundary_div = np.maximum(1., abs(boundary_y))
    pair_div = np.maximum(1., np.maximum(abs(boundary_y[:-1]), abs(boundary_y[1:])))
    # Per-block normalization keeps endpoint supervision strong without changing
    # inference arithmetic. Ridge regularizes all coefficients identically.
    config = dict(seed=SEED, source_checkpoint=str(source), source_sha256=sha(source),
                  architecture='Unchanged ScalarLogPolynomial, including matrix readout evaluation order',
                  dtype='float64', original_weights=original_w.tolist(),
                  boundaries='2**k and actual normalization boundaries .75*2**k; k=-1073..1023',
                  train_boundary_neighbor_width=1, evaluate_boundary_neighbor_width=32,
                  tie_semantics='User permits teacher-rounded ties; zero downward steps; teacher-strict pairs must remain strict',
                  optimizer='Active-constraint linear least-squares residual corrections with line search; exact-anchor supervised SGD',
                  ordinary_weight=1., boundary_weight=16., consistency_weight=16., ordered_pair_weight=64.,
                  generic_ridge=1e-10, max_updates=32, plateau_patience=6,
                  selection='Best training composite objective; original validation not used for selection',
                  heldout_note='Original random held-out examples retained. Required all-exponent boundary anchors enter original magnitude bands.',
                  no_snapping=True, no_epsilon=True, protected_hashes=protected)
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez_compressed(OUT / 'training_data.npz', broad_v=broad_v, broad_y=broad_y,
                        boundary_v=boundary_v, boundary_y=boundary_y, consistency_v=consistency_v)

    def anchor_update():
        # Real supervised optimizer step on g(1)=0, updating every parameter;
        # no direct assignment or coefficient-specific constraint.
        opt = torch.optim.SGD(model.parameters(), lr=1.)
        opt.zero_grad()
        output = model(torch.ones(1, dtype=torch.float64))
        loss = .5 * torch.sum(output ** 2)
        loss.backward()
        opt.step()

    def training_state():
        pb, pe = predict(model, broad_v), predict(model, boundary_v)
        consistency = predict(model, 2. * consistency_v) - predict(model, consistency_v)
        deficits = np.maximum(0., pair_gap - np.diff(pe))
        residuals = [(broad_y - pb) / broad_div / np.sqrt(len(broad_v)),
                     4. * (boundary_y - pe) / boundary_div / np.sqrt(len(boundary_v)),
                     4. * (log2_teacher - consistency) / np.sqrt(len(consistency_v)),
                     8. * deficits / pair_div / np.sqrt(len(pair_gap))]
        w = model.weights.detach().numpy()
        objective = float(sum(np.sum(r * r) for r in residuals) + (config['generic_ridge'] ** 2) * np.sum(w * w))
        return objective, residuals, deficits, pe

    anchor_update()
    history = []
    best_w = model.weights.detach().numpy().copy()
    best_objective = training_state()[0]
    stale = 0
    for update in range(1, config['max_updates'] + 1):
        objective, residuals, deficits, pe = training_state()
        w = model.weights.detach().numpy().copy()
        active = deficits > 0
        blocks = [fb / broad_div[:, None] / np.sqrt(len(broad_v)),
                  4. * fe / boundary_div[:, None] / np.sqrt(len(boundary_v)),
                  4. * fc / np.sqrt(len(consistency_v)),
                  8. * fp[active] / pair_div[active, None] / np.sqrt(len(pair_gap)),
                  config['generic_ridge'] * np.eye(len(w))]
        rhs = np.concatenate((residuals[0], residuals[1], residuals[2], residuals[3][active], -config['generic_ridge'] * w))
        matrix = np.concatenate(blocks)
        column_scale = np.linalg.norm(matrix, axis=0)
        delta = np.linalg.lstsq(matrix / column_scale, rhs, rcond=None)[0] / column_scale
        accepted = False
        for step in [1., .5, .25, .125]:
            with torch.no_grad():
                model.weights.copy_(torch.from_numpy(w + step * delta))
            anchor_update()
            trial = training_state()[0]
            if trial < objective:
                accepted = True
                break
        if not accepted:
            with torch.no_grad():
                model.weights.copy_(torch.from_numpy(w))
        objective, _, _, pe = training_state()
        if objective < best_objective:
            best_objective = objective
            best_w = model.weights.detach().numpy().copy()
            stale = 0
        else:
            stale += 1
        history.append(dict(update=update, objective=objective, accepted=accepted,
                            **pair_report(boundary_v, pe), weights=json.dumps(model.weights.detach().tolist())))
        print(json.dumps({k: v for k, v in history[-1].items() if k != 'weights'}), flush=True)
        if stale >= config['plateau_patience']:
            break
    with torch.no_grad():
        model.weights.copy_(torch.from_numpy(best_w))
    checkpoint = OUT / 'log_refined_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
                             status='SAVED_BEFORE_EVALUATION', updates=len(history), training_objective=best_objective))
    write_csv(OUT / 'training_history.csv', history)
    print(json.dumps(dict(event='saved_before_evaluation', sha256=digest)), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval()
    archived = np.load(SOURCE / 'heldout_data.npz')
    identities = np.load(SOURCE / 'identity_inputs.npz')
    old_model = ScalarLogPolynomial()
    old_model.load_state_dict(torch.load(source, weights_only=False)['state_dict'])
    old_model.eval()
    v, truth, tags = archived['v'], archived['targets'], archived['groups']
    p = predict(model, v)
    old_p = predict(old_model, v)
    rows = [dict(group='all', **metrics(truth, p))]
    for name in sorted(set(tags)):
        mask = tags == name
        rows.append(dict(group=str(name), **metrics(truth[mask], p[mask])))
    write_csv(OUT / 'log_refinement_validation.csv', rows)
    monotonic_rows, failure_rows = [], []
    for name, x in [('original_46164', identities['monotonic_inputs']),
                    ('dense_powers_of_two', adjacent_cloud(powers, 32)),
                    ('dense_actual_boundaries', adjacent_cloud(actual, 32))]:
        x = x[(x > 0) & np.isfinite(x)]
        xp = predict(model, x)
        monotonic_rows.append(dict(group=name, **pair_report(x, xp)))
        np.savez_compressed(OUT / f'{name}.npz', inputs=x, predictions=xp)
        td, pd = np.diff(np.log(x)), np.diff(xp)
        for i in np.flatnonzero((pd < 0) | ((td > 0) & (pd == 0))):
            failure_rows.append(dict(group=name, failure='reversal' if pd[i] < 0 else 'collapsed_teacher_strict',
                                     left=float(x[i]), right=float(x[i+1]), prediction_left=float(xp[i]),
                                     prediction_right=float(xp[i+1]), teacher_gap=float(td[i]), predicted_gap=float(pd[i])))
    write_csv(OUT / 'log_boundary_monotonicity.csv', monotonic_rows)
    if failure_rows:
        write_csv(OUT / 'log_boundary_failures.csv', failure_rows)
    xy = identities['product_pairs']
    px, py = predict(model, xy[:, 0]), predict(model, xy[:, 1])
    product_error = predict(model, xy[:, 0] * xy[:, 1]) - px - py
    rx = identities['reciprocal_inputs']
    reciprocal_error = predict(model, 1. / rx) + predict(model, rx)
    consistency_error = predict(model, 2. * consistency_v) - predict(model, consistency_v) - log2_teacher
    log_one = predict(model, np.array([1.]))
    properties = dict(log_one=float(log_one[0]), log_one_positive_zero=bool(log_one.view(np.uint64)[0] == 0),
                      product_max_abs=float(abs(product_error).max()),
                      product_max_normalized=float(np.max(abs(product_error) / np.maximum(1., abs(px) + abs(py)))),
                      reciprocal_max_abs=float(abs(reciprocal_error).max()),
                      reciprocal_max_normalized=float(np.max(abs(reciprocal_error) / np.maximum(1., abs(np.log(rx))))),
                      doubling_consistency_max_abs=float(abs(consistency_error).max()))
    old_metrics = metrics(truth, old_p)
    accuracy_not_worse = all(rows[0][key] <= old_metrics[key] for key in ['rmse', 'mae', 'max_absolute_error', 'max_normalized_error'])
    passed = bool(accuracy_not_worse and properties['log_one_positive_zero'] and
                  all(r['reversals'] == 0 and r['collapsed_teacher_strict_pairs'] == 0 for r in monotonic_rows) and
                  properties['product_max_normalized'] <= 128 * EPS and properties['reciprocal_max_normalized'] <= 128 * EPS)
    result = dict(operator='log_operator', status='DIAGNOSTIC_ONLY', passed=passed, checkpoint=str(checkpoint), sha256=digest,
                  source_sha256=sha(source), architecture_unchanged=True, updates=len(history),
                  validation=rows[0], previous_validation=old_metrics, accuracy_not_worse=bool(accuracy_not_worse),
                  monotonicity=monotonic_rows, properties=properties)
    if passed:
        lock = json.loads((ROOT / 'orientation_transform_endpoint_refine/orientation_transform_operator_locked.json').read_text())
        opath = Path(lock['checkpoint'])
        mpath = ROOT / 'multiplication_operator_smoke/multiplication_before_evaluation.pt'
        orientation, mul = OrientationTransform(), QuadraticArithmetic()
        orientation.load_state_dict(torch.load(opath, weights_only=False)['state_dict'])
        mul.load_state_dict(torch.load(mpath, weights_only=False)['state_dict'])
        orientation.eval()
        mul.eval()
        ax = inputs(np.concatenate((rng.uniform(0, 1, 1024), np.power(10., rng.uniform(-300, -1, 256)), np.array([0., .5, 1.]))))
        exact_v = orientation_teacher(ax)
        keep = exact_v > 0
        ax, exact_v = ax[keep], exact_v[keep]
        beta = rng.choice([-1., 1.], len(ax)) * np.power(10., rng.uniform(-50, 50, len(ax)))
        beta[::101] = 0.
        expected = beta * np.log(exact_v)
        with torch.no_grad():
            logged = model(orientation(torch.from_numpy(ax)))
            got = mul(torch.stack((torch.from_numpy(beta), logged), dim=1)).numpy()
        assembly = arithmetic_metrics(expected, got)
        assembly['passed'] = bool(assembly['nonfinite'] == 0 and assembly['zero_failures'] == 0 and assembly['max_relative_error'] <= 128 * EPS)
        write_csv(OUT / 'log_refined_assembly.csv', [assembly])
        result['assembly'] = assembly
        result['passed'] = assembly['passed']
        if assembly['passed']:
            result['status'] = 'LOCKED'
    result['assembly_run'] = 'assembly' in result
    assert all(sha(p) == h for p, h in protected.items())
    result['protected_hashes_unchanged'] = True
    result['seconds'] = time.monotonic() - start
    (OUT / 'log_refinement_manifest.json').write_text(json.dumps(result, indent=2))
    if result['status'] == 'LOCKED':
        (OUT / 'log_operator_locked.json').write_text(json.dumps(result, indent=2))
    (OUT / 'log_refinement_summary.txt').write_text(
        'Saved log checkpoint continued without architecture or inference-order changes.\n'
        'Teacher boundary labels, ordered pairs, doubling consistency, broad regression, supervised log(1)=0 anchor.\n'
        'Teacher-rounding ties permitted by explicit user clarification; reversals forbidden.\n'
        'No coefficient snapping, output clipping or boundary patching. Best checkpoint saved before evaluation.\n'
        'Boundary anchors span all exponents, including original held-out magnitude bands; old random held-out examples reused only for evaluation.\n\n'
        + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
