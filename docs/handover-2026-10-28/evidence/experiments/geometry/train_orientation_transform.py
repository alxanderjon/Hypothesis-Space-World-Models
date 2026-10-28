"""Synthetic-only raw cut/orientation -> learned bilinear transform."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.train_inner_quadratic_operator import PolynomialArithmetic
from experiments.geometry.train_sqrt_operator import ScalarPower
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic, metrics as arithmetic_metrics
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'orientation_transform_smoke'
SEED = 27372026
EPS = np.finfo(np.float64).eps
HOLDOUT = [(0.125, 0.25), (0.45, 0.55), (0.8, 0.9)]


class OrientationTransform(nn.Module):
    """Generic bilinear readout, evaluated as intercept(o) + slope(o)*c.

    All four coefficients learned. Orientation is categorical bookkeeping:
    forward=0, reverse=1. No complementary coordinate supplied as input.
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(.2 * torch.randn(4, dtype=torch.float64))

    def forward(self, x):
        c, o = x[:, 0], x[:, 1]
        w = self.weights
        return (w[0] + w[1] * o) + (w[2] + w[3] * o) * c


def inputs(c):
    return np.column_stack((np.repeat(c, 2), np.tile([0., 1.], len(c))))


def teacher(x):
    return np.where(x[:, 1] == 0, x[:, 0], 1. - x[:, 0])


def predict(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def metrics(y, p):
    e = p - y
    return dict(examples=len(y), rmse=float(np.sqrt(np.mean(e * e))),
                mae=float(np.mean(abs(e))), max_error=float(abs(e).max()),
                exact_agreement=int(np.sum(y == p)),
                bit_exact=int(np.sum(y.view(np.uint64) == p.view(np.uint64))),
                nonfinite=int(np.sum(~np.isfinite(p))),
                outside_unit_interval=int(np.sum((p < 0) | (p > 1))))


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    c = rng.uniform(0, 1, 5000)
    keep = np.ones(len(c), dtype=bool)
    for lo, hi in HOLDOUT:
        keep &= ~((c >= lo) & (c <= hi))
    c = np.concatenate((c[keep][:2048], np.tile([0., 1.], 32)))
    x = inputs(c)
    y = teacher(x)
    phi = np.column_stack((np.ones(len(x)), x[:, 1], x[:, 0], x[:, 0] * x[:, 1]))
    model = OrientationTransform()
    w = model.weights.detach().numpy().copy()
    initial = w.copy()
    config = dict(seed=SEED, dtype='float64', inputs=['c', 'orientation'],
                  orientation_encoding={'forward': 0, 'reverse': 1},
                  model='(w0+w1*o)+(w2+w3*o)*c', initial_weights=initial.tolist(),
                  training_examples=len(x), heldout_ranges=HOLDOUT,
                  optimizer='L1 cyclic coordinate descent; supervised least-squares debias and training residual correction',
                  l1=1e-6, max_sweeps=10000, max_residual_corrections=16,
                  coefficient_rules='No teacher assignment, coefficient snapping, or output clipping',
                  absolute_gate=32 * EPS, assembly_relative_gate=128 * EPS,
                  exact_output_bounds_required=True, protected_hashes=protected)
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez_compressed(OUT / 'training_data.npz', inputs=x, targets=y)
    gram, rhs = phi.T @ phi / len(x), phi.T @ y / len(x)
    history = []
    for sweep in range(1, config['max_sweeps'] + 1):
        old = w.copy()
        for j in range(len(w)):
            value = w[j] + (rhs[j] - gram[j] @ w) / gram[j, j]
            w[j] = np.sign(value) * max(abs(value) - config['l1'] / gram[j, j], 0.)
        if sweep == 1 or sweep % 25 == 0:
            history.append(dict(phase='sparse_fit', update=sweep, rmse=float(np.sqrt(np.mean((phi @ w - y) ** 2))), weights=json.dumps(w.tolist())))
        if max(abs(w - old)) < 1e-14:
            break
    support = np.flatnonzero(w != 0)
    assert len(support)
    w[:] = 0.
    w[support] = np.linalg.lstsq(phi[:, support], y, rcond=None)[0]
    # Refine only on training residuals using the model's actual evaluation order.
    # This is a generic linear least-squares correction, never teacher snapping.
    def set_weights(weights):
        with torch.no_grad():
            model.weights.copy_(torch.from_numpy(weights))
    set_weights(w)
    for update in range(config['max_residual_corrections']):
        residual = y - predict(model, x)
        loss = float(np.sum(residual ** 2))
        history.append(dict(phase='debias', update=update, rmse=float(np.sqrt(loss / len(x))), weights=json.dumps(w.tolist())))
        if loss == 0:
            break
        delta = np.linalg.lstsq(phi[:, support], residual, rcond=None)[0]
        proposal = w.copy()
        proposal[support] += delta
        set_weights(proposal)
        new_loss = float(np.sum((y - predict(model, x)) ** 2))
        if new_loss < loss:
            w = proposal
        else:
            set_weights(w)
            break
    checkpoint = OUT / 'orientation_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
                             status='SAVED_BEFORE_EVALUATION', learned_support=support.tolist(), sweeps=sweep))
    write_csv(OUT / 'training_history.csv', history)
    print(json.dumps(dict(event='saved_before_evaluation', weights=w.tolist(), sha256=digest)), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval()

    sets = [('new_uniform', rng.uniform(0, 1, 4096))]
    sets += [(f'heldout_{lo}_{hi}', rng.uniform(lo, hi, 2048)) for lo, hi in HOLDOUT]
    sets += [('dense_grid', np.linspace(0, 1, 10001)),
             ('endpoints_and_neighbors', np.array([0., np.nextafter(0., 1.), 1e-300, 1e-16,
                np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), np.nextafter(1., 0.), 1.])),
             ('near_endpoints', np.concatenate((np.power(10., rng.uniform(-300, -1, 1024)),
                                                1. - np.power(10., rng.uniform(-16, -1, 1024)))))]
    vx = np.concatenate([inputs(v) for _, v in sets])
    tags = np.concatenate([np.repeat(name, 2 * len(v)) for name, v in sets])
    truth, pred = teacher(vx), predict(model, vx)
    rows = [dict(grouping='all', group='all', orientation='both', **metrics(truth, pred))]
    for orientation, name in [(0., 'forward'), (1., 'reverse')]:
        mask = vx[:, 1] == orientation
        rows.append(dict(grouping='orientation', group=name, orientation=name, **metrics(truth[mask], pred[mask])))
    for group in sorted(set(tags)):
        for orientation, name in [(None, 'both'), (0., 'forward'), (1., 'reverse')]:
            mask = tags == group
            if orientation is not None:
                mask &= vx[:, 1] == orientation
            rows.append(dict(grouping='data_family', group=group, orientation=name, **metrics(truth[mask], pred[mask])))
    write_csv(OUT / 'orientation_validation.csv', rows)
    np.savez_compressed(OUT / 'heldout_data.npz', inputs=vx, targets=truth, predictions=pred, groups=tags)
    forward, reverse = pred[::2], pred[1::2]
    complementary = np.column_stack((1. - vx[::2, 0], np.zeros(len(forward))))
    properties = dict(complement_sum_max_error=float(abs(forward + reverse - 1.).max()),
                      complemented_input_max_error=float(abs(predict(model, complementary) - reverse).max()),
                      outside_unit_interval=int(np.sum((pred < 0) | (pred > 1))),
                      output_min=float(pred.min()), output_max=float(pred.max()))
    important = inputs(np.array([0., .5, 1.]))
    important_pred = predict(model, important)
    write_csv(OUT / 'orientation_important_values.csv', [dict(c=float(a), orientation='forward' if b == 0 else 'reverse',
               teacher=float(t), learned=float(p)) for (a, b), t, p in zip(important, teacher(important), important_pred)])
    overall = rows[0]
    passed = bool(overall['nonfinite'] == 0 and overall['max_error'] <= config['absolute_gate'] and
                  properties['outside_unit_interval'] == 0 and
                  properties['complement_sum_max_error'] <= config['absolute_gate'] and
                  properties['complemented_input_max_error'] <= config['absolute_gate'])
    result = dict(operator='orientation_transform_operator', status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',
                  passed=passed, checkpoint=str(checkpoint), sha256=digest, learned_weights=w.tolist(),
                  training_metrics=metrics(y, predict(model, x)), validation=overall, properties=properties)
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    if passed:
        paths = [ROOT / 'inner_quadratic_operator_smoke/inner_quadratic_before_evaluation.pt',
                 ROOT / 'sqrt_operator_smoke/sqrt_before_evaluation.pt',
                 ROOT / 'multiplication_operator_smoke/multiplication_before_evaluation.pt']
        inner, root, mul = PolynomialArithmetic(), ScalarPower(), QuadraticArithmetic()
        for component, path in zip((inner, root, mul), paths):
            component.load_state_dict(torch.load(path, weights_only=False)['state_dict'])
            component.eval()
        # Fresh conditions, both orientations for exactly matched coefficients.
        ac = np.concatenate([rng.uniform(lo, hi, 320) for lo, hi in HOLDOUT] + [np.array([0., .5, 1.])])
        ax = inputs(ac)
        B = np.repeat(np.power(10., rng.uniform(-20, 20, len(ac))), 2)
        a = np.repeat(rng.choice([-1., 1.], len(ac)) * np.power(10., rng.uniform(-8, 8, len(ac))), 2)
        A = np.repeat(rng.choice([-1., 1.], len(ac)) * np.power(10., rng.uniform(-50, 50, len(ac))), 2)
        # Include B=0 endpoints; sqrt has q>0 contract, so skip q=0 rather than add epsilon.
        B[-6:] = 0.
        exact_t = teacher(ax)
        exact_q = B + (a * a) * (exact_t * exact_t)
        valid = exact_q > 0
        skipped = int(np.sum(~valid))
        ax, B, a, A, exact_q = ax[valid], B[valid], a[valid], A[valid], exact_q[valid]
        expected = A * np.sqrt(exact_q)
        with torch.no_grad():
            t = model(torch.from_numpy(ax))
            q = inner(torch.stack((torch.from_numpy(B), torch.from_numpy(a), t), dim=1))
            r = root(q)
            assembled = mul(torch.stack((torch.from_numpy(A), r), dim=1)).numpy()
        ar = []
        for orientation, name in [(None, 'both'), (0., 'forward'), (1., 'reverse')]:
            mask = np.ones(len(ax), dtype=bool) if orientation is None else ax[:, 1] == orientation
            ar.append(dict(orientation=name, **arithmetic_metrics(expected[mask], assembled[mask])))
        assembled_pass = bool(ar[0]['nonfinite'] == 0 and ar[0]['sign_failures'] == 0 and
                              ar[0]['max_relative_error'] <= config['assembly_relative_gate'])
        result['assembly'] = dict(passed=assembled_pass, skipped_q_zero_outside_sqrt_domain=skipped, metrics=ar)
        result['upstream_hashes'] = {str(p): sha(p) for p in paths}
        write_csv(OUT / 'orientation_assembled_validation.csv', ar)
        np.savez_compressed(OUT / 'assembled_data.npz', inputs=ax, B=B, a=a, A=A, targets=expected, predictions=assembled)
        if not assembled_pass:
            result['passed'] = False
            result['status'] = 'DIAGNOSTIC_ONLY'
    assert all(sha(p) == h for p, h in protected.items())
    result['protected_hashes_unchanged'] = True
    result['seconds'] = time.monotonic() - start
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    (OUT / 'orientation_summary.txt').write_text(
        'Orientation transform smoke, synthetic c and orientation only.\n'
        'Generic bilinear model; all four coefficients randomly initialized and supervised.\n'
        'No teacher coefficient assignment, coefficient snapping or output clipping.\n'
        'Forward=0, reverse=1; balanced in training and validation.\n'
        'Checkpoint persisted before validation; all pre-existing checkpoint hashes unchanged.\n'
        'Assembled check respects positive-q contract of frozen sqrt; no epsilon.\n\n' + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
