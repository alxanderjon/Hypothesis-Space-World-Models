"""Raw-positive-input log smoke; internal range reduction, learned polynomial.

No log function occurs in model inference. Binary range reduction is explicit
deterministic arithmetic scaffolding; all readout coefficients are supervised.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.train_orientation_transform import OrientationTransform, inputs, teacher as orientation_teacher
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic, metrics as arithmetic_metrics
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'log_operator_smoke'
SEED = 27392026
EPS = np.finfo(np.float64).eps
DEGREE = 24
HOLDOUT = [(-160., -140.), (-80., -60.), (-4., -2.), (2., 4.), (60., 80.), (140., 160.)]


class ScalarLogPolynomial(nn.Module):
    """Raw v only. Generic polynomial in internally range-reduced coordinate.

    frexp: v=m*2**e. Shift m to [.75,1.5). Internal u=5*(m-1)/(m+1).
    y=w_e*e + sum_{j=0}^{24} w_j*u**j.
    The binary normalization, rational coordinate and polynomial degree are
    architecture/scaffolding; neither log(2) nor series coefficients are fixed.
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(.2 * torch.randn(DEGREE + 2, dtype=torch.float64))

    def features(self, v):
        m, e = torch.frexp(v)
        shift = m < .75
        m = torch.where(shift, 2. * m, m)
        e = e.to(torch.float64) - shift.to(torch.float64)
        u = 5. * ((m - 1.) / (m + 1.))
        columns = [torch.ones_like(u)]
        for _ in range(DEGREE):
            columns.append(columns[-1] * u)
        return torch.stack(columns + [e], dim=1)

    def forward(self, v):
        return self.features(v) @ self.weights


def predict(model, v):
    with torch.no_grad():
        return model(torch.from_numpy(v)).numpy()


def metrics(y, p):
    e = p - y
    maximum = float(abs(e).max())
    normalized = abs(e) / np.maximum(1., abs(y))
    nz = y != 0
    relative = abs(e[nz] / y[nz])
    return dict(examples=len(y), rmse=float(maximum * np.sqrt(np.mean((e / maximum) ** 2))) if maximum else 0.,
                mae=float(np.mean(abs(e))), max_absolute_error=maximum,
                max_normalized_error=float(normalized.max()),
                max_relative_error_nonzero=float(relative.max()) if len(relative) else 0.,
                exact_agreement=int(np.sum(y == p)), nonfinite=int(np.sum(~np.isfinite(p))))


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    exponent = rng.uniform(-200, 200, 8000)
    keep = np.ones(len(exponent), dtype=bool)
    for lo, hi in HOLDOUT:
        keep &= ~((exponent >= lo) & (exponent <= hi))
    broad = np.power(10., exponent[keep][:4096])
    near_one = 1. + rng.choice([-1., 1.], 2048) * np.power(10., rng.uniform(-16, -1, 2048))
    v = np.concatenate((broad, rng.uniform(.75, 1.5, 2048), near_one, np.ones(64)))
    assert np.all(v > 0) and np.isfinite(v).all()
    y = np.log(v)
    model = ScalarLogPolynomial()
    initial = model.weights.detach().numpy().copy()
    with torch.no_grad():
        phi = model.features(torch.from_numpy(v)).numpy()
    # Nonzero targets use relative residual; exact log(1)=0 examples use absolute
    # residual. No epsilon is added to input or teacher; only loss is normalized.
    divisor = abs(y).copy()
    divisor[divisor == 0] = 1.
    R, target = phi / divisor[:, None], y / divisor
    scale = np.sqrt(np.mean(R * R, axis=0))
    Z = R / scale
    config = dict(seed=SEED, dtype='float64', input='raw strictly positive v only', degree=DEGREE,
                  architecture='Binary frexp normalization plus generic degree-24 polynomial and learned exponent coefficient',
                  normalization='m in [.75,1.5), u=5*(m-1)/(m+1); deterministic scaffolding',
                  teacher='np.log(v), labels/evaluation only', initial_weights=initial.tolist(),
                  optimizer='Supervised linear least-squares residual updates from random initialization; no coefficient snapping',
                  loss='Squared relative residual for nonzero targets; absolute residual at log(1)=0',
                  train_log10_range=[-200, 200], heldout_log10_bands=HOLDOUT,
                  training_examples=len(v), normalized_error_gate=64 * EPS,
                  identity_normalized_error_gate=128 * EPS, near_one_absolute_gate=64 * EPS,
                  protected_hashes=protected)
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez_compressed(OUT / 'training_data.npz', v=v, targets=y)
    w = initial.copy()
    history = []
    # The first solve minimizes the same supervised objective on all coefficients;
    # applying its correction starts from, and updates, the random parameters.
    normalized_w = w * scale
    delta = np.linalg.lstsq(Z, target - Z @ normalized_w, rcond=None)[0]
    normalized_w += delta
    w = normalized_w / scale
    with torch.no_grad():
        model.weights.copy_(torch.from_numpy(w))
    for update in range(12):
        residual = (y - predict(model, v)) / divisor
        loss = float(np.mean(residual ** 2))
        history.append(dict(update=update, weighted_rmse=float(np.sqrt(loss)), weights=json.dumps(w.tolist())))
        correction = np.linalg.lstsq(Z, residual, rcond=None)[0] / scale
        proposal = w + correction
        with torch.no_grad():
            model.weights.copy_(torch.from_numpy(proposal))
        new_loss = float(np.mean(((y - predict(model, v)) / divisor) ** 2))
        if new_loss < loss:
            w = proposal
        else:
            with torch.no_grad():
                model.weights.copy_(torch.from_numpy(w))
            break
    checkpoint = OUT / 'log_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
                             status='SAVED_BEFORE_EVALUATION', training_metrics=metrics(y, predict(model, v))))
    write_csv(OUT / 'training_history.csv', history)
    write_csv(OUT / 'learned_coefficients.csv', [dict(term=f'u^{i}' if i <= DEGREE else 'binary_exponent', coefficient=float(a)) for i, a in enumerate(w)])
    print(json.dumps(dict(event='saved_before_evaluation', sha256=digest, training=metrics(y, predict(model, v)))), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval()
    groups = [('new_combinations', np.power(10., rng.uniform(-200, 200, 4096))),
              ('tiny_extrapolation', np.power(10., rng.uniform(-300, -200, 2048))),
              ('large_extrapolation', np.power(10., rng.uniform(200, 300, 2048))),
              ('near_one', 1. + rng.choice([-1., 1.], 4096) * np.power(10., rng.uniform(-16, -1, 4096))),
              ('order_one', rng.uniform(.5, 2., 2048))]
    groups += [(f'heldout_{lo:g}_{hi:g}', np.power(10., rng.uniform(lo, hi, 1024))) for lo, hi in HOLDOUT]
    groups += [('float64_extremes', np.array([np.nextafter(0., 1.), np.finfo(np.float64).tiny,
                  np.nextafter(1., 0.), 1., np.nextafter(1., np.inf), np.finfo(np.float64).max]))]
    vx = np.concatenate([values for _, values in groups])
    tags = np.concatenate([np.repeat(name, len(values)) for name, values in groups])
    truth, pred = np.log(vx), predict(model, vx)
    rows = [dict(grouping='all', group='all', **metrics(truth, pred))]
    for name, _ in groups:
        mask = tags == name
        rows.append(dict(grouping='family', group=name, **metrics(truth[mask], pred[mask])))
    exponents = np.log10(vx)
    for lo in range(-340, 320, 20):
        mask = (exponents >= lo) & (exponents < lo + 20)
        if mask.any():
            rows.append(dict(grouping='log10_v_band', group=f'[{lo},{lo+20})', **metrics(truth[mask], pred[mask])))
    write_csv(OUT / 'log_validation.csv', rows)
    np.savez_compressed(OUT / 'heldout_data.npz', v=vx, targets=truth, predictions=pred, groups=tags)

    def identity(left, right, normalization):
        err = left - right
        return dict(examples=len(left), max_absolute_error=float(abs(err).max()),
                    max_normalized_error=float(np.max(abs(err) / normalization)),
                    nonfinite=int(np.sum(~np.isfinite(left)) + np.sum(~np.isfinite(right))))
    xy = np.power(10., rng.uniform(-140, 140, (8192, 2)))
    px, py = predict(model, xy[:, 0]), predict(model, xy[:, 1])
    product_identity = identity(predict(model, xy[:, 0] * xy[:, 1]), px + py,
                                np.maximum(1., abs(px) + abs(py)))
    teacher_product = identity(np.log(xy[:, 0] * xy[:, 1]), np.log(xy[:, 0]) + np.log(xy[:, 1]),
                              np.maximum(1., abs(np.log(xy[:, 0])) + abs(np.log(xy[:, 1]))))
    rx = np.power(10., rng.uniform(-300, 300, 8192))
    reciprocal = identity(predict(model, 1. / rx), -predict(model, rx), np.maximum(1., abs(np.log(rx))))
    teacher_reciprocal = identity(np.log(1. / rx), -np.log(rx), np.maximum(1., abs(np.log(rx))))
    # Include representable neighbors of every range-reduction boundary throughout
    # the finite normal range to expose possible discontinuity/monotonicity errors.
    boundary = np.ldexp(np.full(2045, .75), np.arange(-1021, 1024))
    ordered = np.unique(np.concatenate((vx, np.logspace(-300, 300, 20000), boundary,
                                       np.nextafter(boundary, 0.), np.nextafter(boundary, np.inf))))
    output = predict(model, ordered)
    differences = np.diff(output)
    reversals = differences < 0
    properties = dict(log_one=float(predict(model, np.array([1.]))[0]),
                      product_identity=product_identity, teacher_product_identity=teacher_product,
                      reciprocal_identity=reciprocal, teacher_reciprocal_identity=teacher_reciprocal,
                      monotonicity_queries=len(ordered), monotonicity_reversals=int(np.sum(reversals)),
                      largest_monotonicity_drop=float(-differences[reversals].min()) if reversals.any() else 0.)
    write_csv(OUT / 'log_identities.csv', [dict(test='product', **product_identity), dict(test='reciprocal', **reciprocal),
                                         dict(test='teacher_product_rounding', **teacher_product), dict(test='teacher_reciprocal_rounding', **teacher_reciprocal)])
    (OUT / 'log_properties.json').write_text(json.dumps(properties, indent=2))
    np.savez_compressed(OUT / 'identity_inputs.npz', product_pairs=xy, reciprocal_inputs=rx, monotonic_inputs=ordered)
    passed = bool(rows[0]['nonfinite'] == 0 and rows[0]['max_normalized_error'] <= config['normalized_error_gate'] and
                  abs(properties['log_one']) <= config['near_one_absolute_gate'] and
                  properties['monotonicity_reversals'] == 0 and
                  product_identity['max_normalized_error'] <= config['identity_normalized_error_gate'] and
                  reciprocal['max_normalized_error'] <= config['identity_normalized_error_gate'])
    result = dict(operator='log_operator', status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY', passed=passed,
                  checkpoint=str(checkpoint), sha256=digest, training_examples=len(v), validation=rows[0], properties=properties,
                  learned_binary_exponent_coefficient=float(w[-1]), learned_constant=float(w[0]))
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    if passed:
        lock = json.loads((ROOT / 'orientation_transform_endpoint_refine/orientation_transform_operator_locked.json').read_text())
        orientation_path = Path(lock['checkpoint'])
        mul_path = ROOT / 'multiplication_operator_smoke/multiplication_before_evaluation.pt'
        orientation, mul = OrientationTransform(), QuadraticArithmetic()
        orientation.load_state_dict(torch.load(orientation_path, weights_only=False)['state_dict'])
        mul.load_state_dict(torch.load(mul_path, weights_only=False)['state_dict'])
        orientation.eval()
        mul.eval()
        c = np.concatenate((rng.uniform(0, 1, 1024), np.power(10., rng.uniform(-300, -1, 256)),
                            np.array([0., .5, 1., np.nextafter(1., 0.)])))
        ax = inputs(c)
        exact_v = orientation_teacher(ax)
        mask = exact_v > 0
        excluded = int(np.sum(~mask))
        ax, exact_v = ax[mask], exact_v[mask]
        beta = rng.choice([-1., 1.], len(ax)) * np.power(10., rng.uniform(-50, 50, len(ax)))
        beta[::101] = 0.
        expected = beta * np.log(exact_v)
        with torch.no_grad():
            oriented = orientation(torch.from_numpy(ax))
            logged = model(oriented)
            got = mul(torch.stack((torch.from_numpy(beta), logged), dim=1)).numpy()
        ar = arithmetic_metrics(expected, got)
        normalized = float(np.max(abs(got - expected) / np.maximum(1., abs(expected))))
        ar['max_normalized_error'] = normalized
        ar['excluded_v_zero'] = excluded
        ar['passed'] = bool(ar['nonfinite'] == 0 and normalized <= config['identity_normalized_error_gate'] and ar['zero_failures'] == 0)
        result['assembly'] = ar
        result['upstream_hashes'] = {str(orientation_path): sha(orientation_path), str(mul_path): sha(mul_path)}
        write_csv(OUT / 'log_assembled_validation.csv', [ar])
        np.savez_compressed(OUT / 'assembled_data.npz', inputs=ax, beta=beta, targets=expected, predictions=got)
        if not ar['passed']:
            result['passed'] = False
            result['status'] = 'DIAGNOSTIC_ONLY'
    assert all(sha(p) == h for p, h in protected.items())
    result['protected_hashes_unchanged'] = True
    result['seconds'] = time.monotonic() - start
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    (OUT / 'log_summary.txt').write_text(
        'Raw-positive-input logarithm smoke; no epsilon. No log call inside inference.\n'
        'Binary normalization and rational internal coordinate are deterministic range-reduction scaffolding.\n'
        'All polynomial and binary-exponent coefficients learned from synthetic labels; no coefficient snapping.\n'
        'Entire magnitude bands held out. Magnitude-normalized error uses max(1,abs(log(v))) only in evaluation.\n'
        'Identities also compared to teacher rounding; monotonicity tested at binary range-reduction boundaries.\n'
        'Checkpoint saved before evaluation. All old checkpoints unchanged. No later primitive trained.\n\n'
        + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
