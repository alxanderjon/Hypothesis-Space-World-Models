"""Raw-input, synthetic-only sparse polynomial multiplication smoke."""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.train_inner_quadratic_operator import PolynomialArithmetic, samples, labels
from experiments.geometry.train_sqrt_operator import ScalarPower
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'multiplication_operator_smoke'
SEED = 27362026
EPS = np.finfo(np.float64).eps
POWERS = tuple(p for degree in range(3) for p in itertools.product(range(degree + 1), repeat=2) if sum(p) == degree)
EDGES = np.array([-40., -20., -4., 0., 4., 20., 40.])


class QuadraticArithmetic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(.2 * torch.randn(len(POWERS), dtype=torch.float64))
        self.register_buffer('feature_scale', torch.ones(len(POWERS), dtype=torch.float64))

    def features(self, x, indices):
        columns = []
        for j in indices:
            term = torch.ones_like(x[:, 0])
            for axis, power in enumerate(POWERS[j]):
                for _ in range(power):
                    term = term * x[:, axis]
            columns.append(term)
        return torch.stack(columns, dim=1)

    def forward(self, x):
        # Generic sparse execution: do not evaluate pruned terms. In particular,
        # irrelevant powers could overflow for very imbalanced finite inputs.
        active = torch.nonzero(self.weights.detach() != 0).flatten().tolist()
        if not active:
            return torch.zeros_like(x[:, 0])
        return self.features(x, active) @ (self.weights[active] / self.feature_scale[active])


def pairs(n, seed, family):
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        e = rng.uniform(-40, 40, 2)
        band = np.searchsorted(EDGES[1:-1], e, side='right')
        reserved = int(band.sum()) % 3 == 1
        if family in ('train', 'seen_combinations') and reserved:
            continue
        if family == 'heldout_combinations' and not reserved:
            continue
        if family == 'wide_extrapolation':
            e = rng.uniform(-140, 140, 2)
        elif family == 'imbalanced_extrapolation':
            e = np.array([rng.uniform(150, 300), rng.uniform(-300, -150)])
            if len(rows) % 2:
                e = e[::-1]
        elif family == 'order_one':
            e = rng.uniform(-.3, .3, 2)
        # Cycle signs to guarantee all four combinations are equally represented.
        sign = np.array([(1., 1.), (1., -1.), (-1., 1.), (-1., -1.)][len(rows) % 4])
        rows.append(sign * np.power(10., e))
    return np.array(rows, dtype=np.float64)


def zero_pairs(n, seed, max_exponent):
    rng = np.random.default_rng(seed)
    v = rng.choice([-1., 1.], n) * np.power(10., rng.uniform(-max_exponent, max_exponent, n))
    z = np.copysign(np.zeros(n), rng.choice([-1., 1.], n))
    return np.concatenate((np.column_stack((v, z)), np.column_stack((z, v)), np.zeros((4, 2))))


def predict(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def metrics(y, p):
    e = p - y
    maximum = float(abs(e).max())
    nonzero = y != 0
    relative = e[nonzero] / abs(y[nonzero])
    return dict(examples=len(y), rmse=float(maximum * np.sqrt(np.mean((e / maximum) ** 2))) if maximum else 0.,
                mae=float(np.mean(abs(e))), max_absolute_error=maximum,
                relative_rmse=float(np.sqrt(np.mean(relative ** 2))) if len(relative) else 0.,
                relative_mae=float(np.mean(abs(relative))) if len(relative) else 0.,
                max_relative_error=float(abs(relative).max()) if len(relative) else 0.,
                nonfinite=int(np.sum(~np.isfinite(p))),
                sign_failures=int(np.sum(np.sign(p[nonzero]) != np.sign(y[nonzero]))),
                zero_cases=int(np.sum(~nonzero)), zero_failures=int(np.sum(p[~nonzero] != 0)),
                signed_zero_differences=int(np.sum(np.signbit(p[~nonzero]) != np.signbit(y[~nonzero]))),
                exact_values=int(np.sum(p == y)))


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    x = np.concatenate((pairs(4096, SEED, 'train'), zero_pairs(64, SEED + 1, 40)))
    y = x[:, 0] * x[:, 1]
    model = QuadraticArithmetic()
    initial = model.weights.detach().numpy().copy()
    with torch.no_grad():
        phi = model.features(torch.from_numpy(x), list(range(len(POWERS)))).numpy()
    divisor = abs(y).copy()
    divisor[divisor == 0] = 1.
    R, target = phi / divisor[:, None], y / divisor
    scale = np.sqrt(np.mean(R * R, axis=0))
    assert np.isfinite(scale).all() and np.all(scale > 0)
    Z = R / scale
    gram, rhs = Z.T @ Z / len(x), Z.T @ target / len(x)
    config = dict(seed=SEED, dtype='float64', inputs=['x1', 'x2'], basis_exponents=POWERS,
                  initial_weights=initial.tolist(), training_examples=len(x),
                  architecture='All six monomials through total degree two; generic learned sparse readout',
                  sparse_execution='Only learned nonzero terms evaluated; no teacher-specific term selection',
                  optimizer='Random-initialized L1 coordinate descent, then supervised least-squares debias',
                  l1=1e-5, loss='Squared relative residual for nonzero products; absolute residual at zero',
                  training_log10_range=[-40, 40], combination_edges=EDGES.tolist(),
                  heldout_rule='(x1_magnitude_band_index + x2_magnitude_band_index) % 3 == 1',
                  relative_gate=128 * EPS, scaling_relative_gate=256 * EPS,
                  protected_hashes=protected)
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez_compressed(OUT / 'training_data.npz', inputs=x, targets=y)
    w = initial.copy()
    history = []
    for sweep in range(1, 10001):
        old = w.copy()
        for j in range(len(w)):
            value = w[j] + (rhs[j] - gram[j] @ w) / gram[j, j]
            w[j] = np.sign(value) * max(abs(value) - config['l1'] / gram[j, j], 0.)
        if sweep == 1 or sweep % 25 == 0:
            history.append(dict(sweep=sweep, loss_rmse=float(np.sqrt(np.mean((Z @ w - target) ** 2))),
                                active_terms=int(np.count_nonzero(w))))
        if max(abs(w - old)) < 1e-13:
            break
    support = np.flatnonzero(w != 0)
    assert len(support)
    w[:] = 0.
    w[support] = np.linalg.lstsq(Z[:, support], target, rcond=None)[0]
    for _ in range(3):
        delta = np.linalg.lstsq(Z[:, support], target - Z @ w, rcond=None)[0]
        proposal = w.copy()
        proposal[support] += delta
        if np.linalg.norm(target - Z @ proposal) < np.linalg.norm(target - Z @ w):
            w = proposal
        else:
            break
    with torch.no_grad():
        model.weights.copy_(torch.from_numpy(w))
        model.feature_scale.copy_(torch.from_numpy(scale))
    terms = [dict(exponents=list(POWERS[j]), coefficient=float(w[j] / scale[j])) for j in support]
    checkpoint = OUT / 'multiplication_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
                             status='SAVED_BEFORE_EVALUATION', learned_terms=terms, sweeps=sweep))
    write_csv(OUT / 'training_history.csv', history)
    print(json.dumps(dict(event='saved_before_evaluation', learned_terms=terms, sha256=digest)), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval()

    families = ['seen_combinations', 'heldout_combinations', 'wide_extrapolation', 'imbalanced_extrapolation', 'order_one']
    xx = [pairs(4096, SEED + 100 + i, g) for i, g in enumerate(families)]
    groups = [np.repeat(g, len(v)) for g, v in zip(families, xx)]
    zeros = zero_pairs(1024, SEED + 200, 300)
    xx.append(zeros)
    groups.append(np.repeat('zeros', len(zeros)))
    vx, tags = np.concatenate(xx), np.concatenate(groups)
    truth = vx[:, 0] * vx[:, 1]
    pred = predict(model, vx)
    assert np.isfinite(truth).all()
    rows = [dict(grouping='all', group='all', **metrics(truth, pred))]
    for g in sorted(set(tags)):
        mask = tags == g
        rows.append(dict(grouping='data_family', group=g, **metrics(truth[mask], pred[mask])))
    signs = np.array([('+' if a > 0 else '-' if a < 0 else '0') + ('+' if b > 0 else '-' if b < 0 else '0') for a, b in vx])
    for g in sorted(set(signs)):
        mask = signs == g
        rows.append(dict(grouping='sign_combination', group=g, **metrics(truth[mask], pred[mask])))
    for axis in range(2):
        logmag = np.full(len(vx), -np.inf)
        positive = vx[:, axis] != 0
        logmag[positive] = np.log10(abs(vx[positive, axis]))
        for lo in range(-300, 300, 20):
            mask = (logmag >= lo) & (logmag < lo + 20)
            if mask.any():
                rows.append(dict(grouping=f'log10_abs_x{axis+1}', group=f'[{lo},{lo+20})', **metrics(truth[mask], pred[mask])))
    write_csv(OUT / 'multiplication_validation.csv', rows)
    np.savez_compressed(OUT / 'heldout_data.npz', inputs=vx, targets=truth, predictions=pred, groups=tags)
    swapped = predict(model, vx[:, ::-1].copy())
    sign_checks = []
    for axis in range(2):
        flipped = vx.copy()
        flipped[:, axis] *= -1
        sign_checks.append(metrics(-pred, predict(model, flipped)))
    rng = np.random.default_rng(SEED + 300)
    base = rng.choice([-1., 1.], (8192, 2)) * np.power(10., rng.uniform(-100, 100, (8192, 2)))
    k = rng.choice([-1., 1.], len(base)) * np.power(10., rng.uniform(-40, 40, len(base)))
    k[::101] = 0.
    scaled = base.copy()
    scaled[:, 0] *= k
    scaling = metrics(k * predict(model, base), predict(model, scaled))
    properties = dict(commutativity=metrics(pred, swapped), sign_flip_x1=sign_checks[0],
                      sign_flip_x2=sign_checks[1], scaling=scaling,
                      zero_semantics='Numeric zero required; signed-zero bit differences separately reported',
                      scaling_k_abs_max=float(abs(k).max()))
    (OUT / 'multiplication_properties.json').write_text(json.dumps(properties, indent=2))
    np.savez_compressed(OUT / 'scaling_data.npz', inputs=base, k=k)
    m = rows[0]
    passed = bool(m['nonfinite'] == 0 and m['sign_failures'] == 0 and m['zero_failures'] == 0 and
                  m['max_relative_error'] <= config['relative_gate'] and
                  properties['commutativity']['max_absolute_error'] == 0 and
                  all(r['max_absolute_error'] == 0 for r in sign_checks) and
                  scaling['nonfinite'] == 0 and scaling['zero_failures'] == 0 and
                  scaling['max_relative_error'] <= config['scaling_relative_gate'])
    result = dict(operator='multiplication_operator', status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',
                  passed=passed, checkpoint=str(checkpoint), sha256=digest, learned_terms=terms,
                  training_examples=len(x), training_metrics=metrics(y, predict(model, x)),
                  validation=m, properties=properties)
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    if passed:
        inner_path = ROOT / 'inner_quadratic_operator_smoke/inner_quadratic_before_evaluation.pt'
        sqrt_path = ROOT / 'sqrt_operator_smoke/sqrt_before_evaluation.pt'
        inner, root = PolynomialArithmetic(), ScalarPower()
        inner.load_state_dict(torch.load(inner_path, weights_only=False)['state_dict'])
        root.load_state_dict(torch.load(sqrt_path, weights_only=False)['state_dict'])
        inner.eval()
        root.eval()
        bx = samples(2048, SEED + 400, 'joint_extrapolation')
        A = rng.choice([-1., 1.], len(bx)) * np.power(10., rng.uniform(-100, 100, len(bx)))
        A[::101] = 0.
        expected = A * np.sqrt(labels(bx))
        with torch.no_grad():
            r = root(inner(torch.from_numpy(bx)))
            got = model(torch.stack((torch.from_numpy(A), r), dim=1)).numpy()
        assembled = metrics(expected, got)
        assembled['passed'] = bool(assembled['nonfinite'] == 0 and assembled['sign_failures'] == 0 and
                                 assembled['zero_failures'] == 0 and assembled['max_relative_error'] <= config['relative_gate'])
        result['assembly'] = assembled
        result['upstream_hashes'] = {str(inner_path): sha(inner_path), str(sqrt_path): sha(sqrt_path)}
        write_csv(OUT / 'multiplication_assembled_validation.csv', [assembled])
        np.savez_compressed(OUT / 'assembled_data.npz', B_a_t=bx, A=A, targets=expected, predictions=got)
        if not assembled['passed']:
            result['passed'] = False
            result['status'] = 'DIAGNOSTIC_ONLY'
    assert all(sha(p) == h for p, h in protected.items())
    result['protected_hashes_unchanged'] = True
    result['seconds'] = time.monotonic() - started
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    (OUT / 'multiplication_summary.txt').write_text(
        'Generic multiplication smoke; synthetic raw x1,x2 only.\n'
        'All six polynomial terms through degree two are available; learned L1 support and supervised debias.\n'
        'No prescribed product term, coefficient assignment or coefficient snapping. Random initialization.\n'
        'Saved/hash checkpoint before evaluation. Only learned nonzero terms evaluated at inference.\n'
        'Training magnitudes 1e-40..1e40 with held-out pairs of magnitude bands.\n'
        'Validation: four signs, zeros, 1e-140..1e140, imbalanced magnitudes 1e-300..1e300.\n'
        'Products constrained to finite representable normal values or exact zeros; overflow/underflow semantics not certified.\n'
        'Numerical zero is required; negative-zero bit identity is diagnostic only.\n\n' + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
