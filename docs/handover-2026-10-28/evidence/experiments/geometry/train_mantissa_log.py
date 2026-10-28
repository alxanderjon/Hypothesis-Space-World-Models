"""One global monotone raw-mantissa log model, float64 synthetic supervision."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'mantissa_log_operator_smoke'
REFINE = '--refine-fit' in sys.argv
if REFINE:
    OUT = ROOT / 'mantissa_log_operator_fit_refine'
SEED = 27422026
DEGREE = 32
EPS = np.finfo(np.float64).eps
HOLDOUT = [(1.13, 1.21), (1.43, 1.51), (1.79, 1.87)]


class MonotoneMantissaLog(nn.Module):
    """Raw m -> bias + positive polynomial in u=3*(m-1)/(m+1).

    One global coordinate, no decomposition, log call, interval branches, or
    output correction. Nonnegative coefficients enforced by parameter activation.
    Horner evaluation uses only products/additions of nonnegative operands before
    adding a shared bias; monotonicity is a hypothesis-class property.
    """
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(.01 + .1 * torch.rand(DEGREE, dtype=torch.float64))
        self.bias = nn.Parameter(.2 * torch.randn((), dtype=torch.float64))

    def coordinate(self, m):
        return 3. * ((m - 1.) / (m + 1.))

    def features(self, m):
        u = self.coordinate(m)
        columns = [u]
        for _ in range(1, DEGREE):
            columns.append(columns[-1] * u)
        return torch.stack(columns, dim=1)

    def forward(self, m):
        u = self.coordinate(m)
        coefficients = torch.relu(self.coefficients)
        result = torch.zeros_like(u)
        for coefficient in coefficients.flip(0):
            result = result * u + coefficient
        return self.bias + u * result


def predict(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def metric(y, p):
    e = p - y
    nz = y != 0
    rel = e[nz] / y[nz]
    return dict(examples=len(y), rmse=float(np.sqrt(np.mean(e * e))), mae=float(np.mean(abs(e))),
                max_absolute_error=float(abs(e).max()), relative_rmse=float(np.sqrt(np.mean(rel * rel))) if len(rel) else 0.,
                max_relative_error=float(abs(rel).max()) if len(rel) else 0.,
                exact_agreement=int(np.sum(y == p)), nonfinite=int(np.sum(~np.isfinite(p))))


def is_training_range(x):
    keep = np.ones(len(x), bool)
    for lo, hi in HOLDOUT:
        keep &= ~((x >= lo) & (x <= hi))
    return keep


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    model = MonotoneMantissaLog()
    resume_path = ROOT / 'mantissa_log_operator_smoke/mantissa_log_before_evaluation.pt'
    if REFINE:
        model.load_state_dict(torch.load(resume_path, weights_only=False)['state_dict'])
    initial = {k: v.detach().tolist() for k, v in model.state_dict().items()}
    ordinary = rng.uniform(1., 2., 4096)
    ordinary = ordinary[is_training_range(ordinary)][:2048]
    dense = np.linspace(1., np.nextafter(2., 1.), 2048)
    dense = dense[is_training_range(dense)]
    near_one = 1. + np.power(10., rng.uniform(-16, -2, 512))
    near_two = 2. - np.power(10., rng.uniform(-16, -2, 512))
    near_two = near_two[near_two < 2.]
    # Adjacent pairs throughout the non-held-out training domain.
    pair_left = ordinary[:512]
    pair_right = np.nextafter(pair_left, np.inf)
    x = np.concatenate((ordinary, dense, near_one, near_two, pair_left, pair_right, np.ones(64)))
    assert np.all((x >= 1.) & (x < 2.)) and np.all(is_training_range(x))
    y = np.log(x)
    config = dict(seed=SEED, dtype='float64', model='Global degree-32 positive polynomial in 3*(m-1)/(m+1), plus learned bias',
                  input='raw m only, 1<=m<2', initialization=initial, heldout_subintervals=HOLDOUT,
                  monotonicity='Nonnegative polynomial coefficients; nonnegative global coordinate and monotone Horner arithmetic',
                  parameter_constraint='ReLU parameter activation, with nonnegative active-set regression optimizer',
                  shape_prior='Positive polynomial on a monotone rational coordinate; no prescribed series coefficients or odd-term restriction',
                  anchor='Supervised SGD on L(1)=0, all parameters eligible',
                  optimizer='Warm-start feasible active-set nonnegative least squares from random positive coefficients; training residual correction',
                  training_examples=len(x), absolute_gate=8 * EPS, relative_gate=32 * EPS,
                  ordering_gate='Zero downward reversals; teacher-strict collapsed pairs reported, not required strict in this experiment',
                  no_range_reduction=True, no_decomposition=True, no_log_in_inference=True,
                  protected_hashes=protected)
    config['resumed_from_checkpoint'] = str(resume_path) if REFINE else None
    config['resume_sha256'] = sha(resume_path) if REFINE else None
    config['inactive_gradient_stop'] = 0. if REFINE else 1e-17
    config['least_squares_rcond'] = 1e-18 if REFINE else None
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez_compressed(OUT / 'training_data.npz', m=x, targets=y)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    optimizer.zero_grad()
    anchor_loss = .5 * torch.sum(model(torch.ones(1, dtype=torch.float64)) ** 2)
    anchor_loss.backward()
    optimizer.step()
    assert float(model.bias.detach()) == 0.
    with torch.no_grad():
        phi = model.features(torch.from_numpy(x)).numpy()
    divisor = abs(y).copy()
    divisor[divisor == 0] = 1.
    R = phi / divisor[:, None]
    target = y / divisor
    scale = np.sqrt(np.mean(R * R, axis=0))
    Z = R / scale
    w = model.coefficients.detach().numpy().copy() * scale
    passive = w > 0
    best_w = w.copy()
    best_loss = np.inf
    history = []
    stale = 0
    # Standard feasible active-set NNLS. Entries reach zero only when the
    # optimizer encounters the nonnegativity boundary, never a teacher assignment.
    for iteration in range(1, 257):
        for inner in range(DEGREE + 1):
            trial = np.zeros(DEGREE)
            trial[passive] = np.linalg.lstsq(Z[:, passive], target, rcond=config['least_squares_rcond'])[0]
            invalid = passive & (trial <= 0)
            if not np.any(invalid):
                w = trial
                break
            indices = np.flatnonzero(invalid)
            ratios = w[indices] / (w[indices] - trial[indices])
            hit = indices[int(np.argmin(ratios))]
            fraction = float(np.min(ratios))
            w = w + fraction * (trial - w)
            w[hit] = 0.  # Generic active-set bound, applied to any learned term.
            passive[hit] = False
        with torch.no_grad():
            model.coefficients.copy_(torch.from_numpy(np.maximum(w, 0.) / scale))
        residual = (y - predict(model, x)) / divisor
        loss = float(np.mean(residual * residual))
        history.append(dict(phase='nonnegative_fit', update=iteration, relative_rmse=float(np.sqrt(loss)), active_terms=int(np.sum(w > 0))))
        if loss < best_loss:
            best_loss, best_w = loss, np.maximum(w, 0.).copy()
            stale = 0
        else:
            stale += 1
        gradient = Z.T @ residual
        gradient[passive] = -np.inf
        if np.all(passive) or np.max(gradient) <= config['inactive_gradient_stop'] or stale >= 24:
            break
        entering = int(np.argmax(gradient))
        passive[entering] = True
    w = best_w.copy()
    with torch.no_grad():
        model.coefficients.copy_(torch.from_numpy(w / scale))
    # Debias on learned support using actual Horner residual, preserving positivity.
    support = w > 0
    for update in range(24):
        residual = (y - predict(model, x)) / divisor
        loss = float(np.mean(residual * residual))
        delta = np.zeros(DEGREE)
        delta[support] = np.linalg.lstsq(Z[:, support], residual, rcond=config['least_squares_rcond'])[0]
        accepted = False
        for step in [1., .5, .25, .125]:
            candidate = w + step * delta
            if np.any(candidate < 0):
                continue
            with torch.no_grad():
                model.coefficients.copy_(torch.from_numpy(candidate / scale))
            new_loss = float(np.mean(((y - predict(model, x)) / divisor) ** 2))
            if new_loss < loss:
                w = candidate
                accepted = True
                break
        if not accepted:
            with torch.no_grad():
                model.coefficients.copy_(torch.from_numpy(w / scale))
        history.append(dict(phase='residual_refine', update=update, relative_rmse=float(np.sqrt(min(loss, new_loss))) if 'new_loss' in locals() else float(np.sqrt(loss)), active_terms=int(np.sum(w > 0))))
        if not accepted:
            break
    checkpoint = OUT / 'mantissa_log_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
                             status='SAVED_BEFORE_EVALUATION', training_metrics=metric(y, predict(model, x))))
    write_csv(OUT / 'training_history.csv', history)
    write_csv(OUT / 'learned_coefficients.csv', [dict(power=i + 1, coefficient=float(c)) for i, c in enumerate(model.coefficients.detach().numpy())])
    print(json.dumps(dict(event='saved_before_evaluation', sha256=digest, training=metric(y, predict(model, x)))), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval()
    # Uniform sampling of mantissa bit patterns produces raw float64 values.
    bits = rng.integers(0, 2 ** 52, 16384, dtype=np.uint64) | np.uint64(0x3ff0000000000000)
    groups = [('random_float64', bits.view(np.float64)), ('ordinary_dense', np.linspace(1., np.nextafter(2., 1.), 32769)),
              ('near_one', 1. + np.power(10., rng.uniform(-16, -1, 4096))),
              ('near_two', np.maximum(1., np.nextafter(2., 1.) - np.power(10., rng.uniform(-16, -1, 4096))))]
    groups += [(f'heldout_{lo}_{hi}', rng.uniform(lo, hi, 4096)) for lo, hi in HOLDOUT]
    groups += [('important_points', np.array([1., np.nextafter(1., 2.), 1.5, np.nextafter(2., 1.)]))]
    vx = np.concatenate([values for _, values in groups])
    tags = np.concatenate([np.repeat(name, len(values)) for name, values in groups])
    truth, pred = np.log(vx), predict(model, vx)
    rows = [dict(grouping='all', group='all', **metric(truth, pred))]
    for name, _ in groups:
        mask = tags == name
        rows.append(dict(grouping='data_family', group=name, **metric(truth[mask], pred[mask])))
    for lo in np.arange(1., 2., .1):
        mask = (vx >= lo) & (vx < lo + .1)
        if mask.any():
            rows.append(dict(grouping='subinterval', group=f'[{lo:.1f},{lo+.1:.1f})', **metric(truth[mask], pred[mask])))
    write_csv(OUT / 'mantissa_log_validation.csv', rows)
    np.savez_compressed(OUT / 'heldout_data.npz', m=vx, targets=truth, predictions=pred, groups=tags)
    centers = rng.integers(1, 2 ** 52 - 1, 131072, dtype=np.uint64) | np.uint64(0x3ff0000000000000)
    adjacent = np.concatenate(((centers - np.uint64(1)).view(np.float64), centers.view(np.float64), (centers + np.uint64(1)).view(np.float64)))
    endpoint_bits = np.concatenate((np.arange(8192, dtype=np.uint64) + np.uint64(0x3ff0000000000000),
                                   np.uint64(0x4000000000000000) - np.arange(1, 8193, dtype=np.uint64)))
    ordered = np.unique(np.concatenate((vx, adjacent, endpoint_bits.view(np.float64), np.linspace(1., np.nextafter(2., 1.), 262145))))
    po, yo = predict(model, ordered), np.log(ordered)
    pd, td = np.diff(po), np.diff(yo)
    consecutive = np.diff(ordered.view(np.uint64)) == 1
    anchor = predict(model, np.array([1.]))
    properties = dict(ordered_inputs=len(ordered), downward_reversals=int(np.sum(pd < 0)),
                      teacher_strict_collapsed_pairs=int(np.sum((td > 0) & (pd == 0))),
                      adjacent_float_pairs=int(np.sum(consecutive)),
                      adjacent_float_reversals=int(np.sum((pd < 0) & consecutive)),
                      adjacent_teacher_strict_collapses=int(np.sum((td > 0) & (pd == 0) & consecutive)),
                      log_one=float(anchor[0]), log_one_positive_zero=bool(anchor.view(np.uint64)[0] == 0),
                      last_input=float(np.nextafter(2., 1.)), last_prediction=float(predict(model, np.array([np.nextafter(2., 1.)]))[0]),
                      last_teacher=float(np.log(np.nextafter(2., 1.))), minimum_coefficient=float(torch.relu(model.coefficients).min().detach()))
    (OUT / 'mantissa_log_properties.json').write_text(json.dumps(properties, indent=2))
    np.savez_compressed(OUT / 'ordered_validation.npz', m=ordered, targets=yo, predictions=po)
    # Apply accuracy gate to dense/adjacent values too, not just summary groups.
    dense_metric = metric(yo, po)
    write_csv(OUT / 'mantissa_log_ordered_metrics.csv', [dense_metric])
    passed = bool(rows[0]['nonfinite'] == 0 and dense_metric['nonfinite'] == 0 and
                  max(rows[0]['max_absolute_error'], dense_metric['max_absolute_error']) <= config['absolute_gate'] and
                  max(rows[0]['max_relative_error'], dense_metric['max_relative_error']) <= config['relative_gate'] and
                  properties['downward_reversals'] == 0 and properties['log_one_positive_zero'])
    assert all(sha(p) == h for p, h in protected.items())
    result = dict(operator='mantissa_log_operator', status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',
                  passed=passed, checkpoint=str(checkpoint), sha256=digest,
                  training_examples=len(x), validation=rows[0], dense_validation=dense_metric, properties=properties,
                  protected_hashes_unchanged=True, seconds=time.monotonic() - started)
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, indent=2))
    (OUT / 'mantissa_log_summary.txt').write_text(
        'One global monotone mantissa log model, raw m in [1,2), float64.\n'
        'No binary decomposition, range switches, separate interval models, log calls, or postprocessing in inference.\n'
        'All powers 1..32 available with learned nonnegative coefficients; no prescribed log-series coefficients.\n'
        'Learned bias trained using L(1)=0. All parameters initialized randomly.\n'
        'Saved/hash before held-out evaluation. Whole subintervals excluded from training.\n'
        'Teacher-strict collapses reported; this experiment requires monotone nondecreasing outputs, not strictness.\n'
        'No full logarithm assembly or later primitive training performed.\n\n' + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
