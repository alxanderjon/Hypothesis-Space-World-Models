"""Synthetic-only float64 raw-q power neuron; no upstream model is trained."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv
from experiments.geometry.train_inner_quadratic_operator import PolynomialArithmetic, samples, labels

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'sqrt_operator_smoke'
SEED = 27352026
EPS = np.finfo(np.float64).eps
RESERVED = [(-160., -140.), (-80., -60.), (-4., -2.), (2., 4.), (60., 80.), (140., 160.)]


class ScalarPower(nn.Module):
    """Raw positive q -> exp(b) * q**p; learned p constrained positive.

    This is a power-function architectural prior, not a general MLP. The
    exponent and gain are supervised parameters, never set from teacher values.
    log_forward is the numerically stable training-loss path only.
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.stack((.2 + torch.rand((), dtype=torch.float64),
                                                .3 * torch.randn((), dtype=torch.float64))))

    def forward(self, q):
        p, b = self.weights
        return torch.exp(b) * torch.pow(q, p)

    def log_forward(self, q, weights=None):
        p, b = self.weights if weights is None else weights
        return p * torch.log(q) + b


def predict(model, q):
    with torch.no_grad():
        return model(torch.from_numpy(q)).numpy()


def metrics(truth, pred):
    error = pred - truth
    relative = error / truth
    largest = float(abs(error).max())
    stable_rmse = largest * np.sqrt(np.mean((error / largest) ** 2)) if largest else 0.
    return dict(examples=len(truth), rmse=float(stable_rmse),
                mae=float(np.mean(abs(error))), max_absolute_error=float(abs(error).max()),
                relative_rmse=float(np.sqrt(np.mean(relative * relative))),
                relative_mae=float(np.mean(abs(relative))), max_relative_error=float(abs(relative).max()),
                nonfinite=int(np.sum(~np.isfinite(pred))), nonpositive=int(np.sum(pred <= 0)),
                bit_exact=int(np.sum(truth.view(np.uint64) == pred.view(np.uint64))))


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if '--resume-evaluation' in sys.argv:
        checkpoint = OUT / 'sqrt_before_evaluation.pt'
        saved = torch.load(checkpoint, weights_only=False)
        config = saved['config']
        protected = config['protected_hashes']
        data = np.load(OUT / 'training_data.npz')
        train_q, target = data['q'], data['target']
        rng.uniform(-200, 200, 7000)  # Restore the original validation RNG position.
        model = ScalarPower()
        digest = sha(checkpoint)
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
        exponent = rng.uniform(-200, 200, 7000)
        keep = np.ones(len(exponent), dtype=bool)
        for lo, hi in RESERVED:
            keep &= ~((exponent >= lo) & (exponent <= hi))
        train_q = np.power(10., exponent[keep][:4096])
        assert len(train_q) == 4096
        target = np.sqrt(train_q)
        np.savez_compressed(OUT / 'training_data.npz', q=train_q, target=target)
        model = ScalarPower()
        initial = model.weights.detach().numpy().copy()
        config = dict(seed=SEED, dtype='float64', sole_input='raw positive q',
                      model='exp(b) * pow(q,p); p and b both learned', initial_weights=initial.tolist(),
                      architecture_prior='Generic positive power function, no fixed root exponent',
                      objective='Continuous squared log-output residual; log is internal, not an input feature',
                      teacher='np.sqrt(q); labels only', no_epsilon=True,
                      train_log10_range=[-200, 200], heldout_log10_bands=RESERVED,
                      validation_log10_range=[-300, 300],
                      optimizer='TRF least_squares, analytic parameter Jacobian, positive exponent bound',
                      relative_gate=128 * EPS, scaling_relative_gate=256 * EPS,
                      protected_hashes=protected)
        (OUT / 'config.json').write_text(json.dumps(config, indent=2))
        tq = torch.from_numpy(train_q)
        log_target = np.log(target)
        history = []
        best = [np.inf, None]

        def residual(w):
            with torch.no_grad():
                output = model.log_forward(tq, torch.from_numpy(w)).numpy()
            residuals = output - log_target
            mse = float(np.mean(residuals ** 2))
            history.append(dict(evaluation=len(history) + 1, log_rmse=np.sqrt(mse), p=w[0], b=w[1]))
            if mse < best[0]:
                best[:] = [mse, w.copy()]
            return residuals

        jac = np.column_stack((torch.log(tq).numpy(), np.ones(len(tq))))
        fit = least_squares(residual, initial, jac=lambda _: jac,
                            bounds=([np.finfo(np.float64).tiny, -np.inf], [np.inf, np.inf]),
                            max_nfev=100, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        with torch.no_grad():
            model.weights.copy_(torch.from_numpy(best[1]))
        checkpoint = OUT / 'sqrt_before_evaluation.pt'
        digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
                                 status='SAVED_BEFORE_EVALUATION', training_log_rmse=np.sqrt(best[0]),
                                 optimizer_message=fit.message))
        write_csv(OUT / 'training_history.csv', history)
        print(json.dumps(dict(event='saved_before_evaluation', weights=best[1].tolist(), hash=digest)), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval()

    families = [('new_combinations', rng.uniform(-200, 200, 4096))]
    families += [(f'reserved_{lo:g}_{hi:g}', rng.uniform(lo, hi, 1024)) for lo, hi in RESERVED]
    families += [('tiny_extrapolation', rng.uniform(-300, -200, 2048)),
                 ('large_extrapolation', rng.uniform(200, 300, 2048)),
                 ('order_one', rng.uniform(-.3, .3, 2048))]
    values = [np.power(10., exponents) for _, exponents in families]
    tags = [np.repeat(name, len(v)) for (name, _), v in zip(families, values)]
    extreme = np.array([np.nextafter(0., 1.), np.finfo(np.float64).tiny,
                        np.nextafter(1., 0.), 1., np.nextafter(1., np.inf), np.finfo(np.float64).max])
    values.append(extreme)
    tags.append(np.repeat('float64_extremes', len(extreme)))
    q = np.concatenate(values)
    tags = np.concatenate(tags)
    truth = np.sqrt(q)
    pred = predict(model, q)
    np.savez_compressed(OUT / 'heldout_data.npz', q=q, target=truth, prediction=pred, group=tags)
    rows = [dict(grouping='all', group='all', **metrics(truth, pred))]
    for tag in sorted(set(tags)):
        mask = tags == tag
        rows.append(dict(grouping='family', group=tag, **metrics(truth[mask], pred[mask])))
    for lo in range(-340, 321, 20):
        mask = (np.log10(q) >= lo) & (np.log10(q) < lo + 20)
        if mask.any():
            rows.append(dict(grouping='log10_q_band', group=f'[{lo},{lo+20})', **metrics(truth[mask], pred[mask])))
    write_csv(OUT / 'sqrt_validation.csv', rows)

    ordered_q = np.unique(np.concatenate((q, np.logspace(-300, 300, 20000))))
    ordered_y = predict(model, ordered_q)
    monotonic_reversals = int(np.sum(np.diff(ordered_y) < 0))
    # Equal rounded outputs on adjacent representable q are permitted.
    base = np.power(10., rng.uniform(-200, 200, 8192))
    k = rng.choice([-1., 1.], len(base)) * np.power(10., rng.uniform(-40, 40, len(base)))
    scaled = (k * k) * base
    assert np.all(scaled > 0) and np.isfinite(scaled).all()
    expected = abs(k) * predict(model, base)
    scaling = metrics(expected, predict(model, scaled))
    properties = dict(positivity_failures=int(np.sum(ordered_y <= 0)),
                      monotonicity_queries=len(ordered_q), monotonicity_reversals=monotonic_reversals,
                      rounded_equal_adjacent_outputs=int(np.sum(np.diff(ordered_y) == 0)),
                      scaling=scaling, scaling_k_abs_range=[float(abs(k).min()), float(abs(k).max())])
    (OUT / 'sqrt_properties.json').write_text(json.dumps(properties, indent=2))
    np.savez_compressed(OUT / 'scaling_data.npz', q=base, k=k, scaled_q=scaled)
    overall = rows[0]
    passed = (overall['nonfinite'] == 0 and overall['nonpositive'] == 0 and
              overall['max_relative_error'] <= config['relative_gate'] and monotonic_reversals == 0 and
              scaling['nonfinite'] == 0 and scaling['max_relative_error'] <= config['scaling_relative_gate'])
    manifest = dict(operator='sqrt_operator', status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',
                    passed=bool(passed), checkpoint=str(checkpoint), sha256=digest,
                    learned_exponent=float(model.weights[0].detach()),
                    learned_log_gain=float(model.weights[1].detach()),
                    learned_gain=float(model.weights[1].detach().exp()),
                    training_examples=len(train_q), training_metrics=metrics(target, predict(model, train_q)),
                    validation=overall, properties=properties)
    # Freeze/persist passing standalone result before the only authorized assembly.
    (OUT / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2))
    if passed:
        upstream_path = ROOT / 'inner_quadratic_operator_smoke/inner_quadratic_before_evaluation.pt'
        upstream = PolynomialArithmetic()
        upstream.load_state_dict(torch.load(upstream_path, weights_only=False)['state_dict'])
        upstream.eval()
        x = samples(2048, SEED + 99, 'joint_extrapolation')
        teacher_q = labels(x)
        assert np.all(teacher_q > 0)
        with torch.no_grad():
            learned_q = upstream(torch.from_numpy(x))
            assembled = model(learned_q).numpy()
        assembled_truth = np.sqrt(teacher_q)
        assembly = metrics(assembled_truth, assembled)
        assembly['upstream_sha256'] = sha(upstream_path)
        assembly['passed'] = bool(assembly['nonfinite'] == 0 and assembly['max_relative_error'] <= config['relative_gate'])
        write_csv(OUT / 'sqrt_assembled_validation.csv', [assembly])
        np.savez_compressed(OUT / 'assembled_data.npz', inputs=x, target=assembled_truth, prediction=assembled)
        manifest['assembly'] = assembly
        if not assembly['passed']:
            manifest['passed'] = False
            manifest['status'] = 'DIAGNOSTIC_ONLY'
    assert all(sha(p) == h for p, h in protected.items())
    manifest['protected_hashes_unchanged'] = True
    manifest['invocation_seconds'] = time.monotonic() - start
    manifest['resumed_evaluation_without_retraining'] = '--resume-evaluation' in sys.argv
    (OUT / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'sqrt_summary.txt').write_text(
        'Generic scalar sqrt smoke. Raw q only; all q strictly positive, no epsilon.\n'
        'Architecture: exp(b)*q**p, both parameters random and learned. Power-family prior; not a generic MLP.\n'
        'Only training loss uses internal logs for dynamic-range conditioning. Inference uses learned power and gain.\n'
        'No coefficient snapping, hard-coded root exponent, or transformed external inputs.\n'
        'Checkpoint saved and hashed before evaluation. Existing checkpoints unchanged.\n'
        'Absolute errors reflect output magnitudes; relative precision is the gate.\n'
        'Monotonicity means nondecreasing float64 outputs; adjacent inputs may round to equal roots.\n\n'
        + json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    run()
