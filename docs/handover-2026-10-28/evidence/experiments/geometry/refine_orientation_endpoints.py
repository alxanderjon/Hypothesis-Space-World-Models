"""Continue saved bilinear orientation model with endpoint supervision only."""
import json
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.geometry.train_orientation_transform import OrientationTransform, inputs, teacher, predict, metrics, HOLDOUT
from experiments.geometry.train_inner_quadratic_operator import PolynomialArithmetic
from experiments.geometry.train_sqrt_operator import ScalarPower
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic, metrics as arithmetic_metrics
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv

ROOT = Path('outputs/sho/stage1B')
SOURCE = ROOT / 'orientation_transform_smoke'
OUT = ROOT / 'orientation_transform_endpoint_refine'
SEED = 27382026
EPS = np.finfo(np.float64).eps


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if '--resume-evaluation' in sys.argv:
        checkpoint = OUT / 'orientation_refined_before_evaluation.pt'
        saved = torch.load(checkpoint, weights_only=False)
        config = saved['config']
        protected = config['protected_hashes']
        source_path = Path(config['source_checkpoint'])
        model = OrientationTransform()
        cycle = saved['cycles']
        with (OUT / 'training_history.csv').open() as handle:
            history = list(csv.DictReader(handle))
        digest = sha(checkpoint)
        rng.uniform(0, 1, 6000)  # Restore validation RNG position, no training.
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
        source_path = SOURCE / 'orientation_before_evaluation.pt'
        model = OrientationTransform()
        model.load_state_dict(torch.load(source_path, weights_only=False)['state_dict'])
        initial = model.weights.detach().numpy().copy()
        old_data = np.load(SOURCE / 'training_data.npz')
        c = rng.uniform(0, 1, 6000)
        keep = np.ones(len(c), dtype=bool)
        for lo, hi in HOLDOUT:
            keep &= ~((c >= lo) & (c <= hi))
        broad_x = np.concatenate((old_data['inputs'], inputs(c[keep][:2048])))
        broad_y = teacher(broad_x)
        endpoint_x = inputs(np.array([0., 1.]))
        endpoint_y = teacher(endpoint_x)
        config = dict(seed=SEED, dtype='float64', architecture='Unchanged imported OrientationTransform',
                      initialization='Reload saved candidate; no reinitialization of trained weights',
                      source_checkpoint=str(source_path), source_sha256=sha(source_path),
                      initial_weights=initial.tolist(), heldout_ranges=HOLDOUT,
                      optimizer='SGD on all four coefficients, lr=1, momentum=0, no weight decay',
                      schedule='Alternate four-endpoint half-sum-squared loss and broad half-mean-squared loss',
                      endpoint_weighting='Each endpoint has unit weight; each broad example weight is 1/broad_count',
                      broad_examples=len(broad_x), endpoint_examples=len(endpoint_x), max_cycles=64,
                      stopping='Exact endpoint and broad training recovery; no validation used for training',
                      prohibited_operations='No coefficient assignment/snapping/projection, output clipping, or architecture changes',
                      protected_hashes=protected)
        (OUT / 'config.json').write_text(json.dumps(config, indent=2))
        np.savez_compressed(OUT / 'training_data.npz', endpoint_inputs=endpoint_x, endpoint_targets=endpoint_y,
                            broad_inputs=broad_x, broad_targets=broad_y)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        history = []
        model.train()
        for cycle in range(1, 65):
            for name, x, y in [('endpoints', endpoint_x, endpoint_y), ('broad_identity', broad_x, broad_y)]:
                optimizer.zero_grad()
                residual = model(torch.from_numpy(x)) - torch.from_numpy(y)
                loss = .5 * (torch.sum(residual ** 2) if name == 'endpoints' else torch.mean(residual ** 2))
                before = model.weights.detach().tolist()
                loss.backward()
                gradient = model.weights.grad.detach().tolist()
                optimizer.step()
                history.append(dict(cycle=cycle, batch=name, loss=float(loss.detach()),
                                    weights_before=json.dumps(before), gradient=json.dumps(gradient),
                                    weights_after=json.dumps(model.weights.detach().tolist())))
            if np.array_equal(predict(model, endpoint_x), endpoint_y) and np.array_equal(predict(model, broad_x), broad_y):
                break
            if not np.isfinite(predict(model, broad_x)).all():
                break
        checkpoint = OUT / 'orientation_refined_before_evaluation.pt'
        digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                 config=config, status='SAVED_BEFORE_EVALUATION', cycles=cycle, updates=len(history)))
        write_csv(OUT / 'training_history.csv', history)
        print(json.dumps(dict(event='saved_before_evaluation', weights=model.weights.detach().tolist(), sha256=digest)), flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval()
    archived = np.load(SOURCE / 'heldout_data.npz')
    sets = [('archived_validation', archived['inputs']), ('fresh_uniform', inputs(rng.uniform(0, 1, 4096)))]
    sets += [(f'heldout_{lo}_{hi}', inputs(rng.uniform(lo, hi, 2048))) for lo, hi in HOLDOUT]
    sets += [('dense_grid', inputs(np.linspace(0, 1, 10001))),
             ('endpoint_stress', inputs(np.array([0., np.nextafter(0., 1.), 1e-300, 1e-20,
               np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), np.nextafter(1., 0.), 1.])))]
    vx = np.concatenate([x for _, x in sets])
    groups = np.concatenate([np.repeat(name, len(x)) for name, x in sets])
    y, p = teacher(vx), predict(model, vx)
    rows = [dict(group='all', orientation='both', **metrics(y, p))]
    for name in ['all'] + [s[0] for s in sets]:
        for orientation, label in [(0., 'forward'), (1., 'reverse')]:
            mask = vx[:, 1] == orientation
            if name != 'all':
                mask &= groups == name
            rows.append(dict(group=name, orientation=label, **metrics(y[mask], p[mask])))
    write_csv(OUT / 'orientation_endpoint_refine.csv', rows)
    np.savez_compressed(OUT / 'heldout_data.npz', inputs=vx, targets=y, predictions=p, groups=groups)
    forward, reverse = p[::2], p[1::2]
    complement_inputs = np.column_stack((1. - vx[::2, 0], np.zeros(len(forward))))
    important_x = inputs(np.array([0., .5, 1.]))
    important_p = predict(model, important_x)
    weights = model.weights.detach().numpy().copy()
    checks = dict(w0_positive_zero=bool(weights[:1].view(np.uint64)[0] == 0),
                  w0_hex=float(weights[0]).hex(),
                  forward_zero_exact=bool(important_p[0] == 0.),
                  reverse_one_exact_zero=bool(important_p[-1] == 0.),
                  endpoint_zero_signbits=[bool(np.signbit(important_p[0])), bool(np.signbit(important_p[-1]))],
                  complement_sum_max_error=float(abs(forward + reverse - 1.).max()),
                  complemented_input_max_error=float(abs(predict(model, complement_inputs) - reverse).max()),
                  outside_unit_interval=rows[0]['outside_unit_interval'])
    write_csv(OUT / 'orientation_endpoint_values.csv', [dict(c=float(c), orientation='forward' if o == 0 else 'reverse',
              teacher=float(t), learned=float(g), learned_hex=float(g).hex())
              for (c, o), t, g in zip(important_x, teacher(important_x), important_p)])
    standalone = bool(checks['w0_positive_zero'] and checks['forward_zero_exact'] and checks['reverse_one_exact_zero'] and
                      checks['complement_sum_max_error'] == 0 and checks['complemented_input_max_error'] == 0 and
                      checks['outside_unit_interval'] == 0 and rows[0]['max_error'] == 0 and rows[0]['nonfinite'] == 0)
    result = dict(operator='orientation_transform_operator', status='DIAGNOSTIC_ONLY', standalone_passed=standalone,
                  checkpoint=str(checkpoint), sha256=digest, source_sha256=sha(source_path),
                  learned_weights=weights.tolist(), updates=len(history), cycles=cycle,
                  validation=rows[0], exact_checks=checks)
    if standalone:
        paths = [ROOT / 'inner_quadratic_operator_smoke/inner_quadratic_before_evaluation.pt',
                 ROOT / 'sqrt_operator_smoke/sqrt_before_evaluation.pt',
                 ROOT / 'multiplication_operator_smoke/multiplication_before_evaluation.pt']
        inner, root, mul = PolynomialArithmetic(), ScalarPower(), QuadraticArithmetic()
        old = OrientationTransform()
        old.load_state_dict(torch.load(source_path, weights_only=False)['state_dict'])
        old.eval()
        for component, path in zip((inner, root, mul), paths):
            component.load_state_dict(torch.load(path, weights_only=False)['state_dict'])
            component.eval()
        def assembled(orientation_model, x, B, a, A):
            with torch.no_grad():
                t = orientation_model(torch.from_numpy(x))
                q = inner(torch.stack((torch.from_numpy(B), torch.from_numpy(a), t), dim=1))
                r = root(q)
                return mul(torch.stack((torch.from_numpy(A), r), dim=1)).numpy()
        archived_assembly = np.load(SOURCE / 'assembled_data.npz')
        cx = inputs(np.concatenate([rng.uniform(lo, hi, 320) for lo, hi in HOLDOUT] + [np.array([0., .5, 1.])]))
        B = np.repeat(np.power(10., rng.uniform(-20, 20, len(cx) // 2)), 2)
        a = np.repeat(rng.choice([-1., 1.], len(cx) // 2) * np.power(10., rng.uniform(-8, 8, len(cx) // 2)), 2)
        A = np.repeat(rng.choice([-1., 1.], len(cx) // 2) * np.power(10., rng.uniform(-50, 50, len(cx) // 2)), 2)
        # B>0 keeps the frozen sqrt input positive even at orientation endpoint zeros.
        cases = [('archived', archived_assembly['inputs'], archived_assembly['B'], archived_assembly['a'], archived_assembly['A']),
                 ('fresh', cx, B, a, A)]
        assembly_rows = []
        all_passed = True
        for group, x, B, a, A in cases:
            t = teacher(x)
            truth = A * np.sqrt(B + (a * a) * (t * t))
            new_p, old_p = assembled(model, x, B, a, A), assembled(old, x, B, a, A)
            assert np.isfinite(new_p).all()
            for orientation, label in [(None, 'both'), (0., 'forward'), (1., 'reverse')]:
                mask = np.ones(len(x), dtype=bool) if orientation is None else x[:, 1] == orientation
                row = dict(group=group, orientation=label, **arithmetic_metrics(truth[mask], new_p[mask]))
                old_error = abs(old_p[mask] - truth[mask])
                new_error = abs(new_p[mask] - truth[mask])
                row['exact_output_changes_vs_old'] = int(np.sum(new_p[mask] != old_p[mask]))
                row['examples_with_increased_absolute_error'] = int(np.sum(new_error > old_error))
                row['old_max_relative_error'] = arithmetic_metrics(truth[mask], old_p[mask])['max_relative_error']
                assembly_rows.append(row)
                all_passed &= row['max_relative_error'] <= 128 * EPS and row['examples_with_increased_absolute_error'] == 0
            np.savez_compressed(OUT / f'assembled_{group}.npz', inputs=x, B=B, a=a, A=A, targets=truth,
                                predictions=new_p, previous_predictions=old_p)
        write_csv(OUT / 'orientation_refined_assembly.csv', assembly_rows)
        result['assembly'] = dict(passed=bool(all_passed), rows=assembly_rows)
        result['upstream_hashes'] = {str(p): sha(p) for p in paths}
        if all_passed:
            result['status'] = 'LOCKED'
    assert all(sha(p) == h for p, h in protected.items())
    result['protected_hashes_unchanged'] = True
    result['passed'] = result['status'] == 'LOCKED'
    result['invocation_seconds'] = time.monotonic() - started
    result['resumed_evaluation_without_retraining'] = '--resume-evaluation' in sys.argv
    (OUT / 'orientation_transform_manifest.json').write_text(json.dumps(result, indent=2))
    if result['passed']:
        (OUT / 'orientation_transform_operator_locked.json').write_text(json.dumps(dict(
            operator='orientation_transform_operator', status='LOCKED', checkpoint=str(checkpoint), sha256=digest,
            architecture='experiments.geometry.train_orientation_transform.OrientationTransform',
            orientation_encoding={'forward': 0, 'reverse': 1}, validation_manifest=str(OUT / 'orientation_transform_manifest.json')), indent=2))
    (OUT / 'orientation_endpoint_summary.txt').write_text(
        'Endpoint-only refinement of saved candidate architecture; existing checkpoint preserved.\n'
        'All four parameters updated by supervised SGD. No manually assigned/zeroed coefficients.\n'
        'Endpoint half-sum loss alternated with broad half-mean loss; gradients and updates recorded.\n'
        'Checkpoint persisted before held-out evaluation. Original held-out cut bands remain excluded from training.\n'
        'All other components frozen; no later primitive trained.\n\n' + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
