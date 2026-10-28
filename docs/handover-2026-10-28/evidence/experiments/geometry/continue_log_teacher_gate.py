"""Continue saved log weights using teacher-aware finite-precision coordinate optimization."""
import json
import csv
import sys
import time
from pathlib import Path
import numpy as np
import torch
from experiments.geometry.refine_log_boundaries import centers, adjacent_cloud, pair_report
from experiments.geometry.train_log_operator import ScalarLogPolynomial, predict, metrics, EPS
from experiments.geometry.train_orientation_transform import OrientationTransform, inputs, teacher as orientation_teacher
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic, metrics as arithmetic_metrics
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv

ROOT = Path('outputs/sho/stage1B')
SOURCE = ROOT / 'log_operator_smoke'
OUT = ROOT / 'log_operator_teacher_gate_continue'
SEED = 27412026


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if '--resume-evaluation' in sys.argv:
        config = json.loads((OUT / 'config.json').read_text())
        protected = config['protected_hashes']
        source = Path(config['accuracy_baseline_checkpoint'])
        resume_path = Path(config['resume_checkpoint'])
        model = ScalarLogPolynomial()
        powers, actual = centers()
        consistency_v = np.load(OUT / 'training_data.npz')['consistency_v']
        with (OUT / 'checkpoint_hashes.csv').open() as handle:
            cp_rows = list(csv.DictReader(handle))
        checkpoint = Path(cp_rows[-1]['checkpoint'])
        digest = sha(checkpoint)
        with (OUT / 'training_history.csv').open() as handle:
            history = list(csv.DictReader(handle))
        sweep = int(history[-1]['sweep'])
        total_accepted = sum(int(r['accepted_updates']) for r in history)
        evaluations = 1 + sweep * len(model.weights) * len(config['step_multipliers']) * 2
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
        source = SOURCE / 'log_before_evaluation.pt'
        resume_path = ROOT / 'log_operator_boundary_refine/log_refined_before_evaluation.pt'
        model = ScalarLogPolynomial()
        model.load_state_dict(torch.load(resume_path, weights_only=False)['state_dict'])
        baseline = ScalarLogPolynomial()
        baseline.load_state_dict(torch.load(source, weights_only=False)['state_dict'])
        baseline.eval()
        powers, actual = centers()
        bv = adjacent_cloud(np.concatenate((powers, actual)), 8)
        bv = bv[(bv > 0) & np.isfinite(bv)]
        by = np.log(bv)
        original_training = np.load(SOURCE / 'training_data.npz')
        broad_v = original_training['v']
        broad_y = original_training['targets']
        # Additional ordinary examples stay outside the six original magnitude bands.
        exponents = rng.uniform(-200, 200, 4096)
        allowed = np.ones(len(exponents), bool)
        for lo, hi in [(-160,-140),(-80,-60),(-4,-2),(2,4),(60,80),(140,160)]:
            allowed &= ~((exponents >= lo) & (exponents <= hi))
        broad_v = np.concatenate((broad_v, np.power(10., exponents[allowed])))
        broad_y = np.log(broad_v)
        with torch.no_grad():
            boundary_features = model.features(torch.from_numpy(bv))
            broad_features = model.features(torch.from_numpy(broad_v))
        previous_broad = predict(baseline, broad_v)
        baseline_metrics = metrics(broad_y, previous_broad)
        consistency_v = np.ldexp(rng.uniform(.5, 1., 2048), rng.integers(-1000,1000,2048))
        with torch.no_grad():
            fc1 = model.features(torch.from_numpy(consistency_v))
            fc2 = model.features(torch.from_numpy(2. * consistency_v))
        config = dict(seed=SEED, resume_checkpoint=str(resume_path), resume_sha256=sha(resume_path),
            accuracy_baseline_checkpoint=str(source), accuracy_baseline_sha256=sha(source),
            architecture='Unchanged ScalarLogPolynomial and inference arithmetic', dtype='float64',
            gate='Teacher strict => learned strict; teacher tie => equality allowed; no learned reversal',
            optimizer='Teacher-supervised float64 coordinate descent, all coefficients eligible; supervised g(1)=0 SGD after proposals',
            objective_priority=['ordinary training accuracy no worse than original', 'number of reversed or wrongly collapsed boundary pairs',
                                'teacher-output label error and doubling-consistency error'],
            train_neighbor_width=8, evaluation_neighbor_width=32,
            step_multipliers=[.25,.5,1.,2.,4.], minimum_step_scale='eps*max(1,abs(weight))',
            stopping='Stop after a full coordinate sweep finds no improving update; at most 32 accepted sweeps for smoke',
            initial_weights=model.weights.detach().tolist(), protected_hashes=protected,
            note='No projection, coefficient snapping, teacher formula assignment, clipping, or boundary lookup',
            heldout_note='All-exponent boundary anchors include original held-out magnitude bands; random held-out tuples never enter optimization')
        (OUT / 'config.json').write_text(json.dumps(config,indent=2))
        np.savez_compressed(OUT / 'training_data.npz', boundary_v=bv, boundary_targets=by,
                            ordinary_v=broad_v, ordinary_targets=broad_y, consistency_v=consistency_v)
        anchor_optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        def set_and_anchor(w):
            with torch.no_grad(): model.weights.copy_(torch.from_numpy(w))
            anchor_optimizer.zero_grad()
            loss = .5 * torch.sum(model(torch.ones(1,dtype=torch.float64)) ** 2)
            loss.backward()
            anchor_optimizer.step()
        def training_key():
            with torch.no_grad():
                p = (boundary_features @ model.weights).numpy()
                ordinary = (broad_features @ model.weights).numpy()
                ce = ((fc2 @ model.weights) - (fc1 @ model.weights)).numpy() - np.log(2.)
            m = metrics(broad_y, ordinary)
            accuracy_violation = max(0., *(m[k] / baseline_metrics[k] - 1. for k in ['rmse','mae','max_absolute_error','max_normalized_error']))
            pd, td = np.diff(p), np.diff(by)
            reversed_pairs = int(np.sum(pd < 0))
            collapsed = int(np.sum((td > 0) & (pd == 0)))
            label_error = float(np.mean(((p-by)/np.maximum(1.,abs(by)))**2) + np.mean((ce/np.maximum(1.,abs(np.log(consistency_v))))**2))
            key = (accuracy_violation, reversed_pairs + collapsed, label_error)
            return key, dict(accuracy_violation=accuracy_violation,reversals=reversed_pairs,
                             collapsed_teacher_strict_pairs=collapsed,label_objective=label_error,
                             ordinary_rmse=m['rmse'],ordinary_max_normalized_error=m['max_normalized_error'])
        set_and_anchor(model.weights.detach().numpy().copy())
        best_key, detail = training_key()
        w = model.weights.detach().numpy().copy()
        history = [dict(sweep=0,accepted_updates=0,**detail)]
        checkpoint_rows = []
        total_accepted = 0
        evaluations = 1
        for sweep in range(1,33):
            sweep_accepted = 0
            # Randomized coordinate order deterministic from the saved seed.
            for j in rng.permutation(len(w)):
                coordinate_w = w.copy()
                chosen_w = w.copy()
                chosen_key = best_key
                chosen_detail = detail
                unit = EPS * max(1.,abs(float(w[j])))
                for multiplier in config['step_multipliers']:
                    for sign in (-1.,1.):
                        trial = coordinate_w.copy()
                        trial[j] += sign * multiplier * unit
                        set_and_anchor(trial)
                        key, stats = training_key()
                        evaluations += 1
                        if key < chosen_key:
                            chosen_key, chosen_detail = key, stats
                            chosen_w = model.weights.detach().numpy().copy()
                if chosen_key < best_key:
                    sweep_accepted += 1
                    total_accepted += 1
                    w, best_key, detail = chosen_w, chosen_key, chosen_detail
                set_and_anchor(w)
            history.append(dict(sweep=sweep,accepted_updates=sweep_accepted,**detail))
            print(json.dumps(history[-1]),flush=True)
            if sweep_accepted:
                cp = OUT / f'best_sweep_{sweep:02d}_before_evaluation.pt'
                digest = save_checkpoint(cp,dict(state_dict=model.state_dict(),config=config,
                    status='SAVED_BEFORE_EVALUATION',sweep=sweep,training_key=best_key,accepted_updates=total_accepted))
                checkpoint_rows.append(dict(sweep=sweep,checkpoint=str(cp),sha256=digest))
                checkpoint = cp
            if sweep_accepted == 0 or (best_key[0] == 0 and best_key[1] == 0): break
        if not checkpoint_rows:
            checkpoint = OUT / 'continued_before_evaluation.pt'
            digest = save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,status='SAVED_BEFORE_EVALUATION'))
            checkpoint_rows.append(dict(sweep=0,checkpoint=str(checkpoint),sha256=digest))
        write_csv(OUT/'training_history.csv',history)
        write_csv(OUT/'checkpoint_hashes.csv',checkpoint_rows)
        print(json.dumps(dict(event='saved_before_evaluation',checkpoint=str(checkpoint),sha256=digest,evaluations=evaluations)),flush=True)
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
    consistency_error = predict(model, 2. * consistency_v) - predict(model, consistency_v) - float(np.log(2.))
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
                  source_sha256=sha(source), architecture_unchanged=True, updates=total_accepted, sweeps=sweep, objective_evaluations=evaluations, resume_sha256=sha(resume_path),
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
    result['invocation_seconds'] = time.monotonic() - start
    result['resumed_evaluation_without_retraining'] = '--resume-evaluation' in sys.argv
    (OUT / 'log_refinement_manifest.json').write_text(json.dumps(result, indent=2))
    if result['status'] == 'LOCKED':
        (OUT / 'log_operator_locked.json').write_text(json.dumps(result, indent=2))
    (OUT / 'log_continuation_summary.txt').write_text(
        'Continuation from latest log refinement with unchanged architecture/inference.\n'
        'Exact gate: teacher-strict pairs stay strict, teacher ties permitted, no reversals.\n'
        'Float64 coordinate updates selected solely by synthetic training labels and ordered-pair outcomes.\n'
        'Every improving sweep checkpoint saved before external validation. No manual coefficient assignment/snapping.\n'
        'All other checkpoints preserved; later assembly runs only after complete gate.\n\n'
        + json.dumps(result,indent=2) + '\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    run()
