"""Standalone float64 exponent -> learned no-bias scalar; no log assembly."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn

from experiments.sho.smoke_artifacts import save_checkpoint

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'exponent_log2_operator_smoke'
SEED = 27442026
EPS = np.finfo(np.float64).eps
HELDOUT = [(-1074, -1023), (-900, -801), (-257, -129), (128, 256), (801, 900), (1000, 1023)]


class ExponentLog2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False, dtype=torch.float64)

    def forward(self, exponent):
        return self.linear(exponent.reshape(-1, 1)).reshape(exponent.shape)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def predict(model, exponents):
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(exponents, dtype=np.float64))).numpy()


def metrics(y, p):
    error = p-y
    nz = y != 0
    rel = error[nz]/y[nz]
    return dict(examples=y.size, rmse=float(np.sqrt(np.mean(error*error))),
                mae=float(np.mean(abs(error))), max_error=float(np.max(abs(error))),
                relative_rmse=float(np.sqrt(np.mean(rel*rel))) if rel.size else 0.,
                max_relative_error=float(np.max(abs(rel))) if rel.size else 0.,
                max_normalized_error=float(np.max(abs(error)/np.maximum(1., abs(y)))),
                exact_agreement=int(np.sum(p == y)),
                bit_exact_agreement=int(np.sum(p.view('u8') == y.view('u8'))))


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    exponents = np.arange(-1074, 1024, dtype=np.float64)
    heldout = np.zeros(exponents.size, dtype=bool)
    for lo, hi in HELDOUT:
        heldout |= (exponents >= lo) & (exponents <= hi)
    x = exponents[~heldout]
    y = x*np.log(np.float64(2))
    model = ExponentLog2()
    initial = float(model.linear.weight.detach()[0, 0])
    assert initial != np.log(2.)
    history = []
    best = [np.inf, None]
    input_rms = float(np.sqrt(np.mean(x*x)))
    tx = torch.from_numpy(x)

    def residual(weight):
        with torch.no_grad():
            model.linear.weight.copy_(torch.from_numpy(weight.reshape(1, 1)))
            error = model(tx).numpy()-y
        loss = float(np.mean(error*error))
        history.append(dict(evaluation=len(history)+1, weight=float(weight[0]),
                            training_rmse=float(np.sqrt(loss))))
        if loss < best[0]:
            best[:] = [loss, weight.copy()]
        return error/input_rms

    result = least_squares(residual, np.array([initial]), jac=lambda _: (x/input_rms)[:, None],
                           method='trf', max_nfev=30, ftol=1e-14, xtol=1e-14, gtol=1e-14)
    with torch.no_grad():
        model.linear.weight.copy_(torch.from_numpy(best[1].reshape(1, 1)))
    config = dict(seed=SEED, dtype='float64', model='y_hat = w * e', bias=False,
                  initialization=initial, training_examples=len(x), useful_exponent_range=[-1074, 1023],
                  range_convention='v=m*2**e, 1<=m<2; includes minimum subnormal and maximum finite float64',
                  heldout_ranges=HELDOUT, heldout_examples=int(heldout.sum()),
                  teacher='float64 integer exponent times np.log(float64(2))',
                  optimizer='TRF continuous least squares, analytic input Jacobian, residuals scaled by input RMS',
                  max_evaluations=30, precision_gate_normalized=8*EPS,
                  only_supervised_residual_updates=True, no_coefficient_snapping=True,
                  no_mantissa_loading=True, no_full_log_assembly=True)
    checkpoint = OUT/'exponent_log2_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
                                             training_objective=best[0], objective_calls=len(history),
                                             optimizer_message=result.message, status='SAVED_BEFORE_EVALUATION'))
    print(json.dumps(dict(event='trained_and_saved_before_evaluation', initial=initial,
                          weight=float(best[1][0]), objective_calls=len(history), sha256=digest)), flush=True)
    # Persist training provenance before any held-out evaluation.
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    np.savez_compressed(OUT/'training_data.npz', exponent=x, target=y)
    write_csv(OUT/'training_history.csv', history)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    target = exponents*np.log(np.float64(2))
    prediction = predict(model, exponents)
    rows = []
    selections = [('all_useful_float64_exponents', np.ones(len(exponents), bool)),
                  ('training', ~heldout), ('heldout_all', heldout),
                  ('negative', exponents < 0), ('zero', exponents == 0), ('positive', exponents > 0)]
    for lo, hi in HELDOUT:
        selections.append((f'heldout_{lo}_to_{hi}', (exponents >= lo) & (exponents <= hi)))
    for lo, hi in [(-1074,-513),(-512,-129),(-128,-1),(1,128),(129,512),(513,1023)]:
        selections.append((f'range_{lo}_to_{hi}', (exponents >= lo) & (exponents <= hi)))
    for name, mask in selections:
        rows.append(dict(subset=name, exponent_min=int(exponents[mask].min()),
                         exponent_max=int(exponents[mask].max()), **metrics(target[mask], prediction[mask])))
    write_csv(OUT/'exponent_log2_validation.csv', rows)
    write_csv(OUT/'exponent_log2_predictions.csv', [dict(exponent=int(e), heldout=bool(h), teacher=float(t),
              prediction=float(p), error=float(p-t)) for e,h,t,p in zip(exponents,heldout,target,prediction)])

    zero = float(predict(model, np.array([0.]))[0])
    symmetric_e = np.arange(1,1024,dtype=np.float64)
    odd_error = predict(model,-symmetric_e)+predict(model,symmetric_e)
    properties = [dict(property='zero', examples=1, max_error=abs(zero), exact_positive_zero=zero==0 and not np.signbit(zero)),
                  dict(property='odd_symmetry', examples=len(symmetric_e), max_error=float(abs(odd_error).max()), exact_agreement=int(np.sum(odd_error==0)))]
    # Exhaustive pairs whose operands AND sum are useful binary exponents.
    # Addition here is a property test, not a trained or assembled component.
    pairs = equal_pairs = 0
    max_error = max_scaled = teacher_max = max_residual_difference = 0.
    worst = None
    for left in np.array_split(exponents, 17):
        sums = left[:,None]+exponents[None,:]
        valid = (sums >= -1074) & (sums <= 1023)
        lhs = predict(model,sums[valid])
        rhs = (predict(model,left)[:,None]+prediction[None,:])[valid]
        teacher_lhs = sums[valid]*np.log(np.float64(2))
        teacher_rhs = (left[:,None]*np.log(np.float64(2))+target[None,:])[valid]
        residual = lhs-rhs
        teacher_residual = teacher_lhs-teacher_rhs
        scale = np.maximum(1.,(abs(predict(model,left))[:,None]+abs(prediction)[None,:])[valid])
        pos = int(np.argmax(abs(residual)))
        if worst is None or abs(residual[pos]) > max_error:
            ij = np.argwhere(valid)[pos]
            worst = dict(e1=int(left[ij[0]]),e2=int(exponents[ij[1]]),
                         lhs=float(lhs[pos]),rhs=float(rhs[pos]),residual=float(residual[pos]),
                         teacher_residual=float(teacher_residual[pos]))
        max_error = max(max_error,float(abs(residual).max()))
        max_scaled = max(max_scaled,float((abs(residual)/scale).max()))
        teacher_max = max(teacher_max,float(abs(teacher_residual).max()))
        max_residual_difference = max(max_residual_difference,float(abs(residual-teacher_residual).max()))
        pairs += int(valid.sum()); equal_pairs += int(np.sum(residual==0))
    properties.append(dict(property='additivity_all_in_range_pairs', examples=pairs, max_error=max_error,
                           exact_agreement=equal_pairs,max_normalized_error=max_scaled,
                           teacher_max_error=teacher_max,max_difference_from_teacher_residual=max_residual_difference))
    write_csv(OUT/'exponent_log2_properties.csv', properties)
    (OUT/'additivity_worst_case.json').write_text(json.dumps(worst,indent=2))
    unchanged = all(sha(Path(p))==h for p,h in protected.items())
    assert unchanged
    passed = bool(rows[0]['max_normalized_error'] <= 8*EPS and rows[0]['max_relative_error'] <= 8*EPS
                  and np.all(np.isfinite(prediction)) and zero==0 and not np.signbit(zero)
                  and np.all(odd_error==0) and max_scaled<=8*EPS)
    learned = float(model.linear.weight.detach()[0,0])
    summary = dict(status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',gate_passed=passed,
                   operator='exponent_log2_operator',learned_w=learned,learned_w_hex=learned.hex(),bias=0,
                   initial_w=initial,training_evaluations=len(history),checkpoint=str(checkpoint),sha256=digest,
                   overall=rows[0],heldout=rows[2],properties=properties,
                   protected_checkpoints=len(protected),all_protected_hashes_unchanged=unchanged,
                   elapsed_seconds=time.monotonic()-start,no_full_log_assembly=True)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:
        (OUT/'candidate_manifest.json').write_text(json.dumps(dict(operator='exponent_log2_operator',status='FROZEN_CANDIDATE',
            checkpoint=str(checkpoint),sha256=digest,model='ExponentLog2',module='experiments.geometry.train_exponent_log2_operator',
            valid_integer_exponent_range=[-1074,1023],gate_results=str(OUT/'results.json')),indent=2))
    text = [f"exponent_log2_operator = {summary['status']}",f'Learned w: {learned:.17g} ({learned.hex()}); no bias.',
            f'Random initial w: {initial:.17g}; supervised least-squares objective calls: {len(history)}.',
            f"All {len(exponents)} useful exponents, including {heldout.sum()} held out: RMSE={rows[0]['rmse']:.17g}, MAE={rows[0]['mae']:.17g}, max={rows[0]['max_error']:.17g}.",
            f"Bit-exact teacher agreement: {rows[0]['bit_exact_agreement']}/{len(exponents)}.",
            f'e=0 produces exact positive zero; odd symmetry exact on {len(symmetric_e)} in-range positive/negative pairs.',
            f'Additivity tested on {pairs} in-range exponent pairs: maximum float64 residual {max_error:.17g}; teacher maximum {teacher_max:.17g}.',
            f'Maximum difference from teacher additivity residual: {max_residual_difference:.17g}. Nonzero additive residual is floating-point rounding, not additional model error.',
            f'Checkpoint: {checkpoint}; SHA256: {digest}. Saved before held-out evaluation.',
            f'All {len(protected)} pre-existing checkpoint hashes unchanged. No mantissa model loaded; full log not assembled.',
            f'Elapsed: {summary["elapsed_seconds"]:.3f} seconds.']
    (OUT/'exponent_log2_summary.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    run()
