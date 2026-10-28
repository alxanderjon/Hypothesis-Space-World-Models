"""Refine the same global monotone model through supervised sparse support fitting."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.geometry.train_mantissa_log import MonotoneMantissaLog, predict, metric, HOLDOUT, DEGREE, EPS
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv
ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'mantissa_log_operator_support_refine'
SEED=27432026

def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    source=ROOT/'mantissa_log_operator_fit_refine/mantissa_log_before_evaluation.pt'
    model=MonotoneMantissaLog();model.load_state_dict(torch.load(source,weights_only=False)['state_dict'])
    data=np.load(ROOT/'mantissa_log_operator_smoke/training_data.npz');x=data['m'];y=data['targets']
    divisor=abs(y).copy();divisor[divisor==0]=1.
    with torch.no_grad():phi=model.features(torch.from_numpy(x)).numpy()
    R=phi/divisor[:,None];target=y/divisor;scale=np.sqrt(np.mean(R*R,axis=0));Z=R/scale
    w=model.coefficients.detach().numpy().copy()*scale
    config=dict(seed=SEED,source_checkpoint=str(source),source_sha256=sha(source),architecture_unchanged=True,
        optimizer='Supervised bounded support refinement: all single add/delete/swap proposals, nonnegative least-squares coefficients only; no teacher-selected terms',
        max_rounds=6,training_examples=len(x),dtype='float64',heldout_subintervals=HOLDOUT,
        absolute_gate=8*EPS,relative_gate=32*EPS,protected_hashes=protected,
        selection='Actual global Horner training relative loss only; no held-out labels used for selection')
    (OUT/'config.json').write_text(json.dumps(config,indent=2));np.savez_compressed(OUT/'training_data.npz',m=x,targets=y)
    def set_w(weights):
        with torch.no_grad():model.coefficients.copy_(torch.from_numpy(weights/scale))
    def loss_of(weights):
        set_w(weights);e=(y-predict(model,x))/divisor;return float(np.mean(e*e))
    def fit(indices):
        indices=np.array(sorted(indices),dtype=int)
        if not len(indices):return None
        coeff=np.linalg.lstsq(Z[:,indices],target,rcond=None)[0]
        if np.any(coeff<0):return None
        candidate=np.zeros(DEGREE);candidate[indices]=coeff
        loss=loss_of(candidate)
        # Residual corrections preserve positivity, never snap teacher constants.
        for _ in range(3):
            residual=(y-predict(model,x))/divisor
            delta=np.linalg.lstsq(Z[:,indices],residual,rcond=None)[0]
            trial=candidate.copy();trial[indices]+=delta
            if np.any(trial<0):break
            improved=loss_of(trial)
            if improved<loss:candidate,loss=trial,improved
            else:set_w(candidate);break
        return loss,candidate
    best=loss_of(w);history=[];checkpoints=[];evaluated=0;used=set()
    for round_id in range(1,7):
        active=set(np.flatnonzero(w>0));inactive=set(range(DEGREE))-active
        proposals={tuple(sorted(active-{i})) for i in active}
        proposals|={tuple(sorted(active|{j})) for j in inactive}
        proposals|={tuple(sorted((active-{i})|{j})) for i in active for j in inactive}
        next_w=w.copy();next_loss=best
        for support in sorted(proposals):
            if support in used:continue
            used.add(support);evaluated+=1
            fitted=fit(support)
            if fitted is not None and fitted[0]<next_loss:next_loss,next_w=fitted
        improved=next_loss<best
        w,best=next_w,next_loss;set_w(w)
        history.append(dict(round=round_id,improved=improved,relative_rmse=float(np.sqrt(best)),active_powers=json.dumps((np.flatnonzero(w>0)+1).tolist()),evaluated_supports=evaluated))
        print(json.dumps(history[-1]),flush=True)
        if improved:
            checkpoint=OUT/f'best_round_{round_id:02d}_before_evaluation.pt'
            digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,status='SAVED_BEFORE_EVALUATION',training_metrics=metric(y,predict(model,x))))
            checkpoints.append(dict(round=round_id,checkpoint=str(checkpoint),sha256=digest))
        if not improved or np.sqrt(best)<=4*EPS:break
    if not checkpoints:
        checkpoint=OUT/'mantissa_log_before_evaluation.pt'
        digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,status='SAVED_BEFORE_EVALUATION',training_metrics=metric(y,predict(model,x))))
        checkpoints.append(dict(round=0,checkpoint=str(checkpoint),sha256=digest))
    write_csv(OUT/'training_history.csv',history);write_csv(OUT/'checkpoint_hashes.csv',checkpoints)
    write_csv(OUT/'learned_coefficients.csv',[dict(power=i+1,coefficient=float(a)) for i,a in enumerate(model.coefficients.detach().numpy())])
    print(json.dumps(dict(event='saved_before_evaluation',sha256=digest,training=metric(y,predict(model,x)))),flush=True)
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
