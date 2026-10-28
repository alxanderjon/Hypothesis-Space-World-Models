"""Replay saved log inputs with exact/learned mantissa; no training or mutation."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.geometry.train_mantissa_log import MonotoneMantissaLog, predict

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT/'log_decomposition_diagnostic'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=False)
    protected = {str(p):sha(p) for p in ROOT.rglob('*.pt')}
    source = ROOT/'addition_operator_smoke/assembled_log_check_inputs.npz'
    source_hash = sha(source)
    with np.load(source) as data:
        v = data['ordered']; saved_direct = data['teacher']; saved_learned = data['predictions']
    assert np.all(np.isfinite(v)&(v>0)) and np.all(np.diff(v)>0)
    mantissa_manifest = json.loads((ROOT/'mantissa_log_operator_support_refine/candidate_manifest.json').read_text())
    checkpoint = Path(mantissa_manifest['checkpoint'])
    assert sha(checkpoint)==mantissa_manifest['sha256']
    model = MonotoneMantissaLog()
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    # Check the two assembled scalar operators without retraining or replacing them.
    for folder,key,expected in [('exponent_log2_operator_smoke','linear.weight',np.array([[np.log(2.)]])),
                                ('addition_operator_smoke','weights',np.ones(2))]:
        manifest = json.loads((ROOT/folder/'candidate_manifest.json').read_text())
        cp = Path(manifest['checkpoint']); assert sha(cp)==manifest['sha256']
        weights = torch.load(cp,weights_only=False)['state_dict'][key].numpy()
        assert np.array_equal(weights,expected)
    f,e = np.frexp(v); m,e = 2*f,e-1
    assert np.array_equal(np.ldexp(m,e).view('u8'),v.view('u8'))
    direct = np.log(v)
    exact_m = np.log(m)
    learned_m = predict(model,m)
    exponent = e.astype(np.float64)*np.log(np.float64(2))
    decomp = exact_m+exponent
    learned = learned_m+exponent
    assert np.array_equal(direct.view('u8'),saved_direct.view('u8'))
    assert np.array_equal(learned.view('u8'),saved_learned.view('u8'))
    dt,dd,dl = np.diff(direct),np.diff(decomp),np.diff(learned)
    previous = np.flatnonzero(np.diff(saved_learned)<0)
    assert len(previous)==103
    # Cross-check replay against the original persisted failure CSV, not just predictions.
    old_cases = list(csv.DictReader((ROOT/'addition_operator_smoke/assembled_log_ordering_cases.csv').open()))
    old_pairs = {(float(r['left']),float(r['right'])) for r in old_cases if r['kind']=='reversal'}
    assert old_pairs=={(float(v[i]),float(v[i+1])) for i in previous}
    adjacent = np.diff(v.view('u8'))==1
    boundary = np.diff(e)!=0
    records = []
    metrics = {}
    for name,value,diff in [('exact_decomposed_teacher',decomp,dd),('learned_mantissa_exact_exponent',learned,dl)]:
        error = value-direct
        entry = dict(record_type='aggregate',method=name,inputs=len(v),rmse=float(np.sqrt(np.mean(error*error))),
                     mae=float(np.mean(abs(error))),max_error=float(abs(error).max()),
                     reversals=int(np.sum(diff<0)),teacher_strict_collapses=int(np.sum((dt>0)&(diff==0))),
                     adjacent_float_pairs=int(adjacent.sum()),adjacent_reversals=int(np.sum(adjacent&(diff<0))),
                     teacher_strict_adjacent_collapses=int(np.sum(adjacent&(dt>0)&(diff==0))),
                     binary_boundary_reversals=int(np.sum(boundary&(diff<0))),
                     boundary_ordering_differences=int(np.sum(boundary&(np.sign(diff)!=np.sign(dt)))))
        metrics[name] = entry; records.append(entry)
    delta = learned-decomp
    metrics['isolated_mantissa_effect'] = dict(rmse=float(np.sqrt(np.mean(delta*delta))),mae=float(np.mean(abs(delta))),max_error=float(abs(delta).max()))
    exact_old = int(np.sum(dd[previous]<0)); learned_only_old = int(np.sum(dd[previous]>=0))
    exact_boundary_changes = boundary&(np.sign(dd)!=np.sign(dt))
    learned_only = (dl<0)&(dd>=0)
    relevant = exact_boundary_changes | (dl<0) | ((dt>0)&adjacent&(dd==0))
    for i in np.flatnonzero(relevant):
        row = dict(record_type='ordering_case',pair_index=int(i),previous_reversal=bool(dl[i]<0),
                   adjacent_float=bool(adjacent[i]),binary_boundary=bool(boundary[i]),
                   exact_boundary_order_differs=bool(exact_boundary_changes[i]),
                   exact_decomposed_reversal=bool(dd[i]<0),learned_reversal=bool(dl[i]<0),
                   learned_only_reversal=bool(learned_only[i]),direct_step=float(dt[i]),
                   exact_decomposed_step=float(dd[i]),learned_step=float(dl[i]))
        for side,j in [('left',i),('right',i+1)]:
            row.update({f'v_{side}':float(v[j]),f'v_{side}_hex':float(v[j]).hex(),f'm_{side}':float(m[j]),
                        f'm_{side}_hex':float(m[j]).hex(),f'e_{side}':int(e[j]),f'exact_mantissa_log_{side}':float(exact_m[j]),
                        f'learned_mantissa_log_{side}':float(learned_m[j]),f'exponent_term_{side}':float(exponent[j]),
                        f'direct_log_{side}':float(direct[j]),f'decomposed_log_{side}':float(decomp[j]),f'learned_log_{side}':float(learned[j])})
        records.append(row)
    unchanged = all(sha(Path(p))==h for p,h in protected.items())
    assert unchanged and sha(source)==source_hash
    conclusion = ('All previous reversals persist with exact teacher mantissa logs: float64 decomposition/recomposition rounding is responsible; do not retrain the mantissa network.'
                  if exact_old==103 and not learned_only.any() else
                  'Attribution is mixed; inspect exact-teacher versus learned-only cases before any further work.')
    result = dict(inputs=len(v),metrics=metrics,previous_reversals_replayed=len(previous),
                  previous_reversals_with_exact_mantissa=exact_old,previous_reversals_only_with_learned_mantissa=learned_only_old,
                  all_learned_only_reversals=int(learned_only.sum()),exact_decomposition_verified=True,
                  previous_predictions_replayed_bit_exactly=True,conclusion=conclusion,
                  protected_checkpoints=len(protected),all_checkpoint_hashes_unchanged=unchanged,
                  input_pool=str(source),input_sha256=source_hash,seconds=time.monotonic()-start)
    with (OUT/'log_decomposition_diagnostic.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader(); writer.writerows(records)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    lines=[conclusion,f'Existing inputs replayed: {len(v):,}; exact frexp/ldexp reconstruction and saved prediction bits verified.',
           'Method | RMSE | MAE | max error | reversals | teacher-strict adjacent collapses | boundary ordering differences']
    for name,r in metrics.items():
        if name=='isolated_mantissa_effect': continue
        lines.append(f"{name} | {r['rmse']:.17g} | {r['mae']:.17g} | {r['max_error']:.17g} | {r['reversals']} | {r['teacher_strict_adjacent_collapses']} | {r['boundary_ordering_differences']}")
    lines += [f'Previous 103 locations: {exact_old} reverse with exact log(m); {learned_only_old} reverse only with learned L(m).',
              'CSV contains all exact-decomposed boundary ordering differences, all 103 prior reversals, and teacher-strict adjacent collapses, with raw/hex inputs and each arithmetic term.',
              'The binary input decomposition is exact. The discrepancy arises when the separately rounded log(m) and e*log(2) terms are added in float64.',
              'No training, checkpoint modifications, recursion, or search. Mantissa candidate retained unchanged.',
              f'{len(protected)} checkpoint hashes unchanged. Execution: {result["seconds"]:.3f} seconds.']
    (OUT/'log_decomposition_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
