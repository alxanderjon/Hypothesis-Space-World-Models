"""Train only the sampled sign-pair router; reuse the frozen zero threshold."""
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

OUT = ROOT / 'branch_switch_sign_transition_training'
SOURCE = ROOT / 'refined_branch_switch_availability_training'
SEED = 28402026


class SignTransition(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold.eval().requires_grad_(False)
        self.head = nn.Linear(9, 2, bias=False, dtype=torch.float64)

    def forward(self, margins, details=False):
        if margins.ndim != 2 or margins.shape[1] != 2:
            raise ValueError('Expected only two finite normalized margins per row.')
        if not torch.isfinite(margins).all() or (margins.abs() > 1).any():
            raise ValueError('Margins must be finite and in [-1,1].')
        signs = self.threshold.decision(margins)
        # Generic categorical encoding of all 3x3 states, not an opposite-sign rule.
        one_hot = nn.functional.one_hot((signs + 1).long(), 3).double()
        pair = (one_hot[:, 0, :, None] * one_hot[:, 1, None, :]).reshape(-1, 9)
        logits = self.head(pair)
        return (logits, signs) if details else logits


def teacher(margins):
    # Label generation only. Avoid multiplication underflow at subnormal inputs.
    s = np.sign(margins)
    return ((s[:, 0] < 0) & (s[:, 1] > 0) |
            (s[:, 0] > 0) & (s[:, 1] < 0)).astype(np.int64)


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p): sha(p) for p in ROOT.rglob('*.json')
                      if any(t in p.name for t in ('manifest', 'locked', 'validated'))})
    source_files = [SOURCE / n for n in ('feature_dataset.npz', 'availability_predictions.csv', 'availability_errors.csv')]
    protected.update({str(p): sha(p) for p in source_files})
    manifest_path = ROOT / 'stationary_magnitude_threshold_smoke/zero_anchor_converged/candidate_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cp = Path(manifest['checkpoint'])
    assert manifest['status'] == 'FROZEN_CANDIDATE' and sha(cp) == manifest['sha256']
    threshold = MagnitudeThreshold()
    threshold.load_state_dict(torch.load(cp, weights_only=False)['state_dict'])
    threshold.eval().requires_grad_(False)
    source = np.load(source_files[0])
    # No availability labels, branch features, ownership or coordinates are read as inputs.
    x = source['features'][:, :2].copy()
    trainmask, heldmask = source['trainmask'], source['heldmask']
    meta = list(csv.DictReader(source_files[1].open()))
    assert len(x) == len(meta)
    table = np.array(list(itertools.product([-1., 0., 1.], repeat=2)))
    tiny = np.nextafter(0., 1.)
    near = np.unique(np.r_[tiny, np.arange(1, 65, dtype=np.uint64).view(np.float64),
                            np.power(10., np.linspace(-320, 0, 97)), [1e-16, 1e-12, .1, 1.]])
    synthetic = np.vstack([table * value for value in near])
    # Whole-parent heldout and independent fresh states never enter fitting.
    train = np.vstack([x[trainmask], synthetic, table])
    target = teacher(train)
    model = SignTransition(threshold)
    initial = model.head.weight.detach().clone()
    optim = torch.optim.Adam(model.head.parameters(), lr=.05)
    xt = torch.from_numpy(train)
    yt = torch.from_numpy(target)
    state_id = ((np.sign(train[:, 0]) + 1) * 3 + np.sign(train[:, 1]) + 1).astype(int)
    count = np.bincount(state_id, minlength=9)
    weights = torch.from_numpy(1. / (9 * count[state_id]))
    history = []
    for step in range(1, 1001):
        optim.zero_grad()
        logits = model(xt)
        loss = (nn.functional.cross_entropy(logits, yt, reduction='none') * weights).sum()
        loss.backward()
        optim.step()
        with torch.no_grad():
            tl = model(torch.from_numpy(table))
            error = int((tl.argmax(1) != torch.from_numpy(teacher(table))).sum())
            separation = float((tl.gather(1, torch.from_numpy(teacher(table))[:, None]).squeeze(1) -
                                tl.gather(1, torch.from_numpy(1-teacher(table))[:, None]).squeeze(1)).min())
        history.append(dict(update=step,loss=float(loss.detach()),truth_table_errors=error,min_logit_separation=separation))
        if error == 0 and separation >= 2:
            break
    config = dict(seed=SEED, inputs=['normalized_signed_margin_1','normalized_signed_margin_2'],
        architecture='frozen MagnitudeThreshold twice -> 3x3 categorical pair encoding -> bias-free Linear(9,2)',
        trainable_parameters=18, actions=['NO_SIGN_TRANSITION','SIGN_TRANSITION'],
        exact_zero='Frozen threshold returns TIE for +0 and -0. Five zero-containing pair states are separately supervised NO_SIGN_TRANSITION; zero flags are returned for audit.',
        training_cases=len(train), archived_fit_cases=int(trainmask.sum()), synthetic_fit_cases=len(synthetic)+len(table),
        training='Random head initialization, Adam lr=.05, balanced nine-state cross-entropy; stop after zero table errors and minimum logit separation >=2.',
        frozen_threshold=dict(checkpoint=str(cp),sha256=sha(cp),theta=float(threshold.theta),alpha=float(threshold.alpha())),
        no_availability_labels=True,no_branch_information=True,no_location_prediction=True,no_recursion=True,no_search=True)
    path = OUT / 'router_before_evaluation.pt'
    digest = save_checkpoint(path, dict(state_dict=model.state_dict(),initial_head_weight=initial,
                                       config=config,optimizer_state=optim.state_dict(),updates=step))
    model.load_state_dict(torch.load(path, weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    write_csv(OUT/'training_history.csv',history)
    metrics = []
    def evaluate(a, name):
        with torch.no_grad():
            logits, signs = model(torch.from_numpy(a), details=True)
        signs=signs.numpy(); pred=logits.argmax(1).numpy(); truth=teacher(a)
        zero=(a==0).any(1)
        row=dict(group=name,cases=len(a),sign_errors_sample_1=int(np.sum(signs[:,0]!=np.sign(a[:,0]))),
            sign_errors_sample_2=int(np.sum(signs[:,1]!=np.sign(a[:,1]))),transition_errors=int(np.sum(pred!=truth)),
            teacher_transitions=int(truth.sum()),predicted_transitions=int(pred.sum()),
            exact_zero_pairs=int(zero.sum()),zero_pair_transition_errors=int(np.sum(zero & (pred!=0))))
        metrics.append(row)
        return pred, signs
    pred, signs = evaluate(x,'all_saved_scan_pairs')
    evaluate(x[trainmask],'archived_training')
    evaluate(x[heldmask],'heldout_archived')
    for group in sorted({r['group'] for r in meta}):
        evaluate(x[np.array([r['group']==group for r in meta])],group)
    for name, field in [('known_366','known_366'),('previous_false_positives','previous_false_positive'),
                         ('five_current_failures','error'),('12_smooth_exceptions','smooth_control'),('all_smooth_controls','smooth_present')]:
        evaluate(x[np.array([r[field]=='True' for r in meta])],name)
    for left,right in itertools.product([-1,0,1],repeat=2):
        mask=(np.sign(x[:,0])==left)&(np.sign(x[:,1])==right)
        evaluate(np.vstack([x[mask],table[(table[:,0]==left)&(table[:,1]==right)]]),f'sign_pair_{left}_{right}')
    rng=np.random.default_rng(SEED+1)
    fresh=np.ldexp(rng.uniform(1,2,(30000,2)),rng.integers(-1074,-1,(30000,2)))
    fresh*=rng.choice([-1.,1.],fresh.shape)
    fresh[rng.random(fresh.shape)<.1]=0.
    evaluate(fresh,'fresh_synthetic_heldout_margins')
    subnormal=np.arange(1,4097,dtype=np.uint64).view(np.float64)
    boundary=np.array(list(itertools.product([-tiny,-0.,0.,tiny],repeat=2)))
    evaluate(boundary,'one_ULP_around_zero_signed_zero_pairs')
    evaluate(np.vstack([table*v for v in subnormal]),'first_4096_subnormal_nine_state_pairs')
    with torch.no_grad():
        symmetry_errors=int(np.sum(model(torch.from_numpy(fresh[:,::-1].copy())).argmax(1).numpy()!=model(torch.from_numpy(fresh)).argmax(1).numpy()))
        inversion_errors=int(np.sum(model(torch.from_numpy(-fresh)).argmax(1).numpy()!=model(torch.from_numpy(fresh)).argmax(1).numpy()))
    table_pred, table_signs=evaluate(table,'complete_nine_state_truth_table')
    write_csv(OUT/'truth_table.csv',[dict(margin_1=a[0],margin_2=a[1],sign_1=int(s[0]),sign_2=int(s[1]),
        zero_present=bool((a==0).any()),teacher=int(teacher(table)[i]),prediction=int(table_pred[i])) for i,(a,s) in enumerate(zip(table,table_signs))])
    rows=[dict(group=r['group'],case=r['case'],boundary_side_for_identification_only=r['boundary_side'],
        margin_1=x[i,0],margin_2=x[i,1],sign_1=int(signs[i,0]),sign_2=int(signs[i,1]),
        zero_1=bool(x[i,0]==0),zero_2=bool(x[i,1]==0),teacher_transition=int(teacher(x)[i]),
        prediction=int(pred[i]),error=bool(pred[i]!=teacher(x)[i]),prior_availability_failure=r['error'],
        previous_false_positive=r['previous_false_positive'],smooth_control=r['smooth_control']) for i,r in enumerate(meta)]
    write_csv(OUT/'scan_pair_replay.csv',rows)
    write_csv(OUT/'five_failures_replay.csv',[r for r in rows if r['prior_availability_failure']=='True'])
    write_csv(OUT/'validation.csv',metrics)
    (OUT/'learned_parameters.json').write_text(json.dumps(dict(head_weight=model.head.weight.tolist(),initial_head_weight=initial.tolist(),config=config),indent=2))
    unchanged=all(sha(Path(p))==h for p,h in protected.items())
    assert unchanged
    passed=all(r['transition_errors']==r['sign_errors_sample_1']==r['sign_errors_sample_2']==0 for r in metrics) and symmetry_errors==inversion_errors==0
    result=dict(operator='branch_switch_sign_transition_operator',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        checkpoint=str(path),sha256=digest,implementation=str(Path(__file__)),implementation_sha256=sha(Path(__file__)),
        config=config,metrics=metrics,updates=step,symmetry_errors=symmetry_errors,simultaneous_sign_inversion_errors=inversion_errors,
        protected_files=len(protected),all_protected_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-start)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:
        lock=OUT/'branch_switch_sign_transition_locked.json'
        lock.write_text(json.dumps(result,indent=2))
        lock.with_suffix('.sha256').write_text(sha(lock)+'  '+lock.name+'\n')
    summary=[f"branch_switch_sign_transition_operator = {result['status']}",
        f"18 trained parameters; frozen sign threshold theta={float(threshold.theta)}, alpha={float(threshold.alpha())}.",
        f"Random initialization; {step} Adam updates; saved and reloaded before evaluation.",
        'Only two margins enter inference. Pair encoding is generic; learned logits choose the action.',
        'Exact signed zeros stay explicit TIE signs, and every zero-containing pair learns NO_SIGN_TRANSITION.',
        'Smooth opposite-sign roots correctly return SIGN_TRANSITION here. This is not branch availability.',
        'No validity/ownership or branch-change information is used. Invalid pairs must be handled by later machinery.',
        '\nValidation (slices overlap):']
    summary += [f"{r['group']}: {r['cases']} cases; sign1/sign2/transition errors {r['sign_errors_sample_1']}/{r['sign_errors_sample_2']}/{r['transition_errors']}; transitions {r['teacher_transitions']}; zero pairs {r['exact_zero_pairs']}" for r in metrics]
    summary += [f'\nSymmetry errors: {symmetry_errors}; simultaneous sign-inversion errors: {inversion_errors}.',
        f"All {len(protected)} protected hashes unchanged. Runtime {result['elapsed_seconds']:.3f}s, excluding implementation/report inspection.",
        f'Checkpoint SHA256: {digest}', 'No full availability training, candidate location, recursion, or search.']
    (OUT/'sign_transition_summary.txt').write_text('\n'.join(summary)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','metrics')},indent=2),flush=True)
    print('\n'.join(summary),flush=True)


if __name__=='__main__':
    run()
