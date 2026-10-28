"""Isolated approved three-branch N2-budget diagnostic; production is untouched."""
from pathlib import Path
import argparse
import csv
import inspect
import json
import platform
import sys
import time
import traceback

import numpy as np
import scipy
import torch

from experiments.resume_hybrid_student_after_ckdtree import Log, run_n2_with_best_ever_assembly
from experiments.sho import audit_coordinated_generic_recovery as audit
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.run_companion6_stabilized_residual_live import atomic, bits_equal, clean, sha256
from experiments.sho.validate_generic_update_network_adapter import protected
from experiments.sho.deterministic_full_k_framework import hashes

OUT = Path('outputs/sho/n2_budget_discrimination_three_branch_20261013')
SRC = Path('outputs/sho/generic_update_network_targeted_recovery/companion_6')
DATA = Path('outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz')
BRANCHES = ['n00005_i00004', 'n00004_i00003', 'n00001_i00000']


def read(path):
    return json.loads(Path(path).read_text())


def integrity():
    files = set()
    for root in ['experiments', 'src']:
        files.update(Path(root).rglob('*.py'))
    inputs = {DATA, Path('coordination/DECISIONS.md'), Path('coordination/NEXT_TASK.md'), Path('RESEARCH_HANDOFF.md')}
    for root in ['outputs/deterministic_teacher_frozen', 'outputs/hybrid_student_baseline_frozen',
                 'outputs/sho/homogeneity_peeled_baseline', 'outputs/canonical_frozen_teacher_regression_822',
                 str(SRC.parent), 'outputs/sho/companion6_stabilized_residual_n3_live',
                 'outputs/sho/coordinated_generic_recovery_audit', 'outputs/sho/stabilized_residual_n3_from_n2']:
        inputs.update(p for p in Path(root).rglob('*') if p.is_file())
    return dict(protected=protected(), checkpoints=hashes(),
                sources={str(p): sha256(p) for p in sorted(files)},
                inputs={str(p): sha256(p) for p in sorted(inputs)})


def differences(before, after):
    return {group: [key for key in sorted(set(before[group]) | set(after[group]))
                    if before[group].get(key) != after[group].get(key)] for group in before}


def init():
    OUT.mkdir(exist_ok=False)
    before = integrity()
    atomic(OUT/'integrity_before.json', before)
    previous = read('outputs/sho/coordinated_generic_recovery_audit/integrity_before.json')
    changes = {g: [p for p, value in previous[g].items() if before[g].get(p) != value]
               for g in ['protected', 'checkpoints', 'sources']}
    atomic(OUT/'prior_audit_integrity_comparison.json', changes)
    if any(changes.values()):
        raise RuntimeError('Prior audit protected/checkpoint/source integrity mismatch')
    with open('outputs/sho/coordinated_generic_recovery_audit/endpoint_summary.csv') as f:
        candidates = list(csv.DictReader(f))
    control = next(x for x in candidates if x['branch'] == '1,0')
    assert float(control['late250_relative_improvement']) == min(float(x['late250_relative_improvement']) for x in candidates)
    atomic(OUT/'manifest.json', dict(
        branches=BRANCHES, control_branch='1,0', control=control,
        control_rationale='Smallest final-250 relative Powell improvement among all 11 saved endpoints (1.2088655%); final-100 improvement 0.1118374%. Existing first N1 site only. Contrasts with 52.0%/65.1% final-250 progress of (4,3)/(5,4). Site-0 basin and endpoint loss differ, so this is descriptive discrimination, not a matched causal control for every basin variable.',
        all_control_candidates=candidates,
        settings=dict(xtol=1e-12, ftol=1e-15, maxiter=200, diagnostic_maxfev=3000, production_maxfev=1500,
                      n3_max_accepted_steps=500, handoff=1e-20, n3_direction='stabilized_residual'),
        python=sys.version, torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), script_hash=sha256(Path(__file__)),
        continuation='One unchanged production wrapper invocation with maxfev=3000; exact point/loss checks on calls 1..1500; no restart.',
        n3_screen='Unchanged audit_coordinated_generic_recovery.screen; only diagnostic OUT and TRAJ globals supplied.'))
    print('INITIALIZED: control (1,0); prior audit integrity verified', flush=True)


def history_summary(history, start_loss):
    best = float(start_loss)
    theta = np.asarray(history[0]['phi'], np.float64)
    path = 0.0
    rows = []
    for index, row in enumerate(history, 1):
        if row['loss'] < best:
            new = np.asarray(row['phi'], np.float64)
            path += float(np.linalg.norm(new-theta))
            best, theta = row['loss'], new.copy()
        rows.append(dict(evaluation=index, best_loss=best, best_phi=theta.copy(), best_path=path))
    return rows


def run_n2(stem):
    target = OUT/(stem+'_n2_extended.json')
    trace_path = OUT/(stem+'_n2_evaluations.jsonl')
    if target.exists() or trace_path.exists():
        raise RuntimeError('Refusing to overwrite N2 artifacts')
    saved = read(SRC/(stem+'_n2.json'))
    assert len(saved['history']) == saved['result']['evaluations'] == 1500
    start = np.asarray(saved['history'][0]['phi'], np.float64)
    start_loss = saved['history'][0]['loss']
    adapter = GenericSHOUpdateNetworkAdapter()
    trajectory = torch.from_numpy(np.load(DATA)['prefix_6'].copy())
    history = []
    t0 = time.monotonic()
    with trace_path.open('x') as stream:
        def objective(phi):
            i = len(history)+1
            point = np.asarray(phi, np.float64)
            with torch.no_grad():
                value = float(adapter.objective(point, trajectory))
            row = dict(evaluation=i, phi=point.tolist(), loss=value)
            history.append(row)
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            if i <= 1500:
                reference = saved['history'][i-1]
                checks = dict(phi=bits_equal(point, reference['phi']), loss=bits_equal(value, reference['loss']))
                if not all(checks.values()):
                    atomic(OUT/(stem+'_replay_failure.json'), dict(evaluation=i, checks=checks, actual=row, expected=reference))
                    raise RuntimeError(f'{stem}: exact N2 reproduction failed at evaluation {i}')
            if i == 1500:
                frames = []
                frame = inspect.currentframe().f_back
                while frame:
                    if frame.f_code.co_name in ['minimize_powell_exact', 'line_search_exact', 'bounded_scalar_exact']:
                        scalars = {k: v for k, v in frame.f_locals.items()
                                   if isinstance(v, (int, float, bool, np.ndarray, np.number))}
                        frames.append(dict(function=frame.f_code.co_name, source=frame.f_code.co_filename,
                                           line=frame.f_lineno, numeric_state=scalars))
                    frame = frame.f_back
                atomic(OUT/(stem+'_state_at_evaluation_1500.json'), dict(frames=frames, note='Observed in-flight call stack; execution continues in same invocation.'))
                print(stem, 'EXACT REPLAY 1500/1500', flush=True)
            if i % 500 == 0:
                print(stem, 'N2', i, 'best', min(z['loss'] for z in history), flush=True)
            return value
        result = run_n2_with_best_ever_assembly(objective, start, start_loss,
            np.asarray(saved['box_lower']), np.asarray(saved['box_upper']), Log(), maxfev=3000, trace=True)
    best_rows = history_summary(history, start_loss)
    old = best_rows[1499]
    end = best_rows[-1]
    assert bits_equal(old['best_loss'], saved['result']['loss']) and bits_equal(old['best_phi'], saved['result']['theta'])
    assert bits_equal(end['best_loss'], result['loss']) and bits_equal(end['best_phi'], result['theta'])
    late = {}
    for window in [100, 250, 500]:
        previous = best_rows[-window-1]
        late[str(window)] = dict(start_best_loss=previous['best_loss'], end_best_loss=end['best_loss'],
            relative_improvement=(previous['best_loss']-end['best_loss'])/previous['best_loss'],
            best_point_path=end['best_path']-previous['best_path'])
    payload = dict(branch=stem, result={k:v for k,v in result.items() if k not in ['tracker','powell']},
        box_lower=saved['box_lower'], box_upper=saved['box_upper'], exact_prefix_evaluations=1500,
        uninterrupted_single_invocation=True, original_endpoint=saved['result'],
        objective_loss_at_evaluation_1500=history[1499]['loss'],
        objective_loss_at_final_evaluation=history[-1]['loss'],
        best_loss_at_1500=old['best_loss'], best_loss_at_end=end['best_loss'],
        best_point_displacement_after_1500=float(np.linalg.norm(end['best_phi']-old['best_phi'])),
        best_point_path_after_1500=end['best_path']-old['best_path'], late=late,
        elapsed_seconds=time.monotonic()-t0, source_hash=sha256(SRC/(stem+'_n2.json')),
        trace_hash=sha256(trace_path))
    atomic(OUT/(stem+'_powell_state_and_cycles.json'), result['powell'])
    atomic(target, payload)
    print('N2 COMPLETE', stem, result['evaluations'], result['loss'], flush=True)


def run_n3(stem, arm):
    audit.OUT = OUT
    audit.TRAJ = torch.from_numpy(np.load(DATA)['prefix_6'].copy())
    endpoint = read(SRC/(stem+'_n2.json'))['result'] if arm == 'original' else read(OUT/(stem+'_n2_extended.json'))['result']
    answer = audit.screen(stem+'_'+arm, endpoint)
    if arm == 'original':
        reference = read('outputs/sho/stabilized_residual_n3_from_n2/stabilized_result.json') if stem == 'n00005_i00004' else read(Path('outputs/sho/coordinated_generic_recovery_audit')/(stem+'_result.json'))
        checks = dict(loss=bits_equal(answer['loss'], reference['loss']), phi=bits_equal(answer['phi'], reference['phi']),
                      accepted_steps=answer['accepted_steps']==reference['accepted_steps'],
                      stop_reason=answer['stop_reason']==reference['stop_reason'])
        atomic(OUT/(stem+'_original_screen_replay.json'), checks)
        if not all(checks.values()):
            raise RuntimeError('Original N3 screen differs from validated saved screen')
    print('N3 COMPLETE', stem, arm, answer['accepted_steps'], answer['loss'], answer['stop_reason'], flush=True)


def verify(label):
    before = read(OUT/'integrity_before.json')
    after = integrity()
    delta = differences(before, after)
    atomic(OUT/('integrity_'+label+'.json'), after)
    atomic(OUT/('integrity_check_'+label+'.json'), dict(unchanged=not any(delta.values()), differences=delta))
    if any(delta.values()):
        raise RuntimeError('Integrity changed')
    print('INTEGRITY VERIFIED', label, len(after['checkpoints']), 'checkpoints', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['init','n2','n3','verify'])
    parser.add_argument('--branch', choices=BRANCHES)
    parser.add_argument('--arm', choices=['original','extended'])
    parser.add_argument('--label', default='after')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    try:
        if args.phase == 'init': init()
        elif args.phase == 'n2': run_n2(args.branch)
        elif args.phase == 'n3': run_n3(args.branch, args.arm)
        else: verify(args.label)
    except Exception as exc:
        if OUT.exists():
            atomic(OUT/('failure_'+args.phase+'_'+str(args.branch)+'_'+str(args.arm)+'.json'),
                   dict(error=str(exc), traceback=traceback.format_exc(), arguments=vars(args)))
        raise


if __name__ == '__main__':
    main()
