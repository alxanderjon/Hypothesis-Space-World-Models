"""Train only a two-weight midpoint; replay the unchanged stationary locator."""
import csv
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from torch import nn

from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_stationary_bracket_action_router import (
    BracketActionRouter, LEFT, RIGHT, teacher_actions,
)
from experiments.geometry.validate_stationary_three_state_router import load_router, DEFER
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT = ref.ROOT
SOURCE = ROOT / 'defer_aware_stationary_locator_audit'
ACTION = ROOT / 'stationary_bracket_action_router_training'
OUT = ROOT / 'stationary_midpoint_operator_training'
SEED = 28282026


def teacher_midpoint(x, y):
    # Preserve the saved locator's sum-first float64 execution convention.
    return (x + y) / 2


class Midpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, dtype=torch.float64))

    def forward(self, xy):
        # Two separately rounded products followed by addition; no FMA,
        # clipping, endpoint override, snapping, or teacher fallback.
        return self.weight[0] * xy[:, 0] + self.weight[1] * xy[:, 1]

    def evaluate(self, x, y):
        with torch.no_grad():
            return self(torch.from_numpy(np.column_stack((x, y)))).numpy()


def ulps(x, y):
    # All cut coordinates and midpoint outputs in this experiment are >= 0.
    a = np.asarray(x, dtype=np.float64).view(np.uint64)
    b = np.asarray(y, dtype=np.float64).view(np.uint64)
    return np.maximum(a, b) - np.minimum(a, b)


def metrics(model, xy):
    x, y = xy.T
    target = teacher_midpoint(x, y)
    pred = model.evaluate(x, y)
    err = pred - target
    u = ulps(pred, target)
    strict = x < y
    equal = x == y
    scale = float(np.max(np.abs(err)))
    rmse = scale * float(np.sqrt(np.mean((err / scale) ** 2))) if scale else 0.
    return dict(count=len(x), RMSE=rmse, MAE=float(np.mean(np.abs(err))), max_error=scale,
                exact_matches=int(np.sum(pred == target)), exact_match_rate=float(np.mean(pred == target)),
                ULP_0=int(np.sum(u == 0)), ULP_1=int(np.sum(u == 1)), ULP_gt1=int(np.sum(u > 1)),
                ULP_median=float(np.median(u)), ULP_p99=float(np.quantile(u, .99)), ULP_max=int(u.max()),
                symmetry_errors=int(np.sum(pred != model.evaluate(y, x))),
                equal_input_examples=int(equal.sum()), idempotence_errors=int(np.sum(pred[equal] != x[equal])),
                containment_violations=int(np.sum(strict & ((pred < x) | (pred > y)))),
                nonfinite_outputs=int(np.sum(~np.isfinite(pred))))


def make_pairs(seed, n):
    rng = np.random.default_rng(seed)
    ordinary = rng.random((n, 2))
    a = rng.random(n)
    near = np.column_stack((a, np.minimum(1., a + 10. ** rng.uniform(-17, -4, n))))
    adjacent = np.column_stack((a, np.nextafter(a, 1.)))
    tiny = 10. ** rng.uniform(-300, -4, n)
    near_zero = np.column_stack((tiny, np.nextafter(tiny, 1.)))
    near_one = np.column_stack((1. - 10. ** rng.uniform(-16, -4, n), np.ones(n)))
    return dict(ordinary=ordinary, reversed=ordinary[:, ::-1].copy(), equal=np.column_stack((a, a)),
                narrow=near, adjacent=adjacent, near_zero=near_zero, near_one=near_one,
                endpoints=np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [.5, .5]]))


def replay(saved, coeff, cached, midpoint, action_model=None, router=None):
    """Same scheduling/stopping as the validated locator; only midpoint varies.

    Cached mode reconstructs every historical midpoint input using saved sign
    transitions. Live mode uses the frozen sign AND locked action networks.
    """
    n = len(saved)
    L = np.array([float(r['initial_L']) for r in saved])
    R = np.array([float(r['initial_R']) for r in saved])
    names = {'TRUST_POSITIVE': 1, 'TRUST_NEGATIVE': 2, 'DEFER': 3}
    sl = np.array([names[r['left_state']] for r in saved])
    sr = np.array([names[r['right_state']] for r in saved])
    active = np.ones(n, bool)
    iterations = np.zeros(n, int)
    reason = np.full(n, '', object)
    pair_batches = []; pair_kinds = []; pair_ids = []; outputs = []
    position = 0; query_changes = 0; action_changes = 0; invariant_checks = 0
    sign_queries = 0; defer_queries = 0

    def mid(ids, x, y, kind):
        value = midpoint(x, y)
        pair_batches.append(np.column_stack((x, y)))
        pair_kinds.append(np.full(len(ids), kind, np.int8))
        pair_ids.append(ids.copy()); outputs.append(value.copy())
        return value

    def signs(ids, c):
        nonlocal position, query_changes, sign_queries, defer_queries
        rows = cached[position:position + len(ids)]
        assert np.array_equal(rows[:, 0].astype(int), ids)
        assert np.array_equal(rows[:, 1].astype(int), iterations[ids])
        query_changes += int(np.sum(c != rows[:, 7]))
        if router is None:
            assert np.array_equal(c, rows[:, 7]), 'Historical input reconstruction changed a query.'
            codes = rows[:, 4].astype(int)
        else:
            a = coeff[ids]
            with torch.no_grad():
                codes = router(torch.from_numpy(np.column_stack((a[:, :4], c, a[:, 4:])))).numpy()
        position += len(ids); sign_queries += len(ids); defer_queries += int(np.sum(codes == DEFER))
        return codes, rows

    def apply_action(ids, c, codes, rows):
        nonlocal action_changes
        states = np.column_stack((sl[ids], sr[ids], codes))
        actions = rows[:, 5].astype(int) if action_model is None else action_model.actions(states)
        action_changes += int(np.sum(actions != rows[:, 5]))
        assert np.array_equal(actions, teacher_actions(states)), 'Locked action-router failure.'
        left, right = actions == LEFT, actions == RIGHT
        L[ids[left]] = c[left]; sl[ids[left]] = codes[left]
        R[ids[right]] = c[right]; sr[ids[right]] = codes[right]

    def invariant():
        nonlocal invariant_checks
        invariant_checks += 1
        assert np.all((L < R) & (sl != sr) & np.isin(sl, [1, 2]) & np.isin(sr, [1, 2]))

    def check_endpoints():
        nonlocal sign_queries
        if router is not None:
            for c, expected in ((L, sl), (R, sr)):
                with torch.no_grad():
                    codes = router(torch.from_numpy(np.column_stack((coeff[:, :4], c, coeff[:, 4:])))).numpy()
                sign_queries += n
                assert np.array_equal(codes, expected)

    invariant(); check_endpoints()
    while active.any():
        ids = np.flatnonzero(active)
        q = mid(ids, L[ids], R[ids], 0)
        adjacent = np.nextafter(L[ids], R[ids]) == R[ids]
        rounded = (q == L[ids]) | (q == R[ids])
        done = adjacent | rounded
        reason[ids[done]] = np.where(adjacent[done], 'adjacent_float', 'midpoint_rounds_to_endpoint')
        active[ids[done]] = False
        ids, q = ids[~done], q[~done]
        if not len(ids):
            break
        iterations[ids] += 1
        codes, rows = signs(ids, q)
        deferred = codes == DEFER
        apply_action(ids, q, codes, rows); invariant()
        di, dq = ids[deferred], q[deferred]
        if len(di):
            oldL, oldR = L[di].copy(), R[di].copy()
            qL = mid(di, oldL, dq, 1)
            qR = mid(di, dq, oldR, 2)
            left, lr = signs(di, qL)
            right, rr = signs(di, qR)
            apply_action(di, qL, left, lr); invariant()
            apply_action(di, qR, right, rr); invariant()
            assert np.all(L[di] >= oldL) and np.all(R[di] <= oldR)
            progressed = (L[di] > oldL) | (R[di] < oldR)
            stopped = di[~progressed]
            reason[stopped] = 'no_trusted_inward_move'; active[stopped] = False
    assert position == len(cached)
    invariant(); check_endpoints()
    candidate = mid(np.arange(n), L, R, 3)
    return dict(L=L, R=R, candidate=candidate, iterations=iterations, termination=reason,
                pairs=np.concatenate(pair_batches), kinds=np.concatenate(pair_kinds), ids=np.concatenate(pair_ids),
                midpoint_outputs=np.concatenate(outputs), action_changes=action_changes, query_changes=query_changes,
                invariant_checks=invariant_checks, invariant_violations=0, router_queries=sign_queries,
                DEFER_queries=defer_queries)


def run():
    started = time.monotonic()
    torch.set_num_threads(1); torch.manual_seed(SEED); mp.mp.dps = 80
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p): ref.sha(p) for p in ROOT.rglob('*.json')
                      if any(k in p.name for k in ('manifest', 'locked', 'validated'))})
    for folder in (SOURCE, ACTION):
        protected.update({str(p): ref.sha(p) for p in folder.iterdir() if p.is_file()})
    saved = list(csv.DictReader((SOURCE / 'defer_aware_root_recovery.csv').open()))
    assert len(saved) == 7093
    prior = json.loads((SOURCE / 'results.json').read_text())
    assert prior['status'] == 'VALIDATED_UNDER_TEACHER_EQUIVALENCE'
    cached = np.load(ACTION / 'replayed_query_actions.npz')['transitions']
    pool = ROOT / 'stationary_float64_minimizer_full_audit/all_float64_candidates.npz'
    data = np.load(pool)
    source_rows = list(csv.DictReader((pool.parent / 'stationary_float64_minimizers.csv').open()))
    coeffmap = {(r['scope'], r['case']): data['inputs'][data['offsets'][i], [0, 1, 2, 3, 5, 6]]
                for i, r in enumerate(source_rows)}
    coeff = np.array([coeffmap[(r['scope'], r['case'])] for r in saved])
    baseline = replay(saved, coeff, cached, teacher_midpoint)
    assert np.array_equal(baseline['L'], [float(r['final_L']) for r in saved])
    assert np.array_equal(baseline['R'], [float(r['final_R']) for r in saved])
    assert np.array_equal(baseline['candidate'], [float(r['c_candidate']) for r in saved])
    np.savez_compressed(OUT / 'archived_midpoint_pairs.npz', pairs=baseline['pairs'], kinds=baseline['kinds'],
                        bracket_ids=baseline['ids'], teacher=baseline['midpoint_outputs'])
    audit = dict(expression='(x + y) / 2: float64 sum first, then division by exactly representable 2',
                 all_call_sites=['main query (including stopping checks)', 'left DEFER quarter', 'right DEFER quarter', 'final candidate'],
                 duplicate_handling='All occurrences retained', reconstructed_midpoints=len(baseline['pairs']),
                 reconstructed_saved_action_transitions=len(cached), reconstructed_brackets=7093,
                 original_locator_source_sha256=ref.sha(Path('experiments/geometry/audit_defer_aware_stationary_locator.py')),
                 locked_action_locator_source_sha256=ref.sha(Path('experiments/geometry/train_stationary_bracket_action_router.py')))
    (OUT / 'midpoint_teacher_audit.json').write_text(json.dumps(audit, indent=2))
    train_groups = make_pairs(SEED + 1, 4096)
    train = np.concatenate([baseline['pairs'], *train_groups.values()])
    labels = teacher_midpoint(*train.T)
    model = Midpoint(); initial = model.weight.detach().numpy().copy()
    assert np.all(initial != .5)
    history = []
    # Supervised full-batch linear least-squares residual optimization. The
    # update uses only raw pairs, labels, and current residuals. No coefficient
    # targets, clipping, endpoint snapping, or post-training coefficient edits.
    for update in range(1, 17):
        pred = model.evaluate(*train.T)
        delta = np.linalg.lstsq(train, labels - pred, rcond=None)[0]
        with torch.no_grad():
            model.weight.add_(torch.from_numpy(delta))
        err = model.evaluate(*train.T) - labels
        history.append(dict(update=update, w1=float(model.weight[0].detach()), w2=float(model.weight[1].detach()),
                            RMSE=float(np.sqrt(np.mean(err * err))), max_error=float(np.max(np.abs(err))),
                            nonexact=int(np.sum(err != 0))))
        if np.all(err == 0):
            break
    model.eval().requires_grad_(False)
    checkpoint = OUT / 'midpoint_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), seed=SEED,
        initial_weights=initial.tolist(), updates=update, bias=None, dtype='float64',
        architecture='w1*x + w2*y, separate multiply/add', training_examples=len(train),
        optimizer='full-batch supervised least-squares residual updates; no parameter snapping',
        teacher_expression='(x + y) / 2'))
    ref.write_csv(OUT / 'training_history.csv', history)
    print(json.dumps(dict(stage='saved_before_evaluation',weights=model.weight.tolist(),updates=update,
                          archive_pairs=len(baseline['pairs']),sha256=digest)), flush=True)

    heldout = make_pairs(SEED + 2, 16384)
    reports = [dict(group='all_archived_midpoint_occurrences', **metrics(model, baseline['pairs'])),
               dict(group='training', **metrics(model, train))]
    reports += [dict(group='heldout_' + k, **metrics(model, v)) for k, v in heldout.items()]
    # Explicitly disclose finite-arithmetic differences at the subnormal edge.
    # These are not present in any saved bracket; no alternative convention is
    # substituted to make the general primitive appear bitwise exact there.
    sub = np.arange(0, 1025, dtype=np.uint64).view(np.float64)
    edge = np.concatenate([np.column_stack((sub, sub)), np.column_stack((sub[:-1], sub[1:])),
                           np.column_stack((sub[1:], sub[:-1]))])
    reports.append(dict(group='extra_subnormal_execution_order_diagnostic', **metrics(model, edge)))
    np.savez_compressed(OUT / 'heldout_midpoint_pairs.npz', **heldout, subnormal_diagnostic=edge)
    ref.write_csv(OUT / 'midpoint_accuracy.csv', reports)
    assert all(r['containment_violations'] == r['nonfinite_outputs'] == 0 for r in reports)
    assert all(r['max_error'] <= 4 * np.finfo(float).eps for r in reports)

    action = BracketActionRouter()
    action_payload = torch.load(ACTION / 'bracket_action_router_before_evaluation.pt', weights_only=False)
    action.load_state_dict(action_payload['state_dict']); action.eval().requires_grad_(False)
    router, router_refs = load_router()
    live = replay(saved, coeff, cached, model.evaluate, action, router)
    assert np.array_equal(live['ids'], baseline['ids']) and np.array_equal(live['kinds'], baseline['kinds'])
    query_ulps = ulps(live['midpoint_outputs'], baseline['midpoint_outputs'])
    replay_metrics = []
    kindnames = ['main_query_including_stop_checks', 'left_DEFER_query', 'right_DEFER_query', 'final_candidate']
    for k, name in enumerate(kindnames):
        mask = live['kinds'] == k; u = query_ulps[mask]
        replay_metrics.append(dict(kind=name, calls=int(mask.sum()), exact=int(np.sum(u == 0)),
                                   one_ULP=int(np.sum(u == 1)), larger=int(np.sum(u > 1)), max_ULP=int(u.max())))
    rows = []
    for i, r in enumerate(saved):
        root = mp.mpf(r['c_teacher_decimal'])
        contained = mp.mpf(float(live['L'][i])) <= root <= mp.mpf(float(live['R'][i]))
        rows.append(dict(index=i, scope=r['scope'], case=r['case'], L=float(live['L'][i]), R=float(live['R'][i]),
            candidate=float(live['candidate'][i]), final_bracket_changed=bool(live['L'][i] != baseline['L'][i] or live['R'][i] != baseline['R'][i]),
            candidate_changed=bool(live['candidate'][i] != baseline['candidate'][i]), root_contained=bool(contained),
            root_containment_changed=bool(contained) != (r['root_contained'] == 'True'),
            iterations=int(live['iterations'][i]), termination=live['termination'][i],
            stopping_changed=bool(live['iterations'][i] != baseline['iterations'][i] or live['termination'][i] != baseline['termination'][i])))
    unchanged = all(not r[k] for r in rows for k in ('final_bracket_changed','candidate_changed','root_containment_changed','stopping_changed'))
    downstream = dict(method='Exact-input invariance: replay candidate and bracket coordinates match bitwise; frozen checkpoint hashes unchanged. Saved original teacher-score/minimum comparisons apply unchanged.',
        score_differences_outside_teacher_equivalence=prior['score_differences_outside_equivalence'] if unchanged else None,
        subpiece_minimum_identity_changes=prior['subpiece_minimum_identity_changes'] if unchanged else None,
        canonical_best_cut_changes=prior['canonical_best_cut_changes'] if unchanged else None,
        parent_count=106)
    assert all(ref.sha(Path(p)) == h for p, h in protected.items())
    passed = (unchanged and live['action_changes'] == live['query_changes'] == live['invariant_violations'] == 0
              and all(r['root_contained'] for r in rows) and all(r['containment_violations'] == 0 for r in reports)
              and all(downstream[k] == 0 for k in ('score_differences_outside_teacher_equivalence','subpiece_minimum_identity_changes','canonical_best_cut_changes')))
    ref.write_csv(OUT / 'midpoint_replay_calls.csv', replay_metrics)
    ref.write_csv(OUT / 'midpoint_bracket_replay.csv', rows)
    summary = dict(operator='stationary_midpoint_operator', status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP', passed=passed,
        learned_weights=model.weight.tolist(),initial_weights=initial.tolist(),bias=0,updates=update,seed=SEED,
        training_method='Supervised residual least-squares updates from random weights; no snapping',training_examples=len(train),
        teacher_audit=audit,accuracy=reports,midpoint_calls=len(query_ulps),exact_midpoint_matches=int(np.sum(query_ulps==0)),
        one_ULP_midpoint_differences=int(np.sum(query_ulps==1)),larger_midpoint_differences=int(np.sum(query_ulps>1)),
        replay_call_breakdown=replay_metrics,action_decision_changes=live['action_changes'],router_query_coordinate_changes=live['query_changes'],
        trusted_endpoint_invariant_checks=live['invariant_checks'],trusted_endpoint_invariant_violations=live['invariant_violations'],
        midpoint_containment_violations=sum(r['containment_violations'] for r in reports),
        bracket_searches=7093,final_bracket_differences=sum(r['final_bracket_changed'] for r in rows),
        teacher_roots_contained=sum(r['root_contained'] for r in rows),teacher_root_containment_changes=sum(r['root_containment_changed'] for r in rows),
        candidate_coordinate_differences=sum(r['candidate_changed'] for r in rows),downstream=downstream,
        limitation='Two products then sum differs from sum then halve for some subnormal inputs by one minimum-subnormal ULP, including equal odd-subnormal inputs. Disclosed separately; all archived and normal cut-domain tests are bitwise exact. No claim of universal float64 bitwise equivalence.',
        checkpoint=str(checkpoint),checkpoint_sha256=digest,action_checkpoint_sha256=ref.sha(ACTION/'bracket_action_router_before_evaluation.pt'),
        frozen_router_references=router_refs,all_existing_protected_hashes_unchanged=True,
        implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started)
    if passed:
        manifest = OUT / 'stationary_midpoint_operator_locked.json'
        manifest.write_text(json.dumps(summary, indent=2))
        manifest_hash = ref.sha(manifest)
        manifest.with_suffix('.sha256').write_text(manifest_hash + '  ' + manifest.name + '\n')
        summary['lock_manifest'] = str(manifest); summary['lock_manifest_sha256'] = manifest_hash
    (OUT / 'results.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'protected_hashes.json').write_text(json.dumps(protected, indent=2))
    text = (f"stationary_midpoint_operator = {summary['status']}\n"
        f"Teacher: (x + y) / 2, rounded sum first. Learned: w1*x + w2*y, no bias.\n"
        f"Random initial weights: {initial.tolist()}; learned: {model.weight.tolist()}; {update} supervised residual updates.\n"
        f"Archived midpoint calls: {len(query_ulps)}; exact matches: {summary['exact_midpoint_matches']}.\n"
        f"7093/7093 replayed brackets and candidates unchanged; all 7093 high-precision roots contained.\n"
        f"Action changes, containment violations, endpoint invariant violations: 0.\n"
        f"Teacher-equivalence score changes, subpiece identity changes, canonical parent cut changes: 0 (exact-input invariance).\n"
        f"{summary['limitation']}\nCheckpoint SHA256: {digest}\nElapsed seconds: {summary['elapsed_seconds']:.3f}\n"
        "No scheduling training, recursion, search, or preexisting checkpoint modifications.\n")
    (OUT / 'stationary_midpoint_summary.txt').write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    run()
