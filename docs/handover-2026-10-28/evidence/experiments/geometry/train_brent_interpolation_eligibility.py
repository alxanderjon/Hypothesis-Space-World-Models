"""Train only two comparison margins and a tiny Brent eligibility routing head."""
import csv
import inspect
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq

from experiments.geometry import audit_exact_branch_brent as audit
from experiments.geometry import train_branch_location_residual as fresh_source
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_branch_switch_location import json_write
from experiments.sho.smoke_artifacts import save_checkpoint

OUT = ROOT / 'brent_interpolation_eligibility_training'
SOURCE = ROOT / 'exact_branch_switch_brent_audit'
SEED = 928202617


class BrentEligibility(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.margin_weights = nn.Parameter(.3*torch.randn(2, 2, dtype=torch.float64))
        self.head = nn.Linear(2, 2, dtype=torch.float64)
        self.threshold = threshold.eval().requires_grad_(False)

    def comparisons(self, states):
        # Raw (spre, delta, fcur, fpre). Only absolute value is deterministic.
        x = torch.stack((states[:, 0].abs(), states[:, 1],
                         states[:, 2].abs(), states[:, 3].abs()), 1).reshape(-1, 2, 2)
        margins = self.margin_weights[:, 0]*x[:, :, 0] + self.margin_weights[:, 1]*x[:, :, 1]
        return self.threshold.decision(margins) > 0

    def forward(self, states):
        return self.head(self.comparisons(states).double())

    def predict(self, states):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(states, dtype=np.float64).reshape(-1, 4))
            return self.comparisons(x).numpy(), self(x).argmax(1).numpy()


def truth(x):
    comparison = np.column_stack((abs(x[:, 0]) > x[:, 1], abs(x[:, 2]) < abs(x[:, 3])))
    return comparison, comparison.all(1).astype(np.int64)


def stats(model, x, group):
    t, y = truth(x); c, p = model.predict(x)
    return dict(group=group, cases=len(x), comparison_1_errors=int((t[:, 0] != c[:, 0]).sum()),
                comparison_2_errors=int((t[:, 1] != c[:, 1]).sum()), eligibility_errors=int((p != y).sum()),
                false_INTERPOLATE=int(((p == 1) & (y == 0)).sum()),
                false_BISECT=int(((p == 0) & (y == 1)).sum()))


def boundary_data(seed, count):
    rng = np.random.default_rng(seed)
    bases = np.r_[np.ldexp(rng.uniform(1, 2, count), rng.integers(-1000, 900, count)),
                  0., np.nextafter(0., 1.), np.finfo(float).tiny, .5, 1., 2.]
    out=[]; tags=[]
    for i, b in enumerate(bases):
        f = bases[(i*17+3) % len(bases)]
        around_b=[max(0., np.nextafter(b, -np.inf)), b, np.nextafter(b, np.inf)]
        around_f=[max(0., np.nextafter(f, -np.inf)), f, np.nextafter(f, np.inf)]
        for j, k in itertools.product(range(3), repeat=2):
            out.append([(-1. if i % 2 else 1.)*around_b[j], b,
                        (-1. if i % 3 else 1.)*around_f[k], (-1. if i % 5 else 1.)*f])
            tags.append((j, k))
    return np.array(out), np.array(tags)


def extract(trace):
    return np.array([[r['normalized_state']['spre'], r['delta'],
                      r['normalized_state']['fcur'], r['normalized_state']['fpre']] for r in trace])


def run():
    start=time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED); mp.mp.dps=80
    OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json')
                      if any(s in p.name for s in ('manifest', 'locked', 'validated'))})
    protected.update({str(p):sha(p) for p in SOURCE.iterdir() if p.is_file()})
    protected[str(Path(audit.__file__))]=sha(Path(audit.__file__))
    manifest=json.loads((ROOT/'stationary_magnitude_threshold_smoke/zero_anchor_converged/candidate_manifest.json').read_text())
    cp=Path(manifest['checkpoint']); assert sha(cp)==manifest['sha256']
    threshold=MagnitudeThreshold(); threshold.load_state_dict(torch.load(cp, weights_only=False)['state_dict'])
    model=BrentEligibility(threshold)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    cases=json.loads((SOURCE/'solve_inputs.json').read_text())
    traces=[]; ids=[]; active=[]; targets=[]
    with (SOURCE/'brent_iterations.csv').open() as f:
        for r in csv.DictReader(f):
            s=json.loads(r['normalized_state'])
            traces.append([s['spre'],float(r['delta']),s['fcur'],s['fpre']])
            ids.append(int(r['case_id']));active.append(r['stop']=='False')
            targets.append(dict(query=float(r['query']) if r['query'] else None,
                                function=float(r['new_function']) if r['new_function'] else None,
                                stop=r['stop']=='True', eligible=r['eligible']=='True'))
    x=np.array(traces);ids=np.array(ids);active=np.array(active)
    assert len(x)==27867 and active.sum()==27049
    # Parent grouping is stricter than whole-solve grouping. All 44 independent
    # fresh solves and every duplicate of their parents are excluded from fit.
    keys=[json.dumps(r['parent'],sort_keys=True) for r in cases]
    freshkeys=set(keys[774:]);holdkeys={k for k in keys[:774] if int(__import__('hashlib').sha256(k.encode()).hexdigest()[:8],16)%5==0}
    held=np.array([k in holdkeys or k in freshkeys for k in keys])
    fit=(~held[ids]) & (ids<774);heldarchive=held[ids] & (ids<774);fresh=ids>=774
    assert not {keys[i] for i in set(ids[fit])} & {keys[i] for i in set(ids[~fit])}
    splitrows=[dict(case_id=i,group='fresh' if i>=774 else 'heldout_archive' if held[i] else 'fit',
                    label=r['label'],parent_key=keys[i]) for i,r in enumerate(cases)]
    write_csv(OUT/'solve_splits.csv',splitrows)
    train_boundary,_=boundary_data(SEED+1,96)
    validation_boundary,tags=boundary_data(SEED+2,256)
    # Synthetic ordinary independent states cover all four truth-table states.
    rng=np.random.default_rng(SEED+3)
    ordinary=np.exp2(rng.uniform(-40,40,(4096,4))); ordinary[:,[0,2,3]]*=rng.choice([-1.,1.],(4096,3))
    train=np.concatenate((x[fit],train_boundary,ordinary))
    # Relative continuous margin regression; normalization is training-only.
    # Inference consumes raw magnitudes, avoiding a rounded normalized feature.
    design=abs(train).reshape(-1,2,2)
    history=[]
    for j in range(2):
        d=design[:,j];scale=d.max(1);d=np.divide(d,scale[:,None],out=np.zeros_like(d),where=scale[:,None]>0)
        y=d[:,0]-d[:,1] if j==0 else d[:,1]-d[:,0]
        for step in range(9):
            w=model.margin_weights[j].detach().numpy().copy()
            p=w[0]*d[:,0]+w[1]*d[:,1];res=y-p
            history.append(dict(component=f'comparison_{j+1}',step=step,weights=w.tolist(),max_error=float(abs(res).max())))
            if np.all(res==0):break
            correction=np.linalg.lstsq(d,res,rcond=None)[0]
            proposal=w+correction
            if np.array_equal(w,proposal):break
            with torch.no_grad():model.margin_weights[j].copy_(torch.from_numpy(proposal))
    table=torch.tensor(list(itertools.product((0.,1.),repeat=2)),dtype=torch.float64)
    table_y=torch.tensor([0,0,0,1],dtype=torch.long)
    with torch.no_grad():train_comparisons=model.comparisons(torch.from_numpy(train)).double()
    train_y=torch.from_numpy(truth(train)[1])
    opt=torch.optim.Adam(model.head.parameters(),lr=.05)
    # Head sees the learned comparisons; complete table retains absent states.
    for step in range(2000):
        opt.zero_grad();logits=model.head(train_comparisons)
        loss=nn.functional.cross_entropy(logits,train_y)+nn.functional.cross_entropy(model.head(table),table_y)
        loss.backward();opt.step()
        with torch.no_grad():
            tl=model.head(table);correct=torch.all(tl.argmax(1)==table_y)
            confidence=(tl.gather(1,table_y[:,None]).flatten()-tl.gather(1,(1-table_y)[:,None]).flatten()).min()
        if correct and confidence>4:break
    history.append(dict(component='routing_head',step=step+1,loss=float(loss.detach()),minimum_table_logit_margin=float(confidence)))
    config=dict(seed=SEED,architecture='Two independent bias-free float64 2-weight margins; frozen zero threshold; Linear(2,2) routing head',
                trainable_parameters=10,inputs=['spre','delta','fcur','fpre'],normalization='Training-only per-pair max normalization; inference uses raw absolute magnitudes',
                regression='Supervised least-squares residual corrections from random initialization; no snapping',
                routing='Cross entropy from random initialization; argmax action decoding; no hard-coded AND at inference',
                frozen_threshold=manifest,all_trace_rows=len(x),actual_eligibility_rows=int(active.sum()),terminal_counterfactual_rows=int((~active).sum()),
                fit_trace_rows=int(fit.sum()),heldout_trace_rows=int(heldarchive.sum()),fresh_trace_rows=int(fresh.sum()),
                fit_solves=int(sum(r['group']=='fit' for r in splitrows)),heldout_solves=int(sum(r['group']=='heldout_archive' for r in splitrows)),fresh_solves=44,
                split='Whole-parent groups; therefore complete solves held out. Fresh parents excluded. No validation-based fitting.',
                terminal_handling='Report formula predictions on terminal states; never invoke eligibility after stop in replay',
                no_other_operator_training=True,no_checkpoint_overwrite=True)
    checkpoint=OUT/'eligibility_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),initial_state=initial,config=config,training_history=history))
    json_write(OUT/'config.json',config);json_write(OUT/'training_history.json',history)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    print(json.dumps(dict(event='trained_and_saved',weights=model.margin_weights.tolist(),checkpoint_sha256=digest,fit_solves=config['fit_solves'],heldout_solves=config['heldout_solves'])),flush=True)
    metrics=[stats(model,x,'all_27867_states'),stats(model,x[active],'actual_27049_decisions'),
             stats(model,x[~active],'terminal_counterfactuals'),stats(model,x[fit],'fit_solves'),
             stats(model,x[heldarchive],'heldout_archive_solves'),stats(model,x[fresh],'saved_fresh_44_solves'),
             stats(model,validation_boundary,'heldout_boundary_nextafter')]
    for j in range(3):
        metrics.append(stats(model,validation_boundary[tags[:,0]==j],f'comparison1_boundary_position_{j}'))
        metrics.append(stats(model,validation_boundary[tags[:,1]==j],f'comparison2_boundary_position_{j}'))
    tt=model.head(table).argmax(1).numpy();table_errors=int((tt!=table_y.numpy()).sum())
    write_csv(OUT/'truth_table.csv',[dict(comparison_1=int(t[0]),comparison_2=int(t[1]),teacher=int(y),learned=int(p)) for t,y,p in zip(table.numpy(),table_y,tt)])
    c,p=model.predict(x);t,y=truth(x)
    write_csv(OUT/'trace_decisions.csv',[dict(case_id=int(ids[i]),row=i,executed=bool(active[i]),
                 comparison1_teacher=bool(t[i,0]),comparison1_learned=bool(c[i,0]),comparison2_teacher=bool(t[i,1]),comparison2_learned=bool(c[i,1]),
                 eligibility_teacher=int(y[i]),eligibility_learned=int(p[i])) for i in range(len(x))])
    conflicts=audit.read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv')
    for g in range(21):
        groupids={int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==g}
        metrics.append(stats(model,x[np.isin(ids,list(groupids))],f'conflicting_group_{g}'))
    metrics.append(stats(model,x[ids==765],'recursive_fixture_no_recursion'))
    write_csv(OUT/'validation.csv',metrics)
    if table_errors or any(any(r[k] for k in ('comparison_1_errors','comparison_2_errors','eligibility_errors')) for r in metrics):
        json_write(OUT/'results.json',dict(status='DIAGNOSTIC_ONLY_STOP',metrics=metrics,truth_table_errors=table_errors))
        return

    # Create the audited interpreter with exactly ONE source substitution. The
    # authoritative original file and every other arithmetic/control line stay intact.
    original=inspect.getsource(audit.traced_brent)
    old='eligible=previous_step_large and residual_decreasing'
    new='eligible=eligibility_hook(sp,delta,fc,fp)'
    assert original.count(old)==1
    substituted=original.replace(old,new)
    (OUT/'brent_with_learned_eligibility.py').write_text(substituted)
    actions=[]
    def eligibility_hook(sp,delta,fc,fp):
        cs,ps=model.predict([[sp,delta,fc,fp]])
        actions.append((sp,delta,fc,fp,int(ps[0])))
        return bool(ps[0])
    env=dict(audit.__dict__,eligibility_hook=eligibility_hook)
    exec(compile(substituted,'brent_with_learned_eligibility.py','exec'),env)
    learned_brent=env['traced_brent'];a=audit.Audit.__new__(audit.Audit);replay=[]

    def replay_case(case,case_id,group,saved=None):
        a.setup(case['parent'],case['axis']);count=case['count'];calls=[]
        def evaluate(c):
            f=a.formula(float(c),count);ia=int(f[4][0])
            return float(f[1][0]),dict(q_active=int(f[3][0]),aspect_active=ia,aspect_power=int(f[5][0,ia]),
                       competing_q=f[6][0].tolist(),competing_log_aspect=f[7][0].tolist())
        teacher,tr,tc,reason=audit.traced_brent(evaluate,case['left'],case['right'],case_id)
        def installed(c):
            v,_=evaluate(c);calls.append((float(c),v));return v
        exact,info=brentq(installed,case['left'],case['right'],xtol=audit.XTOL,rtol=audit.RTOL,maxiter=audit.MAXITER,full_output=True)
        assert teacher==exact and (saved is None or teacher==saved)
        before=len(actions);got,lr,lc,lreason=learned_brent(evaluate,case['left'],case['right'],case_id)
        tq=[r for r in tr if not r['stop']];lq=[r for r in lr if not r['stop']]
        exactq=sum(u['query'].hex()==v['query'].hex() for u,v in zip(tq,lq))
        exactf=sum(u['new_function'].hex()==v['new_function'].hex() for u,v in zip(tq,lq))
        first=next((i for i,(u,v) in enumerate(zip(tq,lq)) if u['query'].hex()!=v['query'].hex()),None)
        if first is None and len(tq)!=len(lq):first=min(len(tq),len(lq))
        dm=sum(pred!=bool(abs(sp)>de and abs(fc)<abs(fp)) for sp,de,fc,fp,pred in actions[before:])
        assert len(calls)==tc==info.function_calls
        assert all(r['query'].hex()==calls[i+2][0].hex() and r['new_function'].hex()==calls[i+2][1].hex() for i,r in enumerate(tq))
        if case_id<818:
            oldrows=[targets[i] for i in np.flatnonzero(ids==case_id)]
            assert len(oldrows)==len(tr)
            assert all(z['query']==r['query'] and z['stop']==r['stop'] for z,r in zip(oldrows,tr))
        replay.append(dict(case_id=case_id,group=group,label=case['label'],iterations=len(lr),queries=len(lq),exact_queries=exactq,
                  exact_function_values=exactf,decision_mismatches=dm,first_query_divergence=first,
                  exact_return=got.hex()==teacher.hex(),exact_all_states=tr==lr,teacher_return=teacher,learned_return=got))
        return tr
    for i,case in enumerate(cases):
        replay_case(case,i,splitrows[i]['group'],case['teacher_cut'])
    print(json.dumps(dict(event='saved_818_replayed',exact=sum(r['exact_all_states'] for r in replay))),flush=True)
    # Newly generated independent positives supplement the 44 saved fresh solves.
    oldseed=fresh_source.SEED
    try:
        fresh_source.SEED=SEED+19
        newcases,parents,freshaudit=fresh_source.fresh_data()
    finally:fresh_source.SEED=oldseed
    json_write(OUT/'new_fresh_candidates.json',newcases);json_write(OUT/'new_fresh_parent_generation.json',dict(parents=parents,audit=freshaudit))
    assert newcases and not {json.dumps(p,sort_keys=True) for p in parents}&set(keys)
    newstates=[]
    for j,r in enumerate(newcases):
        case=dict(parent=r['parent'],axis=r['axis'],count=r['k'],left=r['c_left'],right=r['c_right'],label=r['case'])
        tr=replay_case(case,818+j,'new_fresh',r['teacher_cut']);newstates.append(extract(tr))
    metrics.append(stats(model,np.concatenate(newstates),'new_fresh_solves'))
    write_csv(OUT/'validation.csv',metrics);write_csv(OUT/'solve_replay.csv',replay)
    unchanged=all(sha(Path(p))==v for p,v in protected.items());assert unchanged
    passed=not table_errors and all(not r['comparison_1_errors'] and not r['comparison_2_errors'] and not r['eligibility_errors'] for r in metrics) and all(r['exact_all_states'] and not r['decision_mismatches'] for r in replay)
    result=dict(operator='brent_interpolation_eligibility',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
                checkpoint=str(checkpoint),sha256=digest,module='experiments.geometry.train_brent_interpolation_eligibility',model='BrentEligibility',implementation_sha256=sha(Path(__file__)),
                learned_margin_weights=model.margin_weights.tolist(),learned_head_weight=model.head.weight.tolist(),learned_head_bias=model.head.bias.tolist(),
                truth_table_errors=table_errors,metrics=metrics,archived_solves=774,saved_fresh_solves=44,new_fresh_solves=len(newcases),
                replay_solves=len(replay),exact_query_replay_count=sum(r['exact_queries'] for r in replay),
                total_replayed_queries=sum(r['queries'] for r in replay),exact_full_state_replays=sum(r['exact_all_states'] for r in replay),
                replay_decision_mismatches=sum(r['decision_mismatches'] for r in replay),first_query_divergence=next((r for r in replay if r['first_query_divergence'] is not None),None),
                frozen_threshold_checkpoint=str(cp),frozen_threshold_sha256=manifest['sha256'],
                protected_hashes_unchanged=unchanged,protected_files=len(protected),no_recursion=True,no_other_training=True,
                elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    if passed:json_write(OUT/'brent_interpolation_eligibility_locked.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
