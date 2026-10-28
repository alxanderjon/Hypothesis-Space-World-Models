"""Train only a categorical MOVE/KEEP head; validate isolated division assembly.

No Brent execution. Existing checkpoints and residual implementation are read-only.
"""
import csv
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry import train_division_operator as src
from experiments.geometry.compensated_division_residual import CompensatedDivisionResidual
from experiments.geometry.train_reciprocal_operator import PositivePower
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'division_rounding_router_training'
SEED = 928202652


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        rows = [dict(status='none')]
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def bits(a):
    return np.asarray(a, dtype=np.float64).view('u8')


class RoundingRouter(nn.Module):
    """Only two categorical inputs: comparison {-1,0,1}, odd {0,1}."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True, dtype=torch.float64)

    def forward(self, state):
        return self.linear(state).squeeze(-1)

    def decision(self, state):
        # Generic binary-logit decoding, not the teacher's Boolean rule.
        return self(state) > 0


class DivisionAssembly:
    """Audited one-neighbor correction; the high/low residual is kept intact.

Supported scope is finite inputs/nonzero denominator and finite approximate
quotient in the validated residual domain. No assertion of universal IEEE division.
"""
    def __init__(self, router):
        self.router = router.eval().requires_grad_(False)
        self.residual = CompensatedDivisionResidual()
        self.threshold, self.threshold_meta = src.load(
            MagnitudeThreshold, 'stationary_magnitude_threshold_smoke/zero_anchor_converged')
        rec, self.reciprocal_meta = src.load(PositivePower, 'reciprocal_required_domain_audit')
        self.diagnostic_meta = json.loads((ROOT/'division_operator_refinement/results.json').read_text())
        assert src.sha(Path(self.diagnostic_meta['checkpoint'])) == self.diagnostic_meta['sha256']
        self.approximate = src.Division(self.residual.multiply, rec)
        self.approximate.load_state_dict(torch.load(self.diagnostic_meta['checkpoint'], weights_only=False)['state_dict'])
        self.approximate.eval().requires_grad_(False)

    def comparison_states(self, xy, q):
        x, y = xy.T
        expansion = self.residual.construct(x, y, q)
        # The upstream interface remains the unchanged hi/lo expansion. This
        # comparator-local RN sum is exactly the previously audited comparison;
        # it never replaces the stored pair or crosses an upstream interface.
        r = self.residual.rounded_normalized(expansion)
        with torch.no_grad():
            direction = self.threshold.decision(torch.from_numpy(r)).numpy().astype(np.int8)
        Q = abs(q)
        neighbor = Q.copy()
        active = direction != 0
        neighbor[active] = np.nextafter(Q[active], np.where(direction[active] > 0, np.inf, -np.inf))
        used = active & np.isfinite(neighbor)
        indices = np.flatnonzero(used)
        my, ey = np.frexp(abs(y)); _, ex = np.frexp(abs(x)); e = ex-ey
        d = abs(self.residual._call(self.residual.subtract, neighbor[used], Q[used]))
        g = np.ldexp(d, -e[used])
        u = self.residual._call(self.residual.multiply, g, my[used])
        H = self.residual._call(self.residual.multiply, u, .5)
        margin = self.residual._call(self.residual.subtract, abs(r[used]), H)
        with torch.no_grad():
            comparison = self.threshold.decision(torch.from_numpy(margin)).numpy().astype(np.int8)
        odd = (bits(Q[used]) & np.uint64(1)).astype(np.int8)
        states = np.column_stack((comparison, odd)).astype(np.float64)
        return dict(expansion=expansion, direction=direction, neighbor=neighbor, indices=indices,
                    states=states, threshold=H, my=my, exponent=e,
                    zero_residual=int((~active).sum()), nonfinite_neighbor_guard=int((active & ~used).sum()))

    def predict(self, xy, q=None):
        if q is None:
            q = self.approximate.predict(xy)
        info = self.comparison_states(xy, q)
        with torch.no_grad():
            move = self.router.decision(torch.from_numpy(info['states'])).numpy()
        chosen = abs(q).copy()
        j = info['indices'][move]
        chosen[j] = info['neighbor'][j]
        result = np.copysign(chosen, np.where(np.signbit(xy[:,0]) ^ np.signbit(xy[:,1]), -1., 1.))
        return result, info, q


def teacher_actions(states):
    # Supervision/evaluation only; never called by RoundingRouter or predict.
    return (states[:,0] > 0) | ((states[:,0] == 0) & (states[:,1] == 1))


def exact_state_audit(xy, q, info):
    """Independent rational labels, only for validation, never inference."""
    ids = info['indices']; exp = info['expansion']
    expected = np.empty((len(ids), 2), dtype=np.float64)
    direction_errors = 0; comparison_errors = 0; threshold_errors = 0
    # Every case checks direction, including zero-residual bypasses.
    for i in range(len(xy)):
        r = F(float(exp.hi[i])) + F(float(exp.lo[i]))
        direction_errors += int(((r > 0) - (r < 0)) != info['direction'][i])
    for row, i in enumerate(ids):
        r = F(float(exp.hi[i])) + F(float(exp.lo[i]))
        half = abs(F(float(info['neighbor'][i])) - F(float(abs(q[i])))) * F(float(info['my'][i])) * F(2)**(-int(info['exponent'][i])) / 2
        diff = abs(r) - half
        expected[row] = ((diff > 0) - (diff < 0), int(bits(abs(q[i])) & np.uint64(1)))
        comparison_errors += int(expected[row,0] != info['states'][row,0])
        threshold_errors += int(F(float(info['threshold'][row])) != half)
    return expected, dict(residual_direction_errors=direction_errors, comparison_errors=comparison_errors,
                          exact_threshold_errors=threshold_errors)


def run():
    start = time.monotonic()
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(SEED)
    protected = {str(p):src.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):src.sha(p) for p in ROOT.rglob('*.json')
                      if any(k in p.name for k in ('locked', 'manifest', 'validated'))})
    for folder in ('division_operator_refinement', 'division_rounding_diagnosis',
                   'compensated_division_residual_validation', 'division_remaining_primitives_audit'):
        protected.update({str(p):src.sha(p) for p in (ROOT/folder).iterdir() if p.is_file()})
    for name in ('compensated_division_residual.py', 'train_division_operator.py',
                 'train_subtraction_operator.py', 'train_multiplication_operator.py',
                 'train_addition_operator.py', 'train_stationary_magnitude_threshold.py'):
        p = Path(__file__).with_name(name); protected[str(p)] = src.sha(p)
    dump(OUT/'protected_hashes.json', protected)

    # Whole solve groups remain held out. All possible categorical inputs are
    # nevertheless explicitly supervised, as required for this finite router.
    model = RoundingRouter()
    initial = {k:v.tolist() for k,v in model.state_dict().items()}
    assembly = DivisionAssembly(model)
    z = np.load(ROOT/'division_operator_refinement/refinement_inputs.npz')
    ids = np.flatnonzero(np.isfinite(z['teacher']) & np.isfinite(z['before']))
    assert len(ids) == 119210
    required = z['inputs'][ids]
    rq = assembly.approximate.predict(required)
    assert np.array_equal(bits(rq), bits(z['before'][ids]))
    required_info = assembly.comparison_states(required, rq)
    required_states, exact_metrics = exact_state_audit(required, rq, required_info)
    assert not any(exact_metrics.values())
    metadata = read(ROOT/'division_operator_training/operand_metadata.csv')
    roles = np.array([r['group'] for r in metadata])
    state_source_ids = ids[required_info['indices']]
    fit_states = np.array([i < 45582 and roles[i] == 'fit' for i in state_source_ids])
    empirical_fit = required_states[fit_states]
    truth = np.array([(c,p) for c in (-1.,0.,1.) for p in (0.,1.)], dtype=np.float64)
    labels = teacher_actions(truth)
    # 512 repetitions per state, including strong explicit EQ-odd supervision.
    train_states = np.concatenate((empirical_fit, np.repeat(truth, 512, axis=0)))
    train_labels = teacher_actions(train_states)
    # Exact compression of duplicate categorical examples. Inverse-frequency
    # balancing makes each of the six states equally important, including EQ.
    counts = np.array([int(np.all(train_states == s, axis=1).sum()) for s in truth])
    tensor = torch.from_numpy(truth); targets = torch.from_numpy(labels.astype(np.float64))
    model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    history = []
    for update in range(401):
        logits = model(tensor)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        correct = model.decision(tensor).detach().numpy() == labels
        separation = float(((2*targets-1)*logits).min().detach())
        history.append(dict(update=update,loss=float(loss.detach()),errors=int((~correct).sum()),minimum_logit_margin=separation))
        if correct.all() and separation >= 1.0:
            break
        if update == 400:
            break
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    write(OUT/'screen_training_history.csv', history)
    screen_pass = bool(correct.all())
    config = dict(seed=SEED,architecture='single affine binary logit on ordinal comparison and parity',
                  parameters=3,dtype='float64',inputs=['comparison {-1:LT,0:EQ,1:GT}','parity {0:EVEN,1:ODD}'],
                  initialization=initial,weights=model.linear.weight.detach().tolist(),bias=model.linear.bias.detach().tolist(),
                  no_snapping=True,updates=update,training_occurrences=len(train_states),
                  empirical_fit_occurrences=len(empirical_fit),truth_table_repetitions_per_state=512,
                  state_counts=counts.tolist(),objective='class-balanced BCE on all six categorical states',
                  heldout='complete saved solve roles; no held-out numerical state used to fit parameters')
    dump(OUT/'screen.json', dict(passed=screen_pass,config=config,final=history[-1]))
    if not screen_pass:
        dump(OUT/'results.json', dict(status='SCREEN_FAILED_STOP',config=config)); return

    # Freeze/checkpoint immediately after the small architecture screen succeeds.
    cp = OUT/'rounding_router_before_evaluation.pt'
    torch.save(dict(state_dict=model.state_dict(),config=config,optimizer_state=optimizer.state_dict()),cp)
    digest = src.sha(cp); (OUT/'rounding_router_before_evaluation.sha256').write_text(digest+'\n')
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
    print(json.dumps(dict(event='screen_pass_checkpointed',updates=update,parameters=3,sha256=digest)),flush=True)

    # All independent saved inputs reused; recompute the diagnostic quotient,
    # not the intentionally perturbed q values from residual stress validation.
    old = np.load(ROOT/'compensated_division_residual_validation/compensated_residual_outputs.npz')
    fresh_xy = np.column_stack((old['x'][119210:],old['y'][119210:]))
    halfrows = read(ROOT/'division_rounding_diagnosis/explicit_halfway_controls.csv')
    halfway = np.array([(float(r['x']),float(r['y'])) for r in halfrows])
    rng = np.random.default_rng(SEED+1)
    new_xy = src.synthetic(4096,SEED+2,True)
    new_q = assembly.approximate.predict(new_xy)
    mask = np.isfinite(src.teacher(new_xy)) & np.isfinite(new_q)
    new_xy = new_xy[mask]
    pools = [('required',required,rq,required_info,required_states),
             ('fresh_extreme_residual_cases',fresh_xy,None,None,None),
             ('halfway_controls',halfway,None,None,None),
             ('new_independent_finite',new_xy,None,None,None)]
    state_pools = []; audit_rows = []
    for name,xy,q,info,expected in pools:
        if q is None:q=assembly.approximate.predict(xy)
        if info is None:info=assembly.comparison_states(xy,q)
        if expected is None:expected,em=exact_state_audit(xy,q,info)
        else:em=exact_metrics
        audit_rows.append(dict(group=name,cases=len(xy),active_comparisons=len(expected),**em))
        state_pools.append((name,xy,q,info,expected))
    # Actual EQ-odd controls: perturb an even approximate quotient to the odd
    # float on the other side of a known exact halfway. Only validation uses q.
    hx=[];hq=[]
    for row in halfrows:
        if row['grs_true_halfway'] != 'True':continue
        x,y=float(row['x']),float(row['y']); Q=abs(float(row['teacher']))
        exact=abs(F(x)/F(y)); toward=np.inf if exact>F(Q) else -np.inf
        other=np.nextafter(Q,toward)
        assert int(bits(other)&np.uint64(1)) == 1
        hx.append((x,y)); hq.append(np.copysign(other,-1. if np.signbit(x)^np.signbit(y) else 1.))
    hx=np.array(hx);hq=np.array(hq)
    hi=assembly.comparison_states(hx,hq); hs,he=exact_state_audit(hx,hq,hi)
    assert np.all(hs[:,0]==0) and np.all(hs[:,1]==1)
    state_pools.append(('explicit_EQ_odd_halfway',hx,hq,hi,hs))
    audit_rows.append(dict(group='explicit_EQ_odd_halfway',cases=len(hx),active_comparisons=len(hs),**he))

    router_rows=[]
    def router_stats(name,state,expected):
        with torch.no_grad():got=model.decision(torch.from_numpy(state)).numpy()
        target=teacher_actions(expected)
        router_rows.append(dict(group=name,cases=len(state),errors=int((got!=target).sum()),
                                false_MOVE=int((got&~target).sum()),false_KEEP=int((~got&target).sum())))
    router_stats('complete_truth_table',truth,truth)
    # Irrelevant numerical metadata varies widely but is deliberately not passed
    # to the head: the actual categorical states are the complete input interface.
    repeated=truth[rng.integers(0,6,12000)]
    nuisance=np.ldexp(rng.uniform(.5,1,(12000,3)),rng.integers(-1000,1000,(12000,3)))
    np.savez_compressed(OUT/'categorical_invariance_controls.npz',states=repeated,irrelevant_values=nuisance)
    router_stats('repeated_states_varied_irrelevant_metadata',repeated,repeated)
    for name,xy,q,info,expected in state_pools:
        router_stats(name,info['states'],expected)
        if name=='required':
            for group in sorted(set(roles)):
                j=np.array([i<45582 and roles[i]==group for i in state_source_ids])
                router_stats('Brent_'+group,info['states'][j],expected[j])
            j=state_source_ids<45582
            router_stats('all_Brent',info['states'][j],expected[j])
    write(OUT/'router_validation.csv',router_rows);write(OUT/'exact_comparison_validation.csv',audit_rows)
    with torch.no_grad():logits=model(tensor).numpy();actions=model.decision(tensor).numpy()
    write(OUT/'truth_table.csv',[dict(comparison=('LT','EQ','GT')[int(c)+1],parity=('EVEN','ODD')[int(p)],
          teacher=('KEEP','MOVE')[int(y)],learned=('KEEP','MOVE')[int(g)],logit=float(l))
          for (c,p),y,g,l in zip(truth,labels,actions,logits)])
    router_pass=all(r['errors']==0 for r in router_rows) and all(not any(r[k] for k in ('residual_direction_errors','comparison_errors','exact_threshold_errors')) for r in audit_rows)
    router_manifest=dict(operator='division_rounding_router',status='LOCKED' if router_pass else 'VALIDATION_FAILED_STOP',
                         checkpoint=str(cp),sha256=digest,module='experiments.geometry.train_division_rounding_router',
                         model='RoundingRouter',implementation_sha256=src.sha(Path(__file__)),config=config,metrics=router_rows)
    if not router_pass:
        dump(OUT/'results.json',router_manifest);return
    dump(OUT/'division_rounding_router_locked.json',router_manifest)
    print(json.dumps(dict(event='router_LOCKED',all_routing_errors=0)),flush=True)

    # ONE isolated assembly validation. No Brent interpolation or trace replay.
    metrics=[];failures=[];outputs=[]
    for name,xy,q,info,expected in state_pools:
        pred,replayed_info,baseline=assembly.predict(xy,q=q)
        truth_y=src.teacher(xy)
        assert np.array_equal(replayed_info['states'],info['states'])
        def add_metric(group,j):
            row=src.stats(truth_y[j],pred[j],group)
            row['previously_exact_regressed']=int(((bits(baseline[j])==bits(truth_y[j]))&(bits(pred[j])!=bits(truth_y[j]))).sum())
            metrics.append(row)
        add_metric(name,np.arange(len(xy)))
        if name=='required':
            add_metric('all_Brent',np.flatnonzero(ids<45582))
            add_metric('synthetic_validation',np.flatnonzero((ids>=45582)&(ids<119390)))
            add_metric('nextafter_neighborhoods',np.flatnonzero(ids>=119390))
            for group in sorted(set(roles)):
                j=np.array([v<45582 and roles[v]==group for v in ids])
                add_metric('Brent_'+group,np.flatnonzero(j))
        for i in np.flatnonzero(bits(pred)!=bits(truth_y)):
            failures.append(dict(group=name,index=int(i),x_hex=xy[i,0].hex(),y_hex=xy[i,1].hex(),
                q_hex=q[i].hex(),teacher_hex=truth_y[i].hex(),learned_hex=pred[i].hex()))
        outputs.append((name,xy,q,pred,truth_y))
    write(OUT/'division_validation.csv',metrics);write(OUT/'division_failures.csv',failures)
    for name,xy,q,pred,truth_y in outputs:
        np.savez_compressed(OUT/(name+'_division_outputs.npz'),inputs=xy,approximate=q,learned=pred,teacher=truth_y)
    unchanged=all(src.sha(Path(p))==h for p,h in protected.items());assert unchanged
    passed=all(r['errors']==0 and r['previously_exact_regressed']==0 for r in metrics)
    result=dict(operator='division_operator',status='LOCKED' if passed else 'ASSEMBLY_FAILED_STOP',passed=passed,
        module='experiments.geometry.train_division_rounding_router',class_name='DivisionAssembly',
        implementation_sha256=src.sha(Path(__file__)),router_checkpoint=str(cp),router_sha256=digest,
        router_manifest=str(OUT/'division_rounding_router_locked.json'),
        diagnostic_checkpoint_preserved=assembly.diagnostic_meta,
        arithmetic_dependencies=assembly.residual.manifests,threshold_dependency=assembly.threshold_meta,
        reciprocal_dependency=assembly.reciprocal_meta,
        residual_assembly_manifest=str(ROOT/'compensated_division_residual_validation/compensated_division_residual_validated.json'),
        scope='Audited finite division domain, finite approximate quotient, validated residual contract, at-most-one-neighbor correction. Not universal IEEE-754 division.',
        excluded_original_nonfinite_cases=int(len(z['inputs'])-len(ids)),
        new_independent_excluded_nonfinite_cases=int(len(mask)-mask.sum()),
        residual_interface='Unmodified ResidualExpansion(hi,lo,scale_exponent,numerator_negative) retained; comparator-local rounded sum follows audited comparison only.',
        signed_zero_semantics='input-derived quotient sign restored with unchanged copysign scaffold',
        training='Only the three-parameter categorical routing head',config=config,
        router_metrics=router_rows,comparison_metrics=audit_rows,division_metrics=metrics,
        protected_files=len(protected),protected_hashes_unchanged=unchanged,
        Brent_interpolation=False,full_Brent=False,recursion=False,elapsed_seconds=time.monotonic()-start)
    dump(OUT/'results.json',result)
    if passed:dump(OUT/'division_operator_locked.json',result)
    print(json.dumps(dict(status=result['status'],division_metrics=metrics,seconds=result['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':
    run()
