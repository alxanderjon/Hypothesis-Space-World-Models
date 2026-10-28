"""Isolated frozen linear interpolation followed by one limited local replay.

No training. Inverse-quadratic calculation and all later teacher operations are
unchanged. Saved branch-switch solves only; no recursive partitioning.
"""
import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.geometry import audit_exact_branch_brent as audit
from experiments.geometry import train_division_operator as source
from experiments.geometry.train_division_rounding_router import DivisionAssembly, RoundingRouter
from experiments.geometry.train_brent_interpolation_eligibility import BrentEligibility
from experiments.geometry.train_brent_interpolation_branch_selector import BrentBranchSelector
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'brent_linear_locked_assembly_validation'
SAVED=ROOT/'exact_branch_switch_brent_audit'


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def write(p,rows):
    if not rows:rows=[dict(status='none')]
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bits(x):return np.asarray(x,dtype=np.float64).view('u8')


class LockedLinearInterpolation:
    """Inputs (fcur,fpre,xcur,xpre); output stry. No trained parameters here."""
    def __init__(self):
        folder=ROOT/'division_rounding_router_training'
        self.manifest=json.loads((folder/'division_operator_locked.json').read_text())
        rm=json.loads((folder/'division_rounding_router_locked.json').read_text())
        assert self.manifest['status']==rm['status']=='LOCKED'
        assert source.sha(Path(rm['checkpoint']))==rm['sha256']
        import experiments.geometry.train_division_rounding_router as implementation
        assert source.sha(Path(implementation.__file__))==self.manifest['implementation_sha256']
        router=RoundingRouter()
        router.load_state_dict(torch.load(rm['checkpoint'],weights_only=False)['state_dict'])
        self.division=DivisionAssembly(router)
        self.arithmetic=self.division.residual
        assert self.arithmetic.manifests['subtract']['status']=='LOCKED'

    def evaluate(self,states):
        x=np.asarray(states,dtype=np.float64).reshape(-1,4)
        fc,fp,xc,xp=x.T
        dx=self.arithmetic._call(self.arithmetic.subtract,xc,xp)
        df=self.arithmetic._call(self.arithmetic.subtract,fc,fp)
        numerator=self.arithmetic._call(self.arithmetic.multiply,-fc,dx)
        step,_,_=self.division.predict(np.column_stack((numerator,df)))
        return step,dict(dx=dx,df=df,numerator=numerator)

    def predict(self,states):return self.evaluate(states)[0]


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    protected={str(p):source.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):source.sha(p) for p in ROOT.rglob('*.json')
                      if any(t in p.name for t in ('locked','manifest','validated'))})
    for folder in ('exact_branch_switch_brent_audit','division_rounding_router_training',
                   'brent_interpolation_eligibility_training','brent_interpolation_branch_selector_training'):
        protected.update({str(p):source.sha(p) for p in (ROOT/folder).iterdir() if p.is_file()})
    protected.update({str(p):source.sha(p) for p in Path('experiments/geometry').glob('*.py') if p!=Path(__file__)})
    dump(OUT/'protected_hashes.json',protected)
    assembly=LockedLinearInterpolation()
    em=json.loads((ROOT/'brent_interpolation_eligibility_training/brent_interpolation_eligibility_locked.json').read_text())
    bm=json.loads((ROOT/'brent_interpolation_branch_selector_training/brent_interpolation_branch_selector_locked.json').read_text())
    for manifest in (em,bm):
        assert manifest['status']=='LOCKED' and source.sha(Path(manifest['checkpoint']))==manifest['sha256']
    eligibility=BrentEligibility(MagnitudeThreshold())
    eligibility.load_state_dict(torch.load(em['checkpoint'],weights_only=False)['state_dict']);eligibility.eval().requires_grad_(False)
    branch=BrentBranchSelector(MagnitudeThreshold())
    branch.load_state_dict(torch.load(bm['checkpoint'],weights_only=False)['state_dict']);branch.eval().requires_grad_(False)

    # Task 1 consumes saved inputs/labels only, without executing a Brent solve.
    z=np.load(ROOT/'division_operator_training/Brent_division_operands.npz')
    x=z['linear_states'];meta=read(ROOT/'division_operator_training/linear_metadata.csv')
    y=np.array([float(r['teacher_step']) for r in meta]);assert len(x)==len(y)
    fc,fp,xc,xp=x.T
    native=-fc*(xc-xp)/(fc-fp)
    assert np.array_equal(bits(native),bits(y))
    saved=read(SAVED/'brent_iterations.csv')
    linear=[r for r in saved if r['attempt_method']=='LINEAR_INTERPOLATION']
    assert len(linear)==18039
    for j,row in enumerate(linear):
        s=json.loads(row['normalized_state'])
        assert np.array_equal(bits(x[j]),bits([s[k] for k in ('fcur','fpre','xcur','xpre')]))
        assert bits(y[j])==bits(float(row['trial_step']))
    pred,parts=assembly.evaluate(x)
    groups=np.array([r['group'] for r in meta]);case_ids=np.array([int(r['case_id']) for r in meta])
    metrics=[]
    for name,mask in [('all_linear',np.ones(len(x),bool)),('saved_audit',case_ids<818)]+[(g,groups==g) for g in sorted(set(groups))]:
        metrics.append(source.stats(y[mask],pred[mask],name))
    intermediate=[source.stats(xc-xp,parts['dx'],'dx'),source.stats(fc-fp,parts['df'],'df'),
                  source.stats(-fc*(xc-xp),parts['numerator'],'numerator')]
    # Previous two linear division failures are cases 594/596, iteration 9.
    special=[]
    for case in (594,596,765):
        mask=case_ids==case;row=source.stats(y[mask],pred[mask],f'case_{case}')
        row['iterations']=[int(meta[i]['iteration']) for i in np.flatnonzero(mask)];special.append(row)
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv')
    for group in range(21):
        ids={int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==group}
        mask=np.isin(case_ids,list(ids));special.append(dict(**source.stats(y[mask],pred[mask],f'conflict_group_{group}'),iterations=[]))
    failures=[dict(**meta[i],state_hex=[v.hex() for v in x[i]],teacher_hex=y[i].hex(),learned_hex=pred[i].hex())
              for i in np.flatnonzero(bits(y)!=bits(pred))]
    write(OUT/'isolated_validation.csv',metrics);write(OUT/'intermediate_validation.csv',intermediate)
    dump(OUT/'known_problematic_cases.json',special);dump(OUT/'isolated_failures.json',failures)
    np.savez_compressed(OUT/'linear_validation_outputs.npz',states=x,teacher=y,learned=pred,**parts)
    passed=all(r['errors']==0 for r in metrics+intermediate)
    manifest=dict(operator='brent_linear_interpolation',status='LOCKED' if passed else 'ASSEMBLY_FAILED_STOP',
        module='experiments.geometry.validate_locked_brent_linear',class_name='LockedLinearInterpolation',
        implementation_sha256=source.sha(Path(__file__)),inputs=['fcur','fpre','xcur','xpre'],output='stry',
        exact_operation_order=['dx=subtract(xcur,xpre)','df=subtract(fcur,fpre)','numerator=multiply(-fcur,dx)','stry=division(numerator,df)'],
        new_parameters=0,training=False,division_manifest=str(ROOT/'division_rounding_router_training/division_operator_locked.json'),
        division_manifest_sha256=source.sha(ROOT/'division_rounding_router_training/division_operator_locked.json'),
        arithmetic=assembly.arithmetic.manifests,eligibility=em,branch_selector=bm,
        isolated_metrics=metrics,intermediate_metrics=intermediate,replay_status='PENDING' if passed else 'NOT_RUN')
    dump(OUT/'isolated_assembly_result.json',manifest)
    if not passed:
        dump(OUT/'results.json',manifest);return
    # Immediate assembly checkpoint consists of source/dependency hashes, no
    # duplicate or newly trained numerical checkpoint is introduced.
    dump(OUT/'brent_linear_interpolation_locked.json',manifest)
    (OUT/'assembly_checkpoint.sha256').write_text(source.sha(OUT/'brent_linear_interpolation_locked.json')+'\n')
    print(json.dumps(dict(event='isolated_LOCKED',attempts=len(x),exact=len(x),saved_audit=18039)),flush=True)

    # Task 2: only three replacements in a new local interpreter. IQ arithmetic
    # and all acceptance, fallback, state, stopping, return lines remain intact.
    original=inspect.getsource(audit.traced_brent)
    replacements=[('eligible=previous_step_large and residual_decreasing','eligible=eligibility_hook(sp,delta,fc,fp)'),
                  ('if xp==xb:','if branch_hook(xp,xb):'),
                  ('trial=-fc*(xc-xp)/(fc-fp)','trial=linear_hook(fc,fp,xc,xp)')]
    changed=original
    for old,new in replacements:
        assert changed.count(old)==1;changed=changed.replace(old,new)
    restored=changed
    for old,new in reversed(replacements):restored=restored.replace(new,old)
    assert restored==original
    (OUT/'limited_replay_interpreter.py').write_text(changed)
    hook_counts=dict(eligibility=0,branch=0,linear=0,eligibility_errors=0,branch_errors=0,linear_errors=0)
    def eligibility_hook(sp,d,fc,fp):
        got=bool(eligibility.predict([[sp,d,fc,fp]])[1][0]);hook_counts['eligibility']+=1
        hook_counts['eligibility_errors']+=int(got!=(abs(sp)>d and abs(fc)<abs(fp)));return got
    def branch_hook(xp,xb):
        got=int(branch.predict([[xp,xb]])[0])==0;hook_counts['branch']+=1
        hook_counts['branch_errors']+=int(got!=(xp==xb));return got
    def linear_hook(fc,fp,xc,xp):
        got=float(assembly.predict([[fc,fp,xc,xp]])[0]);hook_counts['linear']+=1
        hook_counts['linear_errors']+=int(got.hex()!=(-fc*(xc-xp)/(fc-fp)).hex());return got
    env=dict(audit.__dict__,eligibility_hook=eligibility_hook,branch_hook=branch_hook,linear_hook=linear_hook)
    exec(compile(changed,'limited_linear_replay','exec'),env);learned=env['traced_brent']
    cases=json.loads((SAVED/'solve_inputs.json').read_text())
    splits={int(r['case_id']):r['group'] for r in read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
    for folder,name in [('brent_interpolation_eligibility_training','new_fresh_candidates.json'),
                        ('brent_interpolation_branch_selector_training','new_fresh_candidates.json'),
                        ('subtraction_operator_training','fresh_candidates.json')]:
        for r in json.loads((ROOT/folder/name).read_text()):
            cases.append(dict(parent=r['parent'],axis=r['axis'],count=r['k'],left=r['c_left'],right=r['c_right'],teacher_cut=r['teacher_cut'],label=r['case']))
    saved_queries={}
    for r in saved:
        if r['query']:saved_queries.setdefault(int(r['case_id']),[]).append((float(r['query']).hex(),float(r['new_function']).hex()))
    a=audit.Audit.__new__(audit.Audit);replays=[];first=None
    for i,case in enumerate(cases):
        a.setup(case['parent'],case['axis']);k=case['count']
        def evaluate(c):
            f=a.formula(float(c),k);ia=int(f[4][0])
            return float(f[1][0]),dict(q_active=int(f[3][0]),aspect_active=ia,aspect_power=int(f[5][0,ia]),
                                     competing_q=f[6][0].tolist(),competing_log_aspect=f[7][0].tolist())
        teacher_calls=[];learned_calls=[]
        def teacher_callback(c):
            v,branches=evaluate(c);teacher_calls.append((float(c).hex(),v.hex(),branches));return v,branches
        def learned_callback(c):
            v,branches=evaluate(c);learned_calls.append((float(c).hex(),v.hex(),branches));return v,branches
        root,tr,n,reason=audit.traced_brent(teacher_callback,case['left'],case['right'],i)
        assert root.hex()==float(case['teacher_cut']).hex()
        if i<818:assert [(c,v) for c,v,_ in teacher_calls[2:]]==saved_queries[i]
        before=hook_counts.copy()
        got,lr,ln,lreason=learned(learned_callback,case['left'],case['right'],i)
        bad=next((j for j,(u,v) in enumerate(zip(teacher_calls,learned_calls)) if u!=v),None)
        if bad is None and len(teacher_calls)!=len(learned_calls):bad=min(len(teacher_calls),len(learned_calls))
        exact_queries=sum(u[0]==v[0] for u,v in zip(teacher_calls,learned_calls))
        row=dict(case_id=i,group=splits[i] if i<818 else 'additional_fresh',label=case['label'],
                 teacher_queries=n,learned_queries=ln,exact_query_coordinates=exact_queries,
                 iteration_queries=ln-2,exact_iteration_queries=sum(u[0]==v[0] for u,v in zip(teacher_calls[2:],learned_calls[2:])),
                 first_divergence=bad,exact_full_state_trace=tr==lr,
                 exact_return=root.hex()==got.hex(),teacher_return_hex=root.hex(),learned_return_hex=got.hex(),
                 **{key:hook_counts[key]-before[key] for key in hook_counts})
        replays.append(row)
        if bad is not None or not row['exact_return'] or not row['exact_full_state_trace']:
            first=row;dump(OUT/'first_divergence.json',dict(summary=row,teacher=tr,learned=lr));break
        if (i+1)%100==0:print(json.dumps(dict(event='limited_replay_progress',solves=i+1,exact=i+1)),flush=True)
    write(OUT/'solve_replay.csv',replays)
    summary=[]
    for group in ['all']+sorted({r['group'] for r in replays}):
        rr=[r for r in replays if group=='all' or r['group']==group]
        summary.append(dict(group=group,solves=len(rr),queries=sum(r['learned_queries'] for r in rr),
            exact_query_coordinates=sum(r['exact_query_coordinates'] for r in rr),
            iteration_queries=sum(r['iteration_queries'] for r in rr),exact_iteration_queries=sum(r['exact_iteration_queries'] for r in rr),
            final_coordinate_matches=sum(r['exact_return'] for r in rr),full_state_trace_matches=sum(r['exact_full_state_trace'] for r in rr),
            query_divergent_solves=sum(r['first_divergence'] is not None for r in rr),
            eligibility_errors=sum(r['eligibility_errors'] for r in rr),branch_errors=sum(r['branch_errors'] for r in rr),linear_errors=sum(r['linear_errors'] for r in rr)))
    write(OUT/'replay_summary.csv',summary)
    unchanged=all(source.sha(Path(p))==h for p,h in protected.items());assert unchanged
    passed=first is None and len(replays)==len(cases) and not any(hook_counts[k] for k in ('eligibility_errors','branch_errors','linear_errors'))
    # Keep the immediate Task-1 checkpoint immutable; result records replay gate.
    result=dict(operator='brent_linear_interpolation',status='LOCKED' if passed else 'REPLAY_FAILED_STOP',passed=passed,
        assembly_checkpoint=str(OUT/'brent_linear_interpolation_locked.json'),
        assembly_checkpoint_sha256=source.sha(OUT/'brent_linear_interpolation_locked.json'),
        isolated_metrics=metrics,intermediate_metrics=intermediate,replay_metrics=summary,hook_counts=hook_counts,
        first_divergence=first,protected_files=len(protected),protected_hashes_unchanged=unchanged,
        no_training=True,inverse_quadratic='unchanged deterministic teacher code',
        replay_scope='Saved complete local Brent solves only; only eligibility, formula selector and linear step replaced',
        world_model_recursion=False,partition_recursion=False,elapsed_seconds=time.monotonic()-start)
    dump(OUT/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
