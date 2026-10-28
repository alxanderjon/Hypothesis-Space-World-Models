"""Ordered IQ primitive audit and one gated local Brent replay; no training."""
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.geometry import audit_exact_branch_brent as audit
from experiments.geometry import train_division_operator as source
from experiments.geometry import validate_locked_brent_linear as linear_source
from experiments.geometry.validate_locked_brent_linear import LockedLinearInterpolation, read, write, dump, bits
from experiments.geometry.train_brent_interpolation_eligibility import BrentEligibility
from experiments.geometry.train_brent_interpolation_branch_selector import BrentBranchSelector
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold

ROOT=Path('outputs/sho/stage1B')
SAVED=ROOT/'exact_branch_switch_brent_audit'
OUT=ROOT/'brent_inverse_quadratic_assembly_validation'
INPUTS=('fcur','fpre','fblk','xcur','xpre','xblk')
# Python/C audited order: numerator is evaluated before the denominator.
# The final expression negates fcur before evaluating the numerator's products.
GRAPH=(
 ('fpre_minus_fcur','subtract','fpre','fcur'),
 ('xpre_minus_xcur','subtract','xpre','xcur'),
 ('dpre','divide','fpre_minus_fcur','xpre_minus_xcur'),
 ('fblk_minus_fcur','subtract','fblk','fcur'),
 ('xblk_minus_xcur','subtract','xblk','xcur'),
 ('dblk','divide','fblk_minus_fcur','xblk_minus_xcur'),
 ('minus_fcur','negate','fcur',None),
 ('fblk_times_dblk','multiply','fblk','dblk'),
 ('fpre_times_dpre','multiply','fpre','dpre'),
 ('numerator_inner','subtract','fblk_times_dblk','fpre_times_dpre'),
 ('minus_fcur_times_numerator_inner','multiply','minus_fcur','numerator_inner'),
 ('dblk_times_dpre','multiply','dblk','dpre'),
 ('fblk_minus_fpre','subtract','fblk','fpre'),
 ('denominator','multiply','dblk_times_dpre','fblk_minus_fpre'),
 ('stry','divide','minus_fcur_times_numerator_inner','denominator'),
)


def native(op,a,b=None):
    if op=='negate':return np.negative(a)
    if op=='subtract':return a-b
    if op=='multiply':return a*b
    if op=='divide':return a/b
    raise ValueError(op)


class PrimitiveFailure(RuntimeError):
    def __init__(self,detail):self.detail=detail;super().__init__(str(detail))


class LockedIQInterpolation:
    def __init__(self):
        manifest=ROOT/'brent_linear_locked_assembly_validation/brent_linear_interpolation_locked.json'
        self.linear_manifest=json.loads(manifest.read_text())
        assert self.linear_manifest['status']=='LOCKED'
        assert source.sha(Path(linear_source.__file__))==self.linear_manifest['implementation_sha256']
        self.linear=LockedLinearInterpolation()
        self.arithmetic=self.linear.arithmetic
        self.division=self.linear.division

    def operation(self,op,a,b=None):
        if op=='negate':return np.negative(a)  # Exact audited unary sign flip.
        if op=='divide':return self.division.predict(np.column_stack((a,b)))[0]
        return self.arithmetic._call(getattr(self.arithmetic,op),a,b)

    def evaluate(self,states,observer=None):
        x=np.asarray(states,dtype=np.float64).reshape(-1,6)
        learned={k:x[:,j] for j,k in enumerate(INPUTS)}
        teacher=learned.copy()
        for stage,(name,op,lhs,rhs) in enumerate(GRAPH,1):
            a=learned[lhs];b=learned[rhs] if rhs else None
            target=native(op,teacher[lhs],teacher[rhs] if rhs else None)
            try:got=self.operation(op,a,b)
            except Exception as exc:
                raise PrimitiveFailure(dict(stage=stage,primitive=name,operation=op,exception=repr(exc))) from exc
            errors=np.flatnonzero(bits(got)!=bits(target))
            if observer is not None:observer(stage,name,op,a,b,target,got)
            if len(errors):
                i=int(errors[0])
                raise PrimitiveFailure(dict(stage=stage,primitive=name,operation=op,index=i,
                    state_hex={k:float(x[i,j]).hex() for j,k in enumerate(INPUTS)},
                    lhs_hex=float(a[i]).hex(),rhs_hex=float(b[i]).hex() if b is not None else None,
                    teacher_hex=float(target[i]).hex(),learned_hex=float(got[i]).hex(),
                    teacher_bits=int(bits(target[i])),learned_bits=int(bits(got[i])),errors=len(errors)))
            learned[name]=got;teacher[name]=target
        return learned,teacher


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    protected={str(p):source.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):source.sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('locked','manifest','validated'))})
    for folder in ('brent_linear_locked_assembly_validation','division_rounding_router_training','exact_branch_switch_brent_audit'):
        protected.update({str(p):source.sha(p) for p in (ROOT/folder).iterdir() if p.is_file()})
    protected.update({str(p):source.sha(p) for p in Path('experiments/geometry').glob('*.py')})
    dump(OUT/'protected_hashes.json',protected)
    model=LockedIQInterpolation()
    rows=read(SAVED/'brent_iterations.csv')
    iq=[r for r in rows if r['attempt_method']=='INVERSE_QUADRATIC_EXTRAPOLATION']
    assert len(iq)==7775
    states=[json.loads(r['normalized_state']) for r in iq]
    x=np.array([[s[k] for k in INPUTS] for s in states])
    splits={int(r['case_id']):r['group'] for r in read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
    groups=np.array([splits[int(r['case_id'])] for r in iq])
    # Exact operand-domain membership in the already locked division validation.
    z=np.load(ROOT/'division_operator_training/Brent_division_operands.npz')
    division_operands=z['operands']
    z.close()
    dm=read(ROOT/'division_operator_training/operand_metadata.csv')
    pair_map={(int(r['case_id']),int(r['iteration']),r['operation']):division_operands[i] for i,r in enumerate(dm)}
    stage_metrics=[];domains=[];coverage=[]
    def observer(stage,name,op,a,b,target,got):
        for group,mask in [('all',np.ones(len(x),bool))]+[(g,groups==g) for g in sorted(set(groups))]:
            stage_metrics.append(dict(stage=stage,primitive=name,**source.stats(target[mask],got[mask],group)))
        for role,v in [('left_operand',a)]+([('right_operand',b)] if b is not None else [])+[('output',got)]:
            nz=abs(v[v!=0])
            domains.append(dict(stage=stage,primitive=name,quantity=role,minimum=float(v.min()),maximum=float(v.max()),
                minimum_nonzero_abs=float(nz.min()) if len(nz) else None,negative=int((v<0).sum()),positive=int((v>0).sum()),
                zeros=int((v==0).sum()),subnormals=int(((abs(v)<np.finfo(float).tiny)&(v!=0)).sum()),nonfinite=int((~np.isfinite(v)).sum())))
        if op=='divide':
            tag={'dpre':'iq_dpre','dblk':'iq_dblk','stry':'iq_final'}[name]
            old=np.array([pair_map[(int(r['case_id']),int(r['iteration']),tag)] for r in iq])
            pair=np.column_stack((a,b))
            coverage.append(dict(stage=stage,primitive=name,cases=len(x),
                bit_exact_previously_validated_operand_pairs=int(np.all(bits(pair)==bits(old),axis=1).sum()),
                zero_denominators=int((b==0).sum()),nonfinite_operands=int((~np.isfinite(pair)).any(axis=1).sum()),
                nonfinite_outputs=int((~np.isfinite(got)).sum())))
        # Save each finished stage before proceeding; failed stages never run successors.
        write(OUT/'stage_validation.csv',stage_metrics);write(OUT/'stage_domains.csv',domains)
        write(OUT/'division_domain_coverage.csv',coverage)
    graph=[]
    for stage,(name,op,a,b) in enumerate(GRAPH,1):
        primitive={'subtract':'subtraction_operator','multiply':'multiplication_operator','divide':'division_operator','negate':'deterministic unary negation'}[op]
        formula=f'-{a}' if op=='negate' else f'{a} '+{'subtract':'-','multiply':'*','divide':'/'}[op]+f' {b}'
        graph.append(dict(stage=stage,intermediate=name,formula=formula,left_operand=a,right_operand=b,
            implementation=primitive,constant_handling='none',notes='IEEE unary sign flip; not multiplication by -1' if op=='negate' else 'separate float64 operation; no reassociation'))
    write(OUT/'operation_graph.csv',graph)
    try:got,teacher=model.evaluate(x,observer)
    except PrimitiveFailure as exc:
        detail=exc.detail
        if 'index' in detail:detail['trace_identity']={k:iq[detail['index']][k] for k in ('case_id','iteration')}
        dump(OUT/'first_failure.json',detail)
        assert all(source.sha(Path(p))==h for p,h in protected.items())
        dump(OUT/'results.json',dict(status='FIRST_PRIMITIVE_FAILURE_STOP',failure=detail,replay=False,training=False));return
    for name,col in [('dpre','dpre'),('dblk','dblk'),('stry','trial_step')]:
        assert np.array_equal(bits(teacher[name]),bits([float(r[col]) for r in iq]))
    assert all(r['bit_exact_previously_validated_operand_pairs']==len(x) for r in coverage)
    np.savez_compressed(OUT/'isolated_IQ_outputs.npz',**got)
    special=[]
    for cid in (360,765):
        mask=np.array([int(r['case_id'])==cid for r in iq])
        special.append(source.stats(teacher['stry'][mask],got['stry'][mask],f'case_{cid}'))
    dump(OUT/'known_problematic_cases.json',special)
    manifest=dict(operator='brent_inverse_quadratic_interpolation',status='LOCKED',
        module='experiments.geometry.validate_locked_brent_iq',class_name='LockedIQInterpolation',
        implementation_sha256=source.sha(Path(__file__)),authority=str(Path(audit.__file__)),authority_sha256=source.sha(Path(audit.__file__)),
        inputs=INPUTS,output='stry',ordered_graph=graph,new_parameters=0,new_arithmetic_primitives=0,training=False,
        cases=len(x),all_intermediates_bit_exact=True,division_domain_coverage=coverage,
        linear_dependency=model.linear_manifest,division_dependency=model.linear.manifest,
        arithmetic_dependencies=model.arithmetic.manifests,replay_status='PENDING')
    dump(OUT/'brent_inverse_quadratic_interpolation_locked.json',manifest)
    (OUT/'assembly_checkpoint.sha256').write_text(source.sha(OUT/'brent_inverse_quadratic_interpolation_locked.json')+'\n')
    print(json.dumps(dict(event='isolated_IQ_LOCKED',cases=len(x),stages=len(GRAPH),errors=0)),flush=True)

    # One limited replay; these are the only changes to the exact interpreter.
    em=model.linear_manifest['eligibility'];bm=model.linear_manifest['branch_selector']
    for m in (em,bm):assert source.sha(Path(m['checkpoint']))==m['sha256']
    eligibility=BrentEligibility(MagnitudeThreshold());branch=BrentBranchSelector(MagnitudeThreshold())
    for op,m in ((eligibility,em),(branch,bm)):
        op.load_state_dict(torch.load(m['checkpoint'],weights_only=False)['state_dict']);op.eval().requires_grad_(False)
    counts=dict(eligibility=0,branch=0,linear=0,IQ=0,eligibility_errors=0,branch_errors=0,linear_errors=0)
    iq_stage_counts={name:0 for name,_,_,_ in GRAPH}
    def eligibility_hook(sp,d,fc,fp):
        p=bool(eligibility.predict([[sp,d,fc,fp]])[1][0]);counts['eligibility']+=1
        counts['eligibility_errors']+=int(p!=(abs(sp)>d and abs(fc)<abs(fp)));return p
    def branch_hook(xp,xb):
        p=int(branch.predict([[xp,xb]])[0])==0;counts['branch']+=1;counts['branch_errors']+=int(p!=(xp==xb));return p
    def linear_hook(fc,fp,xc,xp):
        p=float(model.linear.predict([[fc,fp,xc,xp]])[0]);counts['linear']+=1
        counts['linear_errors']+=int(p.hex()!=(-fc*(xc-xp)/(fc-fp)).hex());return p
    def iq_hook(fc,fp,fb,xc,xp,xb):
        v,_=model.evaluate([[fc,fp,fb,xc,xp,xb]])  # stop immediately on first bad primitive
        counts['IQ']+=1
        for key in iq_stage_counts:iq_stage_counts[key]+=1
        return float(v['dpre'][0]),float(v['dblk'][0]),float(v['stry'][0])
    original=inspect.getsource(audit.traced_brent)
    substitutions=[('eligible=previous_step_large and residual_decreasing','eligible=eligibility_hook(sp,delta,fc,fp)'),
        ('if xp==xb:','if branch_hook(xp,xb):'),('trial=-fc*(xc-xp)/(fc-fp)','trial=linear_hook(fc,fp,xc,xp)'),
        ('dp=(fp-fc)/(xp-xc);db=(fb-fc)/(xb-xc)\n                trial=-fc*(fb*db-fp*dp)/(db*dp*(fb-fp))',
         'dp,db,trial=iq_hook(fc,fp,fb,xc,xp,xb)')]
    changed=original
    for old,new in substitutions:assert changed.count(old)==1;changed=changed.replace(old,new)
    restored=changed
    for old,new in reversed(substitutions):restored=restored.replace(new,old)
    assert restored==original
    (OUT/'limited_replay_interpreter.py').write_text(changed)
    env=dict(audit.__dict__,eligibility_hook=eligibility_hook,branch_hook=branch_hook,linear_hook=linear_hook,iq_hook=iq_hook)
    exec(compile(changed,'limited_IQ_replay','exec'),env);learned=env['traced_brent']
    cases=json.loads((SAVED/'solve_inputs.json').read_text())
    for folder,filename in [('brent_interpolation_eligibility_training','new_fresh_candidates.json'),
                            ('brent_interpolation_branch_selector_training','new_fresh_candidates.json'),
                            ('subtraction_operator_training','fresh_candidates.json')]:
        for r in json.loads((ROOT/folder/filename).read_text()):
            cases.append(dict(parent=r['parent'],axis=r['axis'],count=r['k'],left=r['c_left'],right=r['c_right'],teacher_cut=r['teacher_cut'],label=r['case']))
    saved_queries={}
    for r in rows:
        if r['query']:saved_queries.setdefault(int(r['case_id']),[]).append((float(r['query']).hex(),float(r['new_function']).hex()))
    a=audit.Audit.__new__(audit.Audit);replays=[];first=None
    for i,case in enumerate(cases):
        a.setup(case['parent'],case['axis']);k=case['count']
        def evaluate(c):
            f=a.formula(float(c),k);ia=int(f[4][0])
            return float(f[1][0]),dict(q_active=int(f[3][0]),aspect_active=ia,aspect_power=int(f[5][0,ia]),
                                     competing_q=f[6][0].tolist(),competing_log_aspect=f[7][0].tolist())
        tc=[];lc=[]
        def tcall(c):
            v,b=evaluate(c);tc.append((float(c).hex(),v.hex(),b));return v,b
        def lcall(c):
            v,b=evaluate(c);lc.append((float(c).hex(),v.hex(),b));return v,b
        root,tr,n,reason=audit.traced_brent(tcall,case['left'],case['right'],i)
        assert root.hex()==float(case['teacher_cut']).hex()
        if i<818:assert [(c,v) for c,v,_ in tc[2:]]==saved_queries[i]
        before=counts.copy()
        try:gotroot,lr,ln,lreason=learned(lcall,case['left'],case['right'],i)
        except PrimitiveFailure as exc:
            first=dict(case_id=i,**exc.detail);dump(OUT/'first_replay_failure.json',first);break
        bad=next((j for j,(u,v) in enumerate(zip(tc,lc)) if u!=v),None)
        if bad is None and len(tc)!=len(lc):bad=min(len(tc),len(lc))
        row=dict(case_id=i,group=splits[i] if i<818 else 'additional_fresh',queries=ln,
                 exact_queries=sum(u[0]==v[0] for u,v in zip(tc,lc)),iteration_queries=ln-2,
                 first_divergence=bad,exact_full_state_trace=tr==lr,exact_return=root.hex()==gotroot.hex(),
                 teacher_return_hex=root.hex(),learned_return_hex=gotroot.hex(),
                 **{key:counts[key]-before[key] for key in counts})
        replays.append(row)
        if bad is not None or not row['exact_return'] or not row['exact_full_state_trace']:
            first=row;dump(OUT/'first_replay_failure.json',dict(summary=row,teacher=tr,learned=lr));break
        if (i+1)%100==0:print(json.dumps(dict(event='replay_progress',solves=i+1,exact=i+1)),flush=True)
    write(OUT/'solve_replay.csv',replays)
    summary=[]
    for group in ['all']+sorted({r['group'] for r in replays}):
        rr=[r for r in replays if group=='all' or r['group']==group]
        summary.append(dict(group=group,solves=len(rr),queries=sum(r['queries'] for r in rr),exact_queries=sum(r['exact_queries'] for r in rr),
            iteration_queries=sum(r['iteration_queries'] for r in rr),exact_returns=sum(r['exact_return'] for r in rr),
            full_state_matches=sum(r['exact_full_state_trace'] for r in rr),IQ_attempts=sum(r['IQ'] for r in rr),
            divergences=sum(r['first_divergence'] is not None for r in rr)))
    write(OUT/'replay_summary.csv',summary)
    unchanged=all(source.sha(Path(p))==h for p,h in protected.items());assert unchanged
    passed=first is None and len(replays)==len(cases) and not any(counts[k] for k in ('eligibility_errors','branch_errors','linear_errors'))
    result=dict(operator='brent_inverse_quadratic_interpolation',status='LOCKED' if passed else 'REPLAY_FAILED_STOP',passed=passed,
        assembly_manifest=str(OUT/'brent_inverse_quadratic_interpolation_locked.json'),
        assembly_manifest_sha256=source.sha(OUT/'brent_inverse_quadratic_interpolation_locked.json'),
        isolated_cases=len(x),stages=len(GRAPH),isolated_stage_errors=0,new_learned_primitives=0,
        stage_metrics=stage_metrics,division_domain_coverage=coverage,replay_metrics=summary,replay_IQ_stage_exact_counts=iq_stage_counts,
        hook_counts=counts,first_divergence=first,protected_files=len(protected),protected_hashes_unchanged=unchanged,
        training=False,later_Brent_operations='unchanged teacher',recursion=False,elapsed_seconds=time.monotonic()-start)
    dump(OUT/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='stage_metrics'},indent=2),flush=True)


if __name__=='__main__':run()
