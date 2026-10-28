"""Acceptance-only audit: frozen proposals, arithmetic and decision reuse.

No training, no modified operators, no installed acceptance/fallback policy.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.geometry import audit_exact_branch_brent as authority
from experiments.geometry import train_division_operator as source
from experiments.geometry import validate_locked_brent_iq as iq_source
from experiments.geometry.validate_locked_brent_iq import LockedIQInterpolation, INPUTS
from experiments.geometry.validate_locked_brent_linear import read,write,dump,bits
from experiments.geometry.train_brent_interpolation_eligibility import BrentEligibility
from experiments.geometry.train_brent_interpolation_branch_selector import BrentBranchSelector
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold

ROOT=Path('outputs/sho/stage1B')
SAVED=ROOT/'exact_branch_switch_brent_audit'
OUT=ROOT/'brent_acceptance_isolation_audit'


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    protected={str(p):source.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):source.sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('locked','manifest','validated'))})
    protected.update({str(p):source.sha(p) for p in Path('experiments/geometry').glob('*.py')})
    for folder in ('brent_inverse_quadratic_assembly_validation','brent_linear_locked_assembly_validation'):
        protected.update({str(p):source.sha(p) for p in (ROOT/folder).iterdir() if p.is_file()})
    dump(OUT/'protected_hashes.json',protected)
    iq_manifest=json.loads((ROOT/'brent_inverse_quadratic_assembly_validation/brent_inverse_quadratic_interpolation_locked.json').read_text())
    assert iq_manifest['status']=='LOCKED' and source.sha(Path(iq_source.__file__))==iq_manifest['implementation_sha256']
    iq=LockedIQInterpolation();arith=iq.arithmetic;division=iq.division;threshold=division.threshold
    em=iq.linear_manifest['eligibility'];bm=iq.linear_manifest['branch_selector']
    eligibility=BrentEligibility(MagnitudeThreshold());branch=BrentBranchSelector(MagnitudeThreshold())
    for model,m in ((eligibility,em),(branch,bm)):
        assert source.sha(Path(m['checkpoint']))==m['sha256']
        model.load_state_dict(torch.load(m['checkpoint'],weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    split={int(r['case_id']):r['group'] for r in read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
    traces=[];solves=[]
    for row in read(SAVED/'brent_iterations.csv'):
        r=dict(row);r['normalized_state']=json.loads(r['normalized_state'])
        for key in ('case_id','iteration'):r[key]=int(r[key])
        for key in ('stop','eligible','interpolation_accepted'):
            r[key]=None if r.get(key,'')=='' else r[key]=='True'
        for key in ('delta','sbis','trial_step','acceptance_left','acceptance_limit_previous','acceptance_limit_bracket','acceptance_right'):
            r[key]=float(r[key]) if r.get(key,'')!='' else None
        r['split']=split[r['case_id']];traces.append(r)
    # Additional saved fresh traces were not persisted as iteration rows. Rebuild
    # those reference records only, with the unchanged teacher. No learned
    # acceptance is installed or used to choose subsequent states.
    a=authority.Audit.__new__(authority.Audit);fresh_cases=[]
    for folder,name in [('brent_interpolation_eligibility_training','new_fresh_candidates.json'),
                        ('brent_interpolation_branch_selector_training','new_fresh_candidates.json'),
                        ('subtraction_operator_training','fresh_candidates.json')]:
        fresh_cases.extend(json.loads((ROOT/folder/name).read_text()))
    for j,c in enumerate(fresh_cases):
        a.setup(c['parent'],c['axis']);count=c['k']
        def callback(cut):return float(a.formula(float(cut),count)[1][0]),{}
        root,rr,calls,reason=authority.traced_brent(callback,c['c_left'],c['c_right'],818+j)
        assert root.hex()==float(c['teacher_cut']).hex()
        for r in rr:r['split']='additional_fresh'
        traces.extend(rr);solves.append(dict(case_id=818+j,returned=root,queries=calls,exact_saved_return=True))
    active=[r for r in traces if not r['stop']]
    e_inputs=np.array([[r['normalized_state']['spre'],r['delta'],r['normalized_state']['fcur'],r['normalized_state']['fpre']] for r in active])
    _,ep=eligibility.predict(e_inputs)
    et=np.array([r['eligible'] for r in active],dtype=bool)
    assert np.array_equal(ep,et.astype(int))
    records=[r for r in active if r['eligible']]
    states=np.array([[r['normalized_state'][k] for k in INPUTS] for r in records])
    bs=branch.predict(states[:,[4,5]])
    bt=np.array([r['attempt_method']!='LINEAR_INTERPOLATION' for r in records],int)
    assert np.array_equal(bs,bt)
    proposal=np.empty(len(records),dtype=float)
    linear=bs==0;quadratic=bs==1
    proposal[linear]=iq.linear.predict(states[linear][:,[0,1,3,4]])
    proposal[quadratic]=iq.evaluate(states[quadratic])[0]['stry']
    target=np.array([r['trial_step'] for r in records])
    proposal_metrics=[source.stats(target,proposal,'all_proposals'),source.stats(target[linear],proposal[linear],'linear'),source.stats(target[quadratic],proposal[quadratic],'IQ')]
    write(OUT/'proposal_replay.csv',proposal_metrics)
    assert all(r['errors']==0 for r in proposal_metrics)
    print(json.dumps(dict(event='proposal_replay_exact',acceptance_decisions=len(records),solves=818+len(fresh_cases))),flush=True)

    m=lambda a,b:arith._call(arith.multiply,a,b)
    sub=lambda a,b:arith._call(arith.subtract,a,b)
    add=lambda a,b:arith._call(arith.add,a,b)
    div=lambda a,b:division.predict(np.column_stack(np.broadcast_arrays(a,b)))[0]
    comparison_rows=[];arithmetic_rows=[];domain_rows=[]
    def check(name,teacher,learned):
        row=source.stats(np.asarray(teacher),np.asarray(learned),name);arithmetic_rows.append(row)
        z=np.asarray(learned);nz=abs(z[z!=0])
        domain_rows.append(dict(intermediate=name,cases=len(z),minimum=float(z.min()),maximum=float(z.max()),
            minimum_nonzero_abs=float(nz.min()) if len(nz) else None,zeros=int((z==0).sum()),nonfinite=int((~np.isfinite(z)).sum())))
        write(OUT/'arithmetic_validation.csv',arithmetic_rows)
        if row['errors']:
            i=int(np.flatnonzero(bits(teacher)!=bits(learned))[0])
            failure=dict(intermediate=name,index=i,case_id=records[i]['case_id'],iteration=records[i]['iteration'],teacher_hex=float(teacher[i]).hex(),learned_hex=float(learned[i]).hex())
            dump(OUT/'first_arithmetic_failure.json',failure);raise RuntimeError(failure)
        return learned
    def less(a,b,name):
        margin=sub(a,b)
        with torch.no_grad():sgn=threshold.decision(torch.from_numpy(margin)).numpy()
        got=sgn<0;truth=a<b
        comparison_rows.append(dict(comparison=name,cases=len(a),errors=int((got!=truth).sum()),
            true_count=int(truth.sum()),equality_count=int((a==b).sum()),equality_errors=int((got&(a==b)).sum())))
        return got,margin
    fc,fp,fb,xc,xp,xb=states.T
    sp=np.array([r['normalized_state']['spre'] for r in records])
    abs_xc=check('abs_xcur',abs(xc),np.abs(xc))
    rt=check('rtol_times_abs_xcur',authority.RTOL*abs(xc),m(authority.RTOL,abs_xc))
    tol_sum=check('xtol_plus_relative_tolerance',authority.XTOL+authority.RTOL*abs(xc),add(authority.XTOL,rt))
    delta=check('delta',np.array([r['delta'] for r in records]),div(tol_sum,np.full(len(states),2.)))
    diff=check('xblk_minus_xcur',xb-xc,sub(xb,xc))
    sbis=check('sbis',np.array([r['sbis'] for r in records]),div(diff,np.full(len(states),2.)))
    abs_trial=check('abs_stry',abs(target),np.abs(proposal))
    lhs=check('lhs',np.array([r['acceptance_left'] for r in records]),m(2.,abs_trial))
    lim1=check('lim1',np.array([r['acceptance_limit_previous'] for r in records]),np.abs(sp))
    abs_sbis=check('abs_sbis',abs((xb-xc)/2),np.abs(sbis))
    tripled=check('three_times_abs_sbis',3*abs((xb-xc)/2),m(3.,abs_sbis))
    lim2=check('lim2',np.array([r['acceptance_limit_bracket'] for r in records]),sub(tripled,delta))
    choose_left,min_margin=less(lim1,lim2,'lim1_lt_lim2')
    # Exact audited mux; equal inputs choose lim2. No learned min/equivalence
    # reduction with a 1e-12 tie band is suitable here.
    limit=check('selected_limit',np.array([r['acceptance_right'] for r in records]),np.where(choose_left,lim1,lim2))
    literal,final_margin=less(lhs,limit,'lhs_lt_selected_limit')
    c1,margin1=less(lhs,lim1,'lhs_lt_lim1')
    c2,margin2=less(lhs,lim2,'lhs_lt_lim2')
    features=np.column_stack((c1,c2)).astype(float)
    # Reuse the learned AND submodule, not the eligibility operator on raw
    # acceptance values. Weights remain attached to their original checkpoint.
    with torch.no_grad():head_output=eligibility.head(torch.from_numpy(features)).argmax(1).numpy()
    labels=np.array([r['interpolation_accepted'] for r in records],dtype=bool)
    assert np.array_equal(labels,lhs<np.where(lim1<lim2,lim1,lim2))
    acceptance_errors=int((head_output!=labels).sum())
    write(OUT/'comparison_validation.csv',comparison_rows);write(OUT/'arithmetic_domains.csv',domain_rows)
    table=np.array([(a,b) for a in (0.,1.) for b in (0.,1.)])
    with torch.no_grad():tt=eligibility.head(torch.from_numpy(table)).argmax(1).numpy()
    truth_table=[dict(lhs_lt_lim1=bool(a),lhs_lt_lim2=bool(b),teacher='ACCEPT' if a and b else 'REJECT',
                      learned='ACCEPT' if p else 'REJECT',error=int(bool(p)!=bool(a and b))) for (a,b),p in zip(table,tt)]
    write(OUT/'acceptance_truth_table.csv',truth_table)

    # Finite strict-comparison equality/nextafter probes. These are offline
    # local acceptance triples, not root solves or fallback-policy training.
    bases=np.unique(np.r_[lhs[::max(1,len(lhs)//96)],lim1[::max(1,len(lhs)//96)],lim2[::max(1,len(lhs)//96)],0.,np.nextafter(0.,1.),np.finfo(float).tiny,1.])
    boundary=[]
    for b in bases:
        near=[np.nextafter(b,-np.inf),b,np.nextafter(b,np.inf)]
        for a in near:
            for c in near:boundary.append((b,a,c))
    boundary.extend([(0.,0.,-0.),(-0.,-0.,0.),(0.,-0.,-0.)])
    bound=np.array(boundary);b_lhs,b1,b2=bound.T
    take,_=less(b1,b2,'boundary_min_choice');blearn=np.where(take,b1,b2)
    btarget=np.where(b1<b2,b1,b2)
    bc1,_=less(b_lhs,b1,'boundary_lhs_lt_lim1');bc2,_=less(b_lhs,b2,'boundary_lhs_lt_lim2')
    with torch.no_grad():bp=eligibility.head(torch.from_numpy(np.column_stack((bc1,bc2)).astype(float))).argmax(1).numpy()
    by=b_lhs<btarget
    boundary_summary=dict(cases=len(bound),head_errors=int((bp!=by).sum()),
        min_bit_errors=int((bits(blearn)!=bits(btarget)).sum()),equality_rejection_cases=int((b_lhs==btarget).sum()),
        equality_rejection_errors=int(((b_lhs==btarget)&(bp!=0)).sum()),
        includes_subnormal_and_signed_zero=True)
    np.savez_compressed(OUT/'acceptance_boundary_controls.npz',triples=bound,teacher=by,learned=bp)
    dump(OUT/'boundary_validation.json',boundary_summary);write(OUT/'comparison_validation.csv',comparison_rows)
    saved_rows=[]
    for i,r in enumerate(records):
        st={key:float(r['normalized_state'][key]) for key in ('xpre','xcur','xblk','fpre','fcur','fblk','spre','scur')}
        values=dict(**st,xtol=authority.XTOL,rtol=authority.RTOL,delta=float(delta[i]),sbis=float(sbis[i]),
            stry=float(target[i]),locked_stry=float(proposal[i]),abs_xcur=float(abs_xc[i]),relative_tolerance=float(rt[i]),tolerance_sum=float(tol_sum[i]),
            bracket_difference=float(diff[i]),abs_stry=float(abs_trial[i]),lhs=float(lhs[i]),lim1=float(lim1[i]),abs_sbis=float(abs_sbis[i]),
            three_times_abs_sbis=float(tripled[i]),lim2=float(lim2[i]),selected_limit=float(limit[i]),
            min_comparison_margin=float(min_margin[i]),final_comparison_margin=float(final_margin[i]),
            lhs_lim1_margin=float(margin1[i]),lhs_lim2_margin=float(margin2[i]))
        saved_rows.append(dict(solve_id=r['case_id'],iteration=r['iteration'],split=r['split'],interpolation_branch='LINEAR' if linear[i] else 'IQ',
            **values,hex_values=json.dumps({k:v.hex() for k,v in values.items()}),
            lim1_lt_lim2=bool(lim1[i]<lim2[i]),lim1_eq_lim2=bool(lim1[i]==lim2[i]),lhs_lt_lim1=bool(lhs[i]<lim1[i]),lhs_lt_lim2=bool(lhs[i]<lim2[i]),
            lhs_eq_lim1=bool(lhs[i]==lim1[i]),lhs_eq_lim2=bool(lhs[i]==lim2[i]),lhs_lt_selected_limit=bool(labels[i]),
            label='ACCEPT' if labels[i] else 'REJECT',frozen_comparison_label='ACCEPT' if literal[i] else 'REJECT',
            reused_AND_head_label='ACCEPT' if head_output[i] else 'REJECT'))
    write(OUT/'acceptance_decisions.csv',saved_rows)
    summary=[];group_arr=np.array([r['split'] for r in records])
    for group in ['all']+sorted(set(group_arr)):
        for method,mask in [('all',np.ones(len(records),bool)),('LINEAR',linear),('IQ',quadratic)]:
            j=mask & ((group_arr==group) if group!='all' else True)
            summary.append(dict(split=group,branch=method,cases=int(j.sum()),ACCEPT=int(labels[j].sum()),REJECT=int((~labels[j]).sum()),
                literal_comparison_errors=int((literal[j]!=labels[j]).sum()),reused_head_errors=int((head_output[j]!=labels[j]).sum())))
    write(OUT/'decision_counts.csv',summary)
    unchanged=all(source.sha(Path(p))==h for p,h in protected.items());assert unchanged
    result=dict(status='AUDIT_ONLY_COMPLETE',acceptance_decisions=len(records),saved_acceptance_decisions=sum(r['case_id']<818 for r in records),
        source_solves=818+len(fresh_cases),additional_fresh_reference_solves=len(fresh_cases),
        proposal_metrics=proposal_metrics,arithmetic_metrics=arithmetic_rows,comparison_metrics=comparison_rows,
        counts=summary,truth_table=truth_table,boundary_validation=boundary_summary,
        all_reuse_checks_passed=acceptance_errors==0 and not any(r['errors'] for r in comparison_rows) and not any(r['error'] for r in truth_table) and boundary_summary['head_errors']==0 and boundary_summary['min_bit_errors']==0,
        head_reused_from=em['checkpoint'],head_checkpoint_sha256=em['sha256'],
        head_weights=eligibility.head.weight.tolist(),head_bias=eligibility.head.bias.tolist(),
        threshold_dependency=division.threshold_meta,first_missing_learned_primitive=None,
        minimal_categorical_interface=['lhs_lt_lim1','lhs_lt_lim2'],output=['REJECT','ACCEPT'],
        training=False,new_operator=False,new_checkpoint=False,acceptance_installed_in_solver=False,
        fallback_changed=False,state_updates_changed=False,stopping_changed=False,recursion=False,
        protected_files=len(protected),protected_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-started)
    dump(OUT/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('arithmetic_metrics','comparison_metrics')},indent=2),flush=True)


if __name__=='__main__':run()
