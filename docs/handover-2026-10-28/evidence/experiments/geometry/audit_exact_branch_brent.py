"""Trace reconstruction of installed SciPy brentq, checked callback-for-callback."""
import json
import math
import time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import brentq,_zeros
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read
from experiments.sho.audit_fixed_allocation_intervals import Audit

OUT=ROOT/'exact_branch_switch_brent_audit'
XTOL=5e-324
RTOL=1e-15
MAXITER=100
SOURCE_URL='https://raw.githubusercontent.com/scipy/scipy/v1.16.2/scipy/optimize/Zeros/brentq.c'


def opposite(a,b):return math.copysign(1.,a)!=math.copysign(1.,b)


def traced_brent(callback,lo,hi,case_id):
    xp=float(lo);xc=float(hi);xb=0.;fb=0.;sp=0.;sc=0.;eb=None
    fp,ep=callback(xp);fc,ec=callback(xc);calls=2;rows=[]
    def snap():
        values=dict(xpre=xp,xcur=xc,xblk=xb,fpre=fp,fcur=fc,fblk=fb,spre=sp,scur=sc)
        return dict(**values,hex={k:v.hex() for k,v in values.items()},pre_branches=ep,cur_branches=ec,blk_branches=eb)
    if fp==0:return xp,rows,calls,'INITIAL_LEFT_ZERO'
    if fc==0:return xc,rows,calls,'INITIAL_RIGHT_ZERO'
    if not opposite(fp,fc):raise ValueError('Teacher initial sign error')
    for iteration in range(1,MAXITER+1):
        row=dict(case_id=case_id,iteration=iteration,entering=snap())
        reset=fp!=0 and fc!=0 and opposite(fp,fc)
        if reset:
            xb=xp;fb=fp;eb=ep;sc=xc-xp;sp=sc
        row.update(refresh_bracket=reset,after_refresh=snap())
        promote=abs(fb)<abs(fc)
        if promote:
            # C assignments are SEQUENTIAL: xblk gets the newly assigned xpre.
            xp=xc;ep=ec;xc=xb;ec=eb;xb=xp;eb=ep
            fp=fc;fc=fb;fb=fp
        delta=(XTOL+RTOL*abs(xc))/2
        sbis=(xb-xc)/2
        stopzero=fc==0;stopwidth=abs(sbis)<delta
        row.update(promote_block=promote,normalized_state=snap(),delta=delta,sbis=sbis,
                   bracket_lower=min(xb,xc),bracket_upper=max(xb,xc),stop_zero=stopzero,stop_width=stopwidth,
                   stop=stopzero or stopwidth,attempt_method='NONE',effective_method='NONE',eligible=False,
                   interpolation_accepted=None,forced_bisection=False,rejected_to_bisection=False,
                   minimum_step=False,query=None,new_function=None,final_return=None)
        if stopzero or stopwidth:
            row.update(after_step=snap(),final_return=xc,stop_reason='EXACT_ZERO' if stopzero else 'WIDTH_TOLERANCE')
            rows.append(row);return xc,rows,calls,row['stop_reason']
        previous_step_large=abs(sp)>delta;residual_decreasing=abs(fc)<abs(fp)
        eligible=previous_step_large and residual_decreasing
        row.update(eligible=eligible,previous_step_large=previous_step_large,residual_decreasing=residual_decreasing)
        if eligible:
            if xp==xb:
                method='LINEAR_INTERPOLATION'
                trial=-fc*(xc-xp)/(fc-fp)
                row['dpre']=None;row['dblk']=None
            else:
                method='INVERSE_QUADRATIC_EXTRAPOLATION'
                dp=(fp-fc)/(xp-xc);db=(fb-fc)/(xb-xc)
                trial=-fc*(fb*db-fp*dp)/(db*dp*(fb-fp))
                row['dpre']=dp;row['dblk']=db
            lhs=2*abs(trial);lim1=abs(sp);lim2=3*abs(sbis)-delta
            limit=lim1 if lim1<lim2 else lim2
            accepted=lhs<limit
            row.update(attempt_method=method,trial_step=trial,trial_query=xc+trial,
                       acceptance_left=lhs,acceptance_limit_previous=lim1,acceptance_limit_bracket=lim2,
                       acceptance_right=limit,interpolation_accepted=accepted)
            if accepted:sp=sc;sc=trial;effective=method
            else:sp=sbis;sc=sbis;effective='BISECTION';row['rejected_to_bisection']=True
        else:
            sp=sbis;sc=sbis;effective='BISECTION';row['forced_bisection']=True
        row.update(effective_method=effective,fallback_step=sbis if effective=='BISECTION' else None,
                   fallback_query=xc+sbis if effective=='BISECTION' else None,
                   selected_step=sc,after_proposal=snap())
        xp=xc;fp=fc;ep=ec
        fullstep=abs(sc)>delta
        if fullstep:xc+=sc
        else:xc+=(delta if sbis>0 else -delta)
        fc,ec=callback(xc);calls+=1
        row.update(minimum_step=not fullstep,query=xc,new_function=fc,new_branches=ec,
                   after_step=snap(),stop_reason='CONTINUE')
        rows.append(row)
    raise RuntimeError('Nonconverged instrumented teacher')


def run():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    rows=json.loads((ROOT/'branch_switch_locator_assembly/case_states.json').read_text())
    old=json.loads((ROOT/'branch_switch_locator_assembly/records.json').read_text())
    oldtrace=defaultdict(list)
    for r in read(ROOT/'branch_switch_locator_assembly/dynamic_query_trace.csv'):oldtrace[int(r['case_id'])].append(r)
    a=Audit.__new__(Audit);traces=[];solves=[];differences=[];callback_checks=[]
    for i,r in enumerate(rows):
        a.setup(r['parent'],r['axis']);count=r['count'];scipy_calls=[];reconstructed_calls=[]
        def evaluate(x):
            f=a.formula(float(x),count);iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
            return float(f[1][0]),dict(q_active=iq,aspect_active=ia,aspect_power=power,
                                      competing_q=f[6][0].tolist(),competing_log_aspect=f[7][0].tolist())
        def installed(x):
            f,meta=evaluate(x);scipy_calls.append((float(x),f,meta));return f
        target,info=brentq(installed,r['left'],r['right'],xtol=XTOL,rtol=RTOL,full_output=True,maxiter=MAXITER)
        def instrumented(x):
            f,meta=evaluate(x);reconstructed_calls.append((float(x),f,meta));return f,meta
        candidate,trace,calls,reason=traced_brent(instrumented,r['left'],r['right'],i)
        same_trace=len(scipy_calls)==len(reconstructed_calls) and all(x[0].hex()==y[0].hex() and x[1].hex()==y[1].hex() and x[2]==y[2] for x,y in zip(scipy_calls,reconstructed_calls))
        okay=target==r['teacher_cut']==candidate and calls==info.function_calls and len(trace)==info.iterations and same_trace
        if not okay:
            json_write(OUT/'reconstruction_failure.json',dict(case=r,installed=scipy_calls,reconstructed=reconstructed_calls,trace=trace))
            raise AssertionError(f'Exact teacher reconstruction failed at {i}')
        for tr in trace:
            tr.update(label=r['label'],source=r['source'],switch_type=r['switch_type'],final_return_for_solve=candidate)
        traces.extend(trace)
        solves.append(dict(case_id=i,label=r['label'],source=r['source'],switch_type=r['switch_type'],
             dimension=r['parent']['d'],budget=r['parent']['n'],scan_left=r['left'],scan_right=r['right'],
             saved_candidate=r['teacher_cut'],installed_candidate=target,reconstructed_candidate=candidate,
             iterations=info.iterations,function_calls=calls,stop_reason=reason,exact_full_trace=True,
             first_effective_method=next(tr['effective_method'] for tr in trace if not tr['stop'])))
        # Absolute learned locator used the same initial pair. No locator rerun.
        actual=oldtrace[i][2:2+old[i]['queries']]
        initial_left=int(actual[0]['learned_state']);left=r['left'];right=r['right'];learned=[]
        for q in actual[2:-2]:
            c=float(q['c']);state=int(q['learned_state']);assert state!=3
            learned.append(dict(c=c,L=left,R=right,state=state))
            if state==initial_left:left=c
            else:right=c
        teacher=[tr for tr in trace if not tr['stop']];first=None
        for j,(tt,ll) in enumerate(zip(teacher,learned)):
            if tt['query']!=ll['c']:
                samebracket=tt['bracket_lower']==ll['L'] and tt['bracket_upper']==ll['R']
                cause='INTERPOLATION_PROPOSAL' if tt['effective_method']!='BISECTION' else 'MINIMUM_TOLERANCE_STEP' if tt['minimum_step'] else 'BISECTION_ROUNDING_ORDER' if samebracket else 'BRACKET_HISTORY_STATE'
                first=dict(query_index=j,teacher_iteration=tt['iteration'],operation=cause,
                    teacher_method=tt['effective_method'],teacher_query=tt['query'],learned_query=ll['c'],
                    same_bracket=samebracket,teacher_bracket=[tt['bracket_lower'],tt['bracket_upper']],learned_bracket=[ll['L'],ll['R']]);break
        if first is None:
            first=dict(query_index=min(len(teacher),len(learned)),operation='STOPPING_RULE' if len(teacher)!=len(learned) else 'SAME_QUERY_SEQUENCE',
                       teacher_queries=len(teacher),learned_queries=len(learned))
        differences.append(dict(case_id=i,label=r['label'],**first))
    query_rows=[r for r in traces if not r['stop']]
    counters=dict(attempt_methods=dict(Counter(r['attempt_method'] for r in query_rows)),
        effective_methods=dict(Counter(r['effective_method'] for r in query_rows)),
        forced_bisections=sum(r['forced_bisection'] for r in query_rows),
        rejected_interpolations=sum(r['rejected_to_bisection'] for r in query_rows),
        accepted_interpolations=sum(r['interpolation_accepted'] is True for r in query_rows),
        minimum_tolerance_steps=sum(r['minimum_step'] for r in query_rows),
        bracket_refreshes=sum(r['refresh_bracket'] for r in traces),
        block_promotions=sum(r['promote_block'] for r in traces),
        stop_reasons=dict(Counter(r['stop_reason'] for r in solves)),
        first_divergence_operations=dict(Counter(r['operation'] for r in differences)))
    # Logical state counts of individual operations (continuous state is not finite).
    states=dict(bracket_refresh_inputs=dict(Counter(str((r['entering']['fpre']!=0,r['entering']['fcur']!=0,opposite(r['entering']['fpre'],r['entering']['fcur']))) for r in traces)),
        eligibility_predicates=dict(Counter(str((r['previous_step_large'],r['residual_decreasing'])) for r in query_rows)),
        promotion_outputs=dict(Counter(str(r['promote_block']) for r in traces)),
        stop_predicates=dict(Counter(str((r['stop_zero'],r['stop_width'])) for r in traces)),
        acceptance_outputs=dict(Counter(str(r['interpolation_accepted']) for r in query_rows if r['eligible'])),
        movement_modes=dict(Counter('selected_step' if not r['minimum_step'] else 'positive_delta' if r['sbis']>0 else 'negative_delta' for r in query_rows)))
    flatten=lambda row:{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()}
    write_csv(OUT/'brent_iterations.csv',[flatten(r) for r in traces]);write_csv(OUT/'brent_solves.csv',solves)
    write_csv(OUT/'first_locator_divergence.csv',[flatten(r) for r in differences])
    write_csv(OUT/'iteration_count_distribution.csv',[dict(iterations=k,solves=v) for k,v in sorted(Counter(r['iterations'] for r in solves).items())])
    occurrence=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_all.csv')
    write_csv(OUT/'saved_solve_occurrence_map.csv',[dict(occurrence=int(r['index']),unique_case_id=int(r['unique_id'])) for r in occurrence])
    json_write(OUT/'recursive_k_minus6_trace.json',dict(case=rows[765],solve=solves[765],first_divergence=differences[765],iterations=[r for r in traces if r['case_id']==765]))
    nums=np.array([r['iterations'] for r in solves]);queryfreq=[dict(method=k,count=v,fraction=v/len(query_rows)) for k,v in counters['effective_methods'].items()]
    write_csv(OUT/'proposal_frequencies.csv',queryfreq)
    assert all(sha(Path(p))==v for p,v in protected.items())
    result=dict(status='EXACT_TRACE_RECONSTRUCTION_VERIFIED',scipy_version=scipy.__version__,source_url=SOURCE_URL,
        installed_binary=str(_zeros.__file__),installed_binary_sha256=sha(Path(_zeros.__file__)),
        xtol=XTOL,rtol=RTOL,maxiter=MAXITER,unique_solves=len(solves),saved_unique=774,fresh=44,saved_occurrences=len(occurrence),
        exact_saved_candidate_matches=len(solves),exact_installed_callback_trace_matches=len(solves),
        total_iterations=len(traces),total_proposed_queries=len(query_rows),total_function_calls=sum(r['function_calls'] for r in solves),
        iteration_min=int(nums.min()),iteration_median=float(np.median(nums)),iteration_p99=float(np.quantile(nums,.99)),iteration_max=int(nums.max()),
        frequencies=counters,observed_discrete_states=states,recursive_fixture=solves[765],fixture_first_divergence=differences[765],
        no_training=True,no_checkpoint_modifications=True,no_recursion=True,protected_files=len(protected),protected_hashes_unchanged=True,
        elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    print((OUT/'results.json').read_text(),flush=True)


if __name__=='__main__':run()
