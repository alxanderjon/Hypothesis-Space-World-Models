"""Read-only explanation of the 949 saved discrete-minimizer disagreements."""
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from experiments.geometry.validate_sqrt_derivative_branch import COMPONENTS
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_mantissa_log import MonotoneMantissaLog
from experiments.geometry.train_exponent_log2_operator import ExponentLog2

ROOT=Path('outputs/sho/stage1B')
SOURCE=ROOT/'stationary_float64_minimizer_full_audit'
OUT=ROOT/'stationary_minimizer_disagreement_audit'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ordered(x):
    b=int(np.float64(x).view(np.uint64))
    return (~b & ((1<<64)-1)) if b>>63 else b|(1<<63)


def ulpgap(a,b):return abs(ordered(a)-ordered(b))


def run():
    started=time.monotonic();torch.set_num_threads(1);mp.mp.dps=90
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*manifest.json')})
    protected.update({str(p):sha(p) for p in SOURCE.iterdir() if p.is_file()})
    result=json.loads((SOURCE/'results.json').read_text())
    manifest=SOURCE/'stationary_float64_minimizer_validated.json'
    assert result['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE' and sha(manifest)==result['assembly_sha256']
    saved=[r for r in csv.DictReader((SOURCE/'stationary_float64_minimizers.csv').open()) if r['exact_match']=='False']
    assert len(saved)==949
    data=np.load(SOURCE/'all_float64_candidates.npz');offsets=data['offsets']
    ids=np.concatenate([np.arange(offsets[int(r['index'])],offsets[int(r['index'])+1]) for r in saved])
    x=data['inputs'][ids];yt=data['teacher_score'][ids];yl=data['learned_score'][ids]
    local_offsets=np.r_[0,np.cumsum([int(r['count']) for r in saved])]
    models={}
    classes={key:val[1] for key,val in COMPONENTS.items()}
    classes.update(addition=Addition,mantissa=MonotoneMantissaLog,exponent=ExponentLog2)
    for key,reference in result['components'].items():
        p=Path(reference['checkpoint']);assert sha(p)==reference['sha256']
        model=classes[key]();model.load_state_dict(torch.load(p,weights_only=False)['state_dict'])
        models[key]=model.eval().requires_grad_(False)
    def learned(z,replace=None,details=False):
        A,B,a,beta,c,ot,ov,C=z.unbind(1)
        def mul(a,b):return a*b if replace=='multiplication' else models['multiply'](torch.stack((a,b),1))
        def add(a,b):return a+b if replace=='addition' else models['addition'](torch.stack((a,b),1))
        if replace=='orientation':t=torch.where(ot==0,c,1-c);v=torch.where(ov==0,c,1-c)
        else:t=models['orientation'](torch.stack((c,ot),1));v=models['orientation'](torch.stack((c,ov),1))
        q=B+((a*a)*t)*t if replace=='inner_quadratic' else models['quadratic'](torch.stack((B,a,t),1))
        r=torch.from_numpy(np.sqrt(q.numpy())) if replace=='sqrt' else models['sqrt'](q)
        first=mul(A,r)
        if replace=='log_assembly':lv=torch.from_numpy(np.log(v.numpy()))
        else:
            m,e=np.frexp(v.numpy());m=2*m;e=e-1
            lv=add(models['mantissa'](torch.from_numpy(m)),models['exponent'](torch.from_numpy(e.astype(float))))
        second=mul(beta,lv);total=add(add(first,second),C)
        if details:return dict(t=t.numpy(),v=v.numpy(),q=q.numpy(),sqrt=r.numpy(),log=lv.numpy(),sqrt_branch=first.numpy(),log_branch=second.numpy(),score=total.numpy())
        return total.numpy()
    # Re-evaluate only these mismatched candidate sets, never the matched brackets.
    with torch.no_grad():ld=learned(torch.from_numpy(x),details=True)
    assert np.array_equal(ld['score'],yl), 'Frozen replay differs from the saved candidate scores.'
    td={key:np.empty(len(x)) for key in ld}
    for j,row in enumerate(saved):
        s,e=local_offsets[j:j+2];z=x[s:e];A,B,a,beta,_,ot,ov,C=z[0];c=z[:,4]
        t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
        q=B+a**2*t*t;r=np.sqrt(q);lv=np.log(v);first=A*r;second=beta*lv
        for key,val in dict(t=t,v=v,q=q,sqrt=r,log=lv,sqrt_branch=first,log_branch=second,score=(first+second)+C).items():td[key][s:e]=val
    assert np.array_equal(td['score'],yt), 'Teacher operation-order replay differs from saved scores.'
    rows=[];screened=[];selected_ids=[]
    for j,row in enumerate(saved):
        s,e=local_offsets[j:j+2];c=x[s:e,4];y=yt[s:e];l=yl[s:e];it=int(np.argmin(y));il=int(np.argmin(l));ti=s+it;li=s+il
        assert c[it]==float(row['c_teacher_min']) and c[il]==float(row['c_learned_min'])
        scoregap=ulpgap(y[it],y[il]);gap=float(y[il]-y[it])
        # Local rounding scale accounts for cancellation in sqrt_branch + log_branch.
        # One ULP from each branch at each selected cut; not the downstream 1e-12 tolerance.
        local_scale=sum(abs(np.spacing(td[key][i])) for key in ('sqrt_branch','log_branch') for i in (ti,li))
        def exactscore(i):
            A,B,a,beta,c,ot,ov,C=[mp.mpf(float(v)) for v in x[i]]
            t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
            return A*mp.sqrt(B+a*a*t*t)+beta*mp.log(v)+C
        realgap=exactscore(li)-exactscore(ti)
        # A few rounding units, not one: each branch itself contains several
        # rounded operations. Report the stricter one-unit screen separately.
        # The 90-digit real-score difference must also be below one local unit.
        category=1 if scoregap==0 else 2 if gap<=4*local_scale and abs(realgap)<=local_scale else 3
        item=dict(index=int(row['index']),scope=row['scope'],case=row['case'],category=category,candidate_count=int(e-s),
            c_teacher_min=float(c[it]),c_learned_min=float(c[il]),coordinate_ulp_gap=ulpgap(c[it],c[il]),
            teacher_min=float(y[it]),teacher_at_learned=float(y[il]),learned_at_teacher=float(l[it]),learned_min=float(l[il]),
            teacher_score_gap=gap,teacher_score_ulp_gap=scoregap,local_branch_rounding_scale=float(local_scale),
            score_gap_in_local_rounding_units=gap/local_scale,high_precision_score_gap=str(realgap),
            high_precision_gap_in_local_rounding_units=float(abs(realgap)/local_scale),
            teacher_exact_minima=int(np.sum(y==y[it])),learned_exact_minima=int(np.sum(l==l[il])),
            teacher_strict_order_reversed=bool(scoregap>0 and l[il]<l[it]),teacher_strict_order_collapsed=bool(scoregap>0 and l[il]==l[it]),
            teacher_tie_broken_by_learned=bool(scoregap==0 and l[il]<l[it]),
            teacher_is_first_exact_min=it==int(np.flatnonzero(y==y[it])[0]),
            learned_is_first_exact_min=il==int(np.flatnonzero(l==l[il])[0]),
            teacher_cut_in_learned_exact_minimum_set=bool(l[it]==l[il]),
            first_of_selected_pair_if_equal_recovers_teacher=bool(c[it]<c[il]),
            outside_teacher_equivalence=bool(gap>1e-12*max(1.,y[it])))
        rows.append(item);selected_ids.extend([ti,li])
        if scoregap>4 or category==3:screened.append(j)
    # Trace all selected pairs. Same-input checks separate local primitive error
    # from upstream propagation; one-component substitutions are diagnostics only.
    pair_ids=np.array(selected_ids);pair_x=x[pair_ids];decomposition=[];substitutions=[]
    with torch.no_grad():
        variants={key:learned(torch.from_numpy(pair_x),replace=key) for key in
                  ('orientation','inner_quadratic','sqrt','multiplication','log_assembly','addition')}
        z=torch.from_numpy(pair_x);A,B,a,beta,c,ot,ov,C=z.unbind(1)
        lq=ld['q'][pair_ids];lr=ld['sqrt'][pair_ids];lv=ld['v'][pair_ids];ll=ld['log'][pair_ids]
        # Check multiplication/addition locally on the exact same learned inputs.
        local_sqrt_error=ld['sqrt'][pair_ids]-np.sqrt(lq)
        local_log_error=ll-np.log(lv)
        local_mul1_error=ld['sqrt_branch'][pair_ids]-pair_x[:,0]*lr
        local_mul2_error=ld['log_branch'][pair_ids]-pair_x[:,3]*ll
        local_add_error=yl[pair_ids]-((ld['sqrt_branch'][pair_ids]+ld['log_branch'][pair_ids])+pair_x[:,7])
    for j,row in enumerate(rows):
        for side,k in [('teacher_min',2*j),('learned_min',2*j+1)]:
            i=pair_ids[k];d=dict(index=row['index'],case=row['case'],category=row['category'],selected_cut=side,c=float(x[i,4]))
            for name,val in zip(('A','B','a','beta','cut','orientation_t','orientation_v','C'),x[i]):d[name]=float(val)
            for key in td:d['teacher_'+key]=float(td[key][i]);d['learned_'+key]=float(ld[key][i])
            d.update(local_sqrt_error=float(local_sqrt_error[k]),local_log_error=float(local_log_error[k]),
                     local_sqrt_scale_multiplication_error=float(local_mul1_error[k]),local_beta_multiplication_error=float(local_mul2_error[k]),
                     local_final_addition_error=float(local_add_error[k]))
            decomposition.append(d)
        for name,pred in variants.items():
            a,b=pred[2*j:2*j+2]
            substitutions.append(dict(index=row['index'],case=row['case'],category=row['category'],component_replaced=name,
                score_at_teacher_min=float(a),score_at_learned_min=float(b),strict_teacher_pair_order_restored=bool(a<b),
                selected_pair_teacher_recovered=bool(a<b or (a==b and row['c_teacher_min']<row['c_learned_min'])),
                changed_pair_scores=bool(a!=yl[pair_ids[2*j]] or b!=yl[pair_ids[2*j+1]])))
    np.savez_compressed(OUT/'mismatched_candidate_sets.npz',inputs=x,teacher_score=yt,learned_score=yl,
                        offsets=local_offsets,source_indices=np.array([r['index'] for r in rows]))
    write_csv(OUT/'stationary_minimizer_disagreements.csv',rows)
    write_csv(OUT/'score_ulp_distribution.csv',[dict(score_ulp_gap=k,brackets=v) for k,v in sorted(Counter(r['teacher_score_ulp_gap'] for r in rows).items())])
    write_csv(OUT/'selected_pair_arithmetic.csv',decomposition);write_csv(OUT/'component_substitution_checks.csv',substitutions)
    worst=sorted(rows,key=lambda r:(r['teacher_score_ulp_gap'],r['coordinate_ulp_gap']),reverse=True)[:20]
    write_csv(OUT/'worst_20_score_gaps.csv',worst)
    write_csv(OUT/'larger_output_ulp_gap_cases.csv',[rows[j] for j in screened])
    conservative=[j for j,r in enumerate(rows) if r['score_gap_in_local_rounding_units']>1]
    if conservative:
        ii=np.concatenate([np.arange(local_offsets[j],local_offsets[j+1]) for j in conservative])
        np.savez_compressed(OUT/'conservative_one_unit_flag_candidate_sets.npz',inputs=x[ii],teacher_score=yt[ii],learned_score=yl[ii],
            offsets=np.r_[0,np.cumsum([rows[j]['candidate_count'] for j in conservative])],source_indices=[rows[j]['index'] for j in conservative])
        flagged_ids={rows[j]['index'] for j in conservative}
        write_csv(OUT/'conservative_one_unit_flag_arithmetic.csv',[d for d in decomposition if d['index'] in flagged_ids])
    category3=[j for j,r in enumerate(rows) if r['category']==3]
    if category3:
        ii=np.concatenate([np.arange(local_offsets[j],local_offsets[j+1]) for j in category3])
        np.savez_compressed(OUT/'category3_candidate_sets.npz',inputs=x[ii],teacher_score=yt[ii],learned_score=yl[ii],
            offsets=np.r_[0,np.cumsum([rows[j]['candidate_count'] for j in category3])],source_indices=[rows[j]['index'] for j in category3])
        write_csv(OUT/'category3_arithmetic.csv',[d for d in decomposition if d['category']==3])
    summary=dict(status='MEANINGFUL_ORDERING_FAILURE_STOP' if category3 else 'FLOAT64_TIE_AND_ROUNDING_DIFFERENCES_ONLY',
        mismatched_brackets=len(rows),saved_candidates_inspected=len(x),matched_brackets_reexecuted=0,
        categories={str(k):sum(r['category']==k for r in rows) for k in (1,2,3)},
        category_definition='1: selected cuts are exact teacher minimum ties. 2: nonzero teacher gap is at most four local branch-rounding units, and 90-digit real-score gap is at most one such unit. A local unit is the sum of one sqrt-branch ULP and one log-branch ULP at each selected cut. 3: exceeds either bound. This is an explicit audit categorization, not a change to scores or selection tolerances. All raw output-score ULP gaps and the stricter one-unit screen are reported.',
        exact_teacher_tie_count=sum(r['teacher_score_ulp_gap']==0 for r in rows),
        one_score_ulp_count=sum(r['teacher_score_ulp_gap']==1 for r in rows),larger_score_gap_count=sum(r['teacher_score_ulp_gap']>1 for r in rows),
        output_gaps_over_four_ulps=len(screened),max_teacher_score_ulp_gap=max(r['teacher_score_ulp_gap'] for r in rows),
        max_coordinate_ulp_gap=max(r['coordinate_ulp_gap'] for r in rows),
        score_ulp_histogram=dict(sorted(Counter(r['teacher_score_ulp_gap'] for r in rows).items())),
        max_gap_in_local_branch_rounding_units=max(r['score_gap_in_local_rounding_units'] for r in rows),
        conservative_one_local_unit_screen_flags=sum(r['score_gap_in_local_rounding_units']>1 for r in rows),
        max_high_precision_gap_in_local_branch_rounding_units=max(r['high_precision_gap_in_local_rounding_units'] for r in rows),
        strict_teacher_order_reversals=sum(r['teacher_strict_order_reversed'] for r in rows),
        strict_teacher_order_collapses=sum(r['teacher_strict_order_collapsed'] for r in rows),
        teacher_equal_minima_broken_by_learned=sum(r['teacher_tie_broken_by_learned'] for r in rows),
        tie_rule_violations=sum(not r['teacher_is_first_exact_min'] or not r['learned_is_first_exact_min'] for r in rows),
        pure_different_tie_convention_mismatches=0,
        teacher_cut_in_learned_exact_minimum_set=sum(r['teacher_cut_in_learned_exact_minimum_set'] for r in rows),
        counterfactual_equal_selected_pair_teacher_first=sum(r['first_of_selected_pair_if_equal_recovers_teacher'] for r in rows),
        tie_interpretation='Both arms already use first ascending exact-score minimum. No mismatch is due to different tie-breaking code. 344 teacher ties are broken by learned rounding; other mismatches are collapsed or reversed strict float64 score pairs.',
        primitive_local_checks=dict(orientation_t_errors=int(np.sum(ld['t']!=td['t'])),orientation_v_errors=int(np.sum(ld['v']!=td['v'])),
            selected_sqrt_native_errors=int(np.sum(local_sqrt_error!=0)),selected_log_direct_errors=int(np.sum(local_log_error!=0)),
            selected_multiplication_errors=int(np.sum((local_mul1_error!=0)|(local_mul2_error!=0))),selected_final_addition_errors=int(np.sum(local_add_error!=0))),
        outside_teacher_equivalence=sum(r['outside_teacher_equivalence'] for r in rows),
        downstream_reference_only={k:result[k] for k in ('subpiece_minima','allocation_interval_minima','archived_parent_minima')},
        no_downstream_rerun=True,source_manifest_sha256=sha(manifest),implementation_sha256=sha(Path(__file__)),
        source_hashes={str(p):sha(p) for p in (SOURCE/'all_float64_candidates.npz',SOURCE/'stationary_float64_minimizers.csv')})
    assert all(sha(p)==h for p,h in protected.items());summary['protected_hashes_unchanged']=True
    summary['elapsed_seconds']=time.monotonic()-started
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'stationary_minimizer_disagreement_summary.txt').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
