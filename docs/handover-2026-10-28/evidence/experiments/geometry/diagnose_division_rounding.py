"""Read-only model diagnosis; exact-rational rounding tests, no learned correction."""
import csv
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import torch

from experiments.geometry import train_division_operator as src
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_reciprocal_operator import PositivePower
from experiments.geometry.train_subtraction_operator import Subtraction

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'division_rounding_diagnosis'
REF=ROOT/'division_operator_refinement'


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bits(x):return int(np.float64(x).view('u8'))


def ordered(x):
    b=bits(x);return (~b & ((1<<64)-1)) if b>>63 else b|(1<<63)


def power2(e):return F(2)**int(e)


def decode(x,prefix):
    b=bits(x);eb=(b>>52)&2047;sig=b&((1<<52)-1)
    return {prefix+'_decimal':float(x),prefix+'_hex':float(x).hex(),prefix+'_uint64':str(b),
        prefix+'_bits_hex':f'{b:016x}',prefix+'_sign':b>>63,prefix+'_exponent_field':eb,
        prefix+'_unbiased_exponent':eb-1023 if eb else -1022,
        prefix+'_integer_significand':str(sig+(1<<52) if eb else sig),
        prefix+'_normalized_mantissa':math.frexp(float(x))[0],
        prefix+'_subnormal':bool(eb==0 and sig!=0)}


def product_low(a,b,product):
    ca=134217729.*a;ah=ca-(ca-a);al=a-ah
    cb=134217729.*b;bh=cb-(cb-b);bl=b-bh
    return ((ah*bh-product)+ah*bl+al*bh)+al*bl


def diagnostic_rounding_test(q,residual,y):
    """Oracle-only test: exact residual vs half an output ULP, ties-to-even.

    q,y are positive magnitudes. This is not an inference operator or checkpoint.
    Caller explicitly chooses exact or information-losing residual for comparison.
    """
    if not residual:return q,0,F(0),False
    direction=1 if residual>0 else -1
    neighbor=float(np.nextafter(q,np.inf if direction>0 else -np.inf))
    if not math.isfinite(neighbor):return q,0,F(0),False
    half=abs(F(neighbor)-F(q))*y/2
    tie=abs(residual)==half
    move=abs(residual)>half or (tie and bits(q)&1)
    return (neighbor if move else q),direction if move else 0,half,tie


def grs(ratio):
    e=ratio.numerator.bit_length()-ratio.denominator.bit_length()
    if ratio<power2(e):e-=1
    quantum=power2(max(e-52,-1074));scaled=ratio/quantum
    truncated=scaled.numerator//scaled.denominator;fraction=scaled-truncated
    guard=int(fraction*2);rest=fraction*2-guard;roundbit=int(rest*2);sticky=bool(rest*2-roundbit)
    return dict(output_quantum_exponent=max(e-52,-1074),truncated_significand=str(truncated),
        guard_bit=guard,round_bit=roundbit,sticky_bit=sticky,truncated_lsb=truncated&1,
        grs_true_halfway=bool(fraction==F(1,2)),round_up=bool(guard and (roundbit or sticky or truncated&1)))


def run():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    protected={str(p):src.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):src.sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('locked','manifest','validated'))})
    for folder in (src.OUT,REF):
        protected.update({str(p):src.sha(p) for p in folder.iterdir() if p.is_file()})
    mult,mm=src.load(QuadraticArithmetic,'multiplication_operator_smoke')
    rec,rm=src.load(PositivePower,'reciprocal_required_domain_audit')
    sub,sm=src.load(Subtraction,'subtraction_operator_training','subtraction_operator_locked.json')
    config=json.loads((REF/'results.json').read_text());cp=Path(config['checkpoint']);assert src.sha(cp)==config['sha256']
    model=src.Division(mult,rec);model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    checkpoint_info={name:{k:m.get(k) for k in ('status','checkpoint','sha256','passed')} for name,m in
                     [('multiplication',mm),('division_diagnostic',config),('subtraction',sm),('reciprocal',rm)]}
    assert mm['status']=='FROZEN_CANDIDATE' and not config['passed']
    (OUT/'checkpoint_verification.json').write_text(json.dumps(checkpoint_info,indent=2)+'\n')
    z=np.load(REF/'refinement_inputs.npz');x=z['inputs'];teacher=z['teacher'];before=z['before'];features=z['features'];exponent=z['exponents'];n=len(x)
    pred=model.predict(x);assert np.array_equal(pred.view('u8'),before.view('u8'))
    brent_n=45582;old_n=config['prior_validation_cases'];bad=pred.view('u8')!=teacher.view('u8')
    meta=read(src.OUT/'operand_metadata.csv');provenance=read(src.OUT/'failure_split_provenance.csv')
    write('Brent_failure_reconciliation.csv',provenance)
    groups=np.full(n,'original_synthetic',dtype=object);groups[:brent_n]='Brent';groups[old_n:]='nextafter_neighborhood'
    for i in range(brent_n):groups[i]='Brent_'+meta[i]['group']
    metrics=[]
    def multiply(pairs,name,which=None):
        with torch.no_grad():p=mult(torch.from_numpy(pairs)).numpy()
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):t=pairs[:,0]*pairs[:,1]
        ids=np.arange(len(p)) if which is None else which
        metrics.append(src.stats(t[ids],p[ids],name));return p
    mx,ex=np.frexp(abs(x[:,0]));my,ey=np.frexp(abs(x[:,1]))
    with torch.no_grad():rr=rec(torch.from_numpy(my)).numpy()
    q0=multiply(np.column_stack((mx,rr)),'all_mantissa_quotient_products')
    prod0=multiply(np.column_stack((q0,my)),'all_initial_residual_products')
    low0=product_low(q0,my,prod0);res0=(mx-prod0)-low0
    corr=multiply(np.column_stack((res0,rr)),'all_residual_corrections')
    # Algebraic counterfactual ONLY: no parameter assignment or checkpoint save.
    scale=float(model.feature_scale)
    with np.errstate(over='ignore',under='ignore'):
        cf=np.copysign(np.ldexp(features[:,0]+scale*features[:,1],exponent),x[:,0]*np.sign(x[:,1]))
        direct_mantissa=mx/my
        decomp=np.copysign(np.ldexp(direct_mantissa,exponent),x[:,0]*np.sign(x[:,1]))
    for name,ids in [('Brent',np.arange(brent_n)),('original_synthetic',np.arange(brent_n,old_n)),('neighbors',np.arange(old_n,n))]:
        metrics.append(src.stats(teacher[ids],cf[ids],name+'_counterfactual_residual_weight_equals_feature_scale'))
        metrics.append(src.stats(teacher[ids],decomp[ids],name+'_native_mantissa_then_ldexp'))
    finite=np.isfinite(pred)&np.isfinite(teacher)
    ids=np.flatnonzero(finite);q=abs(pred[ids]);xp=abs(x[ids,0]);yp=abs(x[ids,1])
    product=multiply(np.column_stack((q,yp)),'finite_abs_final_quotient_times_denominator')
    rounded_residual=sub.predict(np.column_stack((xp,product)))
    with np.errstate(invalid='ignore'):native_res=xp-product
    metrics.append(src.stats(native_res,rounded_residual,'finite_abs_residual_subtraction'))
    # Required Brent signed operands are tested separately, including signs.
    bproduct=multiply(np.column_stack((pred[:brent_n],x[:brent_n,1])),'Brent_signed_final_quotient_times_denominator')
    bres=sub.predict(np.column_stack((x[:brent_n,0],bproduct)))
    metrics.append(src.stats(x[:brent_n,0]-bproduct,bres,'Brent_signed_residual_subtraction'))
    # Error-free normalized product expansion of the FINAL output candidate.
    with np.errstate(over='ignore',under='ignore'):qn=np.ldexp(q,-exponent[ids])
    pn=multiply(np.column_stack((qn,my[ids])),'normalized_final_quotient_times_denominator')
    pl=product_low(qn,my[ids],pn);rh=mx[ids]-pn;rl=-pl;rc=rh+rl
    # Match every failure to a nearby exact operand pair in its +/-8 grid.
    nm=read(REF/'neighborhood_metadata.csv');matches={};neighbors=defaultdict(list)
    for j,row in enumerate(nm):neighbors[int(row['pair_id'])].append((old_n+j,int(row['x_offset']),int(row['y_offset'])))
    originals=read(src.OUT/'division_failures.csv')+read(src.OUT/'synthetic_failures.csv')
    pair_list=list(dict.fromkeys((float(row['numerator']).hex(),float(row['denominator']).hex()) for row in originals))
    pair_map={k:i for i,k in enumerate(pair_list)}
    for i in np.flatnonzero(bad):
        if i>=old_n:
            row=nm[i-old_n];pid=int(row['pair_id']);xi=int(row['x_offset']);yi=int(row['y_offset'])
        else:pid=pair_map[(float(x[i,0]).hex(),float(x[i,1]).hex())];xi=yi=0
        choices=[(abs(a-xi)+abs(b-yi),j) for j,a,b in neighbors[pid] if not bad[j]]
        if choices:matches[int(i)]=min(choices)[1]
    selected=set(np.flatnonzero(bad).tolist())|set(matches.values())
    counts=defaultdict(Counter);detail={};test_failures=[];rounding_rows=[]
    for j,i in enumerate(ids):
        X,Y,Q=F(float(xp[j])),F(float(yp[j])),F(float(q[j]));exact=X/Y
        true_res=X-Q*Y;local_exp=int(exponent[i]);scale_factor=power2(int(ex[i]))
        exact_norm=F(float(mx[i]))-F(float(qn[j]))*F(float(my[i]))
        expansion=F(float(rh[j]))+F(float(rl[j]));assert expansion==exact_norm
        assert exact_norm*scale_factor==true_res
        exact_choice,action,half,tie=diagnostic_rounding_test(float(q[j]),true_res,Y)
        # Use exact thresholds for all tests to isolate residual information loss.
        naive_choice=diagnostic_rounding_test(float(q[j]),F(float(rounded_residual[j])),Y)[0] if np.isfinite(rounded_residual[j]) else float('nan')
        fma_res=float(true_res) # correctly rounded simulated fused residual, oracle only
        fma_choice=diagnostic_rounding_test(float(q[j]),F(fma_res),Y)[0]
        comp_choice=diagnostic_rounding_test(float(q[j]),F(float(rc[j]))*scale_factor,Y)[0]
        nfma=float(exact_norm)
        nfma_choice=diagnostic_rounding_test(float(q[j]),F(nfma)*scale_factor,Y)[0]
        # Practical normalized float64 half-spacing test, separately from the
        # exact-rational oracle: assess whether rounding this threshold matters.
        comp_float_choice=float(q[j])
        if rc[j]!=0:
            direction=1 if rc[j]>0 else -1
            neighbor=float(np.nextafter(q[j],np.inf if direction>0 else -np.inf))
            if math.isfinite(neighbor):
                gap=float(np.ldexp(abs(neighbor-q[j]),-local_exp))
                limit=(gap*my[i])*.5
                if abs(rc[j])>limit or (abs(rc[j])==limit and bits(q[j])&1):comp_float_choice=neighbor
        sign_only=float(np.nextafter(q[j],np.inf if true_res>0 else -np.inf)) if true_res else float(q[j])
        truth=float(abs(teacher[i]));native_ok=float(exact)==truth
        restored=np.copysign(exact_choice,-1. if np.signbit(x[i,0])^np.signbit(x[i,1]) else 1.)
        assert bits(restored)==bits(teacher[i])
        outcomes=dict(exact_residual_errors=exact_choice!=truth,rounded_multiply_subtract_errors=naive_choice!=truth,
            fused_residual_errors=fma_choice!=truth,normalized_compensated_residual_errors=comp_choice!=truth,
            normalized_fused_residual_errors=nfma_choice!=truth,sign_only_neighbor_errors=sign_only!=truth,
            normalized_compensated_rounded_threshold_errors=comp_float_choice!=truth,
            residual_erased=(rounded_residual[j]==0 and true_res!=0),native_rational_mismatch=not native_ok,
            true_halfway=tie,model_errors=bool(bad[i]),exact_residual_moves=action!=0,
            true_residual_expansion_exact=expansion==exact_norm)
        labels=['all_finite',str(groups[i])]
        if i<brent_n:labels.append('all_Brent')
        if bad[i]:labels.append('model_failures')
        for label in labels:counts[label]['cases']+=1;counts[label].update({k:int(v) for k,v in outcomes.items()})
        if any(outcomes[k] for k in ('exact_residual_errors','fused_residual_errors','normalized_compensated_residual_errors','normalized_fused_residual_errors')):
            test_failures.append(dict(index=int(i),group=groups[i],**outcomes,x=float(x[i,0]),y=float(x[i,1]),teacher=float(teacher[i]),model=float(pred[i])))
        if i in selected:
            # Exact distance to the midpoint separating teacher and wrong output.
            midpoint=(F(abs(teacher[i]))+Q)/2 if bad[i] else Q
            gap=(exact-midpoint)
            out=dict(index=int(i),group=groups[i],role='failure' if bad[i] else 'matched_exact',
                case_id=meta[i]['case_id'] if i<brent_n else '',iteration=meta[i]['iteration'] if i<brent_n else '',
                operation=meta[i]['operation'] if i<brent_n else '',matched_exact_index=matches.get(int(i),''),
                signed_ULP_error=ordered(pred[i])-ordered(teacher[i]),
                **decode(x[i,0],'x'),**decode(x[i,1],'y'),**decode(teacher[i],'native'),**decode(pred[i],'model'),
                q0=float(q0[i]),reciprocal_mantissa=float(rr[i]),initial_product_low=float(low0[i]),
                initial_residual=float(res0[i]),correction=float(corr[i]),feature_scale=scale,
                learned_weight_ratio=float(model.weights[1])/scale,
                learned_mantissa=float(model.weights[0])*features[i,0]+float(model.weights[1])*features[i,1],
                native_normalized_quotient=float(direct_mantissa[i]),exponent_difference=int(exponent[i]),
                native_mantissa_then_ldexp=float(decomp[i]),counterfactual_equal_scale=float(cf[i]),
                rounded_abs_product=float(product[j]),rounded_residual=float(rounded_residual[j]),
                residual_convention='magnitude residual |x|-|model|*|y|',
                signed_naive_residual=float(x[i,0]-pred[i]*x[i,1]),
                signed_exact_residual_rounded=float(-true_res if np.signbit(x[i,0]) else true_res),
                exact_residual_numerator=str(true_res.numerator),exact_residual_denominator=str(true_res.denominator),
                exact_residual_rounded=fma_res,normalized_residual_high=float(rh[j]),normalized_residual_low=float(rl[j]),
                rounded_compensated_residual=float(rc[j]),residual_halfspacing_ratio=float(abs(true_res)/half) if half else 0.,
                mathematical_side_of_teacher_model_midpoint=(gap>0)-(gap<0),
                midpoint_gap_numerator=str(gap.numerator),midpoint_gap_denominator=str(gap.denominator),
                **(grs(exact) if exact else {}),**outcomes)
            detail[int(i)]=out
        if j and j%30000==0:print('exact rational cases',j,flush=True)
    # All selected records here are nonzero finite; maintain one CSV schema.
    write('failure_and_matched_case_details.csv',list(detail.values()))
    comparisons=[]
    for a,b in matches.items():
        aa,bb=detail[a],detail[b]
        comparisons.append(dict(failure_index=a,matched_exact_index=b,x_ULP_distance=abs(ordered(x[a,0])-ordered(x[b,0])),
            y_ULP_distance=abs(ordered(x[a,1])-ordered(x[b,1])),failure_output_ULPs=aa['signed_ULP_error'],
            failure_residual_halfspacing_ratio=aa['residual_halfspacing_ratio'],exact_case_residual_halfspacing_ratio=bb['residual_halfspacing_ratio'],
            failure_rounded_residual=aa['rounded_residual'],exact_case_rounded_residual=bb['rounded_residual'],
            failure_true_halfway=aa['true_halfway'],exact_case_true_halfway=bb['true_halfway'],
            failure_sticky=aa['sticky_bit'],exact_case_sticky=bb['sticky_bit']))
    write('matched_pair_comparisons.csv',comparisons)
    write('component_and_counterfactual_metrics.csv',metrics)
    write('residual_information_tests.csv',[dict(group=k,**v) for k,v in counts.items()])
    if test_failures:write('residual_test_failures.csv',test_failures)
    # Explicit true halfway controls, absent from the original failures. These
    # exercise subnormal nearest-even rounding without training any operation.
    tiny=np.nextafter(0.,1.)
    tx=np.array([(sx*odd*tiny,sy*den) for odd in (1,3,5,7,9,15,17,31)
                 for den in (2.,4.) for sx in (-1.,1.) for sy in (-1.,1.)])
    ty=src.teacher(tx);tp=model.predict(tx);tie_rows=[]
    for (a,b),truth,p in zip(tx,ty,tp):
        exact=abs(F(a)/F(b));res=abs(F(a))-abs(F(p))*abs(F(b))
        oracle=diagnostic_rounding_test(abs(float(p)),res,abs(F(b)))[0]
        oracle=np.copysign(oracle,-1. if np.signbit(a)^np.signbit(b) else 1.)
        assert bits(oracle)==bits(truth)
        tie_rows.append(dict(x=float(a),y=float(b),teacher=float(truth),model=float(p),
            native_hex=float(truth).hex(),model_hex=float(p).hex(),oracle_hex=float(oracle).hex(),
            model_exact=bits(truth)==bits(p),**grs(exact)))
    write('explicit_halfway_controls.csv',tie_rows)
    unchanged=all(src.sha(Path(p))==h for p,h in protected.items());assert unchanged
    result=dict(status='DIAGNOSIS_ONLY',checkpoint_verification=checkpoint_info,cases=n,Brent_cases=brent_n,
        original_validation=old_n,nextafter_cases=n-old_n,model_failures=int(bad.sum()),
        detailed_failure_cases=int(bad.sum()),matched_failures=len(matches),unique_matched_exact=len(set(matches.values())),
        metrics=metrics,residual_tests={k:dict(v) for k,v in counts.items()},
        required_domain_note='Counterfactual weight evaluation is algebraic diagnosis only; no parameters assigned or saved.',
        checkpoints_unchanged=unchanged,protected_files=len(protected),training=False,architecture_changed=False,
        explicit_halfway_control_cases=len(tie_rows),true_halfway_controls=sum(v['grs_true_halfway'] for v in tie_rows),
        halfway_control_model_errors=sum(not v['model_exact'] for v in tie_rows),halfway_control_oracle_errors=0,
        Brent_assembly=False,recursion=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
