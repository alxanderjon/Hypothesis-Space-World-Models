"""Validate only the residual expansion; no half-ULP test or division router."""
import csv
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch

from experiments.geometry import train_division_operator as source
from experiments.geometry.compensated_division_residual import CompensatedDivisionResidual
from experiments.geometry.diagnose_division_rounding import product_low
from experiments.geometry.train_reciprocal_operator import PositivePower

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'compensated_division_residual_validation'


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bits(x):return np.asarray(x,dtype=np.float64).view('u8')


def run():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    protected={str(p):source.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):source.sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('locked','manifest','validated'))})
    for folder in ('division_operator_training','division_operator_refinement','division_rounding_diagnosis','product_tail_primitive_screen'):
        protected.update({str(p):source.sha(p) for p in (ROOT/folder).iterdir() if p.is_file()})
    assembly=CompensatedDivisionResidual()
    rec,_=source.load(PositivePower,'reciprocal_required_domain_audit')
    dc=json.loads((ROOT/'division_operator_refinement/results.json').read_text());assert source.sha(Path(dc['checkpoint']))==dc['sha256']
    division=source.Division(assembly.multiply,rec)
    division.load_state_dict(torch.load(dc['checkpoint'],weights_only=False)['state_dict']);division.eval().requires_grad_(False)
    z=np.load(ROOT/'division_operator_refinement/refinement_inputs.npz')
    xy=z['inputs'];q=division.predict(xy);assert np.array_equal(bits(q),bits(z['before']))
    ids=np.flatnonzero(np.isfinite(q)&np.isfinite(z['teacher']));assert len(ids)==119210
    x=xy[ids,0];y=xy[ids,1];q=q[ids];original_n=len(ids)
    rng=np.random.default_rng(928202648);chunks=[];qs=[];count=0
    while count<8192:
        pairs=np.ldexp(rng.uniform(.5,1.,(8192,2)),rng.integers(-1073,1025,(8192,2)))*rng.choice([-1.,1.],(8192,2))
        with np.errstate(over='ignore',under='ignore'):t=pairs[:,0]/pairs[:,1]
        pred=division.predict(pairs);mask=np.isfinite(t)&np.isfinite(pred)
        chunks.append(pairs[mask]);qs.append(pred[mask]);count+=int(mask.sum())
    fresh=np.concatenate(chunks)[:8192];fq=np.concatenate(qs)[:8192]
    # Nonzero normal quotient perturbations test approximate, not necessarily
    # correctly rounded quotients; signs and factor-two proximity are preserved.
    for i in range(0,len(fq),4):
        if abs(fq[i])>=np.finfo(float).tiny and abs(fq[i])<np.finfo(float).max:
            fq[i]=np.nextafter(fq[i],np.inf if i%8 else -np.inf)
    tiny=np.nextafter(0.,1.);maxv=np.finfo(float).max
    controls=np.array([(a,b) for a in (0.,-0.,tiny,-tiny,np.finfo(float).tiny,-np.finfo(float).tiny,1.,-1.,maxv,-maxv)
                       for b in (tiny,-tiny,1.,-1.,maxv,-maxv)])
    cq=division.predict(controls)
    with np.errstate(over='ignore',under='ignore'):ct=controls[:,0]/controls[:,1]
    valid=np.isfinite(cq)&np.isfinite(ct);controls=controls[valid];cq=cq[valid]
    x=np.concatenate((x,fresh[:,0],controls[:,0]));y=np.concatenate((y,fresh[:,1],controls[:,1]));q=np.concatenate((q,fq,cq))
    exp=assembly.construct(x,y,q);rounded=assembly.rounded_normalized(exp)
    mx,ex=np.frexp(abs(x));my,ey=np.frexp(abs(y));a=np.ldexp(abs(q),-(ex-ey))
    native_ph=a*my;native_pl=product_low(a,my,native_ph)
    native_hi=mx-native_ph;native_lo=-native_pl;native_rounded=native_hi+native_lo
    got_ph,got_pl=assembly.product_pair(a,my)
    # Replay all independent product-tail pairs from the preceding screen too.
    oldtails=np.load(ROOT/'product_tail_primitive_screen/isolated_product_tail_targets.npz')
    ph,pl=assembly.product_pair(oldtails['a'],oldtails['b'])
    tail_replay_errors=int(((bits(ph)!=bits(oldtails['p_hi']))|(bits(pl)!=bits(oldtails['target_tail']))).sum())
    n=len(x);errors=np.zeros(n,dtype=bool);normal_errors=errors.copy();high_errors=errors.copy();tail_errors=errors.copy()
    signed_exact_zero=np.zeros(n,dtype=bool);rounded_losses=np.zeros(n,dtype=bool)
    rounded_teacher=np.zeros(n);unscaled_underflow=np.zeros(n,dtype=bool);unscaled_subnormal=np.zeros(n,dtype=bool)
    naive_lost=np.zeros(n,dtype=bool);high_zero_nonzero=np.zeros(n,dtype=bool)
    nonzero_min=None;lost_examples=[]
    for i in range(n):
        exact_product=F(float(a[i]))*F(float(my[i]));norm=F(float(mx[i]))-exact_product
        pair=F(float(exp.hi[i]))+F(float(exp.lo[i]))
        high_errors[i]=F(float(exp.hi[i]))!=F(float(mx[i]))-F(float(native_ph[i]))
        tail_errors[i]=F(float(got_pl[i]))!=exact_product-F(float(got_ph[i]))
        normal_errors[i]=pair!=norm
        sign=-1 if exp.numerator_negative[i] else 1
        original_exact=F(float(x[i]))-F(float(q[i]))*F(float(y[i]))
        restored=sign*pair*F(2)**int(exp.scale_exponent[i]);errors[i]=restored!=original_exact
        rounded_teacher[i]=float(norm);rounded_losses[i]=F(float(rounded[i]))!=norm
        signed_exact_zero[i]=original_exact==0
        naive_lost[i]=F(float(exp.hi[i]))!=norm
        high_zero_nonzero[i]=exp.hi[i]==0 and norm!=0
        unscaled=float(original_exact)
        unscaled_underflow[i]=unscaled==0 and original_exact!=0
        unscaled_subnormal[i]=unscaled!=0 and abs(unscaled)<np.finfo(float).tiny
        if original_exact and (nonzero_min is None or abs(original_exact)<nonzero_min):nonzero_min=abs(original_exact)
        if (rounded_losses[i] or unscaled_underflow[i]) and len(lost_examples)<40:
            lost_examples.append(dict(index=i,x=x[i],y=y[i],q=q[i],hi=exp.hi[i],lo=exp.lo[i],rounded=rounded[i],
                scale_exponent=int(exp.scale_exponent[i]),numerator_negative=bool(exp.numerator_negative[i]),
                exact_normalized_numerator=str(norm.numerator),exact_normalized_denominator=str(norm.denominator),
                rounded_scalar_loses_information=bool(rounded_losses[i]),unscaled_underflow=bool(unscaled_underflow[i])))
        if i and i%40000==0:print('exact residual cases',i,flush=True)
    meta=read(ROOT/'division_operator_training/operand_metadata.csv')
    groups={'all_required':np.arange(original_n),'all_Brent':np.flatnonzero(ids<45582),
            'synthetic_validation':np.flatnonzero((ids>=45582)&(ids<119390)),
            'all_nextafter_neighborhoods':np.flatnonzero(ids>=119390),
            'all_previous_synthetic_failures':np.flatnonzero((ids>=45582)&(ids<119390)&(bits(z['before'][ids])!=bits(z['teacher'][ids]))),
            'all_previous_Brent_failures':np.flatnonzero((ids<45582)&(bits(z['before'][ids])!=bits(z['teacher'][ids]))),
            'fresh_independent':np.arange(original_n,original_n+len(fresh)),
            'explicit_zero_extreme_controls':np.arange(original_n+len(fresh),n),'all_tested':np.arange(n)}
    for name in sorted({r['group'] for r in meta}):groups['Brent_'+name]=np.array([j for j,i in enumerate(ids) if i<45582 and meta[i]['group']==name],dtype=int)
    rows=[]
    phbad=bits(got_ph)!=bits(native_ph);plbad=bits(got_pl)!=bits(native_pl)
    hibad=bits(exp.hi)!=bits(native_hi);lobad=bits(exp.lo)!=bits(native_lo);rcbad=bits(rounded)!=bits(native_rounded)
    zero_hi=(exp.hi==0)&(native_hi==0)&hibad;zero_lo=(exp.lo==0)&(native_lo==0)&lobad
    zero_rc=(rounded==0)&(native_rounded==0)&rcbad
    for name,j in groups.items():
        rows.append(dict(group=name,cases=len(j),pair_bit_exact=int((~(hibad[j]|lobad[j])).sum()),
            pair_bit_mismatches=int((hibad[j]|lobad[j]).sum()),product_hi_bit_errors=int(phbad[j].sum()),product_tail_bit_errors=int(plbad[j].sum()),
            exact_normalized_pair_errors=int(normal_errors[j].sum()),exact_original_residual_errors=int(errors[j].sum()),
            high_subtraction_exactness_errors=int(high_errors[j].sum()),product_tail_exactness_errors=int(tail_errors[j].sum()),
            rounded_normalized_bit_errors=int(rcbad[j].sum()),rounded_vs_exact_RN_errors=int((bits(rounded[j])!=bits(rounded_teacher[j])).sum()),
            signed_zero_hi_errors=int(zero_hi[j].sum()),signed_zero_lo_errors=int(zero_lo[j].sum()),signed_zero_rounded_errors=int(zero_rc[j].sum()),
            negative_zero_lo_count=int(((exp.lo[j]==0)&np.signbit(exp.lo[j])).sum()),
            naive_high_loses_information=int(naive_lost[j].sum()),naive_high_zero_but_residual_nonzero=int(high_zero_nonzero[j].sum()),
            single_rounded_normalized_loses_exact_bits=int(rounded_losses[j].sum()),
            unscaled_float_residual_underflows=int(unscaled_underflow[j].sum()),unscaled_float_residual_subnormals=int(unscaled_subnormal[j].sum()),
            nonfinite_pair_outputs=int((~np.isfinite(exp.hi[j])|~np.isfinite(exp.lo[j])).sum())))
    write('compensated_residual_validation.csv',rows)
    if lost_examples:write('residual_information_loss_examples.csv',lost_examples)
    np.savez_compressed(OUT/'compensated_residual_outputs.npz',x=x,y=y,q=q,hi=exp.hi,lo=exp.lo,
        scale_exponent=exp.scale_exponent,numerator_negative=exp.numerator_negative,rounded_normalized=rounded,required_source_indices=ids)
    passed=not any(np.any(v) for v in (errors,normal_errors,high_errors,tail_errors,hibad,lobad,rcbad,phbad,plbad)) and tail_replay_errors==0
    unchanged=all(source.sha(Path(p))==h for p,h in protected.items());assert unchanged
    implementation=Path(__file__).with_name('compensated_division_residual.py')
    result=dict(operator='compensated_division_residual',status='VALIDATED_ASSEMBLY' if passed else 'DIAGNOSTIC_FAILURE',passed=passed,
        module='experiments.geometry.compensated_division_residual',class_name='CompensatedDivisionResidual',
        implementation=str(implementation),implementation_sha256=source.sha(implementation),validator_sha256=source.sha(Path(__file__)),
        interface=['hi','lo','scale_exponent','numerator_negative'],semantics='x-q*y = (-1 if numerator_negative else +1) * 2**scale_exponent * (hi+lo), exact real sum/scaling',
        optional_rounded_normalized='RN64(hi+lo), diagnostic compatibility only; not the information-preserving interface',
        unary_negation='exact IEEE sign flip of p_lo, including +0 -> -0; deterministic scaffold',
        normalization='frexp(abs(x)), frexp(abs(y)), ldexp(abs(q),-(ex-ey)); deterministic scaffold',
        dependency_checkpoints={k:{s:v.get(s) for s in ('status','checkpoint','sha256')} for k,v in assembly.manifests.items()},
        diagnostic_division={k:dc[k] for k in ('status','checkpoint','sha256')},
        required_cases=original_n,Brent_cases=45582,fresh_cases=len(fresh),control_cases=len(controls),metrics=rows,
        isolated_product_tail_replay_cases=len(oldtails['a']),isolated_product_tail_replay_errors=tail_replay_errors,
        protected_files=len(protected),protected_hashes_unchanged=unchanged,training=False,new_checkpoint=False,
        half_ULP_comparison=False,rounding_router=False,Brent_interpolation=False,full_Brent=False,recursion=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
    if passed:(OUT/'compensated_division_residual_validated.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
