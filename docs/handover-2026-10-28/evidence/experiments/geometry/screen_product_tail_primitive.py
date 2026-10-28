"""Isolated product-tail feasibility/reuse screen. No training or division routing."""
import csv
import json
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import torch

from experiments.geometry import train_division_operator as src
from experiments.geometry.diagnose_division_rounding import product_low
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_reciprocal_operator import PositivePower
from experiments.geometry.train_subtraction_operator import Subtraction

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'product_tail_primitive_screen'
K=134217729.0


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    protected={str(p):src.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):src.sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('locked','manifest','validated'))})
    for folder in ('division_operator_training','division_operator_refinement','division_rounding_diagnosis'):
        protected.update({str(p):src.sha(p) for p in (ROOT/folder).iterdir() if p.is_file()})
    models={};manifests={}
    for name,cls,folder,manifest in [('multiply',QuadraticArithmetic,'multiplication_operator_smoke','candidate_manifest.json'),
                                   ('subtract',Subtraction,'subtraction_operator_training','subtraction_operator_locked.json'),
                                   ('add',Addition,'addition_operator_smoke','candidate_manifest.json'),
                                   ('reciprocal',PositivePower,'reciprocal_required_domain_audit','candidate_manifest.json')]:
        models[name],manifests[name]=src.load(cls,folder,manifest)
    dmeta=json.loads((ROOT/'division_operator_refinement/results.json').read_text())
    cp=Path(dmeta['checkpoint']);assert src.sha(cp)==dmeta['sha256']
    division=src.Division(models['multiply'],models['reciprocal'])
    division.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);division.eval().requires_grad_(False)
    z=np.load(ROOT/'division_operator_refinement/refinement_inputs.npz');xy=z['inputs'];teacher=z['teacher']
    qp=division.predict(xy);assert np.array_equal(qp.view('u8'),z['before'].view('u8'))
    finite=np.isfinite(qp)&np.isfinite(teacher);ids=np.flatnonzero(finite)
    mx,ex=np.frexp(abs(xy[:,0]));my,ey=np.frexp(abs(xy[:,1]))
    a=np.ldexp(abs(qp[ids]),-(ex[ids]-ey[ids]));b=my[ids]
    assert len(ids)==119210
    meta=read(ROOT/'division_operator_training/operand_metadata.csv')
    groups=np.array(['Brent_'+meta[i]['group'] if i<45582 else 'synthetic_validation' if i<119390 else 'failure_nextafter' for i in ids],dtype=object)
    # Independent small raw-product domain screen, no fitting or root solves.
    rng=np.random.default_rng(928202647)
    fresh_a=rng.uniform(float(a[a>0].min()),float(a.max()),4096)
    fresh_b=rng.uniform(.5,1.,4096)
    edges=[]
    for aa in (0.,.5,1.,2.,float(a.max())):
        for bb in (.5,.75,np.nextafter(1.,0.)):
            for shift in (-np.inf,np.inf):
                edges.append((np.nextafter(aa,shift) if aa!=0 else aa,bb))
    fresh=np.concatenate((np.column_stack((fresh_a,fresh_b)),np.array(edges)))
    original_n=len(a);a=np.concatenate((a,fresh[:,0]));b=np.concatenate((b,fresh[:,1]))
    groups=np.concatenate((groups,np.full(len(fresh),'fresh_independent_pairs',dtype=object)))
    native={};actual={};checks=[]
    def operation(name,kind,left,right):
        left=np.broadcast_to(left,a.shape);right=np.broadcast_to(right,a.shape)
        with torch.no_grad():p=models[kind](torch.from_numpy(np.column_stack((left,right)))).numpy()
        t=left*right if kind=='multiply' else left-right if kind=='subtract' else left+right
        native[name]=t;actual[name]=p
        checks.append(src.stats(t,p,name))
        if not np.array_equal(p.view('u8'),t.view('u8')):
            print('intermediate mismatch',checks[-1],flush=True)
        assert np.array_equal(p,t),name
        return p
    # This is the diagnostic's exact evaluation order, not a new algorithm.
    ph=operation('p_hi','multiply',a,b)
    ca=operation('ca','multiply',K,a)
    ca_minus_a=operation('ca_minus_a','subtract',ca,a)
    ah=operation('a_hi','subtract',ca,ca_minus_a)
    al=operation('a_lo','subtract',a,ah)
    cb=operation('cb','multiply',K,b)
    cb_minus_b=operation('cb_minus_b','subtract',cb,b)
    bh=operation('b_hi','subtract',cb,cb_minus_b)
    bl=operation('b_lo','subtract',b,bh)
    hh=operation('hi_hi','multiply',ah,bh)
    s0=operation('s0','subtract',hh,ph)
    hl=operation('hi_lo','multiply',ah,bl)
    s1=operation('s1','add',s0,hl)
    lh=operation('lo_hi','multiply',al,bh)
    s2=operation('s2','add',s1,lh)
    ll=operation('lo_lo','multiply',al,bl)
    tail=operation('p_lo','add',s2,ll)
    authority=product_low(a,b,ph)
    assert np.array_equal(tail.view('u8'),authority.view('u8'))
    naive=operation('naive_repeated_product_minus_p_hi','subtract',operation('naive_repeated_product','multiply',a,b),ph)
    labels=np.empty(len(a));proof_errors=[];zero=0
    for i,(aa,bb,pp,lo) in enumerate(zip(a,b,ph,tail)):
        exact=F(float(aa))*F(float(bb))-F(float(pp))
        labels[i]=float(exact)
        if F(float(lo))!=exact:proof_errors.append(i)
        if not exact:zero+=1
    assert not proof_errors
    assert np.array_equal(labels.view('u8'),tail.view('u8'))
    metrics=[]
    indices={g:np.flatnonzero(groups==g) for g in sorted(set(groups))}
    indices['all_Brent']=np.flatnonzero(np.char.startswith(groups.astype(str),'Brent_'))
    indices['all_required']=np.arange(original_n);indices['all_with_fresh']=np.arange(len(a))
    domain=[]
    for group,ii in indices.items():
        metrics.append(src.stats(labels[ii],tail[ii],group))
        metrics.append(src.stats(labels[ii],naive[ii],group+'_ordinary_multiply_subtract_tail'))
        for name,values in {'a_normalized_output_quotient':a,'b_denominator_mantissa':b,'p_hi':ph,'p_lo':tail,
                            'ca_split_scaled_a':ca,'cb_split_scaled_b':cb,'a_lo':al,'b_lo':bl}.items():
            v=values[ii];nz=abs(v[v!=0]);tiny=np.finfo(float).tiny
            domain.append(dict(group=group,quantity=name,count=len(ii),minimum=float(v.min()),maximum=float(v.max()),
                min_nonzero_absolute=float(nz.min()) if len(nz) else None,max_absolute=float(abs(v).max()),
                median=float(np.median(v)),q01=float(np.quantile(v,.01)),q99=float(np.quantile(v,.99)),
                zeros=int((v==0).sum()),subnormals=int(((abs(v)<tiny)&(v!=0)).sum()),nonfinite=int((~np.isfinite(v)).sum())))
    # Rounded high products are not sufficient input to a tail predictor.
    collisions=defaultdict(dict)
    for i in range(original_n):collisions[int(ph[i].view('u8'))].setdefault(int(tail[i].view('u8')),i)
    conflicting=[list(v.values()) for v in collisions.values() if len(v)>1]
    examples=[]
    for group,ii in enumerate(conflicting[:20]):
        for i in ii[:3]:examples.append(dict(collision_group=group,index=int(ids[i]),a=a[i],b=b[i],a_hex=a[i].hex(),b_hex=b[i].hex(),p_hi=ph[i],p_hi_hex=ph[i].hex(),p_lo=tail[i],p_lo_hex=tail[i].hex()))
    if examples:write('same_high_different_tail.csv',examples)
    # Audit the downstream residual identity, without assembling/training it.
    rh=mx[ids]-ph[:original_n]
    erased=(rh==0)&(tail[:original_n]!=0)
    np.savez_compressed(OUT/'isolated_product_tail_targets.npz',a=a,b=b,p_hi=ph,target_tail=labels,groups=groups,
                        original_source_indices=ids,assembled_tail=tail)
    write('operation_exactness.csv',checks);write('tail_exactness.csv',metrics);write('required_domain.csv',domain)
    unchanged=all(src.sha(Path(p))==h for p,h in protected.items());assert unchanged
    result=dict(status='ISOLATED_PRODUCT_TAIL_REUSE_SCREEN_PASSED',training_performed=False,
        candidate_checkpoint_saved=False,new_operator_locked=False,architecture_changed=False,
        required_cases=original_n,required_Brent_cases=len(indices['all_Brent']),excluded_nonfinite_division_outputs=int((~finite).sum()),
        fresh_cases=len(fresh),all_cases=len(a),exact_rational_tail_errors=len(proof_errors),
        matches_authoritative_diagnostic=True,nonzero_required_tails=int((tail[:original_n]!=0).sum()),
        erased_normalized_residuals=int(erased.sum()),same_high_distinct_tail_groups=len(conflicting),
        splitter_constant=K,splitter_constant_is_existing_deterministic_scaffold=True,
        first_missing_scalar_arithmetic_operation=None,missing_standalone_interface='normalized product-tail (a,b) -> exact(a*b)-RN(a*b)',
        intermediate_signed_zero_differences=[dict(operation=c['group'],count=c['signed_zero_errors']) for c in checks if c['signed_zero_errors']],
        final_tail_signed_zero_errors=0,first_missing_scalar_note='No missing scalar VALUE operation. Some intermediate multiplication zero signs differ from native; they do not affect any final tail bit.',
        proposal='Reuse the exact ordered 17-call scalar graph for the product tail; no redundant scalar arithmetic training. Graph wiring and splitter constant remain deterministic, not claimed learned.',
        manifests={k:{s:v.get(s) for s in ('status','checkpoint','sha256')} for k,v in manifests.items()},
        division_checkpoint={k:dmeta.get(k) for k in ('status','checkpoint','sha256')},
        protected_hashes_unchanged=unchanged,protected_files=len(protected),final_rounding_router=False,Brent_assembly=False,recursion=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
