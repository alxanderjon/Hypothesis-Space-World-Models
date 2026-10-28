"""Refine only the two saved division weights, retaining architecture/state."""
import json
import time
from pathlib import Path
import numpy as np
import torch

from experiments.geometry import train_division_operator as src
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_reciprocal_operator import PositivePower
from experiments.geometry.train_subtraction_operator import Subtraction
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'division_operator_refinement'


def prior_synthetic():
    groups=[('heldout_magnitude_bands',src.synthetic(16384,src.SEED+2,True)),
            ('independent_seen_bands',src.synthetic(16384,src.SEED+3,False))]
    rng=np.random.default_rng(src.SEED+4)
    bases=np.ldexp(rng.uniform(1,2,4096),rng.integers(-1074,1023,4096))*rng.choice([-1.,1.],4096)
    adjacent=[];derived=[]
    for a in bases:
        for toward in (-np.inf,np.inf):
            b=np.nextafter(a,toward)
            if b!=0 and np.isfinite(b):adjacent.extend(((a,b),(b,a)))
            den=b-a
            if den!=0 and np.isfinite(den):derived.extend(((a,den),(-a,den),(den,a)))
    groups.extend((('adjacent_nextafter',np.array(adjacent)),('cancellation_derived_denominators',np.array(derived))))
    small=np.nextafter(0.,1.);mxv=np.finfo(float).max;tiny=np.finfo(float).tiny
    groups.append(('signed_zero_extremes_and_overflow',np.array([(x,y) for x in (0.,-0.,small,-small,tiny,-tiny,1.,-1.,mxv,-mxv)
        for y in (small,-small,tiny,-tiny,1.,-1.,mxv,-mxv)])))
    return groups


def shifted(v,k):
    for _ in range(abs(k)):v=np.nextafter(v,np.inf if k>0 else -np.inf)
    return v


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    old=json.loads((src.OUT/'results.json').read_text());cp=Path(old['checkpoint']);assert sha(cp)==old['sha256']
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('locked','manifest','validated'))})
    protected.update({str(p):sha(p) for p in src.OUT.iterdir() if p.is_file()})
    protected[str(Path(src.__file__))]=sha(Path(src.__file__))
    mult,_=src.load(QuadraticArithmetic,'multiplication_operator_smoke');rec,_=src.load(PositivePower,'reciprocal_required_domain_audit')
    model=src.Division(mult,rec);payload=torch.load(cp,weights_only=False);model.load_state_dict(payload['state_dict']);model.eval()
    frozen={k:v.clone() for k,v in model.state_dict().items() if k!='weights'}
    initial=model.weights.detach().numpy().copy()
    data=np.load(src.OUT/'Brent_division_operands.npz');bx=data['operands'];meta=read(src.OUT/'operand_metadata.csv')
    groups=[('Brent',bx)]+prior_synthetic();oldn=sum(len(x) for _,x in groups)
    fails=read(src.OUT/'division_failures.csv')+read(src.OUT/'synthetic_failures.csv')
    unique={(float(r['numerator']).hex(),float(r['denominator']).hex()):(float(r['numerator']),float(r['denominator'])) for r in fails}
    neighborhoods=[];nmeta=[]
    for pair_id,(a,b) in enumerate(unique.values()):
        for i in range(-8,9):
            for j in range(-8,9):
                x,y=shifted(a,i),shifted(b,j)
                if y!=0 and np.isfinite(x) and np.isfinite(y):
                    neighborhoods.append((x,y));nmeta.append(dict(pair_id=pair_id,x_offset=i,y_offset=j))
    groups.append(('failure_nextafter_neighborhoods',np.array(neighborhoods)))
    x=np.concatenate([z for _,z in groups]);y=src.teacher(x);features,e,negative=model.features(x)
    base=model.predict(x);base_bad=base.view('u8')!=y.view('u8');previous_exact=~base_bad[:oldn]
    slices={};offset=0
    for name,z in groups:slices[name]=slice(offset,offset+len(z));offset+=len(z)
    assert base_bad[:len(bx)].sum()==3 and base_bad[len(bx):oldn].sum()==11
    for row in old['metrics']:
        if row['group'] in slices:
            s=slices[row['group']];assert int(base_bad[s].sum())==row['errors']
    np.savez_compressed(OUT/'refinement_inputs.npz',inputs=x,teacher=y,before=base,features=features,exponents=e,negative=negative)
    write_csv(OUT/'neighborhood_metadata.csv',nmeta)
    def predict(w):
        # Same eager multiply/add and ldexp/sign evaluation as Division.predict.
        mant=w[0]*features[:,0]+w[1]*features[:,1]
        with np.errstate(over='ignore',under='ignore'):out=np.ldexp(mant,e)
        return np.copysign(out,np.where(negative,-1.,1.))
    assert np.array_equal(predict(initial).view('u8'),base.view('u8'))
    def assess(p):
        bad=p.view('u8')!=y.view('u8')
        return (int((bad[:oldn]&previous_exact).sum()),int(bad[:oldn].sum()),int(bad[oldn:].sum()))
    importance=np.ones(len(x));importance[:oldn][base_bad[:oldn]]=256.;importance[oldn:]=16.
    # Both coefficients are free. Solve weighted local residual regression on
    # FINAL outputs converted to mantissa units; do not substitute log labels.
    finite=np.isfinite(y)&(y!=0)
    design=features[finite]*np.sqrt(importance[finite,None])
    history=[];bestw=initial.copy();bestpred=base.copy();bestscore=assess(base);w=initial.copy()
    best_checkpoint=OUT/'division_refined_best.pt'
    def save_best(step):
        with torch.no_grad():model.weights.copy_(torch.from_numpy(bestw))
        return save_checkpoint(best_checkpoint,dict(state_dict=model.state_dict(),source_checkpoint=str(cp),source_sha256=old['sha256'],
            refinement_update=step,initial_weights=initial.tolist(),score=bestscore,architecture_unchanged=True,feature_scale_unchanged=True))
    digest=save_best(0)
    visited=set();stalled=0
    for step in range(128):
        pred=predict(w);score=assess(pred);history.append(dict(update=step,w0=w[0],w1=w[1],regressions=score[0],prior_pool_errors=score[1],neighborhood_errors=score[2]))
        if score<bestscore:
            bestscore=score;bestw=w.copy();bestpred=pred.copy();digest=save_best(step);stalled=0
        if score==(0,0,0):break
        key=tuple(float(v).hex() for v in w)
        if key in visited:break
        visited.add(key)
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            residual=np.ldexp(abs(y[finite])-abs(pred[finite]),-e[finite])
        assert np.isfinite(residual).all()
        delta=np.linalg.lstsq(design,residual*np.sqrt(importance[finite]),rcond=None)[0]
        choices=[]
        for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
            proposal=w+alpha*delta
            if np.array_equal(proposal,w):continue
            pp=predict(proposal);ss=assess(pp)
            choices.append((ss,proposal))
        if not choices:break
        choices.sort(key=lambda z:z[0]);nextscore,proposal=choices[0]
        if nextscore>score:
            stalled+=1
            # Do not accept a regression in the retention pool.
            break
        w=proposal
        if nextscore==score:stalled+=1
        if stalled>=12:break
    # Save any last improved accepted update before the final evaluation.
    p=predict(w);score=assess(p)
    if score<bestscore:bestscore=score;bestw=w.copy();bestpred=p.copy();digest=save_best(len(history))
    model.load_state_dict(torch.load(best_checkpoint,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
    after=model.predict(x);assert np.array_equal(after.view('u8'),bestpred.view('u8'))
    rows=[]
    def report(name,idx):
        before=src.stats(y[idx],base[idx],name);aa=src.stats(y[idx],after[idx],name)
        regress=int(((base[idx].view('u8')==y[idx].view('u8'))&(after[idx].view('u8')!=y[idx].view('u8'))).sum())
        rows.append(dict(group=name,cases=before['cases'],before_exact=before['exact_matches'],after_exact=aa['exact_matches'],
            before_errors=before['errors'],after_errors=aa['errors'],before_one_ULP=before['one_ULP_errors'],after_one_ULP=aa['one_ULP_errors'],
            before_larger=before['larger_errors'],after_larger=aa['larger_errors'],before_signed_zero=before['signed_zero_errors'],after_signed_zero=aa['signed_zero_errors'],
            before_nonfinite=before['nonfinite_mismatches'],after_nonfinite=aa['nonfinite_mismatches'],regressions=regress))
    for name,s in slices.items():report(name,s)
    for group in sorted({r['group'] for r in meta}):report('Brent_'+group,np.array([i for i,m in enumerate(meta) if m['group']==group]))
    report('all_prior_synthetic',slice(len(bx),oldn));report('all_prior_validation',slice(0,oldn))
    write_csv(OUT/'before_after.csv',rows);write_csv(OUT/'refinement_history.csv',history)
    remain=np.flatnonzero(after.view('u8')!=y.view('u8'))
    write_csv(OUT/'remaining_failures.csv',[dict(index=int(i),x=x[i,0],y=x[i,1],teacher=y[i],before=base[i],after=after[i],teacher_hex=y[i].hex(),after_hex=after[i].hex(),previously_exact=bool(not base_bad[i])) for i in remain] or [dict(status='none')])
    unchanged=all(sha(Path(p))==v for p,v in protected.items());assert unchanged
    passed=not len(remain);assembly=None
    if passed:
        sub,_=src.load(Subtraction,'subtraction_operator_training','subtraction_operator_locked.json')
        st=data['linear_states'];fc,fp,xc,xp=st.T;dx=sub.predict(np.column_stack((xc,xp)));df=sub.predict(np.column_stack((fc,fp)))
        with torch.no_grad():num=mult(torch.from_numpy(np.column_stack((-fc,dx)))).numpy()
        out=model.predict(np.column_stack((num,df)));assembly=src.stats(-fc*(xc-xp)/(fc-fp),out,'linear_primitive_assembly')
        write_csv(OUT/'linear_primitive_assembly.csv',[assembly])
    result=dict(operator='division_operator',status='LOCKED' if passed else 'REFINEMENT_GATE_FAILED',passed=passed,
        checkpoint=str(best_checkpoint),sha256=digest,source_checkpoint=str(cp),source_sha256=old['sha256'],
        initial_weights=initial.tolist(),refined_weights=bestw.tolist(),feature_scale=float(model.feature_scale),
        prior_validation_cases=oldn,neighborhood_cases=len(neighborhoods),unique_failure_pairs=len(unique),refinement_steps=len(history)-1,
        best_score=list(bestscore),before_after=rows,assembly=assembly,architecture_unchanged=True,frozen_state_unchanged=True,
        protected_files=len(protected),protected_hashes_unchanged=True,no_recursion=True,no_Brent_replay=True,elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    if passed:json_write(OUT/'division_operator_locked.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
