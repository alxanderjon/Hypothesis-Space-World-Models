"""Small residual-corrected division candidate; native division labels only."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_reciprocal_operator import PositivePower
from experiments.geometry.train_subtraction_operator import Subtraction
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'division_operator_training'
SEED=928202635
BANDS=[(-800,-601),(-30,-11),(129,256),(800,900)]


def load(cls,folder,filename='candidate_manifest.json'):
    m=json.loads((ROOT/folder/filename).read_text());assert sha(Path(m['checkpoint']))==m['sha256']
    model=cls();model.load_state_dict(torch.load(m['checkpoint'],weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    return model,m


class Division(nn.Module):
    """Two learned coefficients on a quotient estimate and compensated residual.

    No native division of x by y, or mantissa_x by mantissa_y, in inference.
    frexp/ldexp and sign bookkeeping are deterministic. The fixed Dekker split
    constant decomposes a floating-point product; it is not a division target.
    """
    def __init__(self,multiply,reciprocal,scale=1.):
        super().__init__();self.weights=nn.Parameter(.3*torch.randn(2,dtype=torch.float64))
        self.register_buffer('feature_scale',torch.tensor(scale,dtype=torch.float64))
        self.multiply=multiply.eval().requires_grad_(False)
        self.reciprocal=reciprocal.eval().requires_grad_(False)

    def features(self,xy):
        xy=np.asarray(xy,dtype=np.float64)
        assert np.isfinite(xy).all() and np.all(xy[:,1]!=0)
        mx,ex=np.frexp(abs(xy[:,0]));my,ey=np.frexp(abs(xy[:,1]))
        with torch.no_grad():
            rr=self.reciprocal(torch.from_numpy(my)).numpy()
            q0=self.multiply(torch.from_numpy(np.column_stack((mx,rr)))).numpy()
        # Error-free product decomposition in safe, normalized mantissa ranges.
        product=q0*my;splitter=134217729.
        ca=splitter*q0;ah=ca-(ca-q0);al=q0-ah
        cb=splitter*my;bh=cb-(cb-my);bl=my-bh
        low=((ah*bh-product)+ah*bl+al*bh)+al*bl
        residual=(mx-product)-low
        with torch.no_grad():corr=self.multiply(torch.from_numpy(np.column_stack((residual,rr)))).numpy()
        features=np.column_stack((q0,corr/float(self.feature_scale)))
        negative=np.signbit(xy[:,0])^np.signbit(xy[:,1])
        return features,ex-ey,negative

    def forward(self,features):return self.weights[0]*features[:,0]+self.weights[1]*features[:,1]

    def predict(self,xy):
        features,e,negative=self.features(xy)
        with torch.no_grad():mantissa=self(torch.from_numpy(features)).numpy()
        with np.errstate(over='ignore',under='ignore'):
            output=np.ldexp(mantissa,e)
        return np.copysign(output,np.where(negative,-1.,1.))


def teacher(x):
    with np.errstate(over='ignore',under='ignore'):return x[:,0]/x[:,1]


def held(x):
    _,e=np.frexp(abs(x));e-=1;b=np.zeros(x.shape,bool)
    for lo,hi in BANDS:b|=(x!=0)&(e>=lo)&(e<=hi)
    return b.any(1)


def synthetic(n,seed,reserved):
    rng=np.random.default_rng(seed);parts=[];count=0
    while count<n:
        x=np.ldexp(rng.uniform(1,2,(n,2)),rng.integers(-1000,1001,(n,2)))*rng.choice([-1.,1.],(n,2))
        m=(held(x)==reserved);parts.append(x[m]);count+=int(m.sum())
    return np.concatenate(parts)[:n]


def stats(y,p,name):
    a=y.view('u8');b=p.view('u8');same=a==b;bothzero=(y==0)&(p==0)
    zero=bothzero&~same
    oa=np.where(a>>63,~a,a|np.uint64(1<<63));ob=np.where(b>>63,~b,b|np.uint64(1<<63));u=np.maximum(oa,ob)-np.minimum(oa,ob)
    finite=np.isfinite(y)&np.isfinite(p);numeric=~same&finite&~bothzero
    return dict(group=name,cases=len(y),exact_matches=int(same.sum()),exact_match_rate=float(same.mean()),errors=int((~same).sum()),
        one_ULP_errors=int((numeric&(u==1)).sum()),larger_errors=int((numeric&(u>1)).sum()),signed_zero_errors=int(zero.sum()),
        nonfinite_mismatches=int((~same&~finite).sum()),matching_nonfinite=int((same&~finite).sum()),
        maximum_finite_ULP_error=int(u[finite].max()) if finite.any() else 0)


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    reuse=json.loads((OUT/'reuse_audit.json').read_text());assert reuse['errors']>0
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    mult,mm=load(QuadraticArithmetic,'multiplication_operator_smoke');rec,rm=load(PositivePower,'reciprocal_required_domain_audit')
    data=np.load(OUT/'Brent_division_operands.npz');bx=data['operands'];by=data['teacher'];meta=read(OUT/'operand_metadata.csv')
    roles=np.array([m['group'] for m in meta]);fit=(roles=='fit')&~held(bx)
    train=np.concatenate((bx[fit],synthetic(8192,SEED+1,False)))
    model=Division(mult,rec);initial=model.weights.detach().numpy().copy()
    rawfeatures,_,_=model.features(train)
    scale=float(np.sqrt(np.mean(rawfeatures[:,1]**2)));assert scale>0
    model.feature_scale.fill_(scale)
    design,_,_=model.features(train)
    # Exact supervised labels for the normalized quotient. Only teacher code
    # divides mantissas; learner's inputs are q0 and its arithmetic residual.
    mx=np.frexp(abs(train[:,0]))[0];my=np.frexp(abs(train[:,1]))[0];target=mx/my
    history=[];best=None
    for step in range(24):
        with torch.no_grad():pred=model(torch.from_numpy(design)).numpy()
        res=target-pred;loss=float(np.mean(res*res));w=model.weights.detach().numpy().copy()
        history.append(dict(update=step,w0=w[0],w1=w[1],mse=loss,exact_normalized_targets=int((target.view('u8')==pred.view('u8')).sum())))
        if best is None or loss<best[0]:best=(loss,w.copy())
        if np.all(res==0):break
        proposal=w+np.linalg.lstsq(design,res,rcond=None)[0]
        if np.array_equal(proposal,w):break
        with torch.no_grad():model.weights.copy_(torch.from_numpy(proposal))
    with torch.no_grad():model.weights.copy_(torch.from_numpy(best[1]))
    config=dict(seed=SEED,architecture='Two learned coefficients on normalized quotient estimate and compensated residual correction',
        trainable_parameters=2,initial_weights=initial.tolist(),bias=False,dtype='float64',training_examples=len(train),Brent_fit_examples=int(fit.sum()),
        inference='frexp magnitude decomposition -> frozen reciprocal and multiplication -> Dekker residual features -> learned two-weight sum -> sign/ldexp reconstruction',
        direct_division_in_inference=False,deterministic_scaffold='frexp/ldexp, sign, error-free product decomposition, fixed feature scaling; not claimed learned',
        no_snapping=True,split='Complete saved solve groups; all saved fresh solves withheld; synthetic held-out exponent bands excluded from all training operands',
        heldout_magnitude_bands=BANDS,reciprocal_checkpoint=rm['checkpoint'],multiplication_checkpoint=mm['checkpoint'])
    cp=OUT/'division_before_evaluation.pt';digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),config=config,history=history))
    write_csv(OUT/'training_history.csv',history);json_write(OUT/'config.json',config)
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    print(json.dumps(dict(event='trained_and_saved',weights=model.weights.tolist(),scale=scale,sha256=digest)),flush=True)
    bp=model.predict(bx);metrics=[stats(by,bp,'all_Brent_operands')]
    reuse_metrics=[stats(by,data['reuse_predictions'],'all_Brent_operands')]
    for g in sorted(set(roles)):
        m=roles==g;metrics.append(stats(by[m],bp[m],g));reuse_metrics.append(stats(by[m],data['reuse_predictions'][m],g))
    for op in sorted({r['operation'] for r in meta}):
        m=np.array([r['operation']==op for r in meta]);metrics.append(stats(by[m],bp[m],op))
    for name,m in [('positive_denominator',bx[:,1]>0),('negative_denominator',bx[:,1]<0)]:
        reuse_metrics.append(stats(by[m],data['reuse_predictions'][m],name))
    write_csv(OUT/'reuse_validation.csv',reuse_metrics)
    groups=[('heldout_magnitude_bands',synthetic(16384,SEED+2,True)),('independent_seen_bands',synthetic(16384,SEED+3,False))]
    rng=np.random.default_rng(SEED+4)
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
    extreme=np.array([(x,y) for x in (0.,-0.,small,-small,tiny,-tiny,1.,-1.,mxv,-mxv)
                       for y in (small,-small,tiny,-tiny,1.,-1.,mxv,-mxv)])
    groups.append(('signed_zero_extremes_and_overflow',extreme))
    synthetic_failures=[]
    for name,x in groups:
        y=teacher(x);p=model.predict(x);metrics.append(stats(y,p,name))
        bad=np.flatnonzero(y.view('u8')!=p.view('u8'))
        for i in bad:
            synthetic_failures.append(dict(group=name,numerator=x[i,0],denominator=x[i,1],teacher=y[i],learned=p[i],teacher_hex=y[i].hex(),learned_hex=p[i].hex()))
    bad=np.flatnonzero(by.view('u8')!=bp.view('u8'))
    failure=[dict(**meta[i],numerator=bx[i,0],denominator=bx[i,1],teacher=by[i],learned=bp[i],teacher_hex=by[i].hex(),learned_hex=bp[i].hex()) for i in bad]
    write_csv(OUT/'division_failures.csv',failure or [dict(status='none')]);write_csv(OUT/'synthetic_failures.csv',synthetic_failures or [dict(status='none')])
    write_csv(OUT/'division_validation.csv',metrics)
    passed=all(r['errors']==0 for r in metrics);assembly=None
    if passed:
        sub,sm=load(Subtraction,'subtraction_operator_training','subtraction_operator_locked.json')
        st=data['linear_states'];fc,fp,xc,xp=st.T
        dx=sub.predict(np.column_stack((xc,xp)));df=sub.predict(np.column_stack((fc,fp)))
        with torch.no_grad():num=mult(torch.from_numpy(np.column_stack((-fc,dx)))).numpy()
        phat=model.predict(np.column_stack((num,df)));truth=-fc*(xc-xp)/(fc-fp)
        assembly=stats(truth,phat,'linear_step_primitive_assembly');write_csv(OUT/'linear_primitive_assembly.csv',[assembly])
    unchanged=all(sha(Path(p))==h for p,h in protected.items());assert unchanged
    result=dict(operator='division_operator',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        checkpoint=str(cp),sha256=digest,module='experiments.geometry.train_division_operator',model='Division',implementation_sha256=sha(Path(__file__)),
        weights=model.weights.tolist(),feature_scale=scale,training_updates=len(history)-1,metrics=metrics,reuse_metrics=reuse_metrics,assembly=assembly,
        no_complete_Brent_replay=True,no_recursion=True,protected_files=len(protected),protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result)
    if passed:json_write(OUT/'division_operator_locked.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
