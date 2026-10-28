"""One incremental Brent stage: structured linear interpolation, exact gate."""
import csv
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq
from experiments.geometry import audit_exact_branch_brent as audit
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_brent_interpolation_eligibility import BrentEligibility
from experiments.geometry.train_brent_interpolation_branch_selector import BrentBranchSelector
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'brent_linear_interpolation_training'
SOURCE=ROOT/'exact_branch_switch_brent_audit'
ELIG=ROOT/'brent_interpolation_eligibility_training'
BRANCH=ROOT/'brent_interpolation_branch_selector_training'
SEED=928202625


class LinearInterpolation(nn.Module):
    """Audited arithmetic graph with learned coefficients and rational powers.

    Native signed division remains the rational layer's arithmetic primitive;
    it is not replaced by reciprocal multiplication or algebraic reassociation.
    """
    def __init__(self,multiply):
        super().__init__()
        self.affine=nn.Parameter(.3*torch.randn(5,dtype=torch.float64))
        self.rational=nn.Parameter(.3*torch.randn(3,dtype=torch.float64))
        self.multiply=multiply.eval().requires_grad_(False)

    def intermediates(self,x):
        fc,fp,xc,xp=x.T
        neg=self.affine[0]*fc
        dx=self.affine[1]*xc+self.affine[2]*xp
        numerator=self.multiply(torch.stack((neg,dx),1))
        denominator=self.affine[3]*fc+self.affine[4]*fp
        return neg,dx,numerator,denominator

    def quotient(self,n,d):
        b,p,q=self.rational
        numerator=torch.exp(b)*torch.copysign(torch.pow(abs(n),p),n)
        denominator=torch.copysign(torch.pow(abs(d),q),d)
        return numerator/denominator

    def forward(self,x):
        _,_,n,d=self.intermediates(x)
        return self.quotient(n,d)

    def predict(self,x):
        with torch.no_grad():return self(torch.from_numpy(np.asarray(x,dtype=np.float64).reshape(-1,4))).numpy()


def teacher(x):
    fc,fp,xc,xp=x.T
    return -fc*(xc-xp)/(fc-fp)


def ulps(a,b):
    def index(v):
        bits=np.array(v,dtype=np.float64).view(np.uint64).item()
        return (~bits & ((1<<64)-1)) if bits>>63 else bits|(1<<63)
    return np.array([abs(index(v)-index(w)) for v,w in zip(a,b)],dtype=object)


def metric(x,y,p,name):
    u=ulps(y,p);coords=x[:,2]+y;pc=x[:,2]+p;cu=ulps(coords,pc)
    return dict(group=name,cases=len(x),exact_steps=int((u==0).sum()),step_errors=int((u!=0).sum()),
        one_ULP_step_differences=int((u==1).sum()),larger_step_differences=int((u>1).sum()),
        max_step_ULP_error=int(max(u)),max_step_absolute_error=float(max(abs(y-p))),
        max_step_relative_error=float(max(abs((p-y)/y))),exact_proposal_coordinates=int((cu==0).sum()),
        proposal_coordinate_errors=int((cu!=0).sum()),one_ULP_coordinate_differences=int((cu==1).sum()),
        larger_coordinate_differences=int((cu>1).sum()),max_coordinate_ULP_error=int(max(cu)),
        max_coordinate_absolute_error=float(max(abs(coords-pc))),nonfinite=int((~np.isfinite(p)).sum()))


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(v in p.name for v in ('locked','manifest','validated'))})
    for folder in (SOURCE,ELIG,BRANCH):protected.update({str(p):sha(p) for p in folder.iterdir() if p.is_file()})
    em=json.loads((ELIG/'brent_interpolation_eligibility_locked.json').read_text())
    bm=json.loads((BRANCH/'brent_interpolation_branch_selector_locked.json').read_text())
    for m in (em,bm):assert m['status']=='LOCKED' and sha(Path(m['checkpoint']))==m['sha256']
    eligibility=BrentEligibility(MagnitudeThreshold());eligibility.load_state_dict(torch.load(em['checkpoint'],weights_only=False)['state_dict']);eligibility.eval().requires_grad_(False)
    branch=BrentBranchSelector(MagnitudeThreshold());branch.load_state_dict(torch.load(bm['checkpoint'],weights_only=False)['state_dict']);branch.eval().requires_grad_(False)
    mm=json.loads((ROOT/'multiplication_operator_smoke/candidate_manifest.json').read_text())
    assert sha(Path(mm['checkpoint']))==mm['sha256']
    multiply=QuadraticArithmetic();multiply.load_state_dict(torch.load(mm['checkpoint'],weights_only=False)['state_dict'])
    model=LinearInterpolation(multiply);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    cases=json.loads((SOURCE/'solve_inputs.json').read_text())
    splits={int(r['case_id']):r['group'] for r in audit.read(ELIG/'solve_splits.csv')}
    for folder in (ELIG,BRANCH):
        for r in json.loads((folder/'new_fresh_candidates.json').read_text()):
            cases.append(dict(parent=r['parent'],axis=r['axis'],count=r['k'],left=r['c_left'],right=r['c_right'],teacher_cut=r['teacher_cut'],label=r['case']))
    a=audit.Audit.__new__(audit.Audit);states=[];labels=[];metadata=[];teacher_traces=[]
    for i,r in enumerate(cases):
        a.setup(r['parent'],r['axis']);k=r['count']
        def callback(c):
            f=a.formula(c,k);ia=int(f[4][0])
            return float(f[1][0]),dict(q_active=int(f[3][0]),aspect_active=ia,aspect_power=int(f[5][0,ia]),competing_q=f[6][0].tolist(),competing_log_aspect=f[7][0].tolist())
        root,tr,_,_=audit.traced_brent(callback,r['left'],r['right'],i);assert root==r['teacher_cut'];teacher_traces.append(tr)
        for row in tr:
            if row['attempt_method']!='LINEAR_INTERPOLATION':continue
            s=row['normalized_state'];states.append([s['fcur'],s['fpre'],s['xcur'],s['xpre']]);labels.append(row['trial_step'])
            metadata.append(dict(case_id=i,iteration=row['iteration'],split=splits[i] if i<818 else 'additional_fresh',label=r['label']))
    x=np.array(states);y=np.array(labels);ids=np.array([r['case_id'] for r in metadata]);fit=np.array([r['split']=='fit' for r in metadata])
    assert np.array_equal(teacher(x).view(np.uint64),y.view(np.uint64))
    # Verify frozen multiplication on the needed signed, nonzero numerator pairs.
    n=-x[:,0]*(x[:,2]-x[:,3]);d=x[:,0]-x[:,1]
    assert np.all(n!=0) and np.all(d!=0)
    with torch.no_grad():pm=multiply(torch.from_numpy(np.column_stack((-x[:,0],x[:,2]-x[:,3])))).numpy()
    assert np.array_equal(pm.view(np.uint64),n.view(np.uint64))
    train=x[fit];history=[]
    # Continuous intermediate supervision is derived from training states only.
    for name,ix,design,target in (
        ('negation',[0],train[:,0,None],-train[:,0]),
        ('coordinate_difference',[1,2],train[:,2:4],train[:,2]-train[:,3]),
        ('function_difference',[3,4],train[:,:2],train[:,0]-train[:,1])):
        scale=abs(design).max(1);des=design/scale[:,None];t=target/scale
        # Exact normalized labels preserve the intended subtraction structure.
        if len(ix)==2:t=des[:,0]-des[:,1]
        else:t=-des[:,0]
        for step in range(12):
            w=model.affine[ix].detach().numpy().copy();pred=w[0]*des[:,0]
            if len(ix)==2:pred=pred+w[1]*des[:,1]
            res=t-pred;history.append(dict(part=name,update=step,weights=w.tolist(),max_error=float(abs(res).max())))
            if np.all(res==0):break
            proposed=w+np.linalg.lstsq(des,res,rcond=None)[0]
            if np.array_equal(proposed,w):break
            with torch.no_grad():model.affine[ix]=torch.from_numpy(proposed)
    nt=torch.from_numpy(n[fit]);dt=torch.from_numpy(d[fit]);yt=y[fit]
    design=np.column_stack((np.ones(fit.sum()),np.log(abs(n[fit])),-np.log(abs(d[fit]))))
    target=np.log(abs(yt))
    # Fit the compact signed rational family from random initialization, then
    # refine its actual relative output residual, never assign desired powers.
    for step in range(8):
        w=model.rational.detach().numpy().copy();res=target-design@w
        history.append(dict(part='quotient_log_regression',update=step,weights=w.tolist(),rmse=float(np.sqrt(np.mean(res*res)))))
        proposed=w+np.linalg.lstsq(design,res,rcond=None)[0]
        if np.array_equal(proposed,w):break
        with torch.no_grad():model.rational.copy_(torch.from_numpy(proposed))
    best=None
    for step in range(20):
        with torch.no_grad():pred=model.quotient(nt,dt).numpy()
        residual=(pred-yt)/yt;loss=float(np.mean(residual*residual));w=model.rational.detach().numpy().copy()
        history.append(dict(part='quotient_relative_regression',update=step,weights=w.tolist(),rmse=float(np.sqrt(loss)),max_error=float(abs(residual).max())))
        if best is None or loss<best[0]:best=(loss,w.copy())
        if loss==0:break
        jac=design*(pred/yt)[:,None];proposed=w+np.linalg.lstsq(jac,-residual,rcond=None)[0]
        if np.array_equal(proposed,w):break
        with torch.no_grad():model.rational.copy_(torch.from_numpy(proposed))
    with torch.no_grad():model.rational.copy_(torch.from_numpy(best[1]))
    config=dict(seed=SEED,stage='linear_interpolation_only',trainable_parameters=8,inputs=['fcur','fpre','xcur','xpre'],
        architecture='Learned negation and two 2-weight differences; frozen multiplication; 3-parameter signed power-rational quotient',
        quotient='exp(b)*copysign(abs(n)**p,n) / copysign(abs(d)**q,d)',
        deterministic_arithmetic='Native signed division inside rational layer; no reciprocal-multiplication reassociation. Native abs/copysign/pow and tensor operations remain primitives.',
        operation_order='negate fcur; subtract coordinates; multiply numerator; subtract function values; quotient',
        split='Same complete-parent/solve split as locked components; all independent fresh states excluded from training and checkpoint selection',
        fit_examples=int(fit.sum()),heldout_examples=int(sum(r['split']=='heldout_archive' for r in metadata)),fresh_examples=int(sum('fresh' in r['split'] for r in metadata)),
        frozen_multiplication=mm['checkpoint'],frozen_eligibility=em['checkpoint'],frozen_branch_selector=bm['checkpoint'],
        teacher='-fcur*(xcur-xpre)/(fcur-fpre) in exact audited order',exact_gate=True)
    cp=OUT/'linear_interpolation_before_evaluation.pt';digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),initial_state=initial,config=config,history=history))
    json_write(OUT/'config.json',config);json_write(OUT/'training_history.json',history)
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    p=model.predict(x);metrics=[]
    for group,mask in [('all',np.ones(len(x),bool)),('saved_18039',ids<818),('fit',fit),('heldout_archive',np.array([r['split']=='heldout_archive' for r in metadata])),
        ('saved_fresh_44',(ids>=774)&(ids<818)),('additional_fresh',ids>=818)]:metrics.append(metric(x[mask],y[mask],p[mask],group))
    errors=np.flatnonzero(p.view(np.uint64)!=y.view(np.uint64));coord_bad=np.flatnonzero((x[:,2]+p).view(np.uint64)!=(x[:,2]+y).view(np.uint64))
    with torch.no_grad():parts=[v.numpy() for v in model.intermediates(torch.from_numpy(x))]
    isolation=dict(negation_bit_errors=int(np.sum(parts[0].view('u8')!=(-x[:,0]).view('u8'))),
        coordinate_difference_bit_errors=int(np.sum(parts[1].view('u8')!=(x[:,2]-x[:,3]).view('u8'))),
        numerator_bit_errors=int(np.sum(parts[2].view('u8')!=n.view('u8'))),denominator_bit_errors=int(np.sum(parts[3].view('u8')!=d.view('u8'))))
    failure_rows=[]
    for i in errors:
        failure_rows.append(dict(**metadata[i],fcur=x[i,0],fpre=x[i,1],xcur=x[i,2],xpre=x[i,3],numerator=n[i],denominator=d[i],
            teacher_step=y[i],learned_step=p[i],teacher_step_hex=y[i].hex(),learned_step_hex=p[i].hex(),step_ULPs=int(ulps(y[i:i+1],p[i:i+1])[0]),
            teacher_proposal=x[i,2]+y[i],learned_proposal=x[i,2]+p[i],proposal_ULPs=int(ulps(x[i:i+1,2]+y[i:i+1],x[i:i+1,2]+p[i:i+1])[0])))
    write_csv(OUT/'validation.csv',metrics);write_csv(OUT/'failure_cases.csv',failure_rows or [dict(status='none')])
    json_write(OUT/'stage_isolation.json',isolation)
    unchanged=all(sha(Path(k))==v for k,v in protected.items());assert unchanged
    passed=not len(errors) and not len(coord_bad)
    result=dict(operator='brent_linear_interpolation',status='OPERATION_GATE_PASSED_REPLAY_PENDING' if passed else 'DIAGNOSTIC_ONLY_STOP',
        operation_gate_passed=passed,checkpoint=str(cp),sha256=digest,affine_weights=model.affine.tolist(),rational_weights=model.rational.tolist(),metrics=metrics,
        isolation=isolation,exact_step_errors=len(errors),exact_coordinate_errors=len(coord_bad),replay_run=False,
        stop_reason=None if passed else 'Exact interpolation proposal gate failed; do not replay or train a later operation.',
        protected_files=len(protected),protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
