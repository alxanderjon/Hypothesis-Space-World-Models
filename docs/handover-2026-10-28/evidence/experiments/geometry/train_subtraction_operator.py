"""Generic two-weight subtraction; exact validation and only dx/df assembly."""
import csv
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from torch import nn

from experiments.geometry import audit_exact_branch_brent as audit
from experiments.geometry import train_branch_location_residual as fresh_source
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'subtraction_operator_training'
SOURCE=ROOT/'exact_branch_switch_brent_audit'
SEED=928202629
HELD_BANDS=[(-900,-801),(-30,-11),(129,256),(800,900)]


class Subtraction(nn.Module):
    def __init__(self):
        super().__init__();self.weights=nn.Parameter(.3*torch.randn(2,dtype=torch.float64))

    def forward(self,x):
        return self.weights[0]*x[...,0]+self.weights[1]*x[...,1]

    def predict(self,x):
        with torch.no_grad():return self(torch.from_numpy(np.asarray(x,dtype=np.float64))).numpy()


def held_band(x):
    _,exponent=np.frexp(abs(x));exponent-=1
    held=np.zeros(x.shape,dtype=bool)
    for lo,hi in HELD_BANDS:held|=(x!=0)&(exponent>=lo)&(exponent<=hi)
    return held.any(1)


def synthetic(seed,n,held=None):
    rng=np.random.default_rng(seed);chunks=[];count=0
    while count<n:
        x=np.ldexp(rng.uniform(1,2,(n,2)),rng.integers(-1074,1023,(n,2)))
        x*=rng.choice([-1.,1.],x.shape)
        with np.errstate(over='ignore'):finite=np.isfinite(x[:,0]-x[:,1])
        mask=finite if held is None else finite&(held_band(x)==held)
        chunks.append(x[mask]);count+=int(mask.sum())
    return np.concatenate(chunks)[:n]


def cancellation(seed,n):
    rng=np.random.default_rng(seed)
    a=np.ldexp(rng.uniform(1,2,n),rng.integers(-1074,1023,n))*rng.choice([-1.,1.],n)
    pairs=[];tags=[]
    for v in a:
        for j in range(-4,5):
            w=v
            for _ in range(abs(j)):w=np.nextafter(w,np.inf if j>0 else -np.inf)
            pairs.append((v,w));tags.append('equality' if j==0 else 'one_ULP' if abs(j)==1 else 'nextafter_2_to_4')
        pairs.append((v,v*(1-np.ldexp(1.,-int(rng.integers(10,50))))));tags.append('near_cancellation')
    return np.array(pairs),np.array(tags)


def metric(x,p,group):
    y=x[:,0]-x[:,1];bits=y.view('u8');pb=p.view('u8');same=bits==pb
    # Count signed-zero mismatches separately from numeric ULP errors.
    zero=(y==0)&(p==0)&(~same)
    def ordered(b):return np.where((b>>63)!=0,~b,b|np.uint64(1<<63))
    a,b=ordered(bits),ordered(pb);u=np.maximum(a,b)-np.minimum(a,b)
    nonzero_bad=(~same)&(~zero)
    err=abs(p-y)
    return dict(group=group,cases=len(x),bit_exact_matches=int(same.sum()),exact_match_rate=float(same.mean()),
        errors=int((~same).sum()),one_ULP_errors=int((nonzero_bad&(u==1)).sum()),larger_errors=int((nonzero_bad&(u>1)).sum()),
        signed_zero_mismatches=int(zero.sum()),max_ULP_error=int(u.max()),max_absolute_error=float(err.max()),
        nonfinite=int((~np.isfinite(p)).sum()))


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);mp.mp.dps=80
    OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('locked','manifest','validated'))})
    protected.update({str(p):sha(p) for p in SOURCE.iterdir() if p.is_file()})
    splits={int(r['case_id']):r['group'] for r in audit.read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
    operands=[];meta=[];assembly=[];assemblymeta=[]
    def add(a,b,case,iteration,kind,group):
        operands.append((float(a),float(b)));meta.append(dict(case_id=case,iteration=iteration,operation=kind,group=group))
    def collect(row,case,group):
        iteration=int(row['iteration']);s=row['normalized_state'];enter=row['entering']
        assembly.extend(((s['xcur'],s['xpre']),(s['fcur'],s['fpre'])))
        assemblymeta.extend((dict(case_id=case,iteration=iteration,operation='dx',group=group),dict(case_id=case,iteration=iteration,operation='df',group=group)))
        if row['refresh_bracket']:add(enter['xcur'],enter['xpre'],case,iteration,'refresh_step',group)
        add(s['xblk'],s['xcur'],case,iteration,'sbis_difference',group)
        if row['stop']:return
        method=row['attempt_method']
        if method=='LINEAR_INTERPOLATION':
            add(s['xcur'],s['xpre'],case,iteration,'linear_dx',group)
            add(s['fcur'],s['fpre'],case,iteration,'linear_df',group)
        elif method=='INVERSE_QUADRATIC_EXTRAPOLATION':
            for aa,bb,kind in ((s['fpre'],s['fcur'],'iq_dpre_num'),(s['xpre'],s['xcur'],'iq_dpre_den'),
                              (s['fblk'],s['fcur'],'iq_dblk_num'),(s['xblk'],s['xcur'],'iq_dblk_den'),
                              (s['fblk']*row['dblk'],s['fpre']*row['dpre'],'iq_cross_difference'),(s['fblk'],s['fpre'],'iq_function_difference')):
                add(aa,bb,case,iteration,kind,group)
        if row['eligible']:add(3*abs(row['sbis']),row['delta'],case,iteration,'acceptance_limit_difference',group)
    with (SOURCE/'brent_iterations.csv').open() as f:
        for raw in csv.DictReader(f):
            row={k:json.loads(v) for k,v in raw.items() if k in ('normalized_state','entering')}
            row.update({k:raw[k]=='True' for k in ('refresh_bracket','stop','eligible')})
            row.update(iteration=int(raw['iteration']),attempt_method=raw['attempt_method'])
            for k in ('dpre','dblk','sbis','delta'):row[k]=float(raw[k]) if raw.get(k) else None
            i=int(raw['case_id']);collect(row,i,splits[i])
    archived_operands=len(operands);archived_assembly=len(assembly)
    # Fresh teacher solves are data generation only. No learned Brent replay,
    # interpolation training or recursive region splitting is executed.
    oldfresh=[]
    for folder in ('brent_interpolation_eligibility_training','brent_interpolation_branch_selector_training'):
        oldfresh+=json.loads((ROOT/folder/'new_fresh_candidates.json').read_text())
    oldseed=fresh_source.SEED
    try:
        fresh_source.SEED=SEED+19
        newfresh,parents,fa=fresh_source.fresh_data()
    finally:fresh_source.SEED=oldseed
    assert newfresh
    json_write(OUT/'fresh_candidates.json',newfresh);json_write(OUT/'fresh_generation.json',dict(parents=parents,audit=fa))
    auditobj=audit.Audit.__new__(audit.Audit)
    for j,r in enumerate(oldfresh+newfresh):
        auditobj.setup(r['parent'],r['axis']);k=r['k']
        def callback(c):
            f=auditobj.formula(c,k);return float(f[1][0]),{}
        root,tr,_,_=audit.traced_brent(callback,r['c_left'],r['c_right'],818+j)
        assert root==r['teacher_cut']
        for row in tr:collect(row,818+j,'additional_saved_fresh' if j<len(oldfresh) else 'brand_new_fresh')
    bx=np.array(operands);ax=np.array(assembly)
    assert np.isfinite(bx).all() and np.isfinite(bx[:,0]-bx[:,1]).all()
    roles=np.array([m['group'] for m in meta]);fit=(roles=='fit')&~held_band(bx)
    ordinary=synthetic(SEED+1,8192,False)
    train_c,_=cancellation(SEED+2,1024);train_c=train_c[~held_band(train_c)]
    # Strong independent axis anchors prevent the near-collinear Brent pairs
    # from leaving a common multiplicative scale weakly identified.
    values=ordinary[:4096,0]
    left=np.column_stack((values,np.zeros(len(values))));right=np.column_stack((np.zeros(len(values)),values))
    equal=np.column_stack((values,values))
    rawtrain=np.concatenate((bx[fit],ordinary,train_c,left,right,equal))
    scale=abs(rawtrain).max(1)
    design=np.divide(rawtrain,scale[:,None],out=np.zeros_like(rawtrain),where=scale[:,None]>0)
    target=design[:,0]-design[:,1]
    # Add explicit linear symmetry constraints S(x,y)+S(y,x)=0. No weights
    # are tied or set to target coefficients; these are supervised examples.
    sums=(design[:8192,0]+design[:8192,1])*.5
    sym=np.column_stack((sums,sums))
    design=np.concatenate((design,np.tile([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],(4096,1)),sym))
    target=np.r_[target,np.tile([1.,-1.,-1.,1.],4096),np.zeros(len(sym))]
    model=Subtraction();initial=model.weights.detach().numpy().copy();history=[]
    assert initial[0]!=1 and initial[1]!=-1
    for step in range(16):
        pred=model.predict(design);res=target-pred;w=model.weights.detach().numpy().copy()
        history.append(dict(update=step,w1=float(w[0]),w2=float(w[1]),rmse=float(np.sqrt(np.mean(res*res))),max_error=float(abs(res).max())))
        if np.all(res==0):break
        proposed=w+np.linalg.lstsq(design,res,rcond=None)[0]
        if np.array_equal(proposed,w):break
        with torch.no_grad():model.weights.copy_(torch.from_numpy(proposed))
    config=dict(seed=SEED,model='w1*x+w2*y',dtype='float64',bias=False,parameters=2,initial_weights=initial.tolist(),
        training='Full-batch supervised residual least-squares updates; no snapping or coefficient targets',
        preprocessing='Training-only common scaling creates ordinary float64 subtraction examples; inference is raw scalar inputs with no normalization or zero correction.',
        structural_supervision='S(x,0),S(0,y),S(x,x),antisymmetry constraints; strong independent axis anchors',
        training_rows=len(design),archived_training_operands=int(fit.sum()),heldout_magnitude_bands=HELD_BANDS,
        heldout_rule='Complete parent/solve groups from locked Brent split; exclude every operand in held-out magnitude bands from fitting; all independent fresh solves held out',
        source_saved_iteration_rows=27867,archived_executed_subtraction_operands=archived_operands,
        sources='All executed subtraction sites in saved Brent: refresh, sbis, linear, inverse-quadratic, acceptance. Separate dx/df assembly pairs from every saved/fresh state.',
        no_Brent_training=True,no_checkpoint_changes=True,no_recursion=True)
    checkpoint=OUT/'subtraction_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,training_history=history))
    json_write(OUT/'config.json',config);write_csv(OUT/'training_history.csv',history)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    print(json.dumps(dict(event='saved',weights=model.weights.tolist(),sha256=digest,training_rows=len(design))),flush=True)
    # Save complete operands and provenance for repeatable independent evaluation.
    np.savez_compressed(OUT/'Brent_subtraction_operands.npz',operands=bx,assembly_operands=ax)
    write_csv(OUT/'operand_metadata.csv',meta);write_csv(OUT/'assembly_metadata.csv',assemblymeta)
    metrics=[];pred=model.predict(bx)
    metrics.append(metric(bx,pred,'all_executed_Brent_subtractions'))
    metrics.append(metric(bx[:archived_operands],pred[:archived_operands],'all_saved_Brent_subtractions'))
    for group in sorted(set(roles)):
        m=roles==group;metrics.append(metric(bx[m],pred[m],group))
    for op in sorted({r['operation'] for r in meta}):
        m=np.array([r['operation']==op for r in meta]);metrics.append(metric(bx[m],pred[m],op))
    heldsyn=synthetic(SEED+3,16384,True);broad=synthetic(SEED+4,16384)
    near,tags=cancellation(SEED+5,4096)
    # Exhaust every finite exponent boundary +/-4 representable values.
    powers=np.ldexp(np.ones(2098),np.arange(-1074,1024));neighbors=[]
    for value in powers:
        for sign in (-1.,1.):
            v=sign*value
            for direction in (-np.inf,np.inf):
                w=v
                for _ in range(4):
                    w=np.nextafter(w,direction)
                    if np.isfinite(w):neighbors.extend(((v,w),(w,v)))
    neighbors=np.array(neighbors)
    zeros=np.array([[a,b] for a in (0.,-0.) for b in (0.,-0.)])
    extremes=np.array([[a,b] for a in (np.finfo(float).max,-np.finfo(float).max,np.finfo(float).tiny,-np.finfo(float).tiny,0.,-0.)
                       for b in (0.,-0.,np.nextafter(0.,1.),-np.nextafter(0.,1.))])
    for name,data in [('heldout_magnitude_bands',heldsyn),('broad_random',broad),('cancellation',near),('binary_boundaries_nextafter',neighbors),('signed_zero_pairs',zeros),('extreme_finite_cases',extremes)]:
        metrics.append(metric(data,model.predict(data),name))
    for tag in sorted(set(tags)):
        data=near[tags==tag];metrics.append(metric(data,model.predict(data),tag))
    # Structural signed-zero semantics are reported against native arithmetic.
    # Antisymmetry at exact cancellation is numerically zero on both sides;
    # unary negation changes its sign bit, as it does for the native teacher.
    values=np.r_[broad[:,0],0.,-0.];z=np.zeros(len(values))
    structural=[]
    for name,data in [('S(x,0)',np.column_stack((values,z))),('S(0,y)',np.column_stack((z,values))),('S(x,x)',np.column_stack((values,values)))]:
        pp=model.predict(data);mm=metric(data,pp,name);metrics.append(mm)
        structural.append(dict(identity=name,examples=len(data),bit_errors=mm['errors'],negative_zero_outputs=int(((pp==0)&np.signbit(pp)).sum())))
    p1=model.predict(broad);p2=-model.predict(broad[:,::-1].copy());t1=broad[:,0]-broad[:,1];t2=-(broad[:,1]-broad[:,0])
    structural.append(dict(identity='antisymmetry',examples=len(broad),bit_errors=int((p1.view('u8')!=p2.view('u8')).sum()),
        native_teacher_bit_differences=int((t1.view('u8')!=t2.view('u8')).sum()),numeric_errors=int((p1!=p2).sum())))
    write_csv(OUT/'validation.csv',metrics);write_csv(OUT/'structural_identities.csv',structural)
    failures=np.flatnonzero(pred.view('u8')!=(bx[:,0]-bx[:,1]).view('u8'))
    write_csv(OUT/'Brent_failures.csv',[dict(**meta[i],x=bx[i,0],y=bx[i,1],teacher=bx[i,0]-bx[i,1],learned=pred[i]) for i in failures] or [dict(status='none')])
    passed=all(r['errors']==0 for r in metrics)
    assembly_rows=[]
    if passed:
        # The ONLY assembly check: replace dx and df independently. No product,
        # quotient, interpolation proposal or learned Brent replay is assembled.
        ap=model.predict(ax)
        for name in ('dx','df'):
            m=np.array([r['operation']==name for r in assemblymeta]);assembly_rows.append(metric(ax[m],ap[m],name))
        for name in sorted({r['group'] for r in assemblymeta}):
            m=np.array([r['group']==name for r in assemblymeta]);assembly_rows.append(metric(ax[m],ap[m],name))
        write_csv(OUT/'two_subtraction_assembly.csv',assembly_rows)
        passed=all(r['errors']==0 for r in assembly_rows)
    unchanged=all(sha(Path(p))==v for p,v in protected.items());assert unchanged
    result=dict(operator='subtraction_operator',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        module='experiments.geometry.train_subtraction_operator',model='Subtraction',implementation_sha256=sha(Path(__file__)),
        checkpoint=str(checkpoint),sha256=digest,learned_weights=model.weights.tolist(),training_updates=len(history)-1,
        saved_Brent_solves=818,additional_saved_fresh_solves=len(oldfresh),brand_new_fresh_solves=len(newfresh),metrics=metrics,
        structural_identities=structural,assembly=assembly_rows,full_interpolation_assembled=False,
        protected_files=len(protected),protected_hashes_unchanged=unchanged,no_recursion=True,elapsed_seconds=time.monotonic()-started)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    if passed:json_write(OUT/'subtraction_operator_locked.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
