"""Identifiability, containment, and modest recovery; no checkpoint training."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from experiments.kepler import run_four_parameter_peelback as old
from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.sho import continuous_surrogate_edirect as search
from experiments.sho.deterministic_full_k_framework import hashes

OUT=Path('outputs/known_structure_audit')
GT=np.array([.125,.5,1.,1.,3.])
LOW=np.array([-.25,-1.,-2.,-.5,1.5])
HIGH=np.array([.5,2.,3.,2.5,4.5])

def step(s,t,lib):
    c1,c2,eta,drift,p=t
    dt=float(old.kepler.DT)
    half_dt=0.5*dt
    stack=lambda a: lib.stack(a,dim=-1) if lib is torch else lib.stack(a,axis=-1)
    def field(v):
        dx=v[...,4]-v[...,0]; dy=v[...,5]-v[...,1]
        radius_squared=dx*dx+dy*dy
        radius=lib.sqrt(radius_squared)
        den=lib.pow(radius,p) if lib is torch else np.power(radius,p)
        # Preserve production: common=(G*strength)/denominator at strength=1,
        # then mass scaling.  The new identifiable amplitudes enter afterward
        # as exact multiply-by-one factors at their teacher values.
        one=(torch.as_tensor(1.,dtype=v.dtype) if lib is torch else np.float64(1.))
        g=(torch.as_tensor(float(old.kepler.G),dtype=v.dtype) if lib is torch else np.float64(old.kepler.G))
        m1=(torch.as_tensor(float(old.kepler.MASS_1),dtype=v.dtype) if lib is torch else np.float64(old.kepler.MASS_1))
        m2=(torch.as_tensor(float(old.kepler.MASS_2),dtype=v.dtype) if lib is torch else np.float64(old.kepler.MASS_2))
        common=g*one/den
        base1=common*m2; base2=common*m1
        scale1=base1*(c1/GT[0]);scale2=base2*(c2/GT[1])
        return scale1*dx,scale1*dy,-scale2*dx,-scale2*dy
    a1x,a1y,a2x,a2y=field(s)
    vx=s[...,2]+half_dt*a1x; vy=s[...,3]+half_dt*a1y
    wx=s[...,6]+half_dt*a2x; wy=s[...,7]+half_dt*a2y
    # kd=1 executes multiply-by-one before the original dt multiplication.
    x=s[...,0]+dt*(drift*vx); y=s[...,1]+dt*(drift*vy)
    z=s[...,4]+dt*(drift*wx); w=s[...,5]+dt*(drift*wy)
    mid=stack((x,y,vx,vy,z,w,wx,wy))
    b1x,b1y,b2x,b2y=field(mid)
    # eta=1 executes multiply-by-one before the original half_dt product.
    return stack((x,y,vx+half_dt*(eta*b1x),vy+half_dt*(eta*b1y),
                  z,w,wx+half_dt*(eta*b2x),wy+half_dt*(eta*b2y)))

def rank(fn,theta):
    jac=torch.autograd.functional.jacobian(fn,torch.tensor(theta)).detach().numpy().reshape(-1,len(theta))
    sv=np.linalg.svd(jac,compute_uv=False)
    tol=max(jac.shape)*np.finfo(float).eps*sv[0]
    return dict(rank=int(sum(sv>tol)),dimension=len(theta),singular_values=sv.tolist(),
                condition=float(sv[0]/sv[-1]),rank_tolerance=float(tol))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=hashes(); report={'checkpoint_count':len(before),'sho':[],'ball':[],'kepler':[]}
    data=np.load('outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz')
    for i in (0,51,100,200):
        states=data[f'prefix_{i}']; gt=data['teachers'].reshape(-1,4)[i]
        x=torch.tensor(states[:-1]); expected=states[1:]
        pred=(x@torch.tensor(gt).reshape(2,2).T).numpy()
        truth=states[-1].copy(); value=torch.tensor(truth); rollout_error=0.
        for _ in range(4096):
            truth=gt.reshape(2,2)@truth
            value=torch.tensor(gt).reshape(2,2)@value
            rollout_error=max(rollout_error,float(np.max(abs(value.numpy()-truth))))
        report['sho'].append(dict(case=i,teacher=gt.tolist(),
            one_step_max=float(np.max(abs(pred-expected))),rollout_max=rollout_error,
            jacobian=rank(lambda t:(x@t.reshape(2,2).T).flatten(),gt)))
    bt=ball._trajectories()
    for i in (0,1,3,11):
        x=torch.tensor(bt[i,:127])
        report['ball'].append(dict(case=i,impacts=sum(len(ball.event_signature(s,ball.TEACHER)) for s in bt[i,:127]),
            jacobian=rank(lambda t:torch.stack([ball.torch_step(s,t) for s in x]).flatten(),ball.TEACHER)))
    report['ball_containment']=ball.validate_teacher(bt)
    trajectories=old.trajectories()
    for i,tr in enumerate(trajectories):
        x=torch.tensor(tr[:127]); gt=torch.tensor(GT)
        n=step(tr[:127],GT,np); t=step(x,gt,torch).detach().numpy()
        v=tr[0].copy(); maximum=0.
        for target in tr[1:]:
            v=step(v,GT,np);maximum=max(maximum,float(np.max(abs(v-target))))
        report['kepler'].append(dict(case=i,numpy_one_step=float(np.max(abs(n-tr[1:128]))),
            torch_one_step=float(np.max(abs(t-tr[1:128]))),rollout_max=maximum,
            jacobian=rank(lambda t:step(x,t,torch).flatten(),GT)))
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    # Existing Kepler containment gate was bit-exact. Stop before search if
    # reassociation of the identifiable coefficient products changes its output.
    if any(r['numpy_one_step'] or r['torch_one_step'] or r['rollout_max'] for r in report['kepler']):
        report['status']='STOPPED_KEPLER_CONTAINMENT'
    else:
        report['recovery']=[]
        for i in (0,3,7):
            tr=trajectories[i];x=torch.tensor(tr[:127]);y=torch.tensor(tr[1:128])
            def obj(t):
                with np.errstate(all='ignore'):
                    f=float(np.mean((step(tr[:127],t,np)-tr[1:128])**2))
                return f if np.isfinite(f) else 1e100
            def tor(t):return (step(x,t,torch)-y).square().mean()
            # Historical absolute loss gates are diagnostic here.  Preserve the
            # search policy and terminate this modest screen at its resource cap.
            started=time.time();n1=search.search(obj,LOW,HIGH,91000+i,cap=1000,gate=-math.inf)
            _,_,run=old._make_n3(tor,LOW,HIGH)
            n1_direct=old._run_n3(n1['theta'],n1['loss'],'n1_endpoint',run)
            n2=old._run_n2(obj,n1['theta'],n1['loss'],n1['lower'],n1['upper'])
            n3=old._run_n3(n2['theta'],n2['loss'],'known_structure',run)
            final=np.array(n3['final_theta']);v=tr[127].copy();mx=0.
            for target in tr[128:]:
                v=step(v,final,np);mx=max(mx,float(np.max(abs(v-target))))
            row=dict(case=i,n1_evals=n1['evaluations'],n1_loss=n1['loss'],
                n1_theta=n1['theta'].tolist(),n1_newton_correctable=n1_direct['recovered'],
                teacher_inside_n2_box=bool(np.all(GT>=n1['lower']) and np.all(GT<=n1['upper'])),
                n2_evals=n2['evaluations'],n2_loss=n2['loss'],n3=n3,
                parameter_error=(final-GT).tolist(),rollout_error=mx,seconds=time.time()-started)
            report['recovery'].append(row);print(json.dumps(row),flush=True)
            (OUT/'audit.json').write_text(json.dumps(report,indent=2))
            if not n3['recovered']:break
        report['status']=('RECOVERY_SCREEN_COMPLETE' if report['recovery']
                          and all(row['n3']['recovered'] for row in report['recovery'])
                          else 'FAILED_RECOVERY_SCREEN_N1_LOCALIZATION')
    report['checkpoint_hashes_unchanged']=before==hashes()
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
