#!/usr/bin/env python3
"""Frozen one-step retained-N2-head versus exact-refitted-head preflight."""
from fractions import Fraction
import fcntl, hashlib, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize._lsq.common import CL_scaling_vector

from experiments.sho.exact_rational_bvls import exact_bvls
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import residual_vector, residual_and_jacobian
from experiments.sho.run_checkpoint_controller_factorial16_initializers import clean, save, sha, bits_equal

BASE=Path('outputs/sho/autonomous_generic_sho_20261014')
PROTOCOL=BASE/'protocols/generic_n2_n3_head_state_persistence16.json'
DESIGN=BASE/'reviews/generic_n2_n3_head_state_persistence16_design.md'
MATRIX=BASE/'reviews/generic_n2_n3_post_anchor_ranked_hypothesis_matrix.md'
FACTORIAL=BASE/'screens/generic_n2_n3_checkpoint_controller_factorial16'
DATA=Path('outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz')
OUT=BASE/'diagnostics/generic_n2_n3_head_state_persistence16'
INNER=200

def pack(x):
    if isinstance(x,Fraction):return f'{x.numerator}/{x.denominator}'
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x

def append(p,x):
    with p.open('a') as f:f.write(json.dumps(clean(x),allow_nan=False)+'\n')

def first_profile(directory):
    with (directory/'profiles.jsonl').open() as f:return json.loads(next(f))

def exact_profile(adapter,trajectory,hidden,lower,upper,out,label,index):
    h=torch.tensor(np.asarray(hidden,np.float64),dtype=torch.float64)
    zero=torch.cat((h,torch.zeros(10,dtype=torch.float64)))
    _,J0=residual_and_jacobian(adapter,trajectory,zero);A=J0[:,12:];anchor=-trajectory[1:].reshape(-1)
    sol=exact_bvls([[Fraction(v) for v in row] for row in A.tolist()],
        [Fraction(v) for v in anchor.tolist()],[Fraction(v) for v in lower[12:].tolist()],
        [Fraction(v) for v in upper[12:].tolist()],max_iterations=INNER)
    cert=f'exact_head_{index:02d}_{label}.json';save(out/cert,{'certificate':pack(sol)})
    if not all(sol[k] is True for k in ('exact_feasible','exact_KKT','full_rank_exact')):raise RuntimeError('TECHNICAL_STOP: exact head certificate failed')
    head=torch.tensor([float(v) for v in sol['direction']],dtype=torch.float64);phi=torch.cat((h,head))
    with torch.no_grad():loss=float(adapter.objective(phi,trajectory))
    return {'phi':phi,'head':head,'loss':loss,'active':sol['active'],'certificate':cert,'certificate_sha256':sha(out/cert)}

def arm(adapter,trajectory,center_phi,expected_loss,lower,upper,out,label,index):
    phi=torch.tensor(np.asarray(center_phi,np.float64),dtype=torch.float64)
    with torch.no_grad():replay=float(adapter.objective(phi,trajectory))
    if not bits_equal(replay,expected_loss):raise RuntimeError('ALERT: center native replay mismatch')
    r,J=residual_and_jacobian(adapter,trajectory,phi)
    if not bits_equal(replay,float((r*r).sum()/r.numel())):raise RuntimeError('ALERT: center residual mismatch')
    g=J[:,:12].T@r;v,dv=CL_scaling_vector(phi[:12].numpy(),g.numpy(),lower[:12].numpy(),upper[:12].numpy())
    if not np.all(np.isfinite(v)) or np.any(v<0):return {'status':'NUMERICAL_UNAVAILABLE_INVALID_SCALE','center_loss':replay,'v':v}
    d=np.sqrt(v)
    if np.any(d==0):return {'status':'NUMERICAL_UNAVAILABLE_ZERO_INITIAL_SCALE','center_loss':replay,'v':v,'d':d}
    Delta=float(np.linalg.norm(phi[:12].numpy()/d))
    if Delta==0:Delta=1.0
    if not math.isfinite(Delta) or Delta<=0:return {'status':'NUMERICAL_UNAVAILABLE_INVALID_RADIUS','center_loss':replay,'v':v,'d':d,'radius':Delta}
    dt=torch.from_numpy(d.copy());lo,hi=lower.clone(),upper.clone()
    lo[:12]=torch.maximum(lower[:12],phi[:12]-Delta*dt);hi[:12]=torch.minimum(upper[:12],phi[:12]+Delta*dt)
    fixed=torch.where(lo==hi)[0];mask=torch.zeros(22,dtype=torch.bool);mask[fixed]=True;keep=torch.where(~mask)[0]
    rel_lo,rel_hi=lo-phi,hi-phi
    sol=exact_bvls([[Fraction(x) for x in row] for row in J[:,keep].tolist()],
        [Fraction(x) for x in r.tolist()],[Fraction(x) for x in rel_lo[keep].tolist()],
        [Fraction(x) for x in rel_hi[keep].tolist()],max_iterations=INNER)
    cert=f'exact_joint_{index:02d}_{label}.json';save(out/cert,{'kept_indices':keep,'fixed_indices':fixed,'certificate':pack(sol)})
    if not all(sol[k] is True for k in ('exact_feasible','exact_KKT','full_rank_exact')):raise RuntimeError('TECHNICAL_STOP: exact joint certificate failed')
    step=torch.zeros(22,dtype=torch.float64);step[keep]=torch.tensor([float(x) for x in sol['direction']],dtype=torch.float64)
    candidate=phi+step;lh=step==rel_lo;uh=step==rel_hi;candidate=torch.where(lh,lo,candidate);candidate=torch.where(uh,hi,candidate);p=candidate-phi
    if not bool(((candidate>=lo)&(candidate<=hi)).all()):raise RuntimeError('NUMERICAL_STOP: candidate infeasible')
    scaled=p[:12].abs()/dt;step_norm=float(scaled.max());model_r=r+J@p;pred=float(.5*((r@r)-(model_r@model_r)))
    with torch.no_grad():direct_loss=float(adapter.objective(candidate,trajectory))
    prof=exact_profile(adapter,trajectory,candidate[:12].numpy(),lower,upper,out,label,index)
    actual_direct=(replay-direct_loss)*r.numel()/2;actual_profiled=(replay-prof['loss'])*r.numel()/2
    return {'status':'COMPLETE','center_phi':phi,'center_loss':replay,'center_residual':r,'J':J,
        'hidden_gradient':g,'ColemanLi_v':v,'ColemanLi_dv':dv,'d':d,'radius':Delta,
        'candidate_phi':candidate,'step':p,'hidden_step':p[:12],'scaled_hidden_step_norm':step_norm,
        'joint_prediction_half_sum':pred,'joint_prediction_MSE':2*pred/r.numel(),
        'direct_candidate_loss':direct_loss,'profiled_candidate_phi':prof['phi'],'profiled_candidate_loss':prof['loss'],
        'direct_actual_MSE_decrease':replay-direct_loss,'profiled_actual_MSE_decrease':replay-prof['loss'],
        'direct_fidelity':actual_direct/pred if pred>0 else None,'profiled_fidelity':actual_profiled/pred if pred>0 else None,
        'joint_active':sol['active'],'fixed_indices':fixed,'joint_certificate':cert,'joint_certificate_sha256':sha(out/cert),
        'profile_certificate':prof['certificate'],'profile_certificate_sha256':prof['certificate_sha256'],
        'direct_head_to_profiled_candidate_head_norm':float(torch.linalg.vector_norm(candidate[12:]-prof['head']))}

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if OUT.exists():raise RuntimeError('fresh output required')
    protocol=json.loads(PROTOCOL.read_text());OUT.mkdir(parents=True)
    lock=(BASE/'active.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('TECHNICAL_UNAVAILABLE: numerical lock held')
    inputs={str(p.resolve()):sha(p) for p in (PROTOCOL,DESIGN,MATRIX,DATA,Path(__file__),
        Path('experiments/sho/exact_rational_bvls.py'),Path('experiments/sho/generic_update_network_adapter.py'),
        Path('experiments/sho/generic_update_residual_n3_opt_in.py'),FACTORIAL/'final_integrity.json')}
    rows=[];arm_index=0
    try:
      with np.load(DATA) as corpus:
       for case,spec in protocol_source_specs().items():
        init=BASE/'initializers'/spec['initializer_run_id'];checkpoints=json.loads((init/'checkpoints.json').read_text())
        inputs[str((init/'artifact_manifest.json').resolve())]=sha(init/'artifact_manifest.json')
        trajectory=torch.from_numpy(corpus[spec['trajectory_key']].copy());adapter=GenericSHOUpdateNetworkAdapter()
        lower=torch.from_numpy(adapter.lower.copy());upper=torch.from_numpy(adapter.upper.copy())
        for cp in ('n2_prefix500','n2_endpoint1500'):
            retained=np.asarray(checkpoints[cp]['theta'],np.float64);retained_loss=float(checkpoints[cp]['loss'])
            baseline=FACTORIAL/'cells'/f'{case}__{cp}__unchanged_baseline';fp=first_profile(baseline)
            profiled=np.asarray(fp['phi'],np.float64);profiled_loss=float(fp['loss'])
            if not bits_equal(retained[:12],profiled[:12]):raise RuntimeError('ALERT: paired hidden mismatch')
            pair={'case':case,'checkpoint':cp,'initial_head_gap':float(np.linalg.norm(retained[12:]-profiled[12:])),
                  'common_profiled_center_loss':profiled_loss,'arms':{}}
            for label,center,loss in (('retained_N2_head',retained,retained_loss),('global_exact_refit_head',profiled,profiled_loss)):
                arm_index+=1
                try:pair['arms'][label]=arm(adapter,trajectory,center,loss,lower,upper,OUT,label,arm_index)
                except BaseException as exc:pair['arms'][label]={'status':'TECHNICAL_UNAVAILABLE','error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()}
                append(OUT/'arms.jsonl',{'case':case,'checkpoint':cp,'head_state':label,'result':pair['arms'][label]})
            a,b=pair['arms']['retained_N2_head'],pair['arms']['global_exact_refit_head']
            if a['status']=='COMPLETE' and b['status']=='COMPLETE':
                g0=pair['initial_head_gap'];g1=a['direct_head_to_profiled_candidate_head_norm'];ratio=g1/g0 if g0>0 else None
                loss_ratio=a['profiled_candidate_loss']/b['profiled_candidate_loss']
                ia=a['profiled_candidate_loss']<profiled_loss;ib=b['profiled_candidate_loss']<profiled_loss
                material=loss_ratio>1.01 or loss_ratio<1/1.01
                pair.update(retained_head_persistence_ratio=ratio,profiled_candidate_loss_ratio_retained_over_refit=loss_ratio,
                    retained_profiled_candidate_improves_common_center=ia,refit_profiled_candidate_improves_common_center=ib,
                    improvement_disposition_differs=ia!=ib,material_profiled_loss_difference=material,
                    primary_support_cell=bool(g0>0 and ratio>0.01 and (material or ia!=ib)))
            else:pair.update(retained_head_persistence_ratio=None,profiled_candidate_loss_ratio_retained_over_refit=None,
                improvement_disposition_differs=None,material_profiled_loss_difference=None,primary_support_cell=False)
            rows.append(pair);save(OUT/'progress.json',{'rows':rows})
      prefix=[r for r in rows if r['checkpoint']=='n2_prefix500'];support=sum(r['primary_support_cell'] for r in prefix)
      result={'status':'HEAD_STATE_PERSISTENCE16_COMPLETE','rows':rows,'primary_prefix_support_cases':support,
        'primary_gate_met':support>=3,'declared_arms':16,'complete_arms':sum(a['status']=='COMPLETE' for r in rows for a in r['arms'].values()),
        'unavailable_arms':sum(a['status']!='COMPLETE' for r in rows for a in r['arms'].values()),
        'work':{'center_native_replays':16,'center_derivative_attempts':16,
            'exact_joint_models_completed':sum(a['status']=='COMPLETE' for r in rows for a in r['arms'].values()),
            'direct_candidate_native_objectives':sum(a['status']=='COMPLETE' for r in rows for a in r['arms'].values()),
            'candidate_exact_head_profiles':sum(a['status']=='COMPLETE' for r in rows for a in r['arms'].values()),
            'accepted_updates':0,'trust_updates':0,'rollouts':0},
        'interpretation':'Passing supports only a separately frozen short continuation; failure closes simple retained-N2-head persistence as the primary missing mechanism.'}
      save(OUT/'result.json',result)
      checks={p:sha(p)==h for p,h in inputs.items()};save(OUT/'integrity.json',{'unchanged':all(checks.values()),'checks':checks})
      if not all(checks.values()):raise RuntimeError('ALERT: protected input changed')
      save(OUT/'artifact_manifest.json',{'sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
      print(json.dumps(clean({k:v for k,v in result.items() if k!='rows'}),indent=2))
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

def protocol_source_specs():
    frozen=json.loads((BASE/'protocols/generic_n2_n3_checkpoint_controller_factorial16_design.json').read_text())
    return frozen['source_eligibility']['cases']

if __name__=='__main__':main()
