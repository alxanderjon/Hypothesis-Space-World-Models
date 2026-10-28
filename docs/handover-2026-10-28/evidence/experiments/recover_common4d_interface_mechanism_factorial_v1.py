#!/usr/bin/env python3
"""Prediction-only recovery for common4d_interface_mechanism_factorial_v1."""
from __future__ import annotations
import json,time,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
import common4d_interface_mechanism_factorial_v1 as f
from heterogeneous_entity_context_pilot_v1 import Model,aj,an,sha,rss

PARENT=ROOT/'outputs/common4d_interface_mechanism_factorial_v1'
OUT=ROOT/'outputs/common4d_interface_mechanism_factorial_v1_recovery1'
WALL=120.; RSS_CAP=1<<30; ART_CAP=32<<20
LIMITS={'repeated_generated_transition_rows':9984,'repeated_signed_transform_rows':6144,'checkpoint_loads':4,'new_evaluation_rows':32768,'score_blocks':512,'fits':0,'optimizer_updates':0}
C={k:0 for k in LIMITS}
def charge(k,n=1):
 if C[k]+n>LIMITS[k]: raise RuntimeError(f'PRECALL_CAP:{k}:{C[k]}+{n}>{LIMITS[k]}')
 C[k]+=n
def transform(x): charge('repeated_signed_transform_rows',int(np.prod(x.shape[:-1]))); return x[...,f.PERM]*f.SIGN
def step(x,law): charge('repeated_generated_transition_rows',len(x)); return f.step.__wrapped__(x,law) if hasattr(f.step,'__wrapped__') else raw_step(x,law)
def raw_step(x,law):
 kind,A,a,p=law
 if kind=='cubic': force=x@A.T+a*(f.p3(x)@p)
 else:
  w1,b1,w2,b2=p; force=x@A.T+a*(np.tanh(x@w1+b1)@w2+b2)
 return f.DT*force
def make(laws,cx,qx): return {'cx':np.broadcast_to(cx,(len(laws),)+cx.shape).copy(),'cd':np.stack([step(cx,l) for l in laws]),'qx':np.broadcast_to(qx,(len(laws),)+qx.shape).copy(),'qy':np.stack([step(qx,l) for l in laws])}
def warped(d): return {k:transform(v) for k,v in d.items()}
def predict(m,train,dev,blind=True):
 charge('new_evaluation_rows',len(dev['cx'])*64*(3 if blind else 2))
 cx,cd,qx,em,cm,dt,sd=f.tensors(dev,range(len(dev['cx']))); wrong=np.roll(np.arange(len(dev['cx'])),-1)
 wcx,wcd,_,wem,wcm,wdt,wsd=f.tensors(dev,wrong)
 with torch.no_grad():
  code=m.context_code(cx,cd,em,cm,dt); wcode=m.context_code(wcx,wcd,wem,wcm,wdt)
  correct=m.predict(qx,em,cm,dt,code)*sd
  sxw,_=f.scales(dev); fqx=torch.as_tensor(dev['qx']/sxw[wrong,None])[:,:,None,:]
  full=m.predict(fqx,em,cm,wdt,wcode)*wsd
  out={'correct':correct[:,:,0].numpy(),'fullwrong':full[:,:,0].numpy()}
  if blind:
   tcx,tcd,_,tem,tcm,tdt,_=f.tensors(train,range(len(train['cx']))); mean=m.context_code(tcx,tcd,tem,tcm,tdt).mean(0,keepdim=True).expand(len(dev['cx']),-1)
   out['blind']=(m.predict(qx,em,cm,dt,mean)*sd)[:,:,0].numpy()
 return out
def main():
 t0=time.monotonic()
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 parent_plan=json.load(open(PARENT/'plan.json')); terminal=json.load(open(PARENT/'terminal_error.json'))
 plan={'status':'RECOVERY_PRECHARGED_BEFORE_REPEATED_WORK','limits':LIMITS,'parent_plan_sha256':sha(PARENT/'plan.json'),'parent_terminal_sha256':sha(PARENT/'terminal_error.json'),'parent_checkpoint_sha256':{},'cumulative_after_success':{'generated_transition_rows':19968,'signed_transform_rows':12288,'fits':4,'optimizer_updates':2000,'evaluation_rows_attempted':35840,'score_blocks':512},'ceilings':{'wall_seconds':WALL,'peak_RSS_bytes':RSS_CAP,'artifact_bytes':ART_CAP},'source_hashes':{'driver':sha(Path(__file__)),'recovery_proposal':sha(ROOT/'coordination/COMMON4D_INTERFACE_MECHANISM_FACTORIAL_V1_RECOVERY1.md')}}
 for seed in f.SEEDS:
  for arm in ('shared','scratch'):
   p=PARENT/f'{arm}_seed{seed}.pt'; plan['parent_checkpoint_sha256'][p.name]=sha(p)
 aj(OUT/'plan.json',plan)
 r=np.random.default_rng(2026102917); cx=r.uniform(-1,1,(32,4)); qx=r.uniform(-1,1,(64,4))
 train=make([f.cubic_law(r) for _ in range(48)],cx,qx); scratch=make([f.tanh_law(r) for _ in range(24)],cx,qx); familiar=make([f.cubic_law(r) for _ in range(16)],cx,qx); held=make([f.tanh_law(r) for _ in range(16)],cx,qx); fw,hw=warped(familiar),warped(held)
 models={}
 for seed in f.SEEDS:
  for arm in ('shared','scratch'):
   charge('checkpoint_loads'); m=Model(); m.load_state_dict(torch.load(PARENT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)['state_dict']); m.eval(); models[(arm,seed)]=m
 preds={}
 for seed in f.SEEDS:
  for cell,d in (('familiar_native',familiar),('familiar_transformed',fw),('held_native',held),('held_transformed',hw)):
   for mode,v in predict(models[('shared',seed)],train,d,True).items(): preds[f'shared__{seed}__{cell}__{mode}']=v
  for cell,d in (('held_native',held),('held_transformed',hw)):
   for mode,v in predict(models[('scratch',seed)],scratch,d,False).items(): preds[f'scratch__{seed}__{cell}__{mode}']=v
 an(OUT/'predictions_before_scoring.npz',**preds); aj(OUT/'prediction_barrier.json',{'status':'RECOVERED_PREDICTIONS_PERSISTED_BEFORE_SCORING','sha256':sha(OUT/'predictions_before_scoring.npz'),'counts':C.copy()})
 targets={'familiar_native':familiar['qy'],'familiar_transformed':fw['qy'],'held_native':held['qy'],'held_transformed':hw['qy']}; comparisons=[]; scores=[]
 for seed in f.SEEDS:
  cells={}
  for cell,target in targets.items():
   vals={mode:f.perlaw(preds[f'shared__{seed}__{cell}__{mode}'],target) for mode in ('correct','blind','fullwrong')}; charge('score_blocks',48)
   d={'seed':seed,'cell':cell,'shared_MSE':float(vals['correct'].mean()),'blind_MSE':float(vals['blind'].mean()),'fullwrong_MSE':float(vals['fullwrong'].mean()),'shared_over_blind':float(vals['correct'].sum()/vals['blind'].sum()),'correct_wins_fullwrong':int(np.sum(vals['correct']<vals['fullwrong'])),'zero_MSE':float(np.mean(target**2))}
   if cell.startswith('held'):
    sv=f.perlaw(preds[f'scratch__{seed}__{cell}__correct'],target); charge('score_blocks',32); d['scratch_MSE']=float(sv.mean()); d['shared_over_scratch']=float(vals['correct'].sum()/sv.sum())
   cells[cell]=d; comparisons.append(d)
   for i in range(16): scores.append({'seed':seed,'cell':cell,'law':i,'correct_MSE':float(vals['correct'][i]),'blind_MSE':float(vals['blind'][i]),'fullwrong_MSE':float(vals['fullwrong'][i])})
  summary={'seed':seed,'familiar_interface_ratio':cells['familiar_transformed']['shared_MSE']/cells['familiar_native']['shared_MSE'],'held_interface_ratio':cells['held_transformed']['shared_MSE']/cells['held_native']['shared_MSE'],'held_native_shared_over_scratch':cells['held_native']['shared_over_scratch'],'held_transformed_shared_over_scratch':cells['held_transformed']['shared_over_scratch']}; summary['familiar_native_sanity']=bool(cells['familiar_native']['shared_over_blind']<=.8 and cells['familiar_native']['correct_wins_fullwrong']>=12 and cells['familiar_native']['shared_MSE']/cells['familiar_native']['zero_MSE']<=.5); comparisons.append({'factorial_summary':summary})
 if C!=LIMITS: raise RuntimeError(f'COUNT:{C}!={LIMITS}')
 elapsed=time.monotonic()-t0; art=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()); resources={'wall_seconds':elapsed,'peak_rss_bytes':rss(),'artifact_bytes_before_result':art}
 if elapsed>WALL or resources['peak_rss_bytes']>RSS_CAP or art>ART_CAP: raise RuntimeError(f'RESOURCE:{resources}')
 result={'status':'COMMON4D_INTERFACE_MECHANISM_FACTORIAL_V1_RECOVERY1_COMPLETE','classification':'DEVELOPMENT_FACTORIAL_COMPLETE_RECOVERED','comparisons':comparisons,'law_scores':scores,'recovery_counts':C,'cumulative_counts':plan['cumulative_after_success'],'resources':resources,'qualification':['Original four fits and 2,000 updates were reused, not repeated.','All generated rows and transforms were deterministically repeated and cumulatively counted.','All systems are opened synthetic development; no confirmation.','Unknown signed permutation tests one finite interface change, not arbitrary-coordinate invariance.']}; aj(OUT/'result.json',result); aj(OUT/'completion.json',{'status':'COMPLETE','result_sha256':sha(OUT/'result.json'),'recovery_counts':C,'cumulative_counts':plan['cumulative_after_success']}); print(json.dumps({'classification':result['classification'],'comparisons':comparisons,'recovery_counts':C,'resources':resources},indent=2))
if __name__=='__main__': main()
