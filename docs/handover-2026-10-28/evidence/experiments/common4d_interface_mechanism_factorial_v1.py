#!/usr/bin/env python3
"""Matched familiar/held dynamics x native/transformed interface factorial."""
from __future__ import annotations
import json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
from heterogeneous_entity_context_pilot_v1 import Model,aj,an,sha,rss  # noqa:E402

OUT=ROOT/'outputs/common4d_interface_mechanism_factorial_v1'; DT=.05
SEEDS=(2026102918,2026102919); UPDATES=500; BS=8; BQ=16
WALL=300.; RSS_CAP=1<<30; ART_CAP=64<<20
PERM=np.asarray([2,0,3,1]); SIGN=np.asarray([1.,-1.,1.,-1.])
LIMITS={'generated_transition_rows':9984,'signed_transform_rows':6144,'fits':4,'optimizer_updates':2000,'training_context_rows':512000,'training_query_rows':256000,'evaluation_rows':32768,'score_blocks':512}
ACTUAL={k:0 for k in LIMITS}
def charge(k,n=1):
 if ACTUAL[k]+n>LIMITS[k]: raise RuntimeError(f'PRECALL_CAP:{k}:{ACTUAL[k]}+{n}>{LIMITS[k]}')
 ACTUAL[k]+=n

def transform(x): charge('signed_transform_rows',int(np.prod(x.shape[:-1]))); return x[...,PERM]*SIGN
def p3(x): return (5*x**3-3*x)/2
def stable_A(r):
 a=r.normal(scale=.22,size=(4,4)); a-=np.eye(4)*(.45+.15*r.random()); return a
def cubic_law(r):
 m=r.normal(size=(4,4)); m=m/max(np.linalg.norm(m),1e-12)*2
 return ('cubic',stable_A(r),r.uniform(.6,1.4),m)
def tanh_law(r):
 return ('tanh',stable_A(r),r.uniform(.6,1.4),(r.normal(size=(4,16))/2,r.normal(scale=.2,size=16),r.normal(size=(16,4))/4,r.normal(scale=.1,size=4)))
def step(x,law):
 charge('generated_transition_rows',len(x))
 kind,A,a,p=law
 if kind=='cubic': f=x@A.T+a*(p3(x)@p)
 else:
  w1,b1,w2,b2=p; f=x@A.T+a*(np.tanh(x@w1+b1)@w2+b2)
 return DT*f
def make(laws,cx,qx):
 return {'cx':np.broadcast_to(cx,(len(laws),)+cx.shape).copy(),'cd':np.stack([step(cx,l) for l in laws]),'qx':np.broadcast_to(qx,(len(laws),)+qx.shape).copy(),'qy':np.stack([step(qx,l) for l in laws])}
def warped(d): return {k:transform(v) for k,v in d.items()}
def scales(d): return np.sqrt(np.mean(d['cx']**2,axis=1))+1e-8,np.sqrt(np.mean(d['cd']**2,axis=1))+1e-8
def tensors(d,ids):
 sx,sd=scales(d); i=np.asarray(ids); b=len(i)
 cx=torch.as_tensor(d['cx'][i]/sx[i,None])[:, :, None, :]; cd=torch.as_tensor(d['cd'][i]/sd[i,None])[:, :, None, :]
 qx=torch.as_tensor(d['qx'][i]/sx[i,None])[:, :, None, :]; em=torch.ones((b,1),dtype=torch.bool); cm=torch.ones((b,1,4),dtype=torch.bool); dt=torch.zeros(b)
 return cx,cd,qx,em,cm,dt,torch.as_tensor(sd[i])[:,None,None,:]
def fit(d,seed,name):
 charge('fits')
 torch.manual_seed(seed); m=Model(); opt=torch.optim.Adam(m.parameters(),lr=1e-3); gen=torch.Generator().manual_seed(seed+123); hist=[]
 for _ in range(UPDATES):
  ids=torch.randperm(len(d['cx']),generator=gen)[:BS].numpy(); qi=torch.randperm(64,generator=gen)[:BQ]
  charge('optimizer_updates'); charge('training_context_rows',BS*32); charge('training_query_rows',BS*BQ)
  cx,cd,qx,em,cm,dt,sd=tensors(d,ids); qy=torch.as_tensor(d['qy'][ids][:,qi.numpy()])[:, :, None, :]/sd
  pred=m.predict(qx[:,qi],em,cm,dt,m.context_code(cx,cd,em,cm,dt)); loss=((pred-qy)**2).mean()
  opt.zero_grad(); loss.backward(); opt.step(); hist.append(float(loss.detach()))
 torch.save({'state_dict':m.state_dict()},OUT/f'{name}.pt'); return m,hist
def predict(m,train,dev,include_blind=True):
 charge('evaluation_rows',len(dev['cx'])*64*(3 if include_blind else 2))
 cx,cd,qx,em,cm,dt,sd=tensors(dev,range(len(dev['cx']))); wrong=np.roll(np.arange(len(dev['cx'])),-1)
 wcx,wcd,_,wem,wcm,wdt,wsd=tensors(dev,wrong); tcx,tcd,_,tem,tcm,tdt,_=tensors(train,range(len(train['cx'])))
 with torch.no_grad():
  code=m.context_code(cx,cd,em,cm,dt); wcode=m.context_code(wcx,wcd,wem,wcm,wdt)
  correct=m.predict(qx,em,cm,dt,code)*sd
  # Complete wrong context: donor code and donor context scales.
  sxw,_=scales(dev); fqx=torch.as_tensor(dev['qx']/sxw[wrong,None])[:,:,None,:]
  fullwrong=m.predict(fqx,em,cm,wdt,wcode)*wsd
 out={'correct':correct[:, :, 0].numpy(),'fullwrong':fullwrong[:, :, 0].numpy()}
 if include_blind:
  mean=m.context_code(tcx,tcd,tem,tcm,tdt).mean(0,keepdim=True).expand(len(dev['cx']),-1)
  out['blind']=(m.predict(qx,em,cm,dt,mean)*sd)[:, :, 0].numpy()
 return out
def perlaw(pred,target): return np.mean((pred-target)**2,axis=(1,2))

def main():
 t0=time.monotonic()
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 r=np.random.default_rng(2026102917); cx=r.uniform(-1,1,(32,4)); qx=r.uniform(-1,1,(64,4))
 plan={'status':'PRECHARGED_BEFORE_SCIENCE',**LIMITS,'ceilings':{'wall_seconds':WALL,'peak_RSS_bytes':RSS_CAP,'artifact_bytes':ART_CAP},'transform':{'perm':PERM.tolist(),'sign':SIGN.tolist(),'inverse_visible_to_learner':False},'source_hashes':{'driver':sha(Path(__file__)),'proposal':sha(ROOT/'coordination/COMMON4D_INTERFACE_MECHANISM_FACTORIAL_V1.md')}}; aj(OUT/'plan.json',plan)
 train=make([cubic_law(r) for _ in range(48)],cx,qx); scratch=make([tanh_law(r) for _ in range(24)],cx,qx)
 familiar=make([cubic_law(r) for _ in range(16)],cx,qx); held=make([tanh_law(r) for _ in range(16)],cx,qx)
 fw,hw=warped(familiar),warped(held)
 # Exact conjugacy, norm, dimension and disjoint-bank invariants.
 for native,warped_d in ((familiar,fw),(held,hw)):
  for k in native:
   if native[k].shape!=warped_d[k].shape or not np.allclose(np.sum(native[k]**2,axis=-1),np.sum(warped_d[k]**2,axis=-1),rtol=0,atol=1e-12): raise RuntimeError('TRANSFORM_INVARIANT')
 if np.array_equal(cx,qx[:32]): raise RuntimeError('BANK_OVERLAP')
 models={}; hist={}
 for seed in SEEDS:
  models[('shared',seed)],hist[f'shared_{seed}']=fit(train,seed,f'shared_seed{seed}')
  models[('scratch',seed)],hist[f'scratch_{seed}']=fit(scratch,seed+1000,f'scratch_seed{seed}')
 preds={}
 for seed in SEEDS:
  for mech,d in (('familiar_native',familiar),('familiar_transformed',fw),('held_native',held),('held_transformed',hw)):
   for mode,v in predict(models[('shared',seed)],train,d).items(): preds[f'shared__{seed}__{mech}__{mode}']=v
  for mech,d in (('held_native',held),('held_transformed',hw)):
   pp=predict(models[('scratch',seed)],scratch,d,False); preds[f'scratch__{seed}__{mech}__correct']=pp['correct']; preds[f'scratch__{seed}__{mech}__fullwrong']=pp['fullwrong']
 an(OUT/'predictions_before_scoring.npz',**preds); aj(OUT/'prediction_barrier.json',{'status':'PREDICTIONS_PERSISTED_BEFORE_DEVELOPMENT_SCORING','sha256':sha(OUT/'predictions_before_scoring.npz')})
 rows=[]; comps=[]
 targets={'familiar_native':familiar['qy'],'familiar_transformed':fw['qy'],'held_native':held['qy'],'held_transformed':hw['qy']}
 for seed in SEEDS:
  cell={}
  for mech,target in targets.items():
   vals={mode:perlaw(preds[f'shared__{seed}__{mech}__{mode}'],target) for mode in ('correct','blind','fullwrong')}
   charge('score_blocks',3*16)
   d={'seed':seed,'cell':mech,'shared_MSE':float(vals['correct'].mean()),'blind_MSE':float(vals['blind'].mean()),'fullwrong_MSE':float(vals['fullwrong'].mean()),'shared_over_blind':float(vals['correct'].sum()/vals['blind'].sum()),'correct_wins_fullwrong':int(np.sum(vals['correct']<vals['fullwrong'])),'zero_MSE':float(np.mean(target**2))}
   if mech.startswith('held'):
    sv=perlaw(preds[f'scratch__{seed}__{mech}__correct'],target); d['scratch_MSE']=float(sv.mean()); d['shared_over_scratch']=float(vals['correct'].sum()/sv.sum())
    charge('score_blocks',2*16)
   cell[mech]=d; comps.append(d)
   for i in range(16): rows.append({'seed':seed,'cell':mech,'law':i,'correct_MSE':float(vals['correct'][i]),'blind_MSE':float(vals['blind'][i]),'fullwrong_MSE':float(vals['fullwrong'][i])})
  summary={'seed':seed,'familiar_interface_ratio':cell['familiar_transformed']['shared_MSE']/cell['familiar_native']['shared_MSE'],'held_interface_ratio':cell['held_transformed']['shared_MSE']/cell['held_native']['shared_MSE'],'held_native_shared_over_scratch':cell['held_native']['shared_over_scratch'],'held_transformed_shared_over_scratch':cell['held_transformed']['shared_over_scratch']}
  summary['familiar_native_sanity']=bool(cell['familiar_native']['shared_over_blind']<=.8 and cell['familiar_native']['correct_wins_fullwrong']>=12 and cell['familiar_native']['shared_MSE']/cell['familiar_native']['zero_MSE']<=.5); comps.append({'factorial_summary':summary})
 counts=ACTUAL.copy()
 for k in counts:
  if counts[k]!=LIMITS[k]: raise RuntimeError(f'COUNT:{k}:{counts[k]}!={LIMITS[k]}')
 elapsed=time.monotonic()-t0; art=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()); resources={'wall_seconds':elapsed,'peak_rss_bytes':rss(),'artifact_bytes_before_result':art}
 if elapsed>WALL or resources['peak_rss_bytes']>RSS_CAP or art>ART_CAP: raise RuntimeError(f'RESOURCE:{resources}')
 result={'status':'COMMON4D_INTERFACE_MECHANISM_FACTORIAL_V1_COMPLETE','classification':'DEVELOPMENT_FACTORIAL_COMPLETE','comparisons':comps,'law_scores':rows,'counts':counts,'resources':resources,'parameter_count':sum(p.numel() for p in Model().parameters()),'training_loss_endpoints':{k:v[-1] for k,v in hist.items()},'qualification':['Fixed signed permutation is unknown to the learner; failure is not arbitrary-coordinate non-identifiability.','Cubic-Legendre and tanh residual families, Euler timing and controlled banks are supplied synthetic development assumptions.','All systems are development evidence; no confirmation was opened.']}
 aj(OUT/'result.json',result); aj(OUT/'completion.json',{'status':'COMPLETE','result_sha256':sha(OUT/'result.json'),'counts':counts}); print(json.dumps({'classification':result['classification'],'comparisons':comps,'counts':counts,'resources':resources},indent=2))
if __name__=='__main__': main()
