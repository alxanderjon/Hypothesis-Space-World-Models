#!/usr/bin/env python3
from __future__ import annotations
import fcntl, hashlib, importlib.util, json, os, resource, time
from pathlib import Path
import numpy as np
import torch

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; BASE=ROOT/"outputs/sho/autonomous_generic_sho_20261014"
SOURCE=BASE/"scripts/run_cross_mechanism_alpaca_pilot_v1.py"; PREV=BASE/"diagnostics/sequence_aware_plasticity_v1"
OUT=BASE/"diagnostics/sequence_aware_online_laplace_v1"; PLAN=BASE/"protocols/sequence_aware_online_laplace_v1/plan.json"; LOCK=BASE/"active.lock"
SEEDS=(2026102751,2026102752); START=time.monotonic()
def load():
 s=importlib.util.spec_from_file_location("cross",SOURCE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v=load()
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def jwrite(p,o):
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name("."+p.name+".tmp")
 with open(t,"w") as f:json.dump(o,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def nwrite(p,**a):
 if p.exists():raise FileExistsError(p)
 t=p.with_name("."+p.name+".tmp")
 with open(t,"wb") as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def rss():
 z=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;return int(z if os.uname().sysname=="Darwin" else z*1024)
class Guard:
 def __init__(self,c):self.c=c;self.n={k:0 for k in c if k not in ('wall_seconds','peak_RSS_bytes','artifact_bytes')}
 def add(self,k,a=1):
  q=self.n[k]+a
  if q>self.c[k]:raise RuntimeError(f"PRECALL_{k}:{q}>{self.c[k]}")
  self.n[k]=q
 def resource(self):
  if time.monotonic()-START>self.c['wall_seconds'] or rss()>self.c['peak_RSS_bytes']:raise RuntimeError('RESOURCE_CAP')
def ridge(phi,y,g,key):g.add(key);return v.ridge(phi,y)
def mse(a,b):return torch.mean((a-b)**2)
def flat(xs):return torch.cat([x.reshape(-1) for x in xs])
def assign(model,vec):
 with torch.no_grad():
  off=0
  for p in model.parameters():n=p.numel();p.copy_(vec[off:off+n].reshape_as(p));off+=n
def main():
 if OUT.exists():raise FileExistsError('OUTPUT_NAMESPACE_EXISTS')
 design=ROOT/'coordination/SEQUENCE_AWARE_ONLINE_LAPLACE_V1.md'; previous=json.load(open(PREV/'result.json'))
 caps={'source_importance_gradient_evaluations':144,'source_importance_ridge_solves':144,'calibration_gradient_evaluations':6,'calibration_ridge_solves':6,'target_sequences':4,'target_gradient_evaluations':32,'target_updates':32,'target_ridge_solves':32,'evaluation_ridge_solves':480,'evaluation_prediction_rows':30720,'one_step_simulations':10752,'RK4_RHS_evaluations':43008,'score_blocks':480,'wall_seconds':120,'peak_RSS_bytes':1<<30,'artifact_bytes':64<<20}
 plan={'status':'PROSPECTIVE_BEFORE_EXECUTION','mechanism':'online diagonal empirical-Fisher/Laplace update versus isotropic matched importance','seeds':list(SEEDS),'source_importance':'mean squared per-law gradient of odd16 normalized response loss after even16 ridge fit','damping':'1% of mean source importance, fixed per seed','calibration':'indices 0,24,48 source-training only; median multiplier matches established 0.01 update norm','target_update':'one gradient update per encountered pendulum law; importance accumulates after update','decision':{'pendulum_stage8_over_stage0_max':.8,'diagonal_over_saved_unconstrained_max':1.1,'source_after_over_before_max':1.1,'source_laws_within_10pct_min':20,'diagonal_source_wins_isotropic_min':16,'diagonal_over_isotropic_pendulum_max':1.1},'ceilings':caps,'claim_limits':['Opened controlled oscillator development evidence only.','The empirical Fisher is scale-dependent curvature/importance, not calibrated uncertainty.','The supplied known q/v state, RK4 simulator, normalized response targets, 16D ridge head and one supervised target update remain explicit assumptions.','Passing would not establish arbitrary-mechanism transfer or million-parameter performance.'],'source_hashes':{'driver':sha(HERE),'design':sha(design),'source':sha(SOURCE),'previous_result':sha(PREV/'result.json')}}
 with LOCK.open('a+b') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);jwrite(PLAN,plan);OUT.mkdir(parents=True);g=Guard(caps)
  rows=v.make_laws();train=[r for r in rows if r['role']=='train'];dev=[r for r in rows if r['role']=='represented_dev'];pend=[r for r in rows if r['role']=='heldout_dev'];cx=v.states(32,2026102771);qx=v.states(64,2026102772)
  jwrite(OUT/'systems_and_states_before_responses.json',{'systems':rows,'context_states':cx.tolist(),'query_states':qx.tolist(),'development_query_responses_generated':False})
  train_c=v.simulate(train,cx);train_q=v.simulate(train,qx);dev_c=v.simulate(dev,cx);pend_c=v.simulate(pend,cx);g.add('one_step_simulations',len(train)*96+(len(dev)+len(pend))*32);g.add('RK4_RHS_evaluations',4*(len(train)*96+(len(dev)+len(pend))*32))
  ymean=train_q.reshape(-1,2).mean(0);ystd=np.maximum(train_q.reshape(-1,2).std(0),1e-9);tc=torch.tensor(cx/np.array([1.5,1.]));tq=torch.tensor(qx/np.array([1.5,1.]));tyc=torch.tensor((train_c-ymean)/ystd);dyc=torch.tensor((dev_c-ymean)/ystd);pyc=torch.tensor((pend_c-ymean)/ystd);even=torch.arange(0,32,2);odd=torch.arange(1,32,2)
  importance={};calibration={};persistent={};diagnostics=[]
  def law_grad(model,targets,ridge_key,grad_key):
   pe=model(tc[even]);po=model(tc[odd]);head=ridge(pe,targets[even],g,ridge_key);loss=mse(po@head,targets[odd]);gr=torch.autograd.grad(loss,tuple(model.parameters()));g.add(grad_key);return flat(gr),float(loss)
  for seed in SEEDS:
   model=v.Features();model.load_state_dict(torch.load(PREV/f'retention_aware_seed{seed}.pt',map_location='cpu',weights_only=True));sq=[]
   for i in range(len(train)):
    gr,_=law_grad(model,tyc[i],'source_importance_ridge_solves','source_importance_gradient_evaluations');sq.append(gr.detach()**2)
   imp=torch.stack(sq).mean(0);damping=.01*float(imp.mean());importance[seed]=imp;persistent[f'importance_{seed}']=imp.numpy();etas={'diagonal':[],'isotropic':[]}
   for i in (0,24,48):
    gr,_=law_grad(model,tyc[i],'calibration_ridge_solves','calibration_gradient_evaluations');fn=gr.detach()**2;target=.01*float(torch.linalg.vector_norm(gr));dd=imp+fn+damping;di=gr/dd;iso=gr/float(dd.mean());etas['diagonal'].append(target/max(float(torch.linalg.vector_norm(di)),1e-30));etas['isotropic'].append(target/max(float(torch.linalg.vector_norm(iso)),1e-30))
   calibration[seed]={'damping':damping,'importance_mean':float(imp.mean()),'importance_min':float(imp.min()),'importance_max':float(imp.max()),'eta_diagonal':float(np.median(etas['diagonal'])),'eta_isotropic':float(np.median(etas['isotropic'])),'calibration_etas':etas}
  arrays={};meta=[]
  for arm in ('diagonal','isotropic'):
   for seed in SEEDS:
    g.add('target_sequences');model=v.Features();model.load_state_dict(torch.load(PREV/f'retention_aware_seed{seed}.pt',map_location='cpu',weights_only=True));precision=importance[seed].clone() if arm=='diagonal' else torch.tensor(float(importance[seed].mean()));eta=calibration[seed]['eta_'+arm];damping=calibration[seed]['damping']
    def evaluate(stage):
     with torch.no_grad():
      pc=model(tc);pq=model(tq);pp=[]
      for i in range(8,16):pp.append((pq@ridge(pc,pyc[i],g,'evaluation_ridge_solves')).numpy()*ystd+ymean)
      sp=[]
      if stage in (0,8):
       for i in range(24):sp.append((pq@ridge(pc,dyc[i],g,'evaluation_ridge_solves')).numpy()*ystd+ymean)
     pp=np.asarray(pp);g.add('evaluation_prediction_rows',pp.shape[0]*pp.shape[1]);arrays[f'pend_{arm}_{seed}_stage{stage}']=pp;meta.append({'arm':arm,'seed':seed,'stage':stage,'role':'pendulum','laws':list(range(8,16))})
     if sp:
      sp=np.asarray(sp);g.add('evaluation_prediction_rows',sp.shape[0]*sp.shape[1]);arrays[f'source_{arm}_{seed}_stage{stage}']=sp;meta.append({'arm':arm,'seed':seed,'stage':stage,'role':'source','laws':list(range(24))})
    evaluate(0)
    for encounter in range(8):
     g.resource();gr,loss=law_grad(model,pyc[encounter],'target_ridge_solves','target_gradient_evaluations');fn=gr.detach()**2
     if arm=='diagonal':direction=gr/(precision+fn+damping);precision=precision+fn
     else:
      denom=float(precision+fn.mean()+damping);direction=gr/denom;precision=precision+fn.mean()
     before=flat(tuple(p.detach() for p in model.parameters()));assign(model,before-eta*direction);g.add('target_updates');diagnostics.append({'arm':arm,'seed':seed,'encounter':encounter,'target_loss':loss,'gradient_norm':float(torch.linalg.vector_norm(gr)),'update_norm':float(eta*torch.linalg.vector_norm(direction)),'precision_mean':float(precision.mean())});evaluate(encounter+1)
  nwrite(OUT/'predictions_before_query_labels.npz',**arrays,**persistent);jwrite(OUT/'prediction_barrier.json',{'status':'PREDICTIONS_PERSISTED_BEFORE_DEVELOPMENT_QUERY_LABEL_GENERATION','sha256':sha(OUT/'predictions_before_query_labels.npz'),'metadata':meta,'counts':g.n})
  dev_q=v.simulate(dev,qx);pend_q=v.simulate(pend,qx);g.add('one_step_simulations',(len(dev)+len(pend))*64);g.add('RK4_RHS_evaluations',4*(len(dev)+len(pend))*64)
  scores=[]
  for z in meta:
   key=('pend' if z['role']=='pendulum' else 'source')+f"_{z['arm']}_{z['seed']}_stage{z['stage']}";pred=arrays[key];target=pend_q[8:16] if z['role']=='pendulum' else dev_q
   for i in range(len(pred)):scores.append({**z,'law':z['laws'][i],'MSE':float(np.mean((pred[i]-target[i])**2))});g.add('score_blocks')
  def err(arm,seed,role,stage):return np.array([x['MSE'] for x in scores if x['arm']==arm and x['seed']==seed and x['role']==role and x['stage']==stage])
  comp=[]
  for seed in SEEDS:
   p0,p8=err('diagonal',seed,'pendulum',0),err('diagonal',seed,'pendulum',8);s0,s8=err('diagonal',seed,'source',0),err('diagonal',seed,'source',8);ip=err('isotropic',seed,'pendulum',8);isrc=err('isotropic',seed,'source',8);base=np.array([x['MSE'] for x in previous['scores'] if x['seed']==seed and x['arm']=='retention_aware' and x['role']=='pendulum_never_adapted' and x['stage']==8]);d={'seed':seed,'pendulum_stage8_over_stage0':float(p8.sum()/p0.sum()),'diagonal_over_saved_unconstrained':float(p8.sum()/base.sum()),'source_after_over_before':float(s8.sum()/s0.sum()),'source_laws_within_10pct':int(np.sum(s8<=1.1*s0)),'diagonal_source_over_isotropic':float(s8.sum()/isrc.sum()),'diagonal_source_wins_isotropic':int(np.sum(s8<isrc)),'diagonal_pendulum_over_isotropic':float(p8.sum()/ip.sum()),'finite':bool(np.isfinite(np.r_[p0,p8,s0,s8,ip,isrc]).all())};d['passed']=bool(d['pendulum_stage8_over_stage0']<=.8 and d['diagonal_over_saved_unconstrained']<=1.1 and d['source_after_over_before']<=1.1 and d['source_laws_within_10pct']>=20 and d['diagonal_source_wins_isotropic']>=16 and d['diagonal_pendulum_over_isotropic']<=1.1 and d['finite']);comp.append(d)
  for k,e in caps.items():
   if k in g.n and g.n[k]!=e:raise RuntimeError(f'COUNT_{k}:{g.n[k]}!={e}')
  elapsed=time.monotonic()-START;art=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
  if elapsed>caps['wall_seconds'] or rss()>caps['peak_RSS_bytes'] or art>caps['artifact_bytes']:raise RuntimeError('RESOURCE_CAP')
  result={'status':'SEQUENCE_AWARE_ONLINE_LAPLACE_V1_COMPLETE','classification':'SUPPORTED_DEVELOPMENT_ONLY' if all(x['passed'] for x in comp) else 'NOT_SUPPORTED','comparisons':comp,'calibration':calibration,'diagnostics':diagnostics,'scores':scores,'counts':g.n,'resources':{'wall_seconds':elapsed,'peak_RSS_bytes':rss(),'artifact_bytes_before_result':art},'qualifications':plan['claim_limits'],'source_hashes':{'driver':sha(HERE),'plan':sha(PLAN)}};jwrite(OUT/'result.json',result);jwrite(OUT/'completion.json',{'status':'COMPLETE','classification':result['classification'],'result_sha256':sha(OUT/'result.json'),'prediction_sha256':sha(OUT/'predictions_before_query_labels.npz'),'counts':g.n});print(json.dumps({'classification':result['classification'],'comparisons':comp,'calibration':calibration,'counts':g.n,'resources':result['resources']},indent=2))
if __name__=='__main__':main()
