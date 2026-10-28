#!/usr/bin/env python3
from __future__ import annotations
import fcntl, hashlib, importlib.util, json, os, resource, time
from pathlib import Path
import numpy as np
import torch

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
HERE=Path(__file__).resolve();ROOT=HERE.parents[2];BASE=ROOT/"outputs/sho/autonomous_generic_sho_20261014"
SOURCE=BASE/"scripts/run_cross_mechanism_alpaca_pilot_v1.py";PREV=BASE/"diagnostics/sequence_aware_plasticity_v1"
OUT=BASE/"diagnostics/sequential_head_hyperposterior_v1";PLAN=BASE/"protocols/sequential_head_hyperposterior_v1/plan.json";LOCK=BASE/"active.lock";SEEDS=(2026102751,2026102752);START=time.monotonic()
def load():
 s=importlib.util.spec_from_file_location("cross",SOURCE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v=load()
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def jw(p,o):
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name("."+p.name+".tmp")
 with open(t,"w") as f:json.dump(o,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def nw(p,**a):
 if p.exists():raise FileExistsError(p)
 t=p.with_name("."+p.name+".tmp")
 with open(t,"wb") as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def rss():
 x=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;return int(x if os.uname().sysname=="Darwin" else x*1024)
class G:
 def __init__(self,c):self.c=c;self.n={k:0 for k in c if k not in ('wall_seconds','peak_RSS_bytes','artifact_bytes')}
 def add(self,k,a=1):
  q=self.n[k]+a
  if q>self.c[k]:raise RuntimeError(f'PRECALL_{k}:{q}>{self.c[k]}')
  self.n[k]=q
 def res(self):
  if time.monotonic()-START>self.c['wall_seconds'] or rss()>self.c['peak_RSS_bytes']:raise RuntimeError('RESOURCE_CAP')
def design(x):return torch.kron(x,torch.eye(2))
def posterior(phi,y,mu,cov,noise,g,key):
 g.add(key);z=design(phi);prec=torch.linalg.inv(cov);a=prec+z.T@z/noise;b=prec@mu+z.T@y.reshape(-1)/noise;c=torch.linalg.cholesky(a);pcov=torch.cholesky_inverse(c);pm=torch.cholesky_solve(b[:,None],c)[:,0];return pm,pcov
def predict(phi,beta):return phi@beta.reshape(16,2)
def main():
 if OUT.exists():raise FileExistsError('OUTPUT_NAMESPACE_EXISTS')
 doc=ROOT/'coordination/SEQUENTIAL_HEAD_HYPERPOSTERIOR_V1.md';caps={'source_constructions':2,'source_head_solves':288,'target_sequences':6,'encounter_posterior_solves':48,'evaluation_posterior_solves':720,'evaluation_prediction_rows':46080,'moment_updates':16,'batch_invariant_checks':16,'one_step_simulations':10752,'RK4_RHS_evaluations':43008,'score_blocks':720,'wall_seconds':120,'peak_RSS_bytes':1<<30,'artifact_bytes':64<<20}
 plan={'status':'PROSPECTIVE_BEFORE_EXECUTION','mechanism':'sequential moment-matched Gaussian population posterior over frozen 16x2 task heads','arms':['sequential','frozen','reset'],'seeds':list(SEEDS),'source_noise':'median normalized odd16 residual variance after even16 ridge fit','source_covariance_floor':'1e-3*trace(C)/32','decision':{'sequential_over_frozen_max':.8,'sequential_over_reset_max':.8,'law_wins_min':6,'source_after_over_before_max':1.1,'source_laws_within_10pct_min':20,'batch_tolerance':1e-10},'ceilings':caps,'claim_limits':['Opened controlled-oscillator development evidence only.','Shared features, known q/v state, RK4, normalization and the 16x2 linear head are supplied assumptions.','Every new system fits 32 head quantities from context32; this is supervised per-system inference, not zero-shot prediction.','Gaussian task moments are not calibrated ambiguity or arbitrary-mechanism identification.'],'source_hashes':{'driver':sha(HERE),'design':sha(doc),'source':sha(SOURCE),'previous':sha(PREV/'result.json')}}
 with LOCK.open('a+b') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);jw(PLAN,plan);OUT.mkdir(parents=True);g=G(caps)
  rows=v.make_laws();train=[r for r in rows if r['role']=='train'];dev=[r for r in rows if r['role']=='represented_dev'];pend=[r for r in rows if r['role']=='heldout_dev'];cx=v.states(32,2026102771);qx=v.states(64,2026102772);jw(OUT/'systems_and_states_before_responses.json',{'systems':rows,'context_states':cx.tolist(),'query_states':qx.tolist(),'query_labels_generated':False})
  train_c=v.simulate(train,cx);train_q=v.simulate(train,qx);dev_c=v.simulate(dev,cx);pend_c=v.simulate(pend,cx);g.add('one_step_simulations',len(train)*96+(len(dev)+len(pend))*32);g.add('RK4_RHS_evaluations',4*(len(train)*96+(len(dev)+len(pend))*32))
  ym=train_q.reshape(-1,2).mean(0);ys=np.maximum(train_q.reshape(-1,2).std(0),1e-9);tc=torch.tensor(cx/np.array([1.5,1.]));tq=torch.tensor(qx/np.array([1.5,1.]));yc=torch.tensor((train_c-ym)/ys);dc=torch.tensor((dev_c-ym)/ys);pc=torch.tensor((pend_c-ym)/ys);even=torch.arange(0,32,2);odd=torch.arange(1,32,2)
  priors={};source_heads={};cal=[]
  for seed in SEEDS:
   g.add('source_constructions');m=v.Features();m.load_state_dict(torch.load(PREV/f'retention_aware_seed{seed}.pt',map_location='cpu',weights_only=True));m.eval()
   with torch.no_grad():phi=m(tc);pq=m(tq)
   heads=[];res=[]
   for i in range(72):
    g.add('source_head_solves');h=v.ridge(phi,yc[i]);heads.append(h.reshape(-1));g.add('source_head_solves');he=v.ridge(phi[even],yc[i,even]);res.append((phi[odd]@he-yc[i,odd]).reshape(-1))
   hs=torch.stack(heads);mu=hs.mean(0);center=hs-mu;cov=center.T@center/71;floor=1e-3*torch.trace(cov)/32;cov=cov+floor*torch.eye(32);noise=max(float(torch.median(torch.cat(res)**2)),1e-10);priors[seed]=(mu,cov,noise,phi,pq);source_heads[seed]=hs;cal.append({'seed':seed,'noise':noise,'covariance_floor':float(floor),'covariance_condition':float(torch.linalg.cond(cov))})
  arrays={};meta=[];states=[];invariants=[]
  for arm in ('sequential','frozen','reset'):
   for seed in SEEDS:
    g.add('target_sequences');mu0,cov0,noise,phi,pq=priors[seed]
    if arm=='reset':mu=torch.zeros_like(mu0);cov=torch.eye(32)*torch.trace(cov0)/32
    else:mu=mu0.clone();cov=cov0.clone()
    n=72;stored=[(x.clone(),torch.zeros((32,32))) for x in source_heads[seed]];raw=source_heads[seed]-source_heads[seed].mean(0);source_floor=cov0-raw.T@raw/71
    def ev(stage):
     pp=[];sp=[]
     for i in range(8,16):
      b,_=posterior(phi,pc[i],mu,cov,noise,g,'evaluation_posterior_solves');pp.append((predict(pq,b).numpy()*ys+ym))
     if stage in (0,8):
      for i in range(24):
       b,_=posterior(phi,dc[i],mu,cov,noise,g,'evaluation_posterior_solves');sp.append((predict(pq,b).numpy()*ys+ym))
     pp=np.asarray(pp);g.add('evaluation_prediction_rows',pp.shape[0]*64);arrays[f'pend_{arm}_{seed}_stage{stage}']=pp;meta.append({'arm':arm,'seed':seed,'stage':stage,'role':'pendulum','laws':list(range(8,16))})
     if sp:
      sp=np.asarray(sp);g.add('evaluation_prediction_rows',sp.shape[0]*64);arrays[f'source_{arm}_{seed}_stage{stage}']=sp;meta.append({'arm':arm,'seed':seed,'stage':stage,'role':'source','laws':list(range(24))})
    ev(0)
    for encounter in range(8):
     g.res();pm,ps=posterior(phi,pc[encounter],mu,cov,noise,g,'encounter_posterior_solves')
     if arm=='sequential':
      old=mu.clone();n2=n+1;mu=old+(pm-old)/n2;cov=((n-1)*cov+ps+torch.outer(pm-old,pm-mu))/n;cov=cov+torch.eye(32)*torch.finfo(torch.float64).eps;n=n2;stored.append((pm.clone(),ps.clone()));g.add('moment_updates')
      means=torch.stack([x[0] for x in stored]);bm=means.mean(0);bc=sum((s+torch.outer(m-bm,m-bm) for m,s in stored),torch.zeros((32,32)));bc=(bc+71*source_floor)/(len(stored)-1);bc=bc+torch.eye(32)*torch.finfo(torch.float64).eps;delta=max(float(torch.max(torch.abs(mu-bm))),float(torch.max(torch.abs(cov-bc))));g.add('batch_invariant_checks');invariants.append({'seed':seed,'encounter':encounter,'max_abs_delta':delta})
     states.append({'arm':arm,'seed':seed,'encounter':encounter,'mean_norm':float(torch.linalg.vector_norm(mu)),'cov_trace':float(torch.trace(cov))});ev(encounter+1)
  nw(OUT/'predictions_before_query_labels.npz',**arrays);jw(OUT/'prediction_barrier.json',{'status':'PREDICTIONS_PERSISTED_BEFORE_QUERY_LABELS','sha256':sha(OUT/'predictions_before_query_labels.npz'),'counts':g.n,'metadata':meta})
  dev_q=v.simulate(dev,qx);pend_q=v.simulate(pend,qx);g.add('one_step_simulations',(len(dev)+len(pend))*64);g.add('RK4_RHS_evaluations',4*(len(dev)+len(pend))*64)
  scores=[]
  for z in meta:
   key=('pend' if z['role']=='pendulum' else 'source')+f"_{z['arm']}_{z['seed']}_stage{z['stage']}";pred=arrays[key];truth=pend_q[8:16] if z['role']=='pendulum' else dev_q
   for i in range(len(pred)):scores.append({**z,'law':z['laws'][i],'MSE':float(np.mean((pred[i]-truth[i])**2))});g.add('score_blocks')
  def er(a,s,r,t):return np.array([x['MSE'] for x in scores if x['arm']==a and x['seed']==s and x['role']==r and x['stage']==t])
  comp=[]
  for seed in SEEDS:
   q=er('sequential',seed,'pendulum',8);f=er('frozen',seed,'pendulum',8);r=er('reset',seed,'pendulum',8);s0=er('sequential',seed,'source',0);s8=er('sequential',seed,'source',8);d={'seed':seed,'sequential_over_frozen':float(q.sum()/f.sum()),'sequential_wins_frozen':int(np.sum(q<f)),'sequential_over_reset':float(q.sum()/r.sum()),'sequential_wins_reset':int(np.sum(q<r)),'source_after_over_before':float(s8.sum()/s0.sum()),'source_laws_within_10pct':int(np.sum(s8<=1.1*s0)),'max_batch_invariant':max(x['max_abs_delta'] for x in invariants if x['seed']==seed),'finite':bool(np.isfinite(np.r_[q,f,r,s0,s8]).all())};d['passed']=bool(d['sequential_over_frozen']<=.8 and d['sequential_wins_frozen']>=6 and d['sequential_over_reset']<=.8 and d['sequential_wins_reset']>=6 and d['source_after_over_before']<=1.1 and d['source_laws_within_10pct']>=20 and d['max_batch_invariant']<=1e-10 and d['finite']);comp.append(d)
  for k,e in caps.items():
   if k in g.n and g.n[k]!=e:raise RuntimeError(f'COUNT_{k}:{g.n[k]}!={e}')
  elapsed=time.monotonic()-START;art=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
  if elapsed>caps['wall_seconds'] or rss()>caps['peak_RSS_bytes'] or art>caps['artifact_bytes']:raise RuntimeError('RESOURCE_CAP')
  result={'status':'SEQUENTIAL_HEAD_HYPERPOSTERIOR_V1_COMPLETE','classification':'SUPPORTED_DEVELOPMENT_ONLY' if all(x['passed'] for x in comp) else 'NOT_SUPPORTED','comparisons':comp,'calibration':cal,'states':states,'batch_invariants':invariants,'scores':scores,'counts':g.n,'resources':{'wall_seconds':elapsed,'peak_RSS_bytes':rss(),'artifact_bytes_before_result':art},'qualifications':plan['claim_limits'],'source_hashes':{'driver':sha(HERE),'plan':sha(PLAN)}};jw(OUT/'result.json',result);jw(OUT/'completion.json',{'status':'COMPLETE','classification':result['classification'],'result_sha256':sha(OUT/'result.json'),'predictions_sha256':sha(OUT/'predictions_before_query_labels.npz'),'counts':g.n});print(json.dumps({'classification':result['classification'],'comparisons':comp,'calibration':cal,'counts':g.n,'resources':result['resources']},indent=2))
if __name__=='__main__':main()
