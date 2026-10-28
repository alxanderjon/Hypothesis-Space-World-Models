#!/usr/bin/env python3
from __future__ import annotations
import fcntl, hashlib, importlib.util, json, os, resource, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import nnls

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; BASE=ROOT/"outputs/sho/autonomous_generic_sho_20261014"
SOURCE=BASE/"scripts/run_cross_mechanism_alpaca_pilot_v1.py"; PREV=BASE/"diagnostics/sequence_aware_plasticity_v1"; OUT=BASE/"diagnostics/sequence_aware_soft_function_kl_v1"; PLAN=BASE/"protocols/sequence_aware_soft_function_kl_v1/plan.json"; LOCK=BASE/"active.lock"; START=time.monotonic(); SEEDS=(2026102751,2026102752)
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
def main():
 if OUT.exists():raise FileExistsError('OUTPUT_NAMESPACE_EXISTS')
 design=ROOT/'coordination/SEQUENCE_AWARE_SOFT_FUNCTION_KL_V1.md'; previous=json.load(open(PREV/'result.json'))
 caps={'generator_SVDs':1,'generated_functions':48,'generated_scalar_outputs':9216,'calibration_gradient_evaluations':6,'calibration_ridge_solves':50,'calibration_prediction_rows':3072,'target_sequences':4,'target_updates':32,'target_ridge_solves':32,'regularizer_evaluations':32,'memory_ridge_solves':768,'memory_prediction_rows':49152,'evaluation_ridge_solves':480,'evaluation_prediction_rows':30720,'one_step_simulations':10752,'RK4_RHS_evaluations':43008,'score_blocks':480,'wall_seconds':120,'peak_RSS_bytes':1<<30,'artifact_bytes':64<<20}
 plan={'status':'PROSPECTIVE_BEFORE_EXECUTION','mechanism':'soft Gaussian function-space KL versus gradient-matched mean-only retention','generator_fit':'72 source-training context32+query64 normalized functions; full empirical SVD; no family labels','inducing_bank':'all fixed query64 states; QR-orthonormalized query portion of first 16 source-training PCs','noise_treatment':'deterministic simulator; empirical across-system covariance plus fixed numerical floor 1e-6*trace/16','calibration':'source-training-only gradient norm match; shared coefficient per seed; no target/development outcomes','seeds':list(SEEDS),'decision':{'pendulum_stage8_over_stage0_max':.8,'distribution_over_saved_unconstrained_max':1.1,'source_after_over_before_max':1.1,'source_laws_within_10pct_min':20,'distribution_source_wins_mean_min':16,'distribution_over_mean_pendulum_max':1.1},'ceilings':caps,'claim_limits':['Opened controlled oscillator development evidence only.','The generator uses source true context/query labels offline; target adaptation uses only its context32.','Empirical dispersion is not calibrated uncertainty; the fixed state bank and Gaussian form are declared assumptions.'],'source_hashes':{'driver':sha(HERE),'design':sha(design),'source':sha(SOURCE),'previous_result':sha(PREV/'result.json')}}
 with LOCK.open('a+b') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);jwrite(PLAN,plan);OUT.mkdir(parents=True);g=Guard(caps)
  rows=v.make_laws();train=[r for r in rows if r['role']=='train'];dev=[r for r in rows if r['role']=='represented_dev'];pend=[r for r in rows if r['role']=='heldout_dev'];cx=v.states(32,2026102771);qx=v.states(64,2026102772)
  jwrite(OUT/'systems_and_states_before_responses.json',{'systems':rows,'context_states':cx.tolist(),'query_states':qx.tolist(),'query_responses_generated':False})
  train_c=v.simulate(train,cx);dev_c=v.simulate(dev,cx);pend_c=v.simulate(pend,cx);g.add('one_step_simulations',(len(train)+len(dev)+len(pend))*32);g.add('RK4_RHS_evaluations',4*(len(train)+len(dev)+len(pend))*32)
  # Preserve the original normalization, derived from already-opened source training query responses.
  train_q=v.simulate(train,qx);g.add('one_step_simulations',len(train)*64);g.add('RK4_RHS_evaluations',4*len(train)*64)
  ymean=train_q.reshape(-1,2).mean(0);ystd=np.maximum(train_q.reshape(-1,2).std(0),1e-9);tc=torch.tensor(cx/np.array([1.5,1.]));tq=torch.tensor(qx/np.array([1.5,1.]));tyc=torch.tensor((train_c-ymean)/ystd);tyq=torch.tensor((train_q-ymean)/ystd);dyc=torch.tensor((dev_c-ymean)/ystd);pyc=torch.tensor((pend_c-ymean)/ystd)
  functions=np.concatenate((tyc.numpy().reshape(len(train),-1),tyq.numpy().reshape(len(train),-1)),axis=1);mean=functions.mean(0);_,singular,vh=np.linalg.svd(functions-mean,full_matrices=False);factor=(singular[:71,None]/np.sqrt(len(train)-1))*vh[:71];query_basis,_=np.linalg.qr(vh[:16,64:].T);g.add('generator_SVDs');samples={};calibration={}
  for seed in SEEDS:
   rng=np.random.default_rng(seed);sample=mean+rng.standard_normal((24,71))@factor;samples[seed]=torch.tensor(sample.reshape(24,96,2));g.add('generated_functions',24);g.add('generated_scalar_outputs',24*96*2)
  basis=torch.tensor(query_basis)
  def distribution_losses(model,pseudo_c,pseudo_q,key):
   pc=model(tc);pq=model(tq);pred=[]
   for i in range(24):pred.append(pq@ridge(pc,pseudo_c[i],g,key))
   pred=torch.stack(pred);g.add('calibration_prediction_rows' if key=='calibration_ridge_solves' else 'memory_prediction_rows',24*64);old=(pseudo_q.reshape(24,-1)@basis);cur=(pred.reshape(24,-1)@basis);om=old.mean(0);cm=cur.mean(0);oc=(old-om).T@(old-om)/23;tau=1e-6*torch.trace(oc)/16;eye=torch.eye(16);oc=oc+tau*eye;cc=(cur-cm).T@(cur-cm)/23+tau*eye;delta=cm-om;mean_loss=.5*(delta@torch.linalg.solve(oc,delta))/16;signo,logo=torch.linalg.slogdet(oc);signc,logc=torch.linalg.slogdet(cc);full=.5*(torch.trace(torch.linalg.solve(cc,oc))+delta@torch.linalg.solve(cc,delta)-16+logc-logo)/16
   if signo<=0 or signc<=0:raise RuntimeError('NON_SPD_FUNCTION_COVARIANCE')
   return full,mean_loss,float(tau)
  # Source-only gradient calibration, performed once per seed before either arm.
  for seed in SEEDS:
   model=v.Features();model.load_state_dict(torch.load(PREV/f'retention_aware_seed{seed}.pt',map_location='cpu',weights_only=True));pseudo=samples[seed];full,mean_loss,tau=distribution_losses(model,pseudo[:,:32],pseudo[:,32:],'calibration_ridge_solves');gf=torch.autograd.grad(full,tuple(model.parameters()),retain_graph=True);gm=torch.autograd.grad(mean_loss,tuple(model.parameters()));g.add('calibration_gradient_evaluations',2);nf=torch.linalg.vector_norm(torch.cat([x.reshape(-1) for x in gf]));nm=torch.linalg.vector_norm(torch.cat([x.reshape(-1) for x in gm]));even=torch.arange(0,32,2);odd=torch.arange(1,32,2);pe=model(tc[even]);po=model(tc[odd]);head=ridge(pe,tyc[0,even],g,'calibration_ridge_solves');obs=mse(po@head,tyc[0,odd]);go=torch.autograd.grad(obs,tuple(model.parameters()));g.add('calibration_gradient_evaluations');no=torch.linalg.vector_norm(torch.cat([x.reshape(-1) for x in go]));calibration[seed]={'beta':float(no/nf),'mean_multiplier':float(nf/nm),'tau':tau,'obs_gradient_norm':float(no),'full_gradient_norm':float(nf),'mean_gradient_norm':float(nm)}
  arrays={};meta=[];diagnostics=[]
  for arm in ('distribution','mean_only'):
   for seed in SEEDS:
    g.add('target_sequences');model=v.Features();model.load_state_dict(torch.load(PREV/f'retention_aware_seed{seed}.pt',map_location='cpu',weights_only=True));sample=samples[seed];pseudo_c=sample[:,:32];pseudo_q=sample[:,32:]
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
     g.resource();even=torch.arange(0,32,2);odd=torch.arange(1,32,2);pe=model(tc[even]);po=model(tc[odd]);th=ridge(pe,pyc[encounter,even],g,'target_ridge_solves');tl=mse(po@th,pyc[encounter,odd]);full,mean_loss,_=distribution_losses(model,pseudo_c,pseudo_q,'memory_ridge_solves');reg=full if arm=='distribution' else calibration[seed]['mean_multiplier']*mean_loss;loss=tl+calibration[seed]['beta']*reg;tg=torch.autograd.grad(loss,tuple(model.parameters()));g.add('regularizer_evaluations')
     with torch.no_grad():
      for p,a in zip(model.parameters(),tg):p.add_(a,alpha=-.01)
     g.add('target_updates');diagnostics.append({'arm':arm,'seed':seed,'encounter':encounter,'target_loss':float(tl),'full_KL':float(full),'mean_loss':float(mean_loss),'regularizer':float(reg),'beta':calibration[seed]['beta']});evaluate(encounter+1)
  nwrite(OUT/'predictions_before_query_labels.npz',**arrays);jwrite(OUT/'prediction_barrier.json',{'status':'PREDICTIONS_PERSISTED_BEFORE_DEVELOPMENT_QUERY_LABEL_GENERATION','sha256':sha(OUT/'predictions_before_query_labels.npz'),'metadata':meta,'counts':g.n})
  dev_q=v.simulate(dev,qx);pend_q=v.simulate(pend,qx);g.add('one_step_simulations',(len(dev)+len(pend))*64);g.add('RK4_RHS_evaluations',4*(len(dev)+len(pend))*64)
  scores=[]
  for z in meta:
   key=('pend' if z['role']=='pendulum' else 'source')+f"_{z['arm']}_{z['seed']}_stage{z['stage']}";pred=arrays[key];target=pend_q[8:16] if z['role']=='pendulum' else dev_q
   for i in range(len(pred)):scores.append({**z,'law':z['laws'][i],'MSE':float(np.mean((pred[i]-target[i])**2))});g.add('score_blocks')
  def err(arm,seed,role,stage):return np.array([x['MSE'] for x in scores if x['arm']==arm and x['seed']==seed and x['role']==role and x['stage']==stage])
  comp=[]
  for seed in SEEDS:
   p0,p8=err('distribution',seed,'pendulum',0),err('distribution',seed,'pendulum',8);s0,s8=err('distribution',seed,'source',0),err('distribution',seed,'source',8);meanp=err('mean_only',seed,'pendulum',8);means=err('mean_only',seed,'source',8);base=np.array([x['MSE'] for x in previous['scores'] if x['seed']==seed and x['arm']=='retention_aware' and x['role']=='pendulum_never_adapted' and x['stage']==8]);d={'seed':seed,'pendulum_stage8_over_stage0':float(p8.sum()/p0.sum()),'distribution_over_saved_unconstrained':float(p8.sum()/base.sum()),'source_after_over_before':float(s8.sum()/s0.sum()),'source_laws_within_10pct':int(np.sum(s8<=1.1*s0)),'distribution_source_over_mean':float(s8.sum()/means.sum()),'distribution_source_wins_mean':int(np.sum(s8<means)),'distribution_pendulum_over_mean':float(p8.sum()/meanp.sum()),'finite':bool(np.isfinite(np.r_[p0,p8,s0,s8,means,meanp]).all())};d['passed']=bool(d['pendulum_stage8_over_stage0']<=.8 and d['distribution_over_saved_unconstrained']<=1.1 and d['source_after_over_before']<=1.1 and d['source_laws_within_10pct']>=20 and d['distribution_source_wins_mean']>=16 and d['distribution_pendulum_over_mean']<=1.1 and d['finite']);comp.append(d)
  for k,e in caps.items():
   if k in g.n and g.n[k]!=e:raise RuntimeError(f'COUNT_{k}:{g.n[k]}!={e}')
  elapsed=time.monotonic()-START;art=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
  if elapsed>caps['wall_seconds'] or rss()>caps['peak_RSS_bytes'] or art>caps['artifact_bytes']:raise RuntimeError('RESOURCE_CAP')
  result={'status':'SEQUENCE_AWARE_SOFT_FUNCTION_KL_V1_COMPLETE','classification':'SUPPORTED_DEVELOPMENT_ONLY' if all(x['passed'] for x in comp) else 'NOT_SUPPORTED','comparisons':comp,'calibration':calibration,'diagnostics':diagnostics,'scores':scores,'counts':g.n,'resources':{'wall_seconds':elapsed,'peak_RSS_bytes':rss(),'artifact_bytes_before_result':art},'qualifications':plan['claim_limits'],'source_hashes':{'driver':sha(HERE),'plan':sha(PLAN)}};jwrite(OUT/'result.json',result);jwrite(OUT/'completion.json',{'status':'COMPLETE','classification':result['classification'],'result_sha256':sha(OUT/'result.json'),'prediction_sha256':sha(OUT/'predictions_before_query_labels.npz'),'counts':g.n});print(json.dumps({'classification':result['classification'],'comparisons':comp,'calibration':calibration,'counts':g.n,'resources':result['resources']},indent=2))
if __name__=='__main__':main()
