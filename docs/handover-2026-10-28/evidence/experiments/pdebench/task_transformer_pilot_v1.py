#!/usr/bin/env python3
import copy, json, os, resource, time
from pathlib import Path
import numpy as np, torch
from task_transformer_preflight_v1 import Model, tokens

torch.set_num_threads(1);START=time.monotonic();DATA=Path('/Users/alexanderrichardson/Documents/Codex/2026-10-14/task-2/data/pdebench_range_subset_v1');OUT=DATA/'task_transformer_pilot_v1';SEEDS=(202610281,202610282)
SRC=('advection_beta0.2','advection_beta1.0','advection_beta4.0','burgers_nu0.004','burgers_nu0.04','burgers_nu0.4');TGT=('reacdiff_nu0.5_rho1','reacdiff_nu1_rho2','reacdiff_nu5_rho10')
def load(names):
 out=[]
 for n in names:
  z=np.load(DATA/f'{n}.npz');out.append((n,z['field'],z['x'],z['t']))
 return out
def batch(pool,rng,n,mean,std):
 xs=[]
 for _ in range(n):
  name,a,x,t=pool[int(rng.integers(len(pool)))];i=int(rng.integers(len(a)));start=int(rng.integers(24))*4;xs.append(tokens(a[i],x,t,start,mean,std))
 return tuple(torch.tensor(np.stack([z[j] for z in xs])) for j in range(3))
def loss(m,b):c,q,y=b;return torch.mean((m(c,q)-y)**2)
def fit(m,steps,seed,target,source,mean,std,replay):
 rng=np.random.default_rng(seed);opt=torch.optim.AdamW(m.parameters(),lr=3e-4);hist=[]
 for u in range(steps):
  tb=batch(target,rng,2,mean,std);sb=batch(source,rng,2,mean,std);lt=loss(m,tb);ls=loss(m,sb);z=lt+(.5*ls if replay else 0*ls);opt.zero_grad();z.backward();opt.step()
  if u in (0,steps-1):hist.append({'update':u+1,'target_loss':float(lt),'source_loss':float(ls),'objective':float(z)})
 return hist
def evaluate(m,pool,mean,std,shuffle=False):
 rec=[]
 with torch.no_grad():
  for fi,(name,a,x,t) in enumerate(pool):
   for i in range(len(a)):
    donor=(i+1)%len(a) if shuffle else i
    for start in (0,32,64):
     c,q,y=tokens(a[donor],x,t,start,mean,std);pred=m(torch.tensor(c)[None],torch.tensor(q)[None])[0].numpy();rec.append({'file':name,'trajectory':i,'start':start,'mse':float(np.mean((pred-y)**2))})
 return rec
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();src=load(SRC);tgt=load(TGT);mean=float(np.concatenate([a[:20] for _,a,_,_ in src]).mean());std=float(np.concatenate([a[:20] for _,a,_,_ in src]).std())
 src_train=[(n,a[:20],x,t) for n,a,x,t in src];src_dev=[(n,a[20:25],x,t) for n,a,x,t in src];tgt_adapt=[(n,a[:10],x,t) for n,a,x,t in tgt];tgt_eval=[(n,a[10:25],x,t) for n,a,x,t in tgt]
 allscores=[];logs=[]
 for seed in SEEDS:
  torch.manual_seed(seed);base=Model();rng=np.random.default_rng(seed);opt=torch.optim.AdamW(base.parameters(),lr=3e-4);h=[]
  for u in range(300):
   b=batch(src_train,rng,4,mean,std);z=loss(base,b);opt.zero_grad();z.backward();opt.step()
   if u in (0,299):h.append({'update':u+1,'loss':float(z)})
  arms={'frozen':copy.deepcopy(base),'replay':copy.deepcopy(base),'no_replay':copy.deepcopy(base)};torch.manual_seed(seed+1000);arms['scratch']=Model();logs.append({'seed':seed,'arm':'source','history':h})
  logs.append({'seed':seed,'arm':'replay','history':fit(arms['replay'],100,seed+1,tgt_adapt,src_train,mean,std,True)})
  logs.append({'seed':seed,'arm':'no_replay','history':fit(arms['no_replay'],100,seed+2,tgt_adapt,src_train,mean,std,False)})
  logs.append({'seed':seed,'arm':'scratch','history':fit(arms['scratch'],100,seed+3,tgt_adapt,src_train,mean,std,False)})
  for arm,m in arms.items():
   torch.save(m.state_dict(),OUT/f'{arm}_seed{seed}.pt')
   for role,pool in (('source',src_dev),('target',tgt_eval)):
    for shuffled in (False,True) if role=='target' else (False,):
     for r in evaluate(m,pool,mean,std,shuffled):allscores.append({'seed':seed,'arm':arm,'role':role,'shuffled':shuffled,**r})
 comp=[]
 for seed in SEEDS:
  def vals(a,r,sh=False):return np.array([x['mse'] for x in allscores if x['seed']==seed and x['arm']==a and x['role']==r and x['shuffled']==sh])
  q=vals('replay','target');f=vals('frozen','target');s=vals('scratch','target');nr=vals('no_replay','target');old=vals('frozen','source');ret=vals('replay','source');sh=vals('replay','target',True)
  d={'seed':seed,'replay_over_frozen':float(q.sum()/f.sum()),'replay_wins_frozen':int(np.sum(q<f)),'replay_over_scratch':float(q.sum()/s.sum()),'replay_wins_scratch':int(np.sum(q<s)),'replay_over_no_replay':float(q.sum()/nr.sum()),'source_after_over_before':float(ret.sum()/old.sum()),'source_blocks_within_10pct':int(np.sum(ret<=1.1*old)),'source_blocks':len(old),'shuffled_over_intact':float(sh.sum()/q.sum()),'passed':bool(q.sum()/f.sum()<=.8 and q.sum()/s.sum()<=.8 and np.mean(q<f)>=.6 and np.mean(q<s)>=.6 and ret.sum()/old.sum()<=1.1 and np.mean(ret<=1.1*old)>=.8 and sh.sum()/q.sum()>=1.1)};comp.append(d)
 result={'status':'COMPLETE','classification':'SUPPORTED_DEVELOPMENT_ONLY' if all(x['passed'] for x in comp) else 'NOT_SUPPORTED','architecture':{'parameters':995588,'context_tokens':64,'query_tokens':64},'normalization':{'mean':mean,'std':std,'source_only':True},'counts':{'fits':8,'source_updates':600,'adaptation_updates':600,'training_windows':4800,'evaluation_blocks':len(allscores)},'comparisons':comp,'logs':logs,'scores':allscores,'resources':{'wall_seconds':time.monotonic()-START,'peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},'qualifications':['Official PDEBench subset; scalar-grid full-field interface and preprocessing are supplied assumptions.','Reaction-diffusion is opened development evidence.','Replay/scratch adaptation updates shared weights using target labels; this is not zero-shot inference.','No claim of novel Transformer mechanism, arbitrary physics, or calibrated uncertainty.']}
 p=OUT/'result.json';t=p.with_name('.result.tmp');t.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(t,p);print(json.dumps({'classification':result['classification'],'comparisons':comp,'counts':result['counts'],'resources':result['resources']},indent=2))
if __name__=='__main__':main()
