#!/usr/bin/env python3
import copy, json, os, resource, time
from pathlib import Path
import numpy as np, torch
from torch import nn

torch.set_num_threads(1)
START=time.monotonic()
DATA=Path('/Users/alexanderrichardson/Documents/Codex/2026-10-14/task-2/data/pdebench_range_subset_v1')
OUT=DATA/'task_transformer_markov_v1'
SEEDS=(202610281,202610282)
SRC=('advection_beta0.2','advection_beta1.0','advection_beta4.0','burgers_nu0.004','burgers_nu0.04','burgers_nu0.4')
TGT=('reacdiff_nu0.5_rho1','reacdiff_nu1_rho2','reacdiff_nu5_rho10')

class Model(nn.Module):
 def __init__(self):
  super().__init__();d=128
  self.c=nn.Linear(6,d);self.q=nn.Linear(6,d)
  el=nn.TransformerEncoderLayer(d,4,256,batch_first=True,norm_first=True,dropout=0.0,activation='gelu')
  self.enc=nn.TransformerEncoder(el,3,enable_nested_tensor=False)
  dl=nn.TransformerDecoderLayer(d,4,256,batch_first=True,norm_first=True,dropout=0.0,activation='gelu')
  self.dec=nn.TransformerDecoder(dl,3);self.out=nn.Linear(d,4)
 def forward(self,c,q):return self.out(self.dec(self.q(q),self.enc(self.c(c))))

def load(names):
 out=[]
 for n in names:
  z=np.load(DATA/f'{n}.npz');out.append((n,z['field'],z['x'],z['t']))
 return out

def tokens(field,x,t,start,mean,std,dstd,history_donor=None,current_donor=None):
 h=field if history_donor is None else history_donor
 cur=field if current_donor is None else current_donor
 c=h[start:start+4].reshape(4,16,4);xc=x.reshape(16,4).mean(1)
 ct=np.stack(np.meshgrid(t[start:start+4]/2,xc,indexing='ij'),-1)
 c=np.concatenate(((c-mean)/std,ct),-1).reshape(64,6)
 u=cur[start+3].reshape(16,4);qt=np.stack((xc,np.full(16,t[start+3]/2)),axis=-1)
 q=np.concatenate(((u-mean)/std,qt),-1)
 y=((field[start+4]-field[start+3])/dstd).reshape(16,4)
 return c.astype('float32'),q.astype('float32'),y.astype('float32')

def batch(pool,rng,n,mean,std,dstd):
 xs=[]
 for _ in range(n):
  _,a,x,t=pool[int(rng.integers(len(pool)))];i=int(rng.integers(len(a)));start=int(rng.integers(97));xs.append(tokens(a[i],x,t,start,mean,std,dstd))
 return tuple(torch.tensor(np.stack([z[j] for z in xs])) for j in range(3))
def loss(m,b):c,q,y=b;return torch.mean((m(c,q)-y)**2)

def fit(m,steps,seed,target,source,mean,std,dstd,replay):
 rng=np.random.default_rng(seed);opt=torch.optim.AdamW(m.parameters(),lr=3e-4);hist=[]
 for u in range(steps):
  tb=batch(target,rng,2,mean,std,dstd);sb=batch(source,rng,2,mean,std,dstd)
  lt=loss(m,tb);ls=loss(m,sb);z=lt+(.5*ls if replay else 0*ls)
  opt.zero_grad();z.backward();opt.step()
  if u in (0,steps-1):hist.append({'update':u+1,'target_loss':float(lt),'source_loss':float(ls),'objective':float(z)})
 return hist

def evaluate(m,pool,mean,std,dstd,mode='intact'):
 rec=[]
 with torch.no_grad():
  for name,a,x,t in pool:
   for i in range(len(a)):
    donor=a[(i+1)%len(a)]
    for start in (0,32,64,96):
     hd=donor if mode=='history_shuffle' else None;qd=donor if mode=='current_shuffle' else None
     c,q,y=tokens(a[i],x,t,start,mean,std,dstd,hd,qd)
     pred=m(torch.tensor(c)[None],torch.tensor(q)[None])[0].numpy()
     rec.append({'file':name,'trajectory':i,'start':start,'mse':float(np.mean((pred-y)**2))})
 return rec

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();src=load(SRC);tgt=load(TGT)
 train=np.concatenate([a[:20] for _,a,_,_ in src]);mean=float(train.mean());std=float(train.std());dstd=float(np.diff(train,axis=1).std());assert min(std,dstd)>0
 src_train=[(n,a[:20],x,t) for n,a,x,t in src];src_dev=[(n,a[20:25],x,t) for n,a,x,t in src]
 tgt_adapt=[(n,a[:10],x,t) for n,a,x,t in tgt];tgt_eval=[(n,a[10:25],x,t) for n,a,x,t in tgt]
 allscores=[];logs=[]
 for seed in SEEDS:
  torch.manual_seed(seed);base=Model();rng=np.random.default_rng(seed);opt=torch.optim.AdamW(base.parameters(),lr=3e-4);h=[]
  for u in range(300):
   b=batch(src_train,rng,4,mean,std,dstd);z=loss(base,b);opt.zero_grad();z.backward();opt.step()
   if u in (0,299):h.append({'update':u+1,'loss':float(z)})
  arms={'frozen':copy.deepcopy(base),'replay':copy.deepcopy(base),'no_replay':copy.deepcopy(base)}
  torch.manual_seed(seed+1000);arms['scratch']=Model();logs.append({'seed':seed,'arm':'source','history':h})
  logs.append({'seed':seed,'arm':'replay','history':fit(arms['replay'],100,seed+1,tgt_adapt,src_train,mean,std,dstd,True)})
  logs.append({'seed':seed,'arm':'no_replay','history':fit(arms['no_replay'],100,seed+2,tgt_adapt,src_train,mean,std,dstd,False)})
  logs.append({'seed':seed,'arm':'scratch','history':fit(arms['scratch'],100,seed+3,tgt_adapt,src_train,mean,std,dstd,False)})
  for arm,m in arms.items():
   torch.save(m.state_dict(),OUT/f'{arm}_seed{seed}.pt')
   for r in evaluate(m,src_dev,mean,std,dstd):allscores.append({'seed':seed,'arm':arm,'role':'source','mode':'intact',**r})
   for mode in ('intact','history_shuffle','current_shuffle'):
    for r in evaluate(m,tgt_eval,mean,std,dstd,mode):allscores.append({'seed':seed,'arm':arm,'role':'target','mode':mode,**r})
 comp=[]
 for seed in SEEDS:
  def vals(a,r,mode='intact'):return np.array([x['mse'] for x in allscores if x['seed']==seed and x['arm']==a and x['role']==r and x['mode']==mode])
  q=vals('replay','target');f=vals('frozen','target');s=vals('scratch','target');nr=vals('no_replay','target');old=vals('frozen','source');ret=vals('replay','source');hs=vals('replay','target','history_shuffle');cs=vals('replay','target','current_shuffle')
  d={'seed':seed,'replay_over_frozen':float(q.sum()/f.sum()),'replay_wins_frozen':int(np.sum(q<f)),'replay_over_scratch':float(q.sum()/s.sum()),'replay_wins_scratch':int(np.sum(q<s)),'replay_over_no_replay':float(q.sum()/nr.sum()),'source_after_over_before':float(ret.sum()/old.sum()),'source_blocks_within_10pct':int(np.sum(ret<=1.1*old)),'source_blocks':len(old),'history_shuffle_over_intact':float(hs.sum()/q.sum()),'current_shuffle_over_intact':float(cs.sum()/q.sum())}
  d['passed']=bool(d['replay_over_frozen']<=.8 and d['replay_over_scratch']<=.8 and d['replay_wins_frozen']/len(q)>=.6 and d['replay_wins_scratch']/len(q)>=.6 and d['source_after_over_before']<=1.1 and d['source_blocks_within_10pct']/len(old)>=.8 and d['history_shuffle_over_intact']>=1.05 and d['current_shuffle_over_intact']>=1.1);comp.append(d)
 result={'status':'COMPLETE','classification':'SUPPORTED_DEVELOPMENT_ONLY' if all(x['passed'] for x in comp) else 'NOT_SUPPORTED','architecture':{'parameters':sum(p.numel() for p in Model().parameters()),'context_tokens':64,'query_tokens':16,'query_includes_current_field':True},'normalization':{'field_mean':mean,'field_std':std,'increment_std':dstd,'source_only':True},'counts':{'fits':8,'source_updates':600,'adaptation_updates':600,'training_windows':4800,'evaluation_blocks':len(allscores)},'comparisons':comp,'logs':logs,'scores':allscores,'resources':{'wall_seconds':time.monotonic()-START,'peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},'qualifications':['Opened official PDEBench development subset; no fresh confirmation.','Explicit current field and one-step increment change the v1 observation/task interface.','Full-field scalar-grid Markov structure, coordinates, time, patching and chronology are supplied.','Target adaptation updates all shared weights; replay stores real source examples.','Shuffle controls are incoherent diagnostic interventions, not deployable inputs.']}
 p=OUT/'result.json';tmp=p.with_name('.result.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
 print(json.dumps({'classification':result['classification'],'comparisons':comp,'counts':result['counts'],'resources':result['resources'],'normalization':result['normalization'],'architecture':result['architecture']},indent=2))
if __name__=='__main__':main()
