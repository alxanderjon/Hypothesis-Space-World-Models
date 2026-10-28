#!/usr/bin/env python3
"""Bounded whole-group entity/set context learner comparison."""
from __future__ import annotations
import hashlib,json,math,os,resource,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'outputs/heterogeneous_three_group_package_v1';ROLL=ROOT/'outputs/heterogeneous_rollout5_labels_v1'
OUT=ROOT/'outputs/heterogeneous_entity_context_pilot_v1'
GROUPS=('sho','ball','kepler');SEEDS=(2026102908,2026102909);UPDATES=500;BATCH_SYSTEMS=8;BATCH_QUERIES=16
WALL=600.;RSS=2<<30;ART=128<<20
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def aj(p,o):
 if p.exists():raise FileExistsError(p)
 t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
 with t.open('w') as f:json.dump(o,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def an(p,**a):
 if p.exists():raise FileExistsError(p)
 t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
 with t.open('wb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def rss():
 v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;return int(v if os.uname().sysname=='Darwin' else v*1024)
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.ctx=nn.Sequential(nn.Linear(13,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU());self.q=nn.Sequential(nn.Linear(9,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU());self.out=nn.Sequential(nn.Linear(196,64),nn.SiLU(),nn.Linear(64,4))
 def context_code(self,x,dx,em,cm,dt):
  # x/dx [B,R,E,4], masks [B,E]/[B,E,4], dt [B]
  b,r,e,_=x.shape;d=dt[:,None,None,None].expand(b,r,e,1);m=cm[:,None,:,:].expand(b,r,e,4).to(x.dtype)
  z=self.ctx(torch.cat((x,dx,m,d),-1));w=em[:,None,:,None].expand(b,r,e,1).to(x.dtype);return (z*w).sum((1,2))/w.sum((1,2)).clamp_min(1.)
 def predict(self,qx,em,cm,dt,code):
  # qx [B,Q,E,4]
  b,q,e,_=qx.shape;d=dt[:,None,None,None].expand(b,q,e,1);m=cm[:,None,:,:].expand(b,q,e,4).to(qx.dtype);z=self.q(torch.cat((qx,m,d),-1));w=em[:,None,:,None].expand(b,q,e,1).to(qx.dtype);glob=(z*w).sum(2)/w.sum(2).clamp_min(1.);g=glob[:,:,None,:].expand(b,q,e,64);c=code[:,None,None,:].expand(b,q,e,64);y=self.out(torch.cat((z,g,c,m),-1));return y*w
def norm_for(indices,x,dx,qx,qy,emap):
 active=[];delta=[]
 for i in indices:
  mask=emap[i][:,None].expand(2,4).numpy();active.append(x[i,:,mask]);active.append(qx[i,:,mask]);delta.append(dx[i,:,mask]);delta.append(qy[i,:,mask])
 av=np.concatenate(active);dv=np.concatenate(delta);xs=float(np.sqrt(np.mean(av*av))+1e-8);ys=float(np.sqrt(np.mean(dv*dv))+1e-8);logs=np.log(np.asarray([dt_all[i] for i in indices]));return xs,ys,float(logs.mean()),float(logs.std()+1e-8)
def tensors(indices,xs,ys,lm,ls):
 ii=np.asarray(indices,np.int64);return (X[ii]/xs,DX[ii]/ys,QX[ii]/xs,EM[ii],CM[ii],(torch.log(DT[ii])-lm)/ls)
def fit(train_indices,target_map,seed,name):
 torch.manual_seed(seed);m=Model();opt=torch.optim.Adam(m.parameters(),lr=1e-3);xs,ys,lm,ls=norm_for(train_indices,Xn,DXn,QXn,np.asarray([target_map[i] for i in train_indices]),EM)
 g=torch.Generator().manual_seed(seed+123);losses=[]
 for step in range(UPDATES):
  si=torch.randperm(len(train_indices),generator=g)[:BATCH_SYSTEMS].numpy();ids=[train_indices[int(j)] for j in si];qi=torch.randperm(64,generator=g)[:BATCH_QUERIES]
  cx,cd,qx,em,cm,dt=tensors(ids,xs,ys,lm,ls);qy=torch.stack([torch.as_tensor(target_map[i][qi.numpy()]/ys) for i in ids])
  code=m.context_code(cx,cd,em,cm,dt);pred=m.predict(qx[:,qi],em,cm,dt,code);mask=cm[:,None,:,:].to(pred.dtype);loss=((pred-qy)**2*mask).sum()/mask.sum().clamp_min(1.)/BATCH_QUERIES
  opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
 torch.save({'state_dict':m.state_dict(),'normalizer':(xs,ys,lm,ls),'train_indices':train_indices},OUT/f'{name}.pt');return m,(xs,ys,lm,ls),losses
def predict_modes(m,norm,train_indices,dev_indices):
 xs,ys,lm,ls=norm;cx,cd,qx,em,cm,dt=tensors(dev_indices,xs,ys,lm,ls)
 with torch.no_grad():
  code=m.context_code(cx,cd,em,cm,dt);correct=m.predict(qx,em,cm,dt,code)*ys
  tcx,tcd,_,tem,tcm,tdt=tensors(train_indices,xs,ys,lm,ls);mean_code=m.context_code(tcx,tcd,tem,tcm,tdt).mean(0,keepdim=True).expand(len(dev_indices),-1);blind=m.predict(qx,em,cm,dt,mean_code)*ys
  wrong=m.predict(qx,em,cm,dt,torch.roll(code,-1,0))*ys
  time_blind=m.predict(qx,em,cm,torch.zeros_like(dt),code)*ys
 return [v.numpy() for v in (correct,blind,wrong,time_blind)],code.numpy()
def rollout(m,norm,dev_indices,codes):
 xs,ys,lm,ls=norm;em=EM[dev_indices];cm=CM[dev_indices];dt=(torch.log(DT[dev_indices])-lm)/ls;state=(X[dev_indices,-1]+DX[dev_indices,-1]).clone()/xs;code=torch.as_tensor(codes);rows=[]
 with torch.no_grad():
  for _ in range(5):
   delta=m.predict(state[:,None],em,cm,dt,code)[:,0]*ys;state=state+delta/xs;rows.append((state*xs).numpy())
 return np.stack(rows,1)
def mse_system(pred,target,mask):
 out=[]
 for p,t,m in zip(pred,target,mask):
  mm=np.broadcast_to(m,(p.shape[0],)+m.shape);out.append(float(np.mean((p[mm]-t[mm])**2)))
 return np.asarray(out)
def main():
 global X,DX,QX,EM,CM,DT,Xn,DXn,QXn,dt_all
 started=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 manifest=json.load(open(PKG/'quarantined_split_manifest.json'));records=manifest['records'];byidx={int(r['learner_index']):r for r in records}
 with np.load(PKG/'learner_inputs.npz',allow_pickle=False) as z:X=torch.as_tensor(z['context_x']);DX=torch.as_tensor(z['context_dx']);QX=torch.as_tensor(z['query_x']);EM=torch.as_tensor(z['entity_mask']);CM=torch.as_tensor(z['channel_mask']);DT=torch.as_tensor(z['delta_t'])
 Xn=X.numpy();DXn=DX.numpy();QXn=QX.numpy();dt_all=DT.numpy()
 with np.load(PKG/'training_query_targets.npz',allow_pickle=False) as z:train_target={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 train_by={g:[i for i,r in byidx.items() if r['group']==g and r['role']=='training'] for g in GROUPS};dev_by={g:[i for i,r in byidx.items() if r['group']==g and r['role']=='opened_development'] for g in GROUPS}
 plan={'status':'PRECHARGED_BEFORE_FITS','groups':GROUPS,'seeds':SEEDS,'fits':12,'updates_per_fit':UPDATES,'total_updates':6000,'training_context_rows_encoded':1536000,'training_query_rows':768000,'evaluation_query_rows':15360,'rollout_prediction_steps':480,'ceilings':{'wall_seconds':WALL,'peak_rss_bytes':RSS,'artifact_bytes':ART},'controls':['correct','blind_context','cyclic_wrong_context','time_blind','target_group_scratch'],'prediction_before_development_labels':True,'source_hashes':{'driver':sha(Path(__file__)),'package_barrier':sha(PKG/'prediction_barrier.json'),'rollout_barrier':sha(ROLL/'barrier.json')}};aj(OUT/'plan.json',plan)
 preds={};rollpred={};hist={};meta=[]
 for held in GROUPS:
  shared_train=sum((train_by[g] for g in GROUPS if g!=held),[]);scratch_train=train_by[held];dev=dev_by[held]
  for seed in SEEDS:
   sm,sn,sl=fit(shared_train,train_target,seed,f'shared_held_{held}_seed{seed}');modes,codes=predict_modes(sm,sn,shared_train,dev)
   for mode,val in zip(('correct','blind','wrong','time_blind'),modes):key=f'shared__{held}__{seed}__{mode}';preds[key]=val;meta.append({'key':key,'arm':'shared','held':held,'seed':seed,'mode':mode,'dev_indices':dev})
   rollpred[f'shared__{held}__{seed}']=rollout(sm,sn,dev,codes);hist[f'shared__{held}__{seed}']=sl
   tm,tn,tl=fit(scratch_train,train_target,seed+1000,f'scratch_{held}_seed{seed}');modes2,codes2=predict_modes(tm,tn,scratch_train,dev);key=f'scratch__{held}__{seed}__correct';preds[key]=modes2[0];meta.append({'key':key,'arm':'scratch','held':held,'seed':seed,'mode':'correct','dev_indices':dev});rollpred[f'scratch__{held}__{seed}']=rollout(tm,tn,dev,codes2);hist[f'scratch__{held}__{seed}']=tl
 an(OUT/'predictions_before_development_labels.npz',**preds,**{f'rollout__{k}':v for k,v in rollpred.items()});aj(OUT/'prediction_barrier.json',{'status':'ALL_PREDICTIONS_PERSISTED_BEFORE_DEVELOPMENT_LABELS','prediction_sha256':sha(OUT/'predictions_before_development_labels.npz'),'metadata':meta})
 # Only now open development and rollout labels.
 with np.load(PKG/'protected_development_query_targets.npz',allow_pickle=False) as z:dev_target={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 with np.load(ROLL/'protected_rollout5_targets.npz',allow_pickle=False) as z:roll_target={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['truth_x'])}
 comps=[];scores=[]
 for held in GROUPS:
  dev=dev_by[held];target=np.asarray([dev_target[i] for i in dev]);rt=np.asarray([roll_target[i] for i in dev]);mask=CM[dev].numpy()
  for seed in SEEDS:
   values={mode:mse_system(preds[f'shared__{held}__{seed}__{mode}'],target,mask) for mode in ('correct','blind','wrong','time_blind')};scratch=mse_system(preds[f'scratch__{held}__{seed}__correct'],target,mask);sr=mse_system(rollpred[f'shared__{held}__{seed}'],rt,mask);tr=mse_system(rollpred[f'scratch__{held}__{seed}'],rt,mask)
   d={'held_group':held,'seed':seed,'shared_MSE':float(values['correct'].mean()),'scratch_MSE':float(scratch.mean()),'blind_MSE':float(values['blind'].mean()),'wrong_MSE':float(values['wrong'].mean()),'time_blind_MSE':float(values['time_blind'].mean()),'shared_over_scratch':float(values['correct'].sum()/scratch.sum()),'shared_over_blind':float(values['correct'].sum()/values['blind'].sum()),'correct_wins_wrong':int(np.sum(values['correct']<values['wrong'])),'shared_rollout_MSE':float(sr.mean()),'scratch_rollout_MSE':float(tr.mean()),'shared_rollout_over_scratch':float(sr.sum()/tr.sum())};d['passed_forward_gate']=bool(d['shared_over_scratch']<=.8 and d['shared_over_blind']<=.8 and d['correct_wins_wrong']>=6 and d['shared_rollout_over_scratch']<=1.0);comps.append(d)
   for j,i in enumerate(dev):scores.append({'held_group':held,'seed':seed,'learner_index':i,'shared_MSE':float(values['correct'][j]),'scratch_MSE':float(scratch[j]),'blind_MSE':float(values['blind'][j]),'wrong_MSE':float(values['wrong'][j]),'time_blind_MSE':float(values['time_blind'][j]),'shared_rollout_MSE':float(sr[j]),'scratch_rollout_MSE':float(tr[j])})
 counts={'fits':12,'optimizer_updates':6000,'training_context_rows_encoded':6000*8*32,'training_query_rows':6000*8*16,'evaluation_query_rows':len(preds)*8*64,'rollout_prediction_steps':len(rollpred)*8*5,'score_system_blocks':len(scores)}
 for k in ('training_context_rows_encoded','training_query_rows','evaluation_query_rows','rollout_prediction_steps'):
  if counts[k]!=plan[k]:raise RuntimeError(f'COUNT_{k}:{counts[k]}!={plan[k]}')
 elapsed=time.monotonic()-started;art=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());res={'wall_seconds':elapsed,'peak_rss_bytes':rss(),'artifact_bytes_before_result':art}
 if elapsed>WALL or res['peak_rss_bytes']>RSS or art>ART:raise RuntimeError(f'RESOURCE_CAP:{res}')
 supported=all(d['passed_forward_gate'] for d in comps);result={'status':'HETEROGENEOUS_ENTITY_CONTEXT_PILOT_V1_COMPLETE','classification':'SUPPORTED_FORWARD_DEVELOPMENT_ONLY' if supported else 'NOT_SUPPORTED','comparisons':comps,'scores':scores,'counts':counts,'resources':res,'parameter_count':sum(p.numel() for p in Model().parameters()),'training_loss_endpoints':{k:v[-1] for k,v in hist.items()},'qualification':['Whole-group folds test this fixed common interface and model, not arbitrary mechanism discovery.','Correct-versus-wrong contexts share layout and timestep, limiting family-ID shortcuts.','Scratch sees 24 target-group laws; shared sees zero target-group laws, so the comparison is deliberately stringent.','No persistent chronology was run; a forward pass cannot establish continual accumulation.','All evaluation groups are opened development evidence.']};aj(OUT/'result.json',result);aj(OUT/'completion.json',{'status':'COMPLETE','classification':result['classification'],'result_sha256':sha(OUT/'result.json'),'counts':counts});print(json.dumps({'classification':result['classification'],'comparisons':comps,'counts':counts,'resources':res,'parameter_count':result['parameter_count']},indent=2))
if __name__=='__main__':main()
