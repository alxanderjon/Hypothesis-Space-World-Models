#!/usr/bin/env python3
import json, os
from pathlib import Path
import numpy as np, torch
from torch import nn

torch.set_num_threads(1);torch.manual_seed(20261028)
DATA=Path('/Users/alexanderrichardson/Documents/Codex/2026-10-14/task-2/data/pdebench_range_subset_v1')
SOURCE=('advection_beta0.2','advection_beta1.0','advection_beta4.0','burgers_nu0.004','burgers_nu0.04','burgers_nu0.4')
class Model(nn.Module):
 def __init__(self):
  super().__init__();d=128
  self.c=nn.Linear(6,d);self.q=nn.Linear(2,d)
  el=nn.TransformerEncoderLayer(d,4,256,batch_first=True,norm_first=True,dropout=0.0,activation='gelu');self.enc=nn.TransformerEncoder(el,3,enable_nested_tensor=False)
  dl=nn.TransformerDecoderLayer(d,4,256,batch_first=True,norm_first=True,dropout=0.0,activation='gelu');self.dec=nn.TransformerDecoder(dl,3);self.out=nn.Linear(d,4)
 def forward(self,c,q):return self.out(self.dec(self.q(q),self.enc(self.c(c))))
def tokens(field,x,t,start,mean,std):
 c=field[start:start+4].reshape(4,16,4);y=field[start+4:start+8].reshape(4,16,4);xc=x.reshape(16,4).mean(1);ct=np.stack(np.meshgrid(t[start:start+4]/2,xc,indexing='ij'),-1);qt=np.stack(np.meshgrid(t[start+4:start+8]/2,xc,indexing='ij'),-1)
 c=np.concatenate(((c-mean)/std,ct),-1).reshape(64,6);return c.astype('float32'),qt.reshape(64,2).astype('float32'),((y-mean)/std).reshape(64,4).astype('float32')
def main():
 vals=[]
 for n in SOURCE:
  with np.load(DATA/f'{n}.npz') as z:vals.append(z['field'][:20])
 allv=np.concatenate(vals);mean=float(allv.mean());std=float(allv.std());assert std>0
 batch=[]
 for n in SOURCE[:2]:
  with np.load(DATA/f'{n}.npz') as z:batch.append(tokens(z['field'][0],z['x'],z['t'],0,mean,std))
 c=torch.tensor(np.stack([x[0] for x in batch]));q=torch.tensor(np.stack([x[1] for x in batch]));y=torch.tensor(np.stack([x[2] for x in batch]));m=Model();params=sum(p.numel() for p in m.parameters());pred=m(c,q);loss=torch.mean((pred-y)**2);loss.backward();gn=float(torch.sqrt(sum(torch.sum(p.grad**2) for p in m.parameters())));assert pred.shape==(2,64,4) and np.isfinite([float(loss),gn]).all() and gn>0
 opt=torch.optim.AdamW(m.parameters(),lr=3e-4);before=torch.cat([p.detach().flatten() for p in m.parameters()]);opt.step();after=torch.cat([p.detach().flatten() for p in m.parameters()]);delta=float(torch.linalg.vector_norm(after-before));assert delta>0
 out={'status':'PREFLIGHT_COMPLETE','parameter_count':params,'source_field_mean':mean,'source_field_std':std,'context_shape':list(c.shape),'query_shape':list(q.shape),'target_shape':list(y.shape),'prediction_shape':list(pred.shape),'initial_loss':float(loss),'gradient_norm':gn,'one_step_parameter_delta':delta,'scientific_updates':0,'fits':0,'note':'Disposable gradient-flow update only; no checkpoint or scientific prediction persisted.'}
 p=DATA/'task_transformer_preflight_v1.json'
 if p.exists():raise FileExistsError(p)
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');os.replace(t,p);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
