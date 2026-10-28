#!/usr/bin/env python3
"""Training-only source/target gradient conflict at six source checkpoints."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/heterogeneous_entity_context_pilot_v1.py';PKG=ROOT/'outputs/heterogeneous_three_group_package_v1';SOURCE=ROOT/'outputs/heterogeneous_entity_context_pilot_v1_recovery2';OUT=ROOT/'outputs/heterogeneous_gradient_conflict_v1';GROUPS=('sho','ball','kepler');SEEDS=(2026102908,2026102909)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 spec=importlib.util.spec_from_file_location('hetero_grad_base',BASE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.OUT=OUT;OUT.mkdir(parents=True)
 manifest=json.load(open(PKG/'quarantined_split_manifest.json'));byidx={int(r['learner_index']):r for r in manifest['records']}
 with np.load(PKG/'learner_inputs.npz',allow_pickle=False) as z:m.X=torch.as_tensor(z['context_x']);m.DX=torch.as_tensor(z['context_dx']);m.QX=torch.as_tensor(z['query_x']);m.EM=torch.as_tensor(z['entity_mask']);m.CM=torch.as_tensor(z['channel_mask']);m.DT=torch.as_tensor(z['delta_t'])
 m.dt_all=m.DT.numpy()
 with np.load(PKG/'training_query_targets.npz',allow_pickle=False) as z:targets={int(i):torch.as_tensor(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 train_by={g:sorted([i for i,r in byidx.items() if r['group']==g and r['role']=='training'],key=lambda i:byidx[i]['law_ordinal']) for g in GROUPS}
 m.aj(OUT/'plan.json',{'status':'PRECHARGED','checkpoint_loads':6,'backward_passes':12,'systems_per_source_gradient':48,'systems_per_target_gradient':4,'query_rows_per_system':64,'optimizer_updates':0,'evaluation_labels_used':0,'source_hashes':{'driver':sha(Path(__file__)),'base':sha(BASE),'source':sha(SOURCE/'completion.json')}})
 def gradient(model,norm,ids):
  xs,ys,lm,ls=norm;cx,cd,qx,em,cm,dt=m.tensors(ids,xs,ys,lm,ls);qy=torch.stack([targets[i]/ys for i in ids]);code=model.context_code(cx,cd,em,cm,dt);pred=model.predict(qx,em,cm,dt,code);mask=cm[:,None].to(pred.dtype);loss=((pred-qy)**2*mask).sum()/mask.sum().clamp_min(1.)/64;model.zero_grad();loss.backward();return float(loss.detach()),{name:p.grad.detach().reshape(-1).clone() for name,p in model.named_parameters()}
 def cos(a,b):
  aa=torch.cat(list(a.values()));bb=torch.cat(list(b.values()));return float(torch.dot(aa,bb)/(torch.linalg.vector_norm(aa)*torch.linalg.vector_norm(bb)).clamp_min(1e-30)),float(torch.linalg.vector_norm(aa)),float(torch.linalg.vector_norm(bb))
 rows=[]
 for held in GROUPS:
  src=sum((train_by[g] for g in GROUPS if g!=held),[]);target=train_by[held][:4]
  for seed in SEEDS:
   ck=torch.load(SOURCE/f'shared_held_{held}_seed{seed}.pt',map_location='cpu',weights_only=False);model=m.Model();model.load_state_dict(ck['state_dict']);norm=tuple(float(x) for x in ck['normalizer']);sl,sg=gradient(model,norm,src);tl,tg=gradient(model,norm,target);overall,sn,tn=cos(sg,tg);layers={}
   for prefix in ('ctx','q','out'):
    s={k:v for k,v in sg.items() if k.startswith(prefix+'.')};t={k:v for k,v in tg.items() if k.startswith(prefix+'.')};c,a,b=cos(s,t);layers[prefix]={'cosine':c,'source_norm':a,'target_norm':b,'target_energy_fraction':float(b*b/(tn*tn))}
   rows.append({'held_group':held,'seed':seed,'source_training_loss':sl,'target4_training_loss':tl,'overall_cosine':overall,'source_gradient_norm':sn,'target_gradient_norm':tn,'target_over_source_norm':tn/sn,'layers':layers})
 result={'status':'HETEROGENEOUS_GRADIENT_CONFLICT_V1_COMPLETE','classification':'TRAINING_ONLY_DIAGNOSTIC','rows':rows,'counts':{'checkpoint_loads':6,'backward_passes':12,'source_system_query_blocks':6*48*64,'target_system_query_blocks':6*4*64,'optimizer_updates':0},'qualification':['Gradient cosine is local and parameterization-dependent; it does not prove a universal stability-plasticity barrier.','Target gradients use only the first four prospectively ordered training laws; no development labels.','Negative alignment supports update interference; nonnegative alignment would not establish feature sufficiency.']};m.aj(OUT/'result.json',result);m.aj(OUT/'completion.json',{'status':'COMPLETE','result_sha256':sha(OUT/'result.json'),'counts':result['counts']});print(json.dumps(result,indent=2))
if __name__=='__main__':main()
