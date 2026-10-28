#!/usr/bin/env python3
"""Matched rapid-acquisition test: source-pretrained versus reset weights."""
from __future__ import annotations
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/heterogeneous_entity_context_pilot_v1.py';PKG=ROOT/'outputs/heterogeneous_three_group_package_v1';SOURCE=ROOT/'outputs/heterogeneous_entity_context_pilot_v1_recovery2';OUT=ROOT/'outputs/heterogeneous_rapid_acquisition_v1';GROUPS=('sho','ball','kepler');SEEDS=(2026102908,2026102909);UPDATES=100
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 spec=importlib.util.spec_from_file_location('hetero_rapid_base',BASE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.OUT=OUT;OUT.mkdir(parents=True)
 manifest=json.load(open(PKG/'quarantined_split_manifest.json'));records=manifest['records'];byidx={int(r['learner_index']):r for r in records}
 with np.load(PKG/'learner_inputs.npz',allow_pickle=False) as z:m.X=torch.as_tensor(z['context_x']);m.DX=torch.as_tensor(z['context_dx']);m.QX=torch.as_tensor(z['query_x']);m.EM=torch.as_tensor(z['entity_mask']);m.CM=torch.as_tensor(z['channel_mask']);m.DT=torch.as_tensor(z['delta_t'])
 m.Xn=m.X.numpy();m.DXn=m.DX.numpy();m.QXn=m.QX.numpy();m.dt_all=m.DT.numpy()
 with np.load(PKG/'training_query_targets.npz',allow_pickle=False) as z:train_target={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 train_by={g:sorted([i for i,r in byidx.items() if r['group']==g and r['role']=='training'],key=lambda i:byidx[i]['law_ordinal']) for g in GROUPS};dev_by={g:sorted([i for i,r in byidx.items() if r['group']==g and r['role']=='opened_development'],key=lambda i:byidx[i]['law_ordinal']) for g in GROUPS}
 selected={g:train_by[g][:4] for g in GROUPS};plan={'status':'PRECHARGED','target_laws_per_group':4,'selection':'first four prospective law ordinals, independent of outcomes','seeds':SEEDS,'arms':['source_pretrained_then_100_target_updates','reset_then_100_target_updates'],'fits':12,'new_target_updates':1200,'source_pretraining_updates_reused_not_repeated':3000,'training_context_rows_encoded':153600,'training_query_rows':76800,'evaluation_query_rows':24576,'source_normalization_frozen_for_both_arms':True,'qualification':'Supervised per-mechanism shared-weight adaptation; not zero-shot recognition.','source_hashes':{'driver':sha(Path(__file__)),'base':sha(BASE),'source_completion':sha(SOURCE/'completion.json'),'package':sha(PKG/'completion.json')}};m.aj(OUT/'plan.json',plan)
 def adapt(model,norm,ids,seed):
  xs,ys,lm,ls=norm;opt=torch.optim.Adam(model.parameters(),lr=1e-3);g=torch.Generator().manual_seed(seed+777);loss=[]
  for _ in range(UPDATES):
   qi=torch.randperm(64,generator=g)[:16];cx,cd,qx,em,cm,dt=m.tensors(ids,xs,ys,lm,ls);qy=torch.stack([torch.as_tensor(train_target[i][qi.numpy()]/ys) for i in ids]);code=model.context_code(cx,cd,em,cm,dt);pred=model.predict(qx[:,qi],em,cm,dt,code);mask=cm[:,None].to(pred.dtype);z=((pred-qy)**2*mask).sum()/mask.sum().clamp_min(1.)/16;opt.zero_grad();z.backward();opt.step();loss.append(float(z.detach()))
  return loss
 def predict_pair(model,norm,ids):
  xs,ys,lm,ls=norm;cx,cd,qx,em,cm,dt=m.tensors(ids,xs,ys,lm,ls)
  with torch.no_grad():code=model.context_code(cx,cd,em,cm,dt);return (model.predict(qx,em,cm,dt,code)*ys).numpy(),(model.predict(qx,em,cm,dt,torch.roll(code,-1,0))*ys).numpy()
 preds={};meta=[];losses={}
 for held in GROUPS:
  srcdev=sum((dev_by[g] for g in GROUPS if g!=held),[]);targetdev=dev_by[held]
  for seed in SEEDS:
   checkpoint=torch.load(SOURCE/f'shared_held_{held}_seed{seed}.pt',map_location='cpu',weights_only=False);norm=tuple(float(x) for x in checkpoint['normalizer'])
   pre=m.Model();pre.load_state_dict(checkpoint['state_dict']);base_src,_=predict_pair(pre,norm,srcdev)
   pre_loss=adapt(pre,norm,selected[held],seed);pc,pw=predict_pair(pre,norm,targetdev);post_src,_=predict_pair(pre,norm,srcdev)
   torch.manual_seed(seed);reset=m.Model();reset_loss=adapt(reset,norm,selected[held],seed);rc,rw=predict_pair(reset,norm,targetdev)
   for key,val in ((f'pretrained__{held}__{seed}__correct',pc),(f'pretrained__{held}__{seed}__wrong',pw),(f'reset__{held}__{seed}__correct',rc),(f'reset__{held}__{seed}__wrong',rw),(f'source_before__{held}__{seed}',base_src),(f'source_after__{held}__{seed}',post_src)):preds[key]=val
   meta.append({'held':held,'seed':seed,'target_dev_indices':targetdev,'source_dev_indices':srcdev});losses[f'pretrained__{held}__{seed}']=pre_loss;losses[f'reset__{held}__{seed}']=reset_loss
 m.an(OUT/'predictions_before_rescoring_opened_development.npz',**preds);m.aj(OUT/'prediction_receipt.json',{'status':'PREDICTIONS_PERSISTED_BEFORE_RESCORING_HISTORICALLY_OPENED_LABELS','sha256':sha(OUT/'predictions_before_rescoring_opened_development.npz'),'metadata':meta})
 with np.load(PKG/'protected_development_query_targets.npz',allow_pickle=False) as z:dev_target={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 comps=[];scores=[]
 for held in GROUPS:
  targetids=dev_by[held];sourceids=sum((dev_by[g] for g in GROUPS if g!=held),[]);tt=np.asarray([dev_target[i] for i in targetids]);st=np.asarray([dev_target[i] for i in sourceids]);tm=m.CM[targetids].numpy();sm=m.CM[sourceids].numpy()
  for seed in SEEDS:
   pc=m.mse_system(preds[f'pretrained__{held}__{seed}__correct'],tt,tm);pw=m.mse_system(preds[f'pretrained__{held}__{seed}__wrong'],tt,tm);rc=m.mse_system(preds[f'reset__{held}__{seed}__correct'],tt,tm);rw=m.mse_system(preds[f'reset__{held}__{seed}__wrong'],tt,tm);sb=m.mse_system(preds[f'source_before__{held}__{seed}'],st,sm);sa=m.mse_system(preds[f'source_after__{held}__{seed}'],st,sm)
   d={'held_group':held,'seed':seed,'pretrained_MSE':float(pc.mean()),'reset_MSE':float(rc.mean()),'pretrained_over_reset':float(pc.sum()/rc.sum()),'pretrained_wrong_MSE':float(pw.mean()),'pretrained_correct_over_wrong':float(pc.sum()/pw.sum()),'pretrained_correct_wins_wrong':int(np.sum(pc<pw)),'reset_correct_over_wrong':float(rc.sum()/rw.sum()),'source_after_over_before':float(sa.sum()/sb.sum())};d['passed_accumulation_gate']=bool(d['pretrained_over_reset']<=.8 and d['pretrained_correct_over_wrong']<=.8 and d['pretrained_correct_wins_wrong']>=6 and d['source_after_over_before']<=1.1);comps.append(d)
   for j,i in enumerate(targetids):scores.append({'held_group':held,'seed':seed,'learner_index':i,'pretrained_MSE':float(pc[j]),'reset_MSE':float(rc[j]),'pretrained_wrong_MSE':float(pw[j]),'reset_wrong_MSE':float(rw[j])})
 counts={'fits':12,'new_target_optimizer_updates':1200,'source_pretraining_updates_reused_not_repeated':3000,'training_context_rows_encoded':1200*4*32,'training_query_rows':1200*4*16,'evaluation_query_rows':6*4096,'score_target_system_blocks':len(scores)}
 for k in ('training_context_rows_encoded','training_query_rows','evaluation_query_rows'):
  if counts[k]!=plan[k]:raise RuntimeError(f'COUNT_{k}:{counts[k]}!={plan[k]}')
 result={'status':'HETEROGENEOUS_RAPID_ACQUISITION_V1_COMPLETE','classification':'SUPPORTED_OPENED_ACCUMULATION' if all(x['passed_accumulation_gate'] for x in comps) else 'NOT_SUPPORTED','comparisons':comps,'scores':scores,'counts':counts,'loss_endpoints':{k:v[-1] for k,v in losses.items()},'qualification':['Tests whether source experience accelerates 100-update supervised adaptation to four target-mechanism laws.','The reset control receives identical target observations, updates, architecture and normalization.','Source checkpoints contribute 3,000 prior updates that are reused, not repeated or compute matched away.','All target and retention systems are opened development evidence.','A pass is persistent-weight forward benefit with compact data, not immediate recognition or arbitrary-mechanism discovery.']};m.aj(OUT/'result.json',result);m.aj(OUT/'completion.json',{'status':'COMPLETE','classification':result['classification'],'result_sha256':sha(OUT/'result.json'),'counts':counts});print(json.dumps({'classification':result['classification'],'comparisons':comps,'counts':counts},indent=2))
if __name__=='__main__':main()
