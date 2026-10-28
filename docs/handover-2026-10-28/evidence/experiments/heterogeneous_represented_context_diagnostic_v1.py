#!/usr/bin/env python3
"""Two-fit prerequisite diagnostic for context use with represented mechanisms."""
from __future__ import annotations
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/heterogeneous_entity_context_pilot_v1.py';PKG=ROOT/'outputs/heterogeneous_three_group_package_v1';OUT=ROOT/'outputs/heterogeneous_represented_context_diagnostic_v1';SEEDS=(2026102908,2026102909)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 spec=importlib.util.spec_from_file_location('hetero_base_diag',BASE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.OUT=OUT;OUT.mkdir(parents=True)
 def corrected(indices,x,dx,qx,qy,emap):
  active=[];delta=[]
  for position,i in enumerate(indices):
   mask=emap[i][:,None].expand(2,4).numpy();active.extend((x[i,:,mask].ravel(),qx[i,:,mask].ravel()));delta.extend((dx[i,:,mask].ravel(),qy[position,:,mask].ravel()))
  av=np.concatenate(active);dv=np.concatenate(delta);logs=np.log(np.asarray([m.dt_all[i] for i in indices]));return float(np.sqrt(np.mean(av*av))+1e-8),float(np.sqrt(np.mean(dv*dv))+1e-8),float(logs.mean()),float(logs.std()+1e-8)
 m.norm_for=corrected
 manifest=json.load(open(PKG/'quarantined_split_manifest.json'));records=manifest['records'];byidx={int(r['learner_index']):r for r in records}
 with np.load(PKG/'learner_inputs.npz',allow_pickle=False) as z:m.X=torch.as_tensor(z['context_x']);m.DX=torch.as_tensor(z['context_dx']);m.QX=torch.as_tensor(z['query_x']);m.EM=torch.as_tensor(z['entity_mask']);m.CM=torch.as_tensor(z['channel_mask']);m.DT=torch.as_tensor(z['delta_t'])
 m.Xn=m.X.numpy();m.DXn=m.DX.numpy();m.QXn=m.QX.numpy();m.dt_all=m.DT.numpy()
 with np.load(PKG/'training_query_targets.npz',allow_pickle=False) as z:train_target={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 train=[i for i,r in byidx.items() if r['role']=='training'];dev=[i for i,r in byidx.items() if r['role']=='opened_development']
 plan={'status':'PRECHARGED','fits':2,'updates_per_fit':500,'total_updates':1000,'training_context_rows_encoded':256000,'training_query_rows':128000,'evaluation_query_rows':6144,'qualification':'Reuses already opened development systems only; fixed architecture and settings from the prior negative pilot.','source_hashes':{'driver':sha(Path(__file__)),'base':sha(BASE),'package':sha(PKG/'completion.json')}};m.aj(OUT/'plan.json',plan)
 preds={};meta=[];codes_by={};norms={}
 for seed in SEEDS:
  model,norm,loss=m.fit(train,train_target,seed,f'joint_seed{seed}');modes,codes=m.predict_modes(model,norm,train,dev);codes_by[seed]=codes;norms[seed]=norm
  for mode,val in zip(('correct','blind','wrong','time_blind'),modes):preds[f'{seed}__{mode}']=val;meta.append({'seed':seed,'mode':mode,'dev_indices':dev})
 m.an(OUT/'predictions_before_rescoring_opened_development.npz',**preds);m.aj(OUT/'prediction_receipt.json',{'status':'PREDICTIONS_PERSISTED_BEFORE_RESCORING_HISTORICALLY_OPENED_LABELS','sha256':sha(OUT/'predictions_before_rescoring_opened_development.npz'),'metadata':meta})
 with np.load(PKG/'protected_development_query_targets.npz',allow_pickle=False) as z:targets={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 comparisons=[];scores=[]
 for seed in SEEDS:
  for group in m.GROUPS:
   pos=[j for j,i in enumerate(dev) if byidx[i]['group']==group];ids=[dev[j] for j in pos];target=np.asarray([targets[i] for i in ids]);mask=m.CM[ids].numpy();vals={mode:m.mse_system(preds[f'{seed}__{mode}'][pos],target,mask) for mode in ('correct','blind','wrong','time_blind')};d={'seed':seed,'group':group,'correct_MSE':float(vals['correct'].mean()),'blind_MSE':float(vals['blind'].mean()),'wrong_MSE':float(vals['wrong'].mean()),'time_blind_MSE':float(vals['time_blind'].mean()),'correct_over_blind':float(vals['correct'].sum()/vals['blind'].sum()),'correct_over_wrong':float(vals['correct'].sum()/vals['wrong'].sum()),'correct_wins_wrong':int(np.sum(vals['correct']<vals['wrong']))};d['passed_context_gate']=bool(d['correct_over_blind']<=.8 and d['correct_over_wrong']<=.8 and d['correct_wins_wrong']>=6);comparisons.append(d)
   for k,i in enumerate(ids):scores.append({'seed':seed,'group':group,'learner_index':i,**{m0+'_MSE':float(v[k]) for m0,v in vals.items()}})
 counts={'fits':2,'optimizer_updates':1000,'training_context_rows_encoded':1000*8*32,'training_query_rows':1000*8*16,'evaluation_query_rows':len(preds)*24*64,'score_system_blocks':len(scores)}
 for k in ('training_context_rows_encoded','training_query_rows','evaluation_query_rows'):
  if counts[k]!=plan[k]:raise RuntimeError(f'COUNT_{k}')
 result={'status':'HETEROGENEOUS_REPRESENTED_CONTEXT_DIAGNOSTIC_V1_COMPLETE','classification':'SUPPORTED_OPENED_CONTEXT_PREREQUISITE' if all(x['passed_context_gate'] for x in comparisons) else 'NOT_SUPPORTED','comparisons':comparisons,'scores':scores,'counts':counts,'qualification':['All mechanism types are represented in training; this is not whole-mechanism transfer.','Development labels were historically opened by the preceding pilot; no confirmation claim.','A pass would establish context use only for represented transition types.']};m.aj(OUT/'result.json',result);m.aj(OUT/'completion.json',{'status':'COMPLETE','classification':result['classification'],'result_sha256':sha(OUT/'result.json'),'counts':counts});print(json.dumps({'classification':result['classification'],'comparisons':comparisons,'counts':counts},indent=2))
if __name__=='__main__':main()
