#!/usr/bin/env python3
"""Score persisted predictions after the declared-mode count validation stop."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/'outputs/heterogeneous_three_group_package_v1';PARENT=ROOT/'outputs/heterogeneous_represented_context_diagnostic_v1';OUT=ROOT/'outputs/heterogeneous_represented_context_diagnostic_v1_recovery1';SEEDS=(2026102908,2026102909);GROUPS=('sho','ball','kepler')
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
def mse(pred,target,mask):
 out=[]
 for p,t,m in zip(pred,target,mask):mm=np.broadcast_to(m,(p.shape[0],)+m.shape);out.append(float(np.mean((p[mm]-t[mm])**2)))
 return np.asarray(out)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 if (PARENT/'result.json').exists() or not (PARENT/'predictions_before_rescoring_opened_development.npz').is_file():raise RuntimeError('parent state mismatch')
 OUT.mkdir(parents=True);manifest=json.load(open(PKG/'quarantined_split_manifest.json'));byidx={int(r['learner_index']):r for r in manifest['records']};dev=[i for i,r in byidx.items() if r['role']=='opened_development']
 with np.load(PARENT/'predictions_before_rescoring_opened_development.npz',allow_pickle=False) as z:preds={k:np.asarray(z[k]) for k in z.files}
 with np.load(PKG/'protected_development_query_targets.npz',allow_pickle=False) as z:targets={int(i):np.asarray(y) for i,y in zip(z['learner_index'],z['query_dx'])}
 with np.load(PKG/'learner_inputs.npz',allow_pickle=False) as z:cm=np.asarray(z['channel_mask'])
 comparisons=[];scores=[]
 for seed in SEEDS:
  for group in GROUPS:
   pos=[j for j,i in enumerate(dev) if byidx[i]['group']==group];ids=[dev[j] for j in pos];target=np.asarray([targets[i] for i in ids]);mask=cm[ids];vals={mode:mse(preds[f'{seed}__{mode}'][pos],target,mask) for mode in ('correct','blind','wrong','time_blind')};d={'seed':seed,'group':group,'correct_MSE':float(vals['correct'].mean()),'blind_MSE':float(vals['blind'].mean()),'wrong_MSE':float(vals['wrong'].mean()),'time_blind_MSE':float(vals['time_blind'].mean()),'correct_over_blind':float(vals['correct'].sum()/vals['blind'].sum()),'correct_over_wrong':float(vals['correct'].sum()/vals['wrong'].sum()),'correct_wins_wrong':int(np.sum(vals['correct']<vals['wrong']))};d['passed_context_gate']=bool(d['correct_over_blind']<=.8 and d['correct_over_wrong']<=.8 and d['correct_wins_wrong']>=6);comparisons.append(d)
   for k,i in enumerate(ids):scores.append({'seed':seed,'group':group,'learner_index':i,**{m+'_MSE':float(v[k]) for m,v in vals.items()}})
 counts={'original_fits':2,'original_optimizer_updates':1000,'repeated_fits':0,'repeated_optimizer_updates':0,'persisted_evaluation_query_rows':12288,'recovery_score_system_blocks':len(scores)}
 result={'status':'HETEROGENEOUS_REPRESENTED_CONTEXT_DIAGNOSTIC_V1_RECOVERY1_COMPLETE','classification':'SUPPORTED_OPENED_CONTEXT_PREREQUISITE' if all(x['passed_context_gate'] for x in comparisons) else 'NOT_SUPPORTED','comparisons':comparisons,'scores':scores,'counts':counts,'recovery':{'parent_prediction_sha256':sha(PARENT/'predictions_before_rescoring_opened_development.npz'),'parent_plan_sha256':sha(PARENT/'plan.json'),'original_expected_evaluation_rows':6144,'actual_shape_derived_evaluation_rows':12288,'scientific_calculations_repeated':False},'qualification':['All mechanisms are represented during training.','Labels are historically opened development evidence.','Recovery scores persisted predictions only; it repeats no fit or inference.']};aj(OUT/'result.json',result);aj(OUT/'completion.json',{'status':'COMPLETE','classification':result['classification'],'result_sha256':sha(OUT/'result.json'),'counts':counts});print(json.dumps({'classification':result['classification'],'comparisons':comparisons,'counts':counts},indent=2))
if __name__=='__main__':main()
