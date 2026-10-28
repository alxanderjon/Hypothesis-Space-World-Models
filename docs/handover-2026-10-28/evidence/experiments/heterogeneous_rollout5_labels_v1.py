#!/usr/bin/env python3
"""Acquire protected five-step continuations for the 24 development systems."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
from experiments.bouncing_ball import run_first_transfer as ball
from experiments.kepler import deterministic_teacher as kepler
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'outputs/heterogeneous_three_group_package_v1'
OUT=ROOT/'outputs/heterogeneous_rollout5_labels_v1'
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
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 manifest=json.load(open(PACKAGE/'quarantined_split_manifest.json'))
 with np.load(PACKAGE/'learner_inputs.npz',allow_pickle=False) as z:
  cx=np.asarray(z['context_x']);cd=np.asarray(z['context_dx'])
 dev=[r for r in manifest['records'] if r['role']=='opened_development']
 aj(OUT/'schedule_before_responses.json',{'status':'PERSISTED_BEFORE_RESPONSES','systems':[{'learner_index':r['learner_index'],'group':r['group'],'law_ordinal':r['law_ordinal'],'teacher_parameters':r['teacher_parameters']} for r in dev],'steps_per_system':5,'total_steps':120,'source_hashes':{'package_completion':sha(PACKAGE/'completion.json'),'ball':sha(ROOT/'experiments/bouncing_ball/run_first_transfer.py'),'kepler':sha(ROOT/'experiments/kepler/deterministic_teacher.py')}})
 idx=[];initial=[];truth=[];ball_steps=kepler_steps=sho_steps=ball_scans=ball_events=0
 for r in dev:
  i=int(r['learner_index']);g=r['group'];theta=np.asarray(r['teacher_parameters'],np.float64)
  state=(cx[i,-1]+cd[i,-1])
  if g=='sho':raw=state[0,:2].copy();mat=theta.reshape(2,2)
  elif g=='ball':raw=state[0].copy()
  else:raw=np.concatenate((state[0],state[1]))
  rows=[]
  for _ in range(5):
   if g=='sho':raw=mat@raw;sho_steps+=1;neutral=np.zeros((2,4));neutral[0,:2]=raw
   elif g=='ball':
    sig=ball.numpy_k_event_signature(raw,theta);ball_scans+=1;ball_events+=len(sig)
    raw=ball.numpy_k_step(raw,theta);ball_steps+=1;neutral=np.zeros((2,4));neutral[0]=raw
   else:
    raw=kepler.numpy_step(raw,theta);kepler_steps+=1;neutral=np.stack((raw[:4],raw[4:]))
   rows.append(neutral)
  idx.append(i);initial.append(state);truth.append(rows)
 if sho_steps+ball_steps+kepler_steps!=120:raise RuntimeError('count mismatch')
 an(OUT/'protected_rollout5_targets.npz',learner_index=np.asarray(idx,np.int64),initial_x=np.asarray(initial),truth_x=np.asarray(truth))
 aj(OUT/'barrier.json',{'status':'ROLLOUT_TARGETS_QUARANTINED','target_sha256':sha(OUT/'protected_rollout5_targets.npz'),'prediction_before_label_required':True})
 result={'status':'HETEROGENEOUS_ROLLOUT5_LABELS_V1_COMPLETE','classification':'PROTECTED_LABELS_ACQUIRED_NO_LEARNER_RESULT','counts':{'systems':24,'steps_per_system':5,'total_transition_steps':120,'sho_steps':sho_steps,'ball_steps':ball_steps,'ball_event_signature_scans':ball_scans,'ball_collision_events':ball_events,'kepler_steps':kepler_steps,'kepler_acceleration_evaluations':2*kepler_steps},'qualification':['Rollout labels are opened-development labels and remain protected until predictions persist.','Initial rollout state is the observed endpoint of each passive32 context.']}
 aj(OUT/'result.json',result);aj(OUT/'completion.json',{'status':'COMPLETE','result_sha256':sha(OUT/'result.json'),'barrier_sha256':sha(OUT/'barrier.json'),'counts':result['counts']});print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
