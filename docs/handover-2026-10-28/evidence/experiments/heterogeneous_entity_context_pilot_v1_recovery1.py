#!/usr/bin/env python3
"""Additive recovery of the zero-update fold-local normalization index stop."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'experiments/heterogeneous_entity_context_pilot_v1.py'
FAILED=ROOT/'outputs/heterogeneous_entity_context_pilot_v1'
RECOVERY=ROOT/'outputs/heterogeneous_entity_context_pilot_v1_recovery1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if not (FAILED/'plan.json').is_file() or any(FAILED.glob('*.pt')) or (FAILED/'result.json').exists():raise RuntimeError('failed namespace is not the recorded zero-update stop')
 spec=importlib.util.spec_from_file_location('heterogeneous_pilot_original',ORIGINAL);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 def corrected(indices,x,dx,qx,qy,emap):
  active=[];delta=[]
  for position,i in enumerate(indices):
   mask=emap[i][:,None].expand(2,4).numpy();active.append(x[i,:,mask]);active.append(qx[i,:,mask]);delta.append(dx[i,:,mask]);delta.append(qy[position,:,mask])
  av=np.concatenate(active);dv=np.concatenate(delta);xs=float(np.sqrt(np.mean(av*av))+1e-8);ys=float(np.sqrt(np.mean(dv*dv))+1e-8);logs=np.log(np.asarray([m.dt_all[i] for i in indices]));return xs,ys,float(logs.mean()),float(logs.std()+1e-8)
 m.norm_for=corrected;m.OUT=RECOVERY;m.main()
 evidence={'status':'RECOVERY1_ADDITIVE_EVIDENCE','original_namespace':str(FAILED.relative_to(ROOT)),'original_plan_sha256':sha(FAILED/'plan.json'),'original_optimizer_updates':0,'original_fits':0,'recovery_change':'fold-local target position replaces global learner index in normalization only','recovery_wrapper_sha256':sha(Path(__file__))}
 m.aj(RECOVERY/'recovery_evidence.json',evidence)
if __name__=='__main__':main()
