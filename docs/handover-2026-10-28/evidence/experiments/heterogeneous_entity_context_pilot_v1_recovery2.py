#!/usr/bin/env python3
"""Second additive zero-update recovery: flatten active values for scalar RMS."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ORIGINAL=ROOT/'experiments/heterogeneous_entity_context_pilot_v1.py';FAILED0=ROOT/'outputs/heterogeneous_entity_context_pilot_v1';FAILED1=ROOT/'outputs/heterogeneous_entity_context_pilot_v1_recovery1';OUT=ROOT/'outputs/heterogeneous_entity_context_pilot_v1_recovery2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 for p in (FAILED0,FAILED1):
  if not (p/'plan.json').is_file() or any(p.glob('*.pt')) or (p/'result.json').exists():raise RuntimeError(f'namespace is not a zero-update stop: {p}')
 spec=importlib.util.spec_from_file_location('heterogeneous_pilot_original2',ORIGINAL);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 def corrected(indices,x,dx,qx,qy,emap):
  active=[];delta=[]
  for position,i in enumerate(indices):
   mask=emap[i][:,None].expand(2,4).numpy();active.extend((x[i,:,mask].ravel(),qx[i,:,mask].ravel()));delta.extend((dx[i,:,mask].ravel(),qy[position,:,mask].ravel()))
  av=np.concatenate(active);dv=np.concatenate(delta);logs=np.log(np.asarray([m.dt_all[i] for i in indices]));return float(np.sqrt(np.mean(av*av))+1e-8),float(np.sqrt(np.mean(dv*dv))+1e-8),float(logs.mean()),float(logs.std()+1e-8)
 m.norm_for=corrected;m.OUT=OUT;m.main();m.aj(OUT/'recovery_evidence.json',{'status':'RECOVERY2_ADDITIVE_EVIDENCE','prior_stops':[{'namespace':str(p.relative_to(ROOT)),'plan_sha256':sha(p/'plan.json'),'fits':0,'optimizer_updates':0} for p in (FAILED0,FAILED1)],'recovery_change':'flatten active values before scalar RMS; preserve fold-local target indexing','recovery_wrapper_sha256':sha(Path(__file__))})
if __name__=='__main__':main()
