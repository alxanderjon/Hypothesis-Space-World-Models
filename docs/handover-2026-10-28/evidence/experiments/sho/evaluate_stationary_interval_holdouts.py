"""Requested held-out interval-location/width and multi-root evaluation only."""
import json
import mpmath as mp
import numpy as np
import torch
from experiments.sho.train_subpiece_stationary_solver import OUT,StationarySolver,roots_high_precision,evaluate_generic,root_metrics,Audit
from experiments.sho.train_characteristic_spacing_smoke import write_csv,sha

def run():
 torch.set_num_threads(1);mp.mp.dps=60;model=StationarySolver();manifest=json.loads((OUT/'candidate_manifest.json').read_text());cp=manifest['checkpoints'][-1];assert sha(cp['checkpoint'])==cp['sha256'];model.load_state_dict(torch.load(cp['checkpoint'],weights_only=False)['state_dict']);audit=Audit()
 base=json.loads((OUT/'stage_4_heldout.json').read_text());training=json.loads((OUT/'stage_4_train.json').read_text());assert all(r['lower']>1e-12 and r['upper']<1-1e-12 and r['upper']-r['lower']<.6 for r in training)
 rs=[]
 for i,old in enumerate(base[:192]):
  r=dict(old)
  if i%3==0:r.update(lower=1e-14,upper=.8,coefficient_range='heldout_lower_domain_edge_and_wide_interval')
  elif i%3==1:r.update(lower=.2,upper=1-1e-14,coefficient_range='heldout_upper_domain_edge_and_wide_interval')
  else:r.update(lower=1e-14,upper=1-1e-14,coefficient_range='heldout_both_domain_edges_and_full_width')
  roots=[x for x in roots_high_precision(r) if mp.mpf(r['lower'])<x<mp.mpf(r['upper'])]
  r.update(exists=bool(roots),root_count=len(roots),roots=[float(x) for x in roots])
  if roots:
   def value(c):
    t=c if r['t']==0 else 1-c;v=c if r['v']==0 else 1-c
    return mp.mpf(r['A'])*mp.sqrt(mp.mpf(r['B'])+mp.mpf(r['a'])**2*t*t)+mp.mpf(r['beta'])*mp.log(v)
   c=min(roots,key=lambda c:(value(c),c));r.update(c_star=float(c),root_decimal=mp.nstr(c,55),target_z=float((c-r['lower'])/(r['upper']-r['lower'])))
  else:r.update(c_star=None,root_decimal=None,target_z=0.)
  rs.append(r)
 (OUT/'interval_location_holdouts.json').write_text(json.dumps(rs));recs,p,z=evaluate_generic(model,rs,audit);write_csv(OUT/'interval_location_holdouts.csv',recs)
 summary=[]
 for group in sorted(set(r['coefficient_range'] for r in rs)):
  ids=np.array([i for i,r in enumerate(rs) if r['coefficient_range']==group]);rr=[rs[i] for i in ids];summary.append(dict(group=group,**root_metrics(rr,p[ids],z[ids]),minimum_recovery=float(np.mean([recs[i]['minimum_recovered'] for i in ids])),multi_root_examples=sum(r['root_count']>1 for r in rr)))
 write_csv(OUT/'interval_location_holdout_summary.csv',summary);manifest['interval_location_holdouts']=summary;(OUT/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2))
 with (OUT/'stationary_solver_summary.txt').open('a') as f:
  f.write('\nHELD-OUT INTERVAL LOCATIONS/WIDTHS:\nTraining intervals all have lower>1e-12, upper<1-1e-12, width<0.6. Additional held-out tests use boundaries 1e-14 / 1-1e-14 and widths 0.8 or nearly 1. These jointly test entirely unseen endpoint locations and widths; they are not an isolated coefficient-controlled attribution. No retraining.\n')
  for r in summary:f.write(json.dumps(r)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
