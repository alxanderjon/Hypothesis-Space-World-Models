#!/usr/bin/env python3
import json,sys,hashlib,os
from pathlib import Path
import numpy as np
def states(seed,n): return np.random.default_rng(seed).uniform(-1,1,(n,4))
def metrics(support,query,selected):
 d=np.sqrt(((query[:,None]-support[None])**2).sum(2)); nn=d.min(1)
 pd=np.sqrt(((selected[:,None]-selected[None])**2).sum(2)); pd=pd[np.triu_indices(len(selected),1)]
 return {'query_nn_mean':float(nn.mean()),'query_nn_median':float(np.median(nn)),'query_nn_p95':float(np.quantile(nn,.95)),'query_nn_max':float(nn.max()),'selected_norm_mean':float(np.linalg.norm(selected,axis=1).mean()),'selected_pair_min':float(pd.min()),'selected_pair_median':float(np.median(pd))}
rows=[]
for name,world in [('development',2026105000),('confirmation',2026105100)]:
 root=Path(sys.argv[1])/('common_panel_active_acquisition_v1' if name=='development' else 'common_panel_active_acquisition_v1_confirmation')
 b=json.load(open(root/'selection_barrier.json')); x0=states(world+1,32); pool=states(world+3,256); q=states(world+102,64)
 for arm,key in [('random','random_indices'),('adaptive','active_indices')]:
  sel=pool[np.array(b[key])]; rows.append({'phase':name,'arm':arm,**metrics(np.concatenate([x0,sel]),q,sel)})
out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=False); result={'classification':'DESCRIPTIVE_GEOMETRY_ONLY','counts':{'selection_barriers_read':2,'distance_matrices':8},'rows':rows,'qualification':'Geometry uses no response or query labels and cannot by itself causally explain prediction differences.'}; (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)); h=hashlib.sha256((out/'result.json').read_bytes()).hexdigest(); (out/'completion.json').write_text(json.dumps({'status':'COMPLETE','result_sha256':h,'counts':result['counts']},indent=2,sort_keys=True))
