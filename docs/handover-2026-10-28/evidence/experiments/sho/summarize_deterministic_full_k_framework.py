import json, csv,hashlib
from pathlib import Path
import numpy as np
from experiments.sho import deterministic_full_k_framework as fw
p=fw.OUT;r=json.loads((p/'report.json').read_text());checks=[]
for seed,count,label in ((fw.prior.DATA_SEED,4,'original'),(fw.prior.DATA_SEED+1,8,'fresh')):
 positions,_,_=fw.prior.active_n1.im._make_exact_labeled_split(count,seed)
 for i in range(count):
  pos=positions[i].numpy();s=fw.search(lambda x:float(fw.prior.objective_numpy(x[None,:],pos)[0]),fw.prior.LOWER,fw.prior.UPPER)
  saved=np.load(p/f'{label}_{i}_samples.npz');match=np.array_equal(s['points'],saved['normalized_points']) and np.array_equal(s['values'],saved['values'])
  assert match;checks.append(dict(case=f'{label}_{i}',evaluations=s['evaluations'],exact_replay=match))
r['deterministic_replays']=checks
r['dimension_interface_controls']=json.loads((p/'dimension_interface_controls.json').read_text())
r['limitations']=['Validated on 12 SHO prefixes, not a proof for arbitrary objectives.', 'Retains raw 1e-4 gate; prior cross-system handoff issue is not resolved.', 'Boundary minimizers are not tested; canonical DIRECT approaches domain faces by center subdivision.', 'Downstream is deterministic Powell plus safeguarded Newton, not a claim of learned N2 4D success.', '11/12 Powell endpoints meet 1e-20 within 1500 calls; all 12 are polished by unchanged Newton policy.']
rows=r['cases'];r['summary']=dict(n1_median_evaluations=float(np.median([x['n1_evaluations'] for x in rows])),n1_max_evaluations=max(x['n1_evaluations'] for x in rows),n1_total_evaluations=sum(x['n1_evaluations'] for x in rows),n2_total_evaluations=sum(x['n2_evaluations'] for x in rows),n3_total_evaluations=sum(x['n3_evaluations'] for x in rows),matrix_max_error=max(x['matrix_max_error'] for x in rows),matrix_norm_max=max(x['matrix_norm'] for x in rows),rollout_max_error=max(x['rollout_max_error'] for x in rows),max_partition_volume_error=max(x['volume_error'] for x in rows),teacher_box_containment=sum(x['teacher_inside_n2_box'] for x in rows))
(p/'report.json').write_text(json.dumps(r,indent=2))
print(r['summary'])
