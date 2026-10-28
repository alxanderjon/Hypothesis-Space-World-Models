import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.sho import dimension_general_edirect as ed
p=Path('outputs/sho/stage1B/dimension_general_edirect')
r=json.loads((p/'report.json').read_text());case=r['cases'][-1];name=case['case'];index=6
v=np.load(p/'validation_cases.npz');states=v[f'prefix_{index}'];teacher=v['teachers'][index].ravel()
x=states[:-1];y=states[1:]
t=json.loads((p/f'{name}_trace.json').read_text());s=np.load(p/f'{name}_samples.npz')
P=s['points'];F=s['losses'];last=max(i for i,e in enumerate(t) if e['source']=='greedy')
points=P[:last];losses=F[:last];iteration=t[last]['iteration']
local=ed.local_greedy(points,losses);pool=ed.pool(4,len(points),index,iteration+1000000)
candidates=np.unique(np.concatenate((pool,local)),axis=0)
surface=ed.teacher.GaussianLocalResponse(points,losses);pred=surface.predict(candidates)
novel=cKDTree(points).query(candidates)[0]>1e-12
order=np.argsort(pred);chosen=next(i for i in order if novel[i])
assert np.array_equal(candidates[chosen],P[last])
def values(u):
 ans=[]
 for start in range(0,len(u),1024):
  k=(-1.99+3.98*u[start:start+1024]).reshape(-1,2,2)
  z=np.einsum('bij,nj->bni',k,x)
  ans.extend(np.mean((z-y[None])**2,axis=(1,2)))
 return np.asarray(ans)
true=values(candidates);eligible=np.flatnonzero(novel);oracle=eligible[np.argmin(true[eligible])]
api=ed.composition(4);d=api['_cell_data'](points,losses,index,iteration)
u=(teacher+1.99)/3.98;near=int(cKDTree(points).query(u)[1])
best=int(np.argmin(losses));spacing=cKDTree(points).query(points[best],k=2)[0][1]
result=dict(case=name,diagnostic_only=True,last_greedy_evaluation=last+1,iteration=iteration,
    exact_saved_proposal_replay=True,candidates=len(candidates),novel=int(novel.sum()),
    selected_true_loss=float(true[chosen]),selected_predicted_log_loss=float(pred[chosen]),
    best_available_true_loss=float(true[oracle]),best_available_predicted_log_loss=float(pred[oracle]),
    best_available_surrogate_rank=int(np.flatnonzero(order==oracle)[0]+1),
    novel_candidates_below_handoff=int(np.sum(novel&(true<=1e-4))),
    local_best_true_loss=float(min(values(local))),global_pool_best_true_loss=float(min(values(pool))),
    teacher_nearest_site=near,teacher_nearest_distance=float(np.linalg.norm(points[near]-u)),
    teacher_nearest_cell_selected=bool(near in d.selected),selected_cell_count=len(d.selected),
    incumbent_nearest_spacing=float(spacing),incumbent_teacher_distance=float(np.linalg.norm(points[best]-u)),
    best_loss_before_greedy=float(min(losses)),last_greedy_actual_gain=float(min(losses)-true[chosen]),
    best_loss_by_source={source:float(min(e['loss'] for e in t if e['source']==source)) for source in ('initial','cell','greedy')})
(p/'first_failure_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
