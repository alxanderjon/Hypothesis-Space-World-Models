import sys,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.sho import run_full_k_transfer as old
out=Path('outputs/sho/stage1B/deterministic_full_k_framework');out.mkdir(exist_ok=True)
p,_,c=old.active_n1.im._make_exact_labeled_split(4,old.DATA_SEED)
f=old.FrozenProposals4D(); original=old.objective_numpy
for case in range(4):
 samples=[];losses=[];rounds=[]; teacher=np.array([0,1,-1,float(c[case])]);u=(teacher-old.LOWER)/old.WIDTH
 def objective(t,pos):
  y=original(t,pos);samples.extend(np.atleast_2d(t).tolist());losses.extend(y.tolist());return y
 def observe(frame,event,arg):
  if event=='return' and frame.f_code is old.FrozenProposals4D.proposals.__code__ and arg is not None:
   v=frame.f_locals;s=v['sites'];ix=int(cKDTree(s).query(u)[1]);po=v['po']
   rounds.append(dict(iteration=int(v['iteration']),evaluations=len(s),nearest_teacher_site=ix,nearest_distance=float(np.linalg.norm(s[ix]-u)),teacher_site_selected=ix in po,teacher_site_loss=float(v['losses'][ix]),best_loss=float(min(v['losses'])),po_cells=len(po),proposals=len(arg[0]),boundary_proposals=int(np.sum(np.any((arg[0]==0)|(arg[0]==1),axis=1)))))
  return observe
 old.objective_numpy=objective;sys.setprofile(observe);start=time.time()
 try:r=old.run_n1(dict(trajectory_id=case,generator_id=old.DATA_SEED+case),p[case].numpy(),f)
 finally:sys.setprofile(None);old.objective_numpy=original
 np.savez_compressed(out/f'old_case_{case}_samples.npz',points=np.array(samples),losses=np.array(losses))
 (out/f'old_case_{case}_rounds.json').write_text(json.dumps(rounds,indent=2))
 summary=dict(case=case,evaluations=r.evaluations,loss=r.loss,decisions=r.decisions,rounds=r.rounds,seconds=time.time()-start,teacher_nearest_cell_selected_rounds=sum(x['teacher_site_selected'] for x in rounds),boundary_proposals=sum(x['boundary_proposals'] for x in rounds),total_proposals=sum(x['proposals'] for x in rounds),final_teacher_nearest_distance=float(np.min(np.linalg.norm((np.array(samples)-teacher)/old.WIDTH,axis=1))))
 (out/f'old_case_{case}_summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
