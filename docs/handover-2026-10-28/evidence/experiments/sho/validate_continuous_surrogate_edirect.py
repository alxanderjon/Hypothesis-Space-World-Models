import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.sho import continuous_surrogate_edirect as ext
from experiments.sho.deterministic_full_k_framework import deterministic_finish,hashes,prior
from experiments.sho.validate_dimension_general_edirect import structural
OUT=Path('outputs/sho/stage1B/continuous_surrogate_edirect')
DATA=Path('outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz')

def controls():
    # Existing d=2 pool/cell/local-stencil oracle compatibility remains exact;
    # the continuous proposal is intentionally different from the old stencil.
    check=structural();rng=np.random.default_rng(20261104);rows=[]
    for d in (2,4):
        points=rng.random((30,d));losses=np.sum((points-.37)**2,axis=1)
        a=[];b=[];u=ext.proposal(points,losses,0,0,a);v=ext.proposal(points,losses,0,0,b)
        assert a==b and np.array_equal(u,v)
        assert u is not None and np.isfinite(u).all() and np.all((u>=0)&(u<=1))
        assert not np.any(np.all(points==u,axis=1))
        rows.append(dict(d=d,exact_replay=True,off_evaluated_grid=True,proposal=u.tolist(),
                         nfev=sum(s['nfev'] for s in a[0]['starts'])))
    return dict(inherited_structural=check,continuous=rows)


def main():
 OUT.mkdir(parents=True,exist_ok=True);started=time.time();before=hashes()
 (OUT/'checkpoint_hashes_before.json').write_text(json.dumps(before,indent=2))
 cs=controls();(OUT/'controls.json').write_text(json.dumps(cs,indent=2));print('controls passed',flush=True)
 data=np.load(DATA);rows=[];stop=None
 # Case 7 first; then nearby companion/interior and one face control before
 # resuming every remaining prepared case. No teacher used in search decisions.
 priority=[6,5,7,32,96];order=priority+[i for i in range(len(data['names'])) if i not in priority]
 for index in order:
    name=str(data['names'][index]);category=str(data['categories'][index]);states=data[f'prefix_{index}'];k=data['teachers'][index]
    x=states[:-1];y=states[1:];xt=torch.from_numpy(x.copy());yt=torch.from_numpy(y.copy())
    def objective(theta):return float(np.mean((np.einsum('ij,nj->ni',np.asarray(theta).reshape(2,2),x)-y)**2))
    def tor(theta):return torch.mean((xt@theta.reshape(2,2).T-yt)**2)
    t0=time.time();r=ext.search(objective,prior.LOWER,prior.UPPER,index)
    np.savez_compressed(OUT/f'{name}_samples.npz',points=r['points'],losses=r['values'])
    (OUT/f'{name}_events.json').write_text(json.dumps(r['events']))
    (OUT/f'{name}_continuous.json').write_text(json.dumps(r['continuous_traces']))
    truth=k.ravel();contained=bool(np.all(truth>=r['lower']) and np.all(truth<=r['upper']))
    row=dict(case=name,category=category,n1_success=r['success'],n1_evaluations=r['evaluations'],n1_loss=r['loss'],
      n1_theta=r['theta'].tolist(),rounds=r['iterations'],teacher_contained=contained,
      continuous_calls=len(r['continuous_traces']),continuous_evaluations=sum(e['source']=='greedy' for e in r['events']),
      duplicate_rejections=sum(t['exact_duplicate_rejected'] for t in r['continuous_traces']),
      surrogate_evaluations=sum(s['nfev'] for t in r['continuous_traces'] for s in t['starts']),
      optimizer_nonconverged_starts=sum(not s['success'] for t in r['continuous_traces'] for s in t['starts']),
      duplicate_physical_evaluations=len(r['points'])-len(np.unique(r['points'],axis=0)))
    if r['success']:
      row.update(deterministic_finish(objective,tor,r));final=np.array(row['final_theta']).reshape(2,2);delta=final-k
      true=states[-1].copy();pred=true.copy();errors=[]
      for _ in range(4096):true=k@true;pred=final@pred;errors.append(pred-true)
      errors=np.asarray(errors);row.update(k_errors=delta.tolist(),matrix_error=float(np.linalg.norm(delta)),
        rollout_max_error=float(np.max(abs(errors))),rollout_mse=float(np.mean(errors**2)))
    row['seconds']=time.time()-t0;rows.append(row);(OUT/'results_partial.json').write_text(json.dumps(rows,indent=2))
    print(name,'N1',r['evaluations'],r['loss'],'continuous',row['continuous_calls'],'contained',contained,'final',row.get('final_loss'),flush=True)
    if not r['success'] or not contained or row.get('final_loss',float('inf'))>=1e-20 or row['duplicate_physical_evaluations']:
      stop='N1_GATE' if not r['success'] else 'HANDOFF_CONTAINMENT' if not contained else 'DOWNSTREAM_OR_DUPLICATES';break
    if index==6:
      replay=ext.search(objective,prior.LOWER,prior.UPPER,index)
      assert np.array_equal(replay['points'],r['points']) and np.array_equal(replay['values'],r['values']) and replay['continuous_traces']==r['continuous_traces']
      row['full_case_replay_exact']=True
 after=hashes();assert before==after
 (OUT/'checkpoint_hashes_after.json').write_text(json.dumps(after,indent=2))
 report=dict(status='STOPPED_'+stop if stop else 'VALIDATED_DETERMINISTIC_TEACHER',cases=rows,planned_cases=len(data['names']),
   tested_cases=len(rows),checkpoints_unchanged=True,checkpoint_count=len(before),seconds=time.time()-started,
   optimizer='multistart full-box bounded scipy Powell; incumbent + 2*d fixed Sobol starts; no global optimality certificate',
   controls=cs,first_failure=stop)
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 if not stop:(OUT/'dimension_general_edirect_teacher.json').write_text(json.dumps(dict(status=report['status'],source=ext.__file__,sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest()),indent=2))
 print(report['status'],flush=True)
if __name__=='__main__':main()
