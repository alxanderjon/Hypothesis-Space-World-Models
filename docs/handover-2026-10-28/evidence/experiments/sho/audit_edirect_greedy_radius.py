"""Audit only: original radius has no dimensional exponent to generalize."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.sho import dimension_general_edirect as ed
from experiments.sho.deterministic_full_k_framework import hashes,prior
OUT=Path('outputs/sho/stage1B/edirect_greedy_radius_audit')
OLD=Path('outputs/sho/stage1B/dimension_general_edirect')

def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.time();before=hashes()
 sources=[Path(ed.__file__),Path(ed.teacher.__file__),Path('experiments/sho/deterministic_full_k_framework.py')]
 source_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
 rng=np.random.default_rng(20261103);controls=[]
 for n in (1,2,10,30,100):
  for layout in ('interior','boundary','clustered'):
   points=rng.random((n,2))
   if layout=='boundary':points[:,0]*=1e-6
   if layout=='clustered':points=.5+1e-7*points
   losses=rng.random(n)
   a=ed.teacher._local_greedy_candidates(points,losses);b=ed.local_greedy(points,losses)
   assert np.array_equal(a,b)
   controls.append(dict(n=n,layout=layout,exact=True,coordinates=int(a.size)))
 # Padding with zero coordinates preserves all pair distances and hence radii:
 # direct evidence the original length already has dimension-generic units.
 padded=[]
 for d in (2,3,4,8):
  p=np.column_stack((np.array([[.5,.5],[.500005,.5],[.7,.8]]),np.zeros((3,d-2))))
  spacing=cKDTree(p).query(p,k=2)[0][:,1]
  padded.append(dict(d=d,spacing=spacing.tolist(),base_radius=np.maximum(spacing,1e-5).tolist()))
 assert all(x['spacing']==padded[0]['spacing'] for x in padded)
 samples=np.load(OLD/'companion_6_samples.npz');points=samples['points'];losses=samples['losses']
 trace=json.loads((OLD/'companion_6_trace.json').read_text());radius_trace=[]
 for i,e in enumerate(trace):
  if e['source']!='greedy':continue
  ids=np.argsort(losses[:i])[:12];sp=cKDTree(points[:i]).query(points[ids],k=2)[0][:,1]
  r=np.maximum(sp,1e-5)
  radius_trace.append(dict(evaluation=i+1,site_ids=ids.tolist(),nearest=sp.tolist(),base_radius=r.tolist(),
   incumbent_base_radius=float(r[0]),floor_count=int(np.sum(sp<=1e-5))))
 (OUT/'radius_trace.json').write_text(json.dumps(radius_trace,indent=2))
 last=radius_trace[-1];n=last['evaluation']-1;s=last['nearest'][0]
 contrast=dict(n=n,nearest=s,original=max(s,1e-5),enhanced_2d=max(s,.5/n**.5),
               enhanced_dimensional_4d=max(s,.5/n**.25),adopted=False,
               reason='Enhanced policy changes 2D behavior; not a dimensional correction to original.')
 (OUT/'controls.json').write_text(json.dumps(dict(two_d=controls,padded=padded,enhanced_contrast=contrast),indent=2))
 print('2D controls exact; no dimension-dependent radius term; replaying unchanged case 7',flush=True)
 dataset=np.load(OLD/'validation_cases.npz');states=dataset['prefix_6'];x=states[:-1];y=states[1:]
 def objective(theta):return float(np.mean((np.einsum('ij,nj->ni',np.asarray(theta).reshape(2,2),x)-y)**2))
 r=ed.search(objective,prior.LOWER,prior.UPPER,6)
 assert np.array_equal(r['points'],points) and np.array_equal(r['values'],losses)
 assert r['events']==trace
 after=hashes();assert before==after
 source_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};assert source_before==source_after
 report=dict(status='STOPPED_NO_SEMANTICS_PRESERVING_DIMENSIONAL_RADIUS_CHANGE',
   historical_rule='fractions*(max(nearest_evaluated_site_distance,1e-5))',fractions=[.5,.25,.125,.0625],
   dimension_dependent_term=None,replacement=None,two_d_controls=len(controls),
   two_d_exact_coordinates=sum(c['coordinates'] for c in controls),padding_dimensions=[2,3,4,8],
   enhanced_contrast=contrast,greedy_events=len(radius_trace),
   events_incumbent_at_floor=sum(e['incumbent_base_radius']==1e-5 for e in radius_trace),
   first_incumbent_floor_evaluation=next(e['evaluation'] for e in radius_trace if e['incumbent_base_radius']==1e-5),
   case7=dict(evaluations=r['evaluations'],loss=r['loss'],success=r['success'],
      full_points_values_events_exact=True,rounds=r['iterations'],duplicates=len(points)-len(np.unique(points,axis=0)),
      finite=bool(np.isfinite(points).all()),inside_box=bool(np.all((points>=0)&(points<=1)))),
   resumed_cases=0,checkpoints_unchanged=True,checkpoint_count=len(before),source_hashes=source_before,
   seconds=time.time()-start)
 (OUT/'checkpoint_hashes_before.json').write_text(json.dumps(before,indent=2));(OUT/'checkpoint_hashes_after.json').write_text(json.dumps(after,indent=2))
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
