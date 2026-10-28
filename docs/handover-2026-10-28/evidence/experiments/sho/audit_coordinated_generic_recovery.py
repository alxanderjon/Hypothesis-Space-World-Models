"""Isolated saved-endpoint diagnostic batch; never used by production recovery."""
from pathlib import Path
import json,os,time
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments.resume_hybrid_student_after_ckdtree import Log
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import OptInResidualN3Runtime,STABILIZED_RESIDUAL,residual_and_jacobian,residual_vector
from experiments.sho.run_companion6_stabilized_residual_live import atomic,sha256,bits_equal
from experiments.sho.validate_generic_update_network_adapter import protected
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho import run_stage1b_n3_gradient_polishing as prod
OUT=Path('outputs/sho/coordinated_generic_recovery_audit')
SRC=Path('outputs/sho/generic_update_network_targeted_recovery/companion_6')
GEO=Path('outputs/sho/generic_update_network_geometry_audit')

def replay(rt,t,loss,g,d):
 if d is None:return dict(accepted=False,reason='missing_direction')
 descent,dot,_=rt.controls.descent.check(g,d)
 if not descent:return dict(accepted=False,reason='no_descent_direction')
 unchanged=False;rejections=0
 for k in range(prod.MAX_BACKTRACKS+1):
  c=rt.controls.candidate(t,d,k)
  if torch.equal(c,t):unchanged=True;break
  if not rt.controls.in_bounds(c,rt.lower.numpy(),rt.upper.numpy()):rejections+=1;continue
  value=rt.scalar_loss(c.numpy())
  if rt.controls.accepts(value,loss):return dict(accepted=True,backtrack=k,rejections=rejections,loss=value,phi=c.tolist(),displacement=float((c-t).norm()),decrease=loss-value)
  rejections+=1
 numerical,_,_=rt.controls.no_further_improvement(unchanged,dot,loss)
 return dict(accepted=False,rejections=rejections,reason='floating_point_no_further_improvement' if numerical else 'backtracking_no_improvement')

def runtime():return OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(),TRAJ,Log(),direction_mode=STABILIZED_RESIDUAL)
def geometry(rt,t,values):
 result,g,H,gn,pre,route,d,qp=values
 r,J=residual_and_jacobian(rt.adapter,TRAJ,t);s=torch.linalg.svdvals(J);e=torch.linalg.eigvalsh(H);tol=22*torch.finfo(t.dtype).eps*e.abs().max()
 return dict(loss=result.loss,gradient_norm=gn,residual_norm=float(r.norm()),J_condition=float(s[0]/s[-1]),J_rank=int(torch.linalg.matrix_rank(J)),J_singular_values=s,H_condition=float(torch.linalg.cond(H)),H_eigenvalues=e,H_negative=int((e < -tol).sum()),H_positive=int((e>tol).sum()),H_nearzero=int((e.abs()<=tol).sum()),minimum_bound_margin=float(torch.minimum(t-rt.lower,rt.upper-t).min()),active_bounds=torch.where((t==rt.lower)|(t==rt.upper))[0],mu=qp.get('damping'),phi=t)

def screen(key,n2):
 rt=runtime();t=torch.tensor(n2['theta']);loss=n2['loss'];path=0.;bt=0;counter=0;accepted=0;start=time.monotonic();snap={};stop=None;contacts=set()
 trace=OUT/(key+'_stabilized.jsonl')
 if trace.exists():raise RuntimeError('refuse to overwrite existing screen')
 with trace.open('x') as f:
  while True:
   vals=rt.derivatives_and_direction(t);result,g,H,gn,pre,route,d,qp=vals
   if not bits_equal(result.loss,loss):raise RuntimeError('loss replay mismatch')
   pg=g.clone();pg[((t==rt.lower)&(g>=0))|((t==rt.upper)&(g<=0))]=0
   active=torch.where((t==rt.lower)|(t==rt.upper))[0].tolist();contacts.update(active)
   row=dict(step=accepted,phi=t.tolist(),loss=loss,gradient_norm=gn,projected_gradient_norm=float(pg.norm()),route=route,mu=qp.get('damping'),direction_norm=float(d.norm()) if d is not None else None,active_bounds=active,path=path,backtracking=bt,objective_evaluations=rt.objective_evaluations,stagnation_count=counter)
   if accepted in [0,1,10,50,100,250,500]:snap[str(accepted)]=row.copy()
   if loss<1e-20:stop='HANDOFF_THRESHOLD_REACHED'
   elif stop is None and pre!='CONTINUE':stop=pre
   elif stop is None and accepted>=500:stop='SCREEN_RESOURCE_LIMIT_500_ACCEPTED_STEPS'
   if stop:row['stop_reason']=stop;f.write(json.dumps(row)+'\n');break
   trial=replay(rt,t,loss,g,d);row['trial']=trial
   if trial['accepted']:
    pred=-float(qp['quadratic_value']);counter=prod._updated_float64_stagnation_count(counter,loss,trial['loss'],pred)
    row['full_predicted_decrease']=pred;t=torch.tensor(trial['phi']);loss=trial['loss'];path+=trial['displacement'];bt+=trial['rejections'];accepted+=1
    if counter>=prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:stop=prod.FLOAT64_STAGNATION_STOP_REASON
   else:stop=trial['reason'];bt+=trial.get('rejections',0)
   f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   if not trial['accepted']:break
   if accepted%100==0:print(key,accepted,loss,flush=True)
  f.flush();os.fsync(f.fileno())
 answer=dict(accepted_steps=accepted,loss=loss,phi=t,stop_reason=stop,handoff=loss<1e-20,parameter_path_length=path,backtracking_rejections=bt,new_objective_evaluations=rt.objective_evaluations,elapsed_seconds=time.monotonic()-start,snapshots=snap,bound_coordinates_ever=sorted(contacts),trace_sha256=sha256(trace),reused=False)
 atomic(OUT/(key+'_result.json'),answer);return answer

def main():
 global TRAJ
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 TRAJ=torch.from_numpy(np.load('outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz')['prefix_6'].copy())
 before=dict(protected=protected(),checkpoints=hashes(),sources={str(p):sha256(p) for p in Path('experiments').rglob('*.py')})
 atomic(OUT/'integrity_before.json',before)
 audit=json.load(open(GEO/'saved_trace_audit.json'))['branches'];jac=json.load(open(GEO/'residual_jacobian_audit.json'))['branches']
 # A: stream saved rows, never reconstruct by optimizing.
 arows=[]
 for branch,last in [('4,3',28136),('4,1',11792),('5,4',24000)]:
  count,index=map(int,branch.split(','));wanted={0,10,1000,10000,last-1,last};saved={}
  for p in sorted(SRC.glob(f'n{count:05d}_i{index:05d}_n3*.jsonl')):
   with p.open() as f:
    for line in f:
     row=json.loads(line);k=row['diagnostics']['accepted_before']
     if k in wanted:saved[k]=row
  for k in sorted(saved):
   row=saved[k];diag=row['diagnostics'];t=torch.tensor(row['history']['theta']);rt=runtime();vals=rt.derivatives_and_direction(t);result,g,H,gn,pre,route,d,qp=vals
   r,J=residual_and_jacobian(rt.adapter,TRAJ,t);old=rt.last_residual_state['existing_direction'] if rt.last_residual_state else None
   checks=dict(loss=bits_equal(result.loss,row['history']['loss']),gradient=bits_equal(g,diag['gradient']),H=bits_equal(H,diag['hessian']),direction=bits_equal(old,diag['direction']) if old is not None else diag['direction'] is None)
   jr=next((z for z in jac[branch] if z['step']==k),None)
   if jr:checks.update(residual=bits_equal(r,jr['residual']),J=bits_equal(J,jr['jacobian']))
   if not all(checks.values()):raise RuntimeError(('A replay mismatch',branch,k,checks))
   jt=J.T@J;G=2/r.numel()*jt;R=H-G
   # Independent residual curvature, fixed residual coefficients (no extra dr term).
   directR=torch.autograd.functional.hessian(lambda v:2/r.numel()*(r@residual_vector(rt.adapter,TRAJ,v)),t)
   es,ev=torch.linalg.eigh(H);neg=ev[:,es < -22*torch.finfo(t.dtype).eps*es.abs().max()]
   actions={}
   for name,v in [('gradient',g),('existing_direction',old)]:
    if v is not None:actions[name]=dict(Rv_norm=float((R@v).norm()),Gv_norm=float((G@v).norm()),H_v_norm=float((H@v).norm()),v_R_v=float(v@R@v),v_G_v=float(v@G@v))
   item=dict(branch=branch,step=k,loss=result.loss,checks=checks,JtJ_norm=float(jt.norm()),G_norm=float(G.norm()),R_norm=float(R.norm()),R_direct_decomposition_error=float((directR-R).norm()),G_eigenvalues=torch.linalg.eigvalsh(G),R_eigenvalues=torch.linalg.eigvalsh(R),H_eigenvalues=es,negative_H_directions_R_curvature=torch.diag(neg.T@R@neg),negative_H_directions_G_curvature=torch.diag(neg.T@G@neg),actions=actions,existing_replay=replay(rt,t,result.loss,g,old),stabilized_replay=replay(rt,t,result.loss,g,d))
   arows.append(item)
 atomic(OUT/'test_A.json',dict(rows=arows,norm='Frobenius',decomposition='G=2/M JtJ;R=H-G; independent residual-weighted Hessian cross-check; differences reflect float64 differentiation/accumulation rounding'))
 print('A DONE',len(arows),flush=True)
 # B/C: all endpoint identities retained; exact duplicated phi would reuse screen.
 brows=[];seen={}
 for p in sorted(SRC.glob('*_n2.json')):
  doc=json.load(open(p));n2=doc['result'];key=p.stem[:-3];count=int(key[1:6]);index=int(key[8:13]);branch=f'{count},{index}';hist=doc['history'];rt=runtime();t=torch.tensor(n2['theta']);vals=rt.derivatives_and_direction(t)
  if not bits_equal(vals[0].loss,n2['loss']):raise RuntimeError(('N2 loss mismatch',key))
  geom=geometry(rt,t,vals);best=[];b=float('inf');btheta=None;updates=[]
  for i,h in enumerate(hist,1):
   if h['loss']<b:
    updates.append(dict(evaluation=i,loss=h['loss'],motion=float(np.linalg.norm(np.array(h['phi'])-btheta)) if btheta is not None else 0.));b=h['loss'];btheta=np.array(h['phi'])
   best.append(b)
  if not bits_equal(b,n2['loss']) or not bits_equal(btheta,n2['theta']):raise RuntimeError('N2 best-ever replay mismatch')
  late={str(w):dict(start_best_loss=best[len(best)-w-1],end_best_loss=b,relative_improvement=(best[len(best)-w-1]-b)/best[len(best)-w-1],improving_evaluations=sum(z['evaluation']>len(hist)-w for z in updates),best_point_path=sum(z['motion'] for z in updates if z['evaluation']>len(hist)-w)) for w in [100,250,500]}
  atomic(OUT/(key+'_endpoint.json'),dict(geometry=geom,N2=n2,late=late))
  token=t.numpy().tobytes()
  if token in seen:res=seen[token];reuse='identical endpoint'
  elif branch=='5,4':
   base=Path('outputs/sho/stabilized_residual_n3_from_n2');ref=json.load(open(base/'start_verification.json'));res=json.load(open(base/'stabilized_result.json'));res['reused']=True;reuse=str(base)
   if not all(bits_equal(a,b) for a,b in [(t,ref['phi']),(vals[1],ref['gradient']),(vals[2],ref['hessian']),(vals[7]['damping'],ref['mu'])]):raise RuntimeError('existing screen mismatch')
   with (base/'stabilized.jsonl').open() as f:first=json.loads(next(f))
   if not bits_equal(vals[6],first['direction']):raise RuntimeError('existing stabilized direction mismatch')
   res['bound_coordinates_ever']=sorted({j for z in res['snapshots'].values() for j in z['bound_contacts']})
  else:res=screen(key,n2);reuse=None
  seen[token]=res
  prior=audit[branch];old=prior.get('result') or {};oldloss=prior['final']['loss'] if 'loss' in prior.get('final',{}) else old.get('loss')
  entry=dict(branch=branch,n1_site=index,n1_loss=prior['site']['loss'],n2_start_loss=hist[0]['loss'],n2_endpoint_loss=n2['loss'],n2_evaluations=n2['evaluations'],n2_message=n2['message'],n2_last_improvement=updates[-1],n2_late=late,geometry=geom,screen=res,reuse=reuse,historical=old,historical_best_loss=oldloss,source_hash=sha256(p))
  brows.append(entry);atomic(OUT/'test_B_C_partial.json',dict(rows=brows))
  print('B DONE',branch,res['accepted_steps'],res['loss'],res['stop_reason'],flush=True)
  if res['handoff']:
   print('EARLY STOP: BASIN_ENTRY_PRIMARY',flush=True);break
 # D: identical horizons distinct from unequal historical horizons.
 predictors=['loss','gradient_norm','J_condition','H_condition','H_negative','minimum_bound_margin','residual_norm']
 correlations={}
 for target in ['screen','historical']:
  y=[z['screen']['loss'] if target=='screen' else z['historical_best_loss'] for z in brows]
  correlations[target]={name:float(spearmanr([float(z['geometry'][name]) for z in brows],y).statistic) for name in predictors}
 for w in [100,250,500]:
  correlations[f'late_N2_relative_{w}']={target:float(spearmanr([z['n2_late'][str(w)]['relative_improvement'] for z in brows],[z['screen']['loss'] if target=='screen' else z['historical_best_loss'] for z in brows]).statistic) for target in ['screen','historical']}
 atomic(OUT/'test_B_C_D.json',dict(rows=brows,distinct_endpoints=len(seen),correlations=correlations,correlation_note='Spearman descriptive only; 11 endpoints from five shared N1 sites, all one trajectory. Historical horizons unequal/censored; screen horizons also shorter if numerical stop. No selector.'))
 after=dict(protected=protected(),checkpoints=hashes(),sources={str(p):sha256(p) for p in Path('experiments').rglob('*.py')})
 if before!=after:raise RuntimeError('integrity changed')
 atomic(OUT/'complete.json',dict(integrity_unchanged=True,A_states=len(arows),B_endpoints=len(brows),all_available_endpoints=len(list(SRC.glob('*_n2.json'))),handoffs=sum(z['screen']['handoff'] for z in brows),script_hash=sha256(Path(__file__))))
 print('BATCH COMPLETE INTEGRITY UNCHANGED',flush=True)
if __name__=='__main__':main()
