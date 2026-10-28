"""Only a recurrent, score-feedback optimizer for one supplied allocation interval."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.sho.audit_fixed_allocation_intervals import Audit,ROOT
from experiments.sho.train_fixed_interval_cut_optimizer import features,parents,targets,score_predictions,metrics
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'fixed_interval_iterative_optimizer_smoke';OLD=ROOT/'fixed_interval_cut_optimizer_smoke'
SEED=27312026;MAX_STEPS=6000;CHECK_EVERY=200
GATES=dict(recovery=.995,small_budget_recovery=1.,worst_scaled_penalty=1e-12)

class Controller(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=nn.Sequential(nn.Linear(3,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh())
  self.initial=nn.Sequential(nn.Linear(59,64),nn.Tanh())
  self.memory=nn.GRUCell(4,64)
  self.action=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
  self.double()
 def start(self,x):
  sides,mask,context=x;h=self.encoder(sides)
  mean=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None];maximum=h.masked_fill(~mask[:,:,None],-torch.inf).max(1).values
  return self.initial(torch.cat((mean,maximum,context),1))
 def propose(self,hidden):return (.5+self.action(hidden).squeeze(1)).clamp(0,1)
 def observe(self,hidden,z,score,previous_score,step):
  logscore=score.log();change=torch.zeros_like(score) if previous_score is None else logscore-previous_score.log()
  history=torch.stack((z,logscore,change,torch.full_like(z,step/12)),1)
  return self.memory(history,hidden)

class FrozenScore:
 """Differentiable execution of the same loaded frozen modules, not a new score.

 Boolean min/max routing uses the locked comparison weights and threshold;
 gradients follow the selected branches. No frozen parameter requires grad.
 """
 def __init__(self,audit):self.a=audit;self.models=audit.models
 def choose(self,left,right,maximum):
  active=left!=right;scale=torch.maximum(left,right);safe=torch.where(active,scale,torch.ones_like(scale))
  w=self.models['margin'].weight[0];margin=w[0]*(left/safe)+w[1]*(right/safe)
  threshold=self.models['threshold'];logit=threshold.alpha()*(margin-threshold.theta())
  replace=active & ((logit<0) if maximum else (logit>0))
  return torch.where(replace,right,left)
 def __call__(self,data,z):
  s,lo,axis,dim,n,counts,bounds,mask=data;dmax=s.shape[1]
  c=bounds[:,0]+z*(bounds[:,1]-bounds[:,0]);c=torch.maximum(bounds[:,0],torch.minimum(c,bounds[:,1]))
  axlen=(s*axis).sum(1);axlo=(lo*axis).sum(1);physical=axlo+c*axlen
  l=physical-axlo;r=axlo+axlen-physical
  lengths=torch.stack((s*(1-axis)+l[:,None]*axis,s*(1-axis)+r[:,None]*axis),1)
  logs=lengths.log().reshape(-1,dmax);childmask=mask[:,None,:].expand(-1,2,-1).reshape(-1,dmax)
  q=self.models['q'](logs,childmask,counts.reshape(-1).log(),dim[:,None].expand(-1,2).reshape(-1)).reshape(-1,2)
  qm=self.choose(q[:,0],q[:,1],True)
  h=self.models['h'](s.log(),mask,n.log(),dim)
  radius=self.models['radius'](torch.stack((qm.log(),h.log()),1))
  hi=lengths[:,:,0];low=hi
  for j in range(1,dmax):
   hi=torch.where(mask[:,j,None],self.choose(hi,lengths[:,:,j],True),hi)
   low=torch.where(mask[:,j,None],self.choose(low,lengths[:,:,j],False),low)
  aspect=self.models['aspect']['log_aspect'](torch.stack((hi.log(),low.log()),-1).reshape(-1,2)).reshape(-1,2)
  am=self.choose(aspect[:,0],aspect[:,1],True);penalty=self.models['aspect']['scale'](am)
  return self.a.finite(torch.stack((radius,penalty),1))

def geometry(rows):
 s=np.ones((len(rows),5));lo=np.zeros_like(s);axis=np.zeros_like(s);mask=np.zeros_like(s,dtype=bool)
 for i,r in enumerate(rows):
  d=r['d'];s[i,:d]=np.array(r['upper'])-r['lower'];lo[i,:d]=r['lower'];axis[i,r['axis']]=1;mask[i,:d]=True
 return tuple(torch.from_numpy(v) for v in (s,lo,axis,np.array([r['d'] for r in rows],float),np.array([r['n'] for r in rows],float),np.array([[r['left_count'],r['right_count']] for r in rows],float),np.array([[r['interval_lower'],r['interval_upper']] for r in rows]),mask))

def rollout(model,x,data,score,queries):
 hidden=model.start(x);zs=[];values=[];prev=None
 for t in range(queries):
  z=model.propose(hidden);v=score(data,z);zs.append(z);values.append(v)
  if t+1<queries:hidden=model.observe(hidden,z,v,prev,t+1)
  prev=v
 return torch.stack(zs,1),torch.stack(values,1)

def authoritative(model,audit,rows,x,queries,history_ablation=False):
 with torch.no_grad():hidden=model.start(x)
 prev=None;zr=[];vr=[];all_records=[]
 for t in range(queries):
  with torch.no_grad():z=model.propose(hidden)
  scored=score_predictions(audit,rows,z.numpy());v=torch.tensor([r['predicted_score'] for r in scored],dtype=torch.float64)
  zr.append(z.numpy());vr.append(v.numpy());all_records.append(scored)
  if t+1<queries:
   feedback=torch.ones_like(v) if history_ablation else v
   with torch.no_grad():hidden=model.observe(hidden,z,feedback,prev,t+1)
   prev=feedback
 zmat=np.column_stack(zr);vmat=np.column_stack(vr);idx=np.argmin(vmat,axis=1);result=[]
 for i,r in enumerate(rows):
  chosen=dict(all_records[idx[i]][i]);chosen.update(selected_query=int(idx[i]+1),queries=queries,query_z=json.dumps(zmat[i].tolist()),query_scores=json.dumps(vmat[i].tolist()),first_query_relative_penalty=all_records[0][i]['relative_penalty'],query_improved=bool(vmat[i,-1]<vmat[i,0]-1e-12*max(1,r['minimum_score'])))
  result.append(chosen)
 return result,zmat,vmat

def passed(rows):
 small=[r for r in rows if r['budget']<=20]
 return np.mean([r['teacher_equivalent'] for r in rows])>=GATES['recovery'] and all(r['teacher_equivalent'] for r in small) and max(r['absolute_penalty']/max(1,r['minimum_score']) for r in rows)<=GATES['worst_scaled_penalty'] and all(r['allocation_preserved'] and r['inside_interval'] for r in rows)

def run():
 start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);rng=np.random.default_rng(SEED);OUT.mkdir(parents=True,exist_ok=False)
 protected={str(p):sha(p) for p in ROOT.glob('*/*.pt')}
 for name in ('audit_fixed_allocation_intervals.py','train_fixed_interval_cut_optimizer.py','validate_recursive_split_assembly.py','train_recursive_region_splitter_poc.py','train_finite_split_score_smoke.py'):
  p=Path('experiments/sho')/name;protected[str(p)]=sha(p)
 config=dict(seed=SEED,query_curriculum=[2,4,8,12],max_steps_per_stage=MAX_STEPS,plateau='Stop after 1200 updates without >=1% cumulative improvement in best selection mean relative score penalty; at least 2400 updates.',gates=GATES,
  model='Shared geometry encoder -> GRU history of (z, log queried score, log score change, query index) -> bounded z.',
  loss='Mean relative regret of best actual query + 0.05 mean relative regret across all queries. Exact teacher optimum score only normalizes loss. No optimum cut or score target is a controller input.',
  data='Cached Stage-1 n=2..5 labels plus fresh synthetic n=6..20; all valid count pairs/all axes; disjoint train/selection/test parents with reserved heldout aspect bands.',
  scoring='Autodiff through unchanged frozen modules with locked comparison routing; validate against authoritative original execution. Final test uses authoritative score feedback at every query.',protected_hashes=protected)
 (OUT/'config.json').write_text(json.dumps(config,indent=2))
 audit=Audit();scorer=FrozenScore(audit);datasets={}
 for group in ('train','selection','heldout'):
  rows=json.loads((OLD/f'stage_1_{group}_intervals.json').read_text())
  more=parents(group,list(range(6,21)));limit=2 if group=='train' else 1
  more=[r for r in more if int(r['parent'].rsplit('_',1)[1])<limit]
  rows+=targets(audit,more);datasets[group]=rows;(OUT/(group+'_intervals.json')).write_text(json.dumps(rows))
 train=datasets['train'];selection=datasets['selection'];test=datasets['heldout'];tx=features(train);vx=features(selection);testx=features(test);td=geometry(train);vd=geometry(selection)
 # Validate differentiable frozen execution at all endpoints, midpoint, and random cuts on selection data.
 validation=[]
 for name,z in [('left',np.zeros(len(selection))),('right',np.ones(len(selection))),('midpoint',np.full(len(selection),.5)),('random',rng.random(len(selection)))]:
  reference=score_predictions(audit,selection,z);truth=np.array([r['predicted_score'] for r in reference])
  with torch.no_grad():p=scorer(vd,torch.from_numpy(z)).numpy()
  scaled=abs(p-truth)/np.maximum(1,abs(truth));assert np.all(scaled<256*np.finfo(float).eps),(name,scaled.max())
  validation.append(dict(cuts=name,examples=len(z),max_absolute_error=float(abs(p-truth).max()),max_scaled_error=float(scaled.max())))
 write_csv(OUT/'frozen_score_execution_validation.csv',validation)
 print(json.dumps(dict(event='data_ready',train=len(train),selection=len(selection),test=len(test),seconds=time.monotonic()-start)),flush=True)
 model=Controller();initial=copy.deepcopy(model.state_dict());reference=torch.tensor([r['minimum_score'] for r in train],dtype=torch.float64);vreference=np.array([r['minimum_score'] for r in selection]);history=[];summaries=[];breakdown=[];checkpoints=[]
 # Balance parent budgets instead of allowing large budgets' many intervals to dominate.
 counts={n:sum(r['n']==n for r in train) for n in range(2,21)};prob=np.array([1/counts[r['n']] for r in train]);prob/=prob.sum()
 for queries in (2,4,8,12):
  if checkpoints:model.load_state_dict(torch.load(checkpoints[-1]['checkpoint'],weights_only=False)['state_dict'])
  opt=torch.optim.Adam(model.parameters(),lr=.0015);best=None;bestkey=np.inf;beststep=0;anchor=np.inf;last_progress=0;plateau=False
  for step in range(1,MAX_STEPS+1):
   lr=3e-5+(.0015-3e-5)*.5*(1+math.cos(math.pi*(step-1)/MAX_STEPS))
   for pg in opt.param_groups:pg['lr']=lr
   ix=torch.from_numpy(rng.choice(len(train),256,replace=True,p=prob));z,v=rollout(model,tuple(x[ix] for x in tx),tuple(x[ix] for x in td),scorer,queries)
   rel=(v-reference[ix,None])/reference[ix,None];loss=rel.min(1).values.mean()+.05*rel.mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
   if step==1 or step%CHECK_EVERY==0:
    with torch.no_grad():vz,vv=rollout(model,vx,vd,scorer,queries)
    bestscores=vv.min(1).values.numpy();penalty=np.maximum(0,bestscores/vreference-1);key=float(penalty.mean());recovery=float(np.mean(bestscores<=vreference+1e-12*np.maximum(1,vreference)))
    if key<bestkey:bestkey=key;best=copy.deepcopy(model.state_dict());beststep=step
    if bestkey<anchor*.99:anchor=bestkey;last_progress=step
    entry=dict(queries=queries,step=step,loss=float(loss),selection_recovery=recovery,selection_mean_relative_penalty=key,selection_worst_relative_penalty=float(penalty.max()),seconds=time.monotonic()-start);history.append(entry);print(json.dumps(entry),flush=True)
    if recovery==1 and penalty.max()<=GATES['worst_scaled_penalty']:break
    if step>=2400 and step-last_progress>=1200:plateau=True;break
  model.load_state_dict(best)
  path=OUT/f'queries_{queries}_best.pt';digest=save_checkpoint(path,dict(state_dict=best,status='UNPROMOTED_SMOKE_CHECKPOINT',queries=queries,seed=SEED,steps=step,best_step=beststep,initial_state_dict=initial if queries==2 else None,previous_checkpoint=checkpoints[-1] if checkpoints else None,config=config))
  cp=dict(queries=queries,checkpoint=str(path),sha256=digest);checkpoints.append(cp);(OUT/'checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2))
  results,zmat,vmat=authoritative(model,audit,test,testx,queries);write_csv(OUT/f'queries_{queries}_heldout.csv',results);np.savez_compressed(OUT/f'queries_{queries}_history.npz',z=zmat,scores=vmat)
  trainresults,_,_=authoritative(model,audit,train,tx,queries);write_csv(OUT/f'queries_{queries}_train.csv',trainresults)
  success=passed(results);m=metrics(results);summary=dict(queries=queries,steps=step,best_step=beststep,plateau=plateau,status='PASSED' if success else 'FAILED_STOP',train_examples=len(train),selection_examples=len(selection),**m,first_query_recovery=float(np.mean(vmat[:,0]<=np.array([r['minimum_score'] for r in test])+1e-12*np.maximum(1,np.array([r['minimum_score'] for r in test])))),queries_improved=sum(r['query_improved'] for r in results),checkpoint_sha256=digest)
  summaries.append(summary)
  for field in ('dimension','budget','minimum_location','interval_kind'):
   for value in sorted(set(r[field] for r in results),key=str):breakdown.append(dict(queries=queries,split='heldout',grouping=field,group=value,**metrics([r for r in results if r[field]==value])))
  breakdown.append(dict(queries=queries,split='train',grouping='all',group='all',**metrics(trainresults)))
  # Verify score history causally affects later actions, without changing or training the model.
  _,ablated_z,_=authoritative(model,audit,test,testx,queries,history_ablation=True)
  summary['score_feedback_ablation_max_z_change']=float(abs(ablated_z[:,1:]-zmat[:,1:]).max());summary['score_feedback_ablation_changed_queries']=int(np.sum(ablated_z[:,1:]!=zmat[:,1:]))
  write_csv(OUT/'iterative_optimizer_curriculum.csv',summaries);write_csv(OUT/'iterative_optimizer_breakdown.csv',breakdown);write_csv(OUT/'iterative_optimizer_training_history.csv',history)
  if not success:break
 assert all(sha(p)==h for p,h in protected.items())
 result=dict(status='FROZEN_CANDIDATE' if success and queries==12 else 'DIAGNOSTIC_ONLY',operator='fixed_interval_iterative_optimizer',passed=bool(success and queries==12),completed_stages=len(summaries),final_stage=summaries[-1],protected_hashes_unchanged=True,seconds=time.monotonic()-start,checkpoints=checkpoints)
 (OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2));(OUT/'iterative_optimizer_summary.txt').write_text(json.dumps(result,indent=2)+'\nNo recursion or SHO search; all existing checkpoints and interval discovery unchanged.\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
