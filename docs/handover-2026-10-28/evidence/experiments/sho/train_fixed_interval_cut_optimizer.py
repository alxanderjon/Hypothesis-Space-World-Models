"""Supervise only one cut within a supplied, fixed integer-allocation interval."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq
from experiments.sho.audit_fixed_allocation_intervals import Audit,ROOT,SOURCE
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'fixed_interval_cut_optimizer_smoke';SEED=27302026
GATES=dict(equivalent_recovery=.995,small_budget_recovery=1.,worst_relative_penalty=1e-8)
MAX_STEPS=4000

class IntervalCutPredictor(nn.Module):
 """Shared dimension encoder, permutation invariant set pooling, bounded output.

 No score, root, norm, or interval optimizer is called by forward(). Axis identity
 enters via a side feature; only normalized geometry, budgets/counts/bounds enter.
 """
 def __init__(self):
  super().__init__()
  self.encoder=nn.Sequential(nn.Linear(3,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh())
  self.head=nn.Sequential(nn.Linear(59,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
  self.double()
 def forward(self,sides,mask,context):
  h=self.encoder(sides);mean=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
  maximum=h.masked_fill(~mask[:,:,None],-torch.inf).max(1).values
  raw=self.head(torch.cat((mean,maximum,context),1)).squeeze(1)
  return torch.clamp(.5+raw,0.,1.)

def features(rows):
 sides=np.zeros((len(rows),5,3));mask=np.zeros((len(rows),5),bool);context=[]
 for i,r in enumerate(rows):
  ell=np.array(r['upper'])-r['lower'];z=np.log(ell);z-=z.mean();d=r['d'];a=r['axis'];n=r['n'];k=r['left_count'];l=r['interval_lower'];u=r['interval_upper']
  sides[i,:d,0]=z;sides[i,:d,1]=ell/ell.max();sides[i,a,2]=1;mask[i,:d]=True
  context.append([d/5,np.log(d),z[a],np.log(n),1/n,k/n,(n-k)/n,np.log(k),np.log(n-k),l,u])
 return torch.from_numpy(sides),torch.from_numpy(mask),torch.tensor(context,dtype=torch.float64)

def parents(group,budgets):
 rng=np.random.default_rng(SEED+{'train':0,'selection':1,'heldout':2}[group]+max(budgets)*100)
 result=[];rep={'train':16,'selection':4,'heldout':8}[group]
 for d in range(1,6):
  for n in budgets:
   for j in range(rep):
    if d==1:ratio=1.
    elif group=='train':
     while True:
      ratio=np.exp(rng.uniform(0,np.log(32)))
      if not (3<=ratio<=4 or 12<=ratio<=18):break
    else:
     band=(3,4) if j%2==0 else (12,18);ratio=rng.uniform(*band)
    z=rng.random(d)
    if d>1:z[0]=0;z[1]=1;rng.shuffle(z)
    else:z[:]=0
    scale=10**rng.uniform(-2,2) if group=='train' else 10**rng.uniform(-6,6)
    ell=scale*np.exp((z-z.mean())*np.log(ratio));lo=rng.uniform(-.5,.5,d)*scale
    result.append(dict(d=d,n=n,lower=lo.tolist(),upper=(lo+ell).tolist(),group=group,parent=f'{group}_d{d}_n{n}_{j}',reason=''))
 if group=='heldout' and 3 in budgets:
  saved=json.loads((SOURCE/'heldout_regions.json').read_text());case=json.loads((SOURCE/'worst_case.json').read_text())['result']['case'];r=saved[case]
  result.append(dict(r,group=group,parent='previous_n3',reason='previous n=3 failure'))
 if group=='heldout' and 4 in budgets:
  a=np.load(ROOT/'split_quality_smoke/previous_n4_failure.npz');result.append(dict(d=2,n=4,lower=a['lower'][0].tolist(),upper=a['upper'][0].tolist(),group=group,parent='previous_n4',reason='previous n=4 failure'))
 return result

def targets(audit,records):
 rows=[]
 for pi,r in enumerate(records):
  for axis in range(r['d']):
   audit.setup(r,axis);n=r['n'];edge=max(1e-14,4*np.spacing(abs(audit.lo[axis])+abs(audit.hi[axis]))/audit.a)
   scan=np.linspace(edge,1-edge,max(257,8*n+1));cnt=audit.counts(scan)[:,0];assert np.all(np.diff(cnt)>=0)
   low=np.array([scan[np.flatnonzero(cnt<k)[-1]] for k in range(1,n+1)]);high=np.array([scan[np.flatnonzero(cnt>=k)[0]] for k in range(1,n+1)])
   for _ in range(60):
    mid=(low+high)/2;left=audit.counts(mid)[:,0]<np.arange(1,n+1);low=np.where(left,mid,low);high=np.where(left,high,mid)
   assert np.array_equal(audit.counts(high)[:,0],np.arange(1,n+1));assert np.array_equal(audit.counts(low)[:,0],np.arange(n))
   candidate=[];metadata=[]
   for k in range(1,n):
    l=high[k-1];u=low[k];br=audit.breaks(l,u,k);opt=list(br)
    for a,b in zip(br[:-1],br[1:]):
     aa=np.nextafter(a,b);bb=np.nextafter(b,a)
     if aa>=bb:continue
     xx=np.linspace(aa,bb,129);der=audit.formula(xx,k)[1]
     for j in np.flatnonzero(der[:-1]*der[1:]<0):opt.append(brentq(lambda x:float(audit.formula(x,k)[1][0]),xx[j],xx[j+1],xtol=5e-324,rtol=1e-15))
    opt=np.unique(opt);grid=np.linspace(l,u,65);start=len(candidate);candidate.extend(opt);metadata.append((k,l,u,opt,start,len(candidate),grid))
   vals,cc=audit.scores(candidate)
   for k,l,u,opt,a,b,grid in metadata:
    assert np.all(cc[a:b,0]==k)
    scores=vals[a:b];best=float(scores.min());tol=1e-12*max(1,best);j=int(np.flatnonzero(scores<=best+tol)[0]);cut=float(opt[j]);z=float(np.clip((cut-l)/(u-l),0,1))
    gv,gc=audit.scores(grid);assert np.all(gc[:,0]==k) and best<=gv.min()+tol
    diff=np.diff(gv);sg=np.sign(diff[abs(diff)>tol]);kind='increasing' if np.all(sg>=0) else 'decreasing' if np.all(sg<=0) else 'unimodal'
    rows.append(dict(r,axis=axis,left_count=k,right_count=n-k,interval_lower=float(l),interval_upper=float(u),target_cut=cut,target_z=z,target_score=float(scores[j]),minimum_score=best,minimum_location='lower_endpoint' if cut==l else 'upper_endpoint' if cut==u else 'interior',interval_kind=kind))
  if pi%100==0:print(json.dumps(dict(event='labels',group=r['group'],parents=pi+1,total=len(records),intervals=len(rows))),flush=True)
 return rows

def score_predictions(audit,rows,z):
 cs=np.array([np.clip(r['interval_lower']+v*(r['interval_upper']-r['interval_lower']),r['interval_lower'],r['interval_upper']) for r,v in zip(rows,z)])
 result=[];i=0
 while i<len(rows):
  j=i+1
  while j<len(rows) and rows[j]['parent']==rows[i]['parent'] and rows[j]['axis']==rows[i]['axis']:j+=1
  audit.setup(rows[i],rows[i]['axis']);values,counts=audit.scores(cs[i:j])
  for p in range(i,j):
   r=rows[p];value=float(values[p-i]);best=r['minimum_score'];tol=1e-12*max(1,best)
   result.append(dict(parent=r['parent'],reason=r['reason'],dimension=r['d'],budget=r['n'],axis=r['axis'],left_count=r['left_count'],right_count=r['right_count'],interval_kind=r['interval_kind'],minimum_location=r['minimum_location'],target_cut=r['target_cut'],predicted_cut=float(cs[p]),target_z=r['target_z'],predicted_z=float(z[p]),cut_error=abs(cs[p]-r['target_cut']),normalized_error=abs(z[p]-r['target_z']),target_score=r['target_score'],minimum_score=best,predicted_score=value,teacher_equivalent=bool(value<=best+tol),relative_penalty=max(0.,value/best-1),absolute_penalty=max(0.,value-best),allocation_preserved=bool(counts[p-i,0]==r['left_count']),inside_interval=bool(r['interval_lower']<=cs[p]<=r['interval_upper'])))
  i=j
 return result

def metrics(rows):
 return dict(examples=len(rows),cut_mae=float(np.mean([r['cut_error'] for r in rows])),cut_max=max(r['cut_error'] for r in rows),normalized_mae=float(np.mean([r['normalized_error'] for r in rows])),normalized_max=max(r['normalized_error'] for r in rows),teacher_equivalent_recovery=float(np.mean([r['teacher_equivalent'] for r in rows])),mean_relative_penalty=float(np.mean([r['relative_penalty'] for r in rows])),worst_relative_penalty=max(r['relative_penalty'] for r in rows),worst_absolute_penalty=max(r['absolute_penalty'] for r in rows),allocation_mistakes=sum(not r['allocation_preserved'] for r in rows),outside_interval=sum(not r['inside_interval'] for r in rows))

def passing(rows):
 m=metrics(rows);small=[r for r in rows if r['budget']<=20]
 return m['teacher_equivalent_recovery']>=GATES['equivalent_recovery'] and all(r['teacher_equivalent'] for r in small) and m['worst_relative_penalty']<=GATES['worst_relative_penalty'] and m['allocation_mistakes']==m['outside_interval']==0

def infer(model,x):
 with torch.no_grad():return model(*x).numpy()

def run():
 start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
 OUT.mkdir(parents=True,exist_ok=False);protected={str(p):sha(p) for p in ROOT.glob('*/*.pt')}
 for p in [Path('experiments/sho/validate_recursive_split_assembly.py'),Path('experiments/sho/train_recursive_region_splitter_poc.py'),Path('experiments/sho/train_finite_split_score_smoke.py')]:protected[str(p)]=sha(p)
 (OUT/'config.json').write_text(json.dumps(dict(seed=SEED,dtype='float64',max_steps=MAX_STEPS,curriculum=[[2,5],[2,10],[2,20],[2,73]],gates=GATES,heldout='Disjoint parents; held-out aspect bands [3,4] and [12,18]; scales 1e-6..1e6 versus training 1e-2..1e2. All dimensions appear in training.',selection='Checkpoint selection uses separate selection parents, not final heldout parents.',features='Shared side encoder(log normalized length, length/max length, split-axis indicator), mean/max pooling; dimension, selected log side, budget/count features, supplied interval bounds. No teacher score or optimum/root inputs.',protected_hashes=protected),indent=2))
 audit=Audit();model=IntervalCutPredictor();initial=copy.deepcopy(model.state_dict());all_history=[];summary=[];aggregates=[];datasets={g:[] for g in ('train','selection','heldout')};manifest=[]
 for stage,new_budgets in enumerate((list(range(2,6)),list(range(6,11)),list(range(11,21)),[29,37,73]),1):
  stage_start=time.monotonic()
  if stage>1:model.load_state_dict(torch.load(manifest[-1]['checkpoint'],weights_only=False)['state_dict'])
  for group in datasets:
   generated=targets(audit,parents(group,new_budgets));datasets[group]+=generated
   (OUT/f'stage_{stage}_{group}_intervals.json').write_text(json.dumps(datasets[group]))
  train,selection,test=[datasets[g] for g in ('train','selection','heldout')];tx=features(train);vx=features(selection);testx=features(test);target=torch.tensor([r['target_z'] for r in train],dtype=torch.float64)
  # Equal interval regression, with modest interior oversampling to prevent endpoint dominance.
  prob=np.array([2. if r['minimum_location']=='interior' else 1. for r in train]);prob/=prob.sum()
  opt=torch.optim.Adam(model.parameters(),lr=.002);best=None;best_key=None;last_improvement=0;selected_step=0;plateau=False
  for step in range(1,MAX_STEPS+1):
   lr=2e-5+(.002-2e-5)*.5*(1+math.cos(math.pi*(step-1)/MAX_STEPS))
   for pg in opt.param_groups:pg['lr']=lr
   ix=torch.from_numpy(rng.choice(len(train),size=min(256,len(train)),replace=True,p=prob));pred=model(*(x[ix] for x in tx));loss=((pred-target[ix])**2).mean()
   opt.zero_grad();loss.backward();opt.step()
   if step==1 or step%100==0:
    zr=infer(model,vx);vr=score_predictions(audit,selection,zr);m=metrics(vr);mse=float(np.mean((zr-np.array([r['target_z'] for r in selection]))**2))
    # Select by continuous geometry penalty first, then coordinate regression.
    key=(m['mean_relative_penalty'],mse)
    if best_key is None or key<best_key:
     meaningful=best_key is None or key[0]<best_key[0]*.995
     if meaningful:last_improvement=step
     best_key=key;best=copy.deepcopy(model.state_dict());selected_step=step
    row=dict(stage=stage,step=step,train_loss=float(loss),selection_normalized_mse=mse,**m);all_history.append(row)
    print(json.dumps(dict(event='train',stage=stage,step=step,recovery=m['teacher_equivalent_recovery'],penalty=m['worst_relative_penalty'],mae=m['normalized_mae'],seconds=time.monotonic()-start)),flush=True)
    if passing(vr):break
    if step>=1600 and step-last_improvement>=800:plateau=True;break
  model.load_state_dict(best)
  # Persist immediately after training, before final heldout evaluation.
  path=OUT/f'stage_{stage}_best.pt';digest=save_checkpoint(path,dict(state_dict=best,initial_state_dict=initial if stage==1 else None,stage=stage,steps_run=step,selected_step=selected_step,architecture='IntervalCutPredictor shared 3->24->24 encoder; mean/max; 59->64->64->1; clamped relative output',status='UNPROMOTED_SMOKE_CHECKPOINT',seed=SEED,previous_checkpoint=manifest[-1] if manifest else None))
  item=dict(stage=stage,checkpoint=str(path),sha256=digest);manifest.append(item);(OUT/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))
  testrows=score_predictions(audit,test,infer(model,testx));trainrows=score_predictions(audit,train,infer(model,tx));write_csv(OUT/f'stage_{stage}_heldout_predictions.csv',testrows);write_csv(OUT/f'stage_{stage}_train_predictions.csv',trainrows)
  passed=passing(testrows);m=metrics(testrows);summary.append(dict(stage=stage,max_budget=max(r['n'] for r in test),steps_run=step,best_step=selected_step,plateau=plateau,train_examples=len(train),selection_examples=len(selection),status='PASSED' if passed else 'FAILED_STOP',seconds=time.monotonic()-stage_start,**m,checkpoint_sha256=digest))
  for field in ('dimension','budget','interval_kind','minimum_location'):
   for value in sorted(set(r[field] for r in testrows),key=str):aggregates.append(dict(stage=stage,split='heldout',grouping=field,group=value,**metrics([r for r in testrows if r[field]==value])))
  aggregates.append(dict(stage=stage,split='train',grouping='all',group='all',**metrics(trainrows)))
  write_csv(OUT/'fixed_interval_training_history.csv',all_history);write_csv(OUT/'fixed_interval_cut_curriculum.csv',summary);write_csv(OUT/'fixed_interval_cut_breakdown.csv',aggregates)
  if not passed:break
 passed=bool(summary[-1]['status']=='PASSED' and len(summary)==4)
 assert all(sha(p)==h for p,h in protected.items())
 result=dict(passed=passed,status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',operator='fixed_interval_cut_optimizer',completed_stages=len(summary),last_stage=summary[-1],protected_hashes_unchanged=True,recursion_run=False,search_run=False,seconds=time.monotonic()-start,checkpoints=manifest)
 (OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2));(OUT/'fixed_interval_cut_summary.txt').write_text(json.dumps(result,indent=2)+'\n\nNo existing checkpoint or recursive scaffold was modified. Failed stages stop the curriculum. Checkpoints are persisted before final heldout evaluation.\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
