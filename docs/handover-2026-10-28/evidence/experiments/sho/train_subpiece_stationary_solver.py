"""Generic coefficient-only supervised stationary solver. No task/search data training."""
import copy,csv,json,math,time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.audit_fixed_allocation_intervals import Audit,ROOT,SOURCE

OUT=ROOT/'split_subpiece_stationary_solver_smoke';SEED=27332026
mp.mp.dps=60

class StationarySolver(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Linear(12,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh());self.exists=nn.Linear(64,1);self.location=nn.Linear(64,1);self.double()
 def forward(self,x):
  h=self.body(x);return self.exists(h).squeeze(1),torch.sigmoid(self.location(h).squeeze(1))

def score(r,c):
 c=np.asarray(c);t=c if r['t']==0 else 1-c;v=c if r['v']==0 else 1-c
 return r['A']*np.sqrt(r['B']+r['a']**2*t*t)+r['beta']*np.log(v)

def derivative(r,c):
 t=c if r['t']==0 else 1-c;v=c if r['v']==0 else 1-c
 return (1 if r['t']==0 else -1)*r['A']*r['a']**2*t/np.sqrt(r['B']+r['a']**2*t*t)+(1 if r['v']==0 else -1)*r['beta']/v

def roots_high_precision(r):
 A,a,B,beta=[mp.mpf(r[k]) for k in ('A','a','B','beta')];st=1 if r['t']==0 else -1;sv=1 if r['v']==0 else -1
 if beta==0 or st*sv*beta>=0:return []
 k=abs(beta)/(A*a);b=B/(a*a);ts=[]
 if r['t']==r['v']:
  y=(k*k+mp.sqrt(k**4+4*k*k*b))/2;t=mp.sqrt(y)
  if 0<t<1:ts=[t]
 elif b==0:
  t=1-k
  if 0<t<1:ts=[t]
 else:
  bf=float(b);peak0=brentq(lambda t:t**3+2*bf*t-bf,0.,.5,xtol=1e-15)
  peak=mp.findroot(lambda t:t**3+2*b*t-b,mp.mpf(peak0),tol=mp.mpf('1e-52'))
  g=lambda t:t*(1-t)/mp.sqrt(b+t*t)
  if k>g(peak):return []
  if k==g(peak):ts=[peak]
  else:
   gf=lambda t:t*(1-t)/math.sqrt(bf+t*t)-float(k)
   for l,u in ((0.,float(peak)),(float(peak),1.)):
    x=brentq(gf,l,u,xtol=5e-324,rtol=1e-15);dx=max(1e-18,min(x,1-x)*1e-6)
    try:t=mp.findroot(lambda t:g(t)-k,(mp.mpf(x-dx),mp.mpf(x+dx)),tol=mp.mpf('1e-50'),verify=True)
    except (ValueError,ZeroDivisionError):
     ml,mu=mp.mpf(l),mp.mpf(u);fl=g(ml)-k
     for _ in range(200):
      mid=(ml+mu)/2;fm=g(mid)-k
      if fm*fl<=0:mu=mid
      else:ml=mid;fl=fm
     t=(ml+mu)/2
    assert 0<t<1 and abs(g(t)-k)<mp.mpf('1e-40')
    ts.append(t)
 roots=sorted([t if r['t']==0 else 1-t for t in ts]);out=[]
 for c in roots:
  if not out or abs(c-out[-1])>mp.mpf('1e-40'):out.append(c)
 return out

def make_data(stage,count,group):
 rng=np.random.default_rng(SEED+stage*100+{'train':0,'selection':1,'heldout':2}[group]);rows=[]
 while len(rows)<count:
  j=len(rows);held=group!='train'
  orientations=[(0,0)] if stage==1 else [(0,0),(1,0)] if stage==2 else ([(0,0),(1,1)] if stage==3 and not held else [(0,0),(0,1),(1,0),(1,1)])
  t,v=orientations[int(rng.integers(len(orientations)))];b=10**rng.uniform(-1,1) if stage<4 else 10**rng.uniform(-3,3)
  range_group='train_range'
  if held:
   if stage==4:
    band=((-6,-3) if j%3==0 else (3,6) if j%3==1 else (-.7,-.4));b=10**rng.uniform(*band);range_group='heldout_coefficient_band'
   else:b=10**rng.uniform(-.7,-.4);range_group='reserved_coefficient_band'
  elif stage==4 and .19<=b<=.4:continue
  elif stage<4 and .19<=b<=.4:continue
  if stage==4 and j%20==0:b=0.;range_group+=':B_zero'
  A=10**rng.uniform(-2,2) if not held else 10**rng.uniform(-5,5);a=10**rng.uniform(-2,2) if not held else 10**rng.uniform(-5,5)
  beta=0. if stage<=2 else float(rng.choice([-1,1]))*A*a*10**rng.uniform(-3,-.05)
  r=dict(A=A,a=a,B=a*a*b,beta=beta,t=t,v=v,stage=stage,group=group,coefficient_range=range_group,orientation=f't{t}_v{v}')
  roots=roots_high_precision(r)
  want_root=stage>=3 and j%2==0
  if want_root and not roots:continue
  if want_root:
   c=float(roots[int(rng.integers(len(roots)))]);rho=float(rng.uniform(.05,.95));width=10**rng.uniform(-1.3,-.3)
   if held and j%6==0:rho=10**rng.uniform(-8,-4) if j%12==0 else 1-10**rng.uniform(-8,-4);range_group+=':endpoint_near'
   if held and j%6==2:width=10**rng.uniform(-4,-2);range_group+=':narrow_interval'
   width=min(width,.95*c/rho,.95*(1-c)/(1-rho));l=c-rho*width;u=l+width
  else:
   width=float(rng.uniform(.05,.5));l=float(rng.uniform(.001,.999-width));u=l+width
   if held and j%5==1:width=10**rng.uniform(-4,-2);l=float(rng.uniform(.001,.999-width));u=l+width;range_group+=':narrow_interval'
  if not 0<l<u<1:continue
  inside=[c for c in roots if mp.mpf(l)<c<mp.mpf(u)]
  if stage>=3 and bool(inside)!=want_root:continue
  r.update(lower=l,upper=u,coefficient_range=range_group,exists=bool(inside),root_count=len(inside),roots=[float(c) for c in inside])
  if inside:
   def ms(c):
    tt=c if t==0 else 1-c;vv=c if v==0 else 1-c
    return mp.mpf(A)*mp.sqrt(mp.mpf(r['B'])+mp.mpf(a)**2*tt**2)+mp.mpf(beta)*mp.log(vv)
   canonical=min(inside,key=lambda c:(ms(c),c));r.update(c_star=float(canonical),root_decimal=mp.nstr(canonical,55),target_z=float((canonical-mp.mpf(l))/(mp.mpf(u)-mp.mpf(l))))
  else:r.update(c_star=None,root_decimal=None,target_z=0.)
  rows.append(r)
 return rows

def features(rows):
 f=[]
 for r in rows:
  A,a,B,beta,l,u=[r[k] for k in ('A','a','B','beta','lower','upper')];b=B/a**2;lam=beta/(A*a)
  f.append([math.log(A)/6,math.log(a)/6,math.log1p(b)/6,b/(1+b),math.asinh(lam),float(beta==0),float(np.sign(beta)),1-2*r['t'],1-2*r['v'],l,u,math.log(u-l)/6])
 return torch.tensor(f,dtype=torch.float64)

def infer(model,rows):
 with torch.no_grad():logit,z=model(features(rows))
 return logit.numpy()>0,z.numpy()

def root_metrics(rows,pred,z):
 yes=np.array([r['exists'] for r in rows]);cuts=np.array([r['lower']+zz*(r['upper']-r['lower']) for r,zz in zip(rows,z)]);err=np.array([abs(c-r['c_star']) for r,c in zip(rows,cuts) if r['exists']]);ne=np.array([abs(zz-r['target_z']) for r,zz in zip(rows,z) if r['exists']]);res=np.array([abs(derivative(r,c)) for r,c in zip(rows,cuts) if r['exists']]);scaled=np.array([abs(derivative(r,c))/max(1,r['A']*r['a'],abs(r['beta'])) for r,c in zip(rows,cuts) if r['exists']])
 return dict(examples=len(rows),true_roots=int(yes.sum()),no_roots=int((~yes).sum()),existence_accuracy=float(np.mean(pred==yes)),false_negatives=int(np.sum(yes&~pred)),false_positives=int(np.sum(~yes&pred)),root_mae=float(err.mean()) if len(err) else None,root_max=float(err.max()) if len(err) else None,normalized_mae=float(ne.mean()) if len(ne) else None,normalized_max=float(ne.max()) if len(ne) else None,derivative_residual_mean=float(res.mean()) if len(res) else None,derivative_residual_max=float(res.max()) if len(res) else None,scaled_residual_max=float(scaled.max()) if len(scaled) else None)

def locked_min(audit,values):
 values=np.asarray(values);offset=max(0.,1-float(values.min()));positive=values+offset;cur=float(positive[0]);owner=0
 for j in range(1,len(values)):
  if positive[j]!=cur and audit.execution.logits(np.array([cur]),np.array([positive[j]]))[0]>0:cur=float(positive[j]);owner=j
 return owner

def evaluate_generic(model,rows,audit):
 pred,z=infer(model,rows);records=[]
 for i,r in enumerate(rows):
  c=r['lower']+z[i]*(r['upper']-r['lower']);candidates=[r['lower'],r['upper']]+([float(c)] if pred[i] else []);oracle=[r['lower'],r['upper']]+r['roots']
  # Arbitrary coefficients need not correspond to actual child boxes. Use their
  # formula terms with the locked finite sum and existing comparison selector.
  ca=np.array(candidates);t=ca if r['t']==0 else 1-ca;v=ca if r['v']==0 else 1-ca
  parts=np.column_stack((r['A']*np.sqrt(r['B']+r['a']**2*t*t),r['beta']*np.log(v)))
  with torch.no_grad():vals=audit.finite(torch.from_numpy(parts)).numpy()
  chosen=locked_min(audit,vals);best=float(np.min(score(r,np.array(oracle))));value=float(vals[chosen]);tol=1e-12*max(1,best)
  records.append(dict(stage=r['stage'],case=i,coefficient_range=r['coefficient_range'],orientation=r['orientation'],root_count=r['root_count'],true_exists=r['exists'],predicted_exists=bool(pred[i]),target_cut=r['c_star'],predicted_cut=float(c),root_error=abs(c-r['c_star']) if r['exists'] else None,normalized_error=abs(z[i]-r['target_z']) if r['exists'] else None,derivative_residual=float(abs(derivative(r,c))),minimum_score=best,chosen_score=value,minimum_recovered=bool(value<=best+tol),scaled_penalty=max(0,value-best)/max(1,abs(best)),absolute_penalty=max(0,value-best),selected_cut=candidates[chosen],endpoint_near_root=bool(r['exists'] and min(r['target_z'],1-r['target_z'])<1e-3)))
 return records,pred,z

def audited_assembly(model,audit):
 ap=ROOT/'fixed_allocation_interval_audit';pieces=list(csv.DictReader((ap/'fixed_interval_score_structure.csv').open()));intervals=list(csv.DictReader((ap/'fixed_allocation_interval_audit.csv').open()))
 parents=[]
 for group in ('train','heldout'):parents+=json.loads((SOURCE/(group+'_regions.json')).read_text())
 old=np.load(ROOT/'split_quality_smoke/previous_n4_failure.npz');parents.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist()))
 coeff=[];used=[];last=None
 for p in pieces:
  if p['has_float64_interior']!='True':continue
  ri=int(p['parent']);axis=int(p['axis']);k=int(p['interval']);key=(ri,axis)
  if key!=last:audit.setup(parents[ri],axis);last=key
  child=int(p['radius_active_child']);nj=k if child==0 else audit.n-k
  coeff.append(dict(A=1/(audit.h*nj**(1/audit.d)),a=audit.a,B=audit.B,beta=.05*int(p['aspect_log_power']),t=child,v=int(p['aspect_active_child']),lower=float(p['lower']),upper=float(p['upper'])))
  used.append(p)
 pred,z=infer(model,coeff);grouped={};piece_results=[]
 for p,r,exists,zz in zip(used,coeff,pred,z):
  key=(int(p['parent']),int(p['axis']),int(p['interval']));c=r['lower']+zz*(r['upper']-r['lower']);candidate=[r['lower'],r['upper']]+([float(c)] if exists else []);oracle=[r['lower'],r['upper']]+json.loads(p['quartic_stationary_roots'])
  grouped.setdefault(key,dict(candidate=[],oracle=[]));grouped[key]['candidate']+=candidate;grouped[key]['oracle']+=oracle
  piece_results.append(dict(parent=key[0],dimension=int(p['dimension']),budget=int(p['budget']),axis=key[1],interval=key[2],piece=int(p['piece']),predicted_root=bool(exists),root_cut=float(c),candidates=candidate,oracle=oracle))
 results=[];allpieces=[]
 for ii,inter in enumerate(intervals):
  if inter['valid']!='True':continue
  key=(int(inter['parent']),int(inter['axis']),int(inter['interval']));audit.setup(parents[key[0]],key[1]);gg=grouped.get(key,dict(candidate=[],oracle=[]))
  ca=np.unique([float(inter['lower']),float(inter['upper'])]+gg['candidate']);oracle=np.unique([float(inter['lower']),float(inter['upper'])]+gg['oracle'])
  val,counts=audit.scores(ca);truth,tc=audit.scores(oracle);assert np.all(counts[:,0]==key[2]) and np.all(tc[:,0]==key[2])
  choice=locked_min(audit,val);best=float(truth.min());value=float(val[choice]);tol=1e-12*max(1,best)
  results.append(dict(parent=key[0],dimension=parents[key[0]]['d'],budget=parents[key[0]]['n'],axis=key[1],interval=key[2],minimum_score=best,predicted_score=value,minimum_recovered=bool(value<=best+tol),relative_penalty=max(0,value/best-1),absolute_penalty=max(0,value-best),predicted_cut=float(ca[choice]),candidate_count=len(ca),oracle_candidate_count=len(oracle)))
 write_csv(OUT/'audited_interval_assembly.csv',results)
 # Individual subpieces use the full authoritative locked split score, not the formula proxy.
 byaxis={}
 for p in piece_results:byaxis.setdefault((p['parent'],p['axis']),[]).append(p)
 for (ri,axis),ps in byaxis.items():
  audit.setup(parents[ri],axis);cuts=[];offsets=[]
  for p in ps:
   a=len(cuts);cuts+=p['candidates'];b=len(cuts);cuts+=p['oracle'];offsets.append((a,b,len(cuts)))
  vv,cc=audit.scores(cuts)
  for p,(a,b,c) in zip(ps,offsets):
   values=vv[a:b];true=vv[b:c];v=float(values[locked_min(audit,values)]);best=float(true.min());allpieces.append(dict(**{k:v for k,v in p.items() if k not in ('candidates','oracle')},minimum_recovered=bool(v<=best+1e-12*max(1,best)),relative_penalty=max(0,v/best-1)))
 write_csv(OUT/'audited_subpiece_assembly.csv',allpieces)
 return dict(intervals=len(results),interval_recovery=float(np.mean([r['minimum_recovered'] for r in results])),interval_worst_relative_penalty=max(r['relative_penalty'] for r in results),subpieces=len(allpieces),subpiece_recovery=float(np.mean([r['minimum_recovered'] for r in allpieces])),subpiece_worst_relative_penalty=max(r['relative_penalty'] for r in allpieces))

def run():
 started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(SEED);audit=Audit()
 protected={str(p):sha(p) for p in ROOT.glob('*/*.pt')};model=StationarySolver();summaries=[];breakdown=[];history=[];manifests=[]
 config=dict(seed=SEED,dtype='float64',labels='60-decimal mpmath analytic roots; classify strict interior membership. If multiple roots, target lowest-scoring stationary point, leftmost exact tie.',
  curriculum=['beta=0,t=c','beta=0,both t orientations','nonzero beta,same t/v orientations; opposite orientations held out','all orientation combinations,broad coefficient ranges'],
  features='Only coefficients, orientations and interval; dimensionless coefficient transforms allowed. No score, derivative, stationary root, SHO data or search labels as inputs.',
  scoring='Generic arbitrary-coefficient formula terms -> frozen finite sum and comparison selection. Audited geometric assembly -> full original frozen split-score chain.',
  training='BCE existence + 5*MSE normalized root location only on true-root examples; no iterative inference or score queries to predict roots.',protected_hashes=protected)
 (OUT/'config.json').write_text(json.dumps(config,indent=2))
 for stage in range(1,5):
  if stage>1:model.load_state_dict(torch.load(manifests[-1]['checkpoint'],weights_only=False)['state_dict'])
  train=make_data(stage,1024 if stage<=2 else 4096,'train');val=make_data(stage,256 if stage<=2 else 1024,'selection');test=make_data(stage,512 if stage<=2 else 1536,'heldout')
  for name,rows in [('train',train),('selection',val),('heldout',test)]:(OUT/f'stage_{stage}_{name}.json').write_text(json.dumps(rows))
  print(json.dumps(dict(event='data',stage=stage,train=len(train),train_roots=sum(r['exists'] for r in train),test_roots=sum(r['exists'] for r in test),seconds=time.monotonic()-started)),flush=True)
  x=features(train);v=features(val);exists=torch.tensor([r['exists'] for r in train],dtype=torch.float64);target=torch.tensor([r['target_z'] for r in train],dtype=torch.float64);vy=torch.tensor([r['exists'] for r in val],dtype=torch.float64);vz=torch.tensor([r['target_z'] for r in val],dtype=torch.float64)
  optimizer=torch.optim.Adam(model.parameters(),lr=.002);best=None;bestloss=np.inf;beststep=0;anchor=np.inf;last=0;cap=400 if stage<=2 else 6000
  for step in range(1,cap+1):
   ids=torch.from_numpy(rng.integers(len(train),size=256));logit,z=model(x[ids]);mask=exists[ids]>0
   loss=nn.functional.binary_cross_entropy_with_logits(logit,exists[ids])+(5*((z[mask]-target[ids][mask])**2).mean() if mask.any() else 0.)
   optimizer.zero_grad();loss.backward();optimizer.step()
   if step%100==0:
    with torch.no_grad():vl,vp=model(v);vm=vy>0;vloss=float(nn.functional.binary_cross_entropy_with_logits(vl,vy)+(5*((vp[vm]-vz[vm])**2).mean() if vm.any() else 0.))
    if vloss<bestloss:bestloss=vloss;best=copy.deepcopy(model.state_dict());beststep=step
    if bestloss<anchor*.99:anchor=bestloss;last=step
    history.append(dict(stage=stage,step=step,validation_loss=vloss,best_loss=bestloss,last_progress=last));
    if step%1000==0:print(json.dumps(dict(event='train',stage=stage,step=step,validation_loss=vloss,seconds=time.monotonic()-started)),flush=True)
    if stage<=2 and np.all((vl.numpy()>0)==vy.numpy().astype(bool)):break
    if stage>=3 and step>=2000 and step-last>=1500:break
  model.load_state_dict(best);cp=OUT/f'stage_{stage}_best.pt';h=save_checkpoint(cp,dict(state_dict=best,stage=stage,steps=step,best_step=beststep,status='UNPROMOTED',config=config,previous_checkpoint=manifests[-1] if manifests else None))
  manifests.append(dict(stage=stage,checkpoint=str(cp),sha256=h));(OUT/'checkpoint_manifest.json').write_text(json.dumps(manifests,indent=2))
  records,pred,z=evaluate_generic(model,test,audit);m=root_metrics(test,pred,z);summary=dict(stage=stage,steps=step,best_step=beststep,**m,generic_minimum_recovery=float(np.mean([r['minimum_recovered'] for r in records])),generic_worst_scaled_penalty=max(r['scaled_penalty'] for r in records),checkpoint_sha256=h);summaries.append(summary)
  write_csv(OUT/f'stage_{stage}_heldout_predictions.csv',records)
  for field in ('coefficient_range','orientation','root_count'):
   for value in sorted(set(r[field] for r in test),key=str):
    ids=np.array([i for i,r in enumerate(test) if r[field]==value]);rr=[test[i] for i in ids];breakdown.append(dict(stage=stage,grouping=field,group=value,**root_metrics(rr,pred[ids],z[ids])))
  for name,ids in [('endpoint_near',np.array([i for i,r in enumerate(records) if r['endpoint_near_root']],int)),('no_root',np.array([i for i,r in enumerate(test) if not r['exists']],int))]:
   if len(ids):breakdown.append(dict(stage=stage,grouping='special',group=name,**root_metrics([test[i] for i in ids],pred[ids],z[ids])))
  write_csv(OUT/'stationary_solver_curriculum.csv',summaries);write_csv(OUT/'stationary_solver_breakdown.csv',breakdown);write_csv(OUT/'training_history.csv',history)
  print(json.dumps(dict(event='stage_complete',**summary)),flush=True)
 assembly=audited_assembly(model,audit)
 last=summaries[-1];passed=last['existence_accuracy']>=.995 and last['generic_minimum_recovery']>=.995 and last['generic_worst_scaled_penalty']<=1e-12 and assembly['interval_recovery']==1 and assembly['interval_worst_relative_penalty']<=1e-12
 assert all(sha(p)==h for p,h in protected.items())
 result=dict(status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',operator='split_subpiece_stationary_solver',passed=bool(passed),stages=summaries,assembly=assembly,protected_hashes_unchanged=True,no_sho_training_data=True,no_recursion=True,no_search=True,checkpoints=manifests,seconds=time.monotonic()-started)
 (OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2));(OUT/'stationary_solver_summary.txt').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
