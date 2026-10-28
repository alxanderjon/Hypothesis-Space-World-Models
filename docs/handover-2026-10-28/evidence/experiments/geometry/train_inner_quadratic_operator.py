"""Synthetic-only sparse polynomial arithmetic smoke: raw (B,a,t) -> q.

Complete degree <=4 polynomial dictionary, all coefficients random. Generic L1
coordinate descent discovers support, followed by unpenalized supervised fitting
on that learned support. No term index or coefficient is specified as a target.
"""
import itertools,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv

ROOT=Path('outputs/sho/stage1B');OUT=ROOT/'inner_quadratic_operator_smoke'
SEED=27342026;EPS=np.finfo(np.float64).eps
POWERS=tuple(p for degree in range(5) for p in itertools.product(range(degree+1),repeat=3) if sum(p)==degree)
GATES=dict(max_relative_error=256*EPS,symmetry_max_absolute_error=0.,scaling_max_relative_error=512*EPS,zero_target_max_absolute_error=0.)

class PolynomialArithmetic(nn.Module):
 def __init__(self):
  super().__init__();self.weights=nn.Parameter(.2*torch.randn(len(POWERS),dtype=torch.float64));self.register_buffer('feature_scale',torch.ones(len(POWERS),dtype=torch.float64))
 def polynomial_features(self,x):
  # The sole network inputs are the three raw scalars. Generate the complete
  # generic basis internally, without privileged or selected teacher terms.
  powers=[torch.ones_like(x)]
  for _ in range(4):powers.append(powers[-1]*x)
  return torch.stack([powers[i][:,0]*powers[j][:,1]*powers[k][:,2] for i,j,k in POWERS],1)
 def forward(self,x):
  return self.polynomial_features(x)@(self.weights/self.feature_scale)

def labels(x):
 # Preserve teacher float64 order exactly; labels only, not network inputs.
 return x[:,0]+(x[:,1]*x[:,1])*(x[:,2]*x[:,2])

def samples(n,seed,group):
 rng=np.random.default_rng(seed);rows=[]
 while len(rows)<n:
  B=10**rng.uniform(-12,12);a=10**rng.uniform(-4,4);t=10**rng.uniform(-5,0)
  if group=='train':
   if 1e-4<=B<=1e-3 or 1e3<=B<=1e4 or .125<=a<=.25 or 2<=a<=4 or .137<=t<=.173 or .6<=t<=.7:continue
  elif group=='reserved_B':B=10**rng.uniform(*((-4,-3) if len(rows)%2 else (3,4)))
  elif group=='reserved_a':a=rng.uniform(*((.125,.25) if len(rows)%2 else (2,4)))
  elif group=='reserved_t':t=rng.uniform(*((.137,.173) if len(rows)%2 else (.6,.7)))
  elif group=='B_extrapolation':B=10**rng.uniform(*((-20,-12.1) if len(rows)%2 else (12.1,20)))
  elif group=='a_extrapolation':a=10**rng.uniform(*((-8,-4.1) if len(rows)%2 else (4.1,8)))
  elif group=='t_extrapolation':t=10**rng.uniform(*((-10,-5.1) if len(rows)%2 else (.1,1)))
  elif group=='joint_extrapolation':B=10**rng.uniform(-20,20);a=10**rng.uniform(-8,8);t=10**rng.uniform(-10,1)
  elif group=='balanced_terms':
   B=(a*a)*(t*t)*10**rng.uniform(-2,2)
  elif group=='zeros':
   mode=len(rows)%7
   if mode in (0,3,4,6):B=0.
   if mode in (1,3,5,6):a=0.
   if mode in (2,4,5,6):t=0.
  if group=='train':
   # Include zero boundaries without encoding symmetry through augmentation.
   mode=len(rows)%23
   if mode in (0,1,2):B=0.
   if mode in (1,3):a=0.
   if mode in (2,4):t=0.
  a*=rng.choice([-1,1]);t*=rng.choice([-1,1]);rows.append((B,a,t))
 return np.array(rows,dtype=np.float64)

def metric(y,p):
 e=p-y;positive=y>0;rel=e[positive]/y[positive];zero=~positive
 return dict(examples=len(y),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),max_error=float(abs(e).max()),relative_rmse=float(np.sqrt(np.mean(rel*rel))) if len(rel) else None,relative_mae=float(np.mean(abs(rel))) if len(rel) else None,max_relative_error=float(abs(rel).max()) if len(rel) else None,zero_targets=int(zero.sum()),zero_target_max_error=float(abs(e[zero]).max()) if zero.any() else 0.,teacher_min=float(y.min()),teacher_max=float(y.max()),bit_exact=int(np.sum(y.view(np.uint64)==p.view(np.uint64))))

def predict(model,x):
 with torch.no_grad():return model(torch.from_numpy(x)).numpy()

def bucket(x):
 a=abs(x)
 if a==0:return '0'
 if a<1e-8:return '(0,1e-8)'
 if a<1e-4:return '[1e-8,1e-4)'
 if a<1:return '[1e-4,1)'
 if a<1e4:return '[1,1e4)'
 if a<1e8:return '[1e4,1e8)'
 if a<1e12:return '[1e8,1e12)'
 return '>=1e12'

def run():
 started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
 OUT.mkdir(parents=True,exist_ok=False);protected={str(p):sha(p) for p in ROOT.glob('*/*.pt')}
 model=PolynomialArithmetic();initial=model.weights.detach().numpy().copy();x=samples(4096,SEED,'train');y=labels(x)
 np.savez_compressed(OUT/'training_data.npz',inputs=x,targets=y)
 with torch.no_grad():phi=model.polynomial_features(torch.from_numpy(x)).numpy()
 denom=y.copy();denom[denom==0]=1.;R=phi/denom[:,None];target=y/denom
 scale=np.sqrt(np.mean(R*R,axis=0));assert np.all(scale>0);Z=R/scale
 with torch.no_grad():model.feature_scale.copy_(torch.from_numpy(scale))
 lam=1e-5;w=initial.copy();history=[];gram=Z.T@Z/len(x);rhs=Z.T@target/len(x)
 config=dict(seed=SEED,dtype='float64',raw_inputs=['B','a','t'],domain='B>=0; signed a and t including zero',training_examples=len(x),basis='Every monomial in 3 raw variables of total degree <=4, generated inside the model',basis_exponents=POWERS,coefficient_count=len(POWERS),initial_weights=initial.tolist(),
  optimizer='Random-initialized cyclic proximal coordinate descent, L1=1e-5 in relative-loss-conditioned coordinates; supervised least-squares debias on learned nonzero support',
  loss='Squared relative residual for positive targets; ordinary absolute residual on exact-zero targets. No epsilon in teacher or inference.',
  term_selection='Generic soft threshold on every coefficient; no selected index, sign, exponent or teacher coefficient is hard-coded. No coefficient rounding or snapping after fit.',
  preconditioning='Per-column RMS of training design divided by positive target, zero-target divisor=1; stored buffer. Inference takes only raw B,a,t.',gates=GATES,protected_hashes=protected)
 (OUT/'config.json').write_text(json.dumps(config,indent=2))
 for sweep in range(1,10001):
  old=w.copy()
  for j in range(len(w)):
   value=w[j]+(rhs[j]-gram[j]@w)/gram[j,j];threshold=lam/gram[j,j]
   w[j]=np.sign(value)*max(abs(value)-threshold,0.)
  if sweep==1 or sweep%25==0:
   residual=Z@w-target;history.append(dict(sweep=sweep,relative_rmse=float(np.sqrt(np.mean(residual**2))),active_coefficients=int(np.count_nonzero(w)),largest_update=float(abs(w-old).max())))
  if abs(w-old).max()<1e-13:break
 support=np.flatnonzero(w!=0);assert len(support)>0
 # Generic debias: learn magnitudes using only the selected training columns.
 fitted=np.linalg.lstsq(Z[:,support],target,rcond=None)[0];w[:]=0.;w[support]=fitted
 # Residual correction solves the same supervised system; never assign teacher
 # values. This improves numerical conditioning without rounding coefficients.
 for _ in range(3):
  residual=target-Z[:,support]@w[support];delta=np.linalg.lstsq(Z[:,support],residual,rcond=None)[0]
  proposal=w.copy();proposal[support]+=delta
  if np.linalg.norm(target-Z@proposal)<np.linalg.norm(target-Z@w):w=proposal
  else:break
 with torch.no_grad():model.weights.copy_(torch.from_numpy(w))
 train_pred=predict(model,x);train_stats=metric(y,train_pred);coef=(model.weights/model.feature_scale).detach().numpy()
 learned=[dict(exponents=list(POWERS[j]),coefficient=float(coef[j]),normalized_weight=float(w[j])) for j in support]
 checkpoint=OUT/'inner_quadratic_before_evaluation.pt';digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),status='SAVED_BEFORE_EVALUATION',config=config,initial_weights=initial.tolist(),learned_terms=learned,training_metrics=train_stats,sweeps=sweep))
 write_csv(OUT/'training_history.csv',history);write_csv(OUT/'learned_coefficients.csv',[dict(B_power=i,a_power=j,t_power=k,coefficient=float(c),active=bool(c!=0)) for (i,j,k),c in zip(POWERS,coef)])
 print(json.dumps(dict(event='trained_and_saved',sweeps=sweep,learned_terms=learned,training=train_stats,hash=digest)),flush=True)
 model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval()
 groups=['new_combinations','reserved_B','reserved_a','reserved_t','B_extrapolation','a_extrapolation','t_extrapolation','joint_extrapolation','balanced_terms','zeros'];xx=[];gg=[]
 for i,g in enumerate(groups):
  q=samples(2048 if g!='zeros' else 700,SEED+100+i,g);xx.append(q);gg.extend([g]*len(q))
 vx=np.concatenate(xx);vy=labels(vx);vp=predict(model,vx);gg=np.array(gg);np.savez_compressed(OUT/'heldout_data.npz',inputs=vx,targets=vy,predictions=vp,group=gg)
 rows=[dict(grouping='all',group='all',**metric(vy,vp))]
 for g in groups:
  ids=gg==g;rows.append(dict(grouping='data_family',group=g,**metric(vy[ids],vp[ids])))
 for j,name in enumerate(['B','abs_a','abs_t']):
  bb=np.array([bucket(v) for v in vx[:,j]])
  for b in sorted(set(bb)):
   ids=bb==b;rows.append(dict(grouping=name,group=b,**metric(vy[ids],vp[ids])))
 write_csv(OUT/'inner_quadratic_validation.csv',rows)
 invariants=[]
 for axis,name in [(1,'a_sign'),(2,'t_sign')]:
  changed=vx.copy();changed[:,axis]*=-1;got=predict(model,changed);e=got-vp
  invariants.append(dict(test=name,examples=len(got),max_absolute_error=float(abs(e).max()),bit_exact=int(np.sum(got.view(np.uint64)==vp.view(np.uint64))),max_relative_error=float(np.max(abs(e[vp!=0]/vp[vp!=0]))),zero_expected_error=float(abs(e[vp==0]).max())))
 rng=np.random.default_rng(SEED+900);base=samples(4096,SEED+901,'joint_extrapolation');base[:,0]=0.;baseline=predict(model,base);k=rng.choice([-1.,1.],len(base))*10**rng.uniform(-6,6,len(base));k[::101]=0
 for axis,name in [(1,'a_scaling'),(2,'t_scaling')]:
  changed=base.copy();changed[:,axis]*=k;got=predict(model,changed);expected=(k*k)*baseline;e=got-expected;active=expected!=0
  invariants.append(dict(test=name,examples=len(got),max_absolute_error=float(abs(e).max()),bit_exact=int(np.sum(got.view(np.uint64)==expected.view(np.uint64))),max_relative_error=float(np.max(abs(e[active]/expected[active]))),zero_expected_error=float(abs(e[~active]).max())))
 write_csv(OUT/'inner_quadratic_symmetry_scaling.csv',invariants)
 m=rows[0];passed=m['max_relative_error']<=GATES['max_relative_error'] and m['zero_target_max_error']==0 and all(r['max_absolute_error']==0 for r in invariants[:2]) and all(r['max_relative_error']<=GATES['scaling_max_relative_error'] and r['zero_expected_error']==0 for r in invariants[2:])
 assert all(sha(p)==h for p,h in protected.items())
 result=dict(operator='inner_quadratic_operator',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',passed=bool(passed),checkpoint=str(checkpoint),sha256=digest,learned_terms=learned,training_metrics=train_stats,heldout_metrics=m,symmetry_scaling=invariants,protected_hashes_unchanged=True,sweeps=sweep,seconds=time.monotonic()-started)
 (OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2))
 lines=['Inner quadratic arithmetic smoke: '+result['status'],
  'Only this new polynomial arithmetic network was trained. No sqrt, A scaling, logarithm, derivative, stationary solver, recursion, search, or existing operator was trained/changed.',
  'Training consumed only synthetic raw B,a,t with labels B+(a*a)*(t*t). B>=0; signs of a and t varied; zeros included. No derived products/powers were supplied as inputs.',
  'The network internally enumerates all 35 monomials of total degree <=4; all 35 output weights initialized independently at random. L1 optimization learned a sparse support. Debiasing learned its coefficient magnitudes. No term index was supplied and no coefficient was rounded/snapped to a teacher value.',
  'The basis degree is an architectural prior; quadratic dependence is represented by learned term selection rather than learned continuous exponents.',
  'Training: B 1e-12..1e12; |a| 1e-4..1e4; |t| 1e-5..1. Entire reserved B/a/t bands excluded. Heldout: new combinations, all reserved bands, magnitude extrapolation to B 1e-20..1e20, |a| 1e-8..1e8, |t| 1e-10..10, balanced terms and zero cases.',
  'Absolute errors scale with q magnitude; numerical-precision gate uses relative error on every positive target and exact zero recovery on zero targets.',
  'Checkpoint persisted before heldout evaluation. Candidate/diagnostic status is in the manifest; all original checkpoint hashes unchanged.',
  '',json.dumps(result,indent=2),'','RANGE BREAKDOWN:']
 for r in rows:lines.append(json.dumps(r))
 (OUT/'inner_quadratic_summary.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
