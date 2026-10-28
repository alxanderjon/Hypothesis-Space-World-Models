import csv,json
from pathlib import Path
import numpy as np, torch
from experiments.geometry import audit_exact_branch_brent as audit
from experiments.geometry.train_reciprocal_operator import PositivePower
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
O=ROOT/'division_operator_training';O.mkdir(exist_ok=False)
torch.set_num_threads(1)
protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('manifest','locked','validated'))})
(O/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
pairs=[];meta=[];linear=[];lm=[]
splits={int(r['case_id']):r['group'] for r in audit.read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
def collect(r,i,group):
 if not r['eligible']:return
 s=r['normalized_state'];fc,fp,fb=s['fcur'],s['fpre'],s['fblk'];xc,xp,xb=s['xcur'],s['xpre'],s['xblk']
 if r['attempt_method']=='LINEAR_INTERPOLATION':
  nums=[(-fc*(xc-xp),fc-fp,'linear')];linear.append([fc,fp,xc,xp]);lm.append(dict(case_id=i,iteration=r['iteration'],group=group,teacher_step=r['trial_step']))
 else:
  dp=r['dpre'];db=r['dblk'];nums=[(fp-fc,xp-xc,'iq_dpre'),(fb-fc,xb-xc,'iq_dblk'),(-fc*(fb*db-fp*dp),db*dp*(fb-fp),'iq_final')]
 for a,b,name in nums:
  pairs.append((a,b));meta.append(dict(case_id=i,iteration=r['iteration'],operation=name,group=group))
with (ROOT/'exact_branch_switch_brent_audit/brent_iterations.csv').open() as f:
 for raw in csv.DictReader(f):
  if raw['stop']=='True' or raw['eligible']!='True':continue
  row=dict(eligible=True,normalized_state=json.loads(raw['normalized_state']),attempt_method=raw['attempt_method'],iteration=int(raw['iteration']),trial_step=float(raw['trial_step']))
  for k in ('dpre','dblk'):row[k]=float(raw[k]) if raw.get(k) else None
  i=int(raw['case_id']);collect(row,i,splits[i])
archived=len(pairs);cases=[]
for folder,name in [('brent_interpolation_eligibility_training','new_fresh_candidates.json'),('brent_interpolation_branch_selector_training','new_fresh_candidates.json'),('subtraction_operator_training','fresh_candidates.json')]:cases+=json.loads((ROOT/folder/name).read_text())
a=audit.Audit.__new__(audit.Audit)
for j,r in enumerate(cases):
 a.setup(r['parent'],r['axis']);k=r['k']
 def callback(c):return float(a.formula(c,k)[1][0]),{}
 root,trace,_,_=audit.traced_brent(callback,r['c_left'],r['c_right'],818+j);assert root==r['teacher_cut']
 for row in trace:collect(row,818+j,'additional_fresh')
x=np.array(pairs);assert np.isfinite(x).all() and np.all(x[:,1]!=0)
models=[];refs=[]
for folder,cls in [('multiplication_operator_smoke',QuadraticArithmetic),('reciprocal_required_domain_audit',PositivePower)]:
 m=json.loads((ROOT/folder/'candidate_manifest.json').read_text());assert sha(Path(m['checkpoint']))==m['sha256'];model=cls();model.load_state_dict(torch.load(m['checkpoint'],weights_only=False)['state_dict']);model.eval().requires_grad_(False);models.append(model);refs.append(m)
mult,rec=models
with torch.no_grad():
 t=torch.from_numpy(x);r=rec(t[:,1]);p=mult(torch.stack((t[:,0],r),1)).numpy()
y=x[:,0]/x[:,1];bad=p.view('u8')!=y.view('u8')
np.savez_compressed(O/'Brent_division_operands.npz',operands=x,teacher=y,reuse_predictions=p,linear_states=np.array(linear))
write_csv(O/'operand_metadata.csv',meta);write_csv(O/'linear_metadata.csv',lm)
write_csv(O/'reuse_failures.csv',[dict(**meta[i],numerator=x[i,0],denominator=x[i,1],teacher=y[i],learned=p[i],teacher_hex=y[i].hex(),learned_hex=p[i].hex()) for i in np.flatnonzero(bad)])
result=dict(total_cases=len(x),saved_cases=archived,saved_solves=818,additional_fresh_solves=len(cases),exact_matches=int((~bad).sum()),errors=int(bad.sum()),positive_denominator_errors=int((bad&(x[:,1]>0)).sum()),negative_denominator_errors=int((bad&(x[:,1]<0)).sum()),reuse_refs=refs,negative_domain_note='Negative denominators were not covered by the positive reciprocal manifest; unmodified checkpoint behavior evaluated explicitly, not assumed validated.')
(O/'reuse_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='reuse_refs'},indent=2))
