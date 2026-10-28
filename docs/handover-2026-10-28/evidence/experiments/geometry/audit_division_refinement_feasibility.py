import json
from pathlib import Path
import numpy as np
R=Path('outputs/sho/stage1B/division_operator_refinement')
z=np.load(R/'refinement_inputs.npz');x=z['inputs'];f=z['features'];e=z['exponents'];y=abs(z['teacher']);initial=json.loads((R/'results.json').read_text())['initial_weights']
anchor=np.flatnonzero((x[:,0]==1)&(x[:,1]==1))[0]
assert f[anchor,0]==1 and f[anchor,1]==0 and e[anchor]==0 and y[anchor]==1
scale=json.loads((R/'results.json').read_text())['feature_scale']
ix=np.flatnonzero(np.isfinite(y)&(y>0)&(f[:,1]!=0))
a=f[ix,0];b=f[ix,1];ex=e[ix];t=y[ix];positive=b>0
H=int(np.float64(2*scale).view('u8'))
def bound(strict):
 lo=np.zeros(len(ix),dtype=np.int64);hi=np.full(len(ix),H+1,dtype=np.int64)
 while np.any(lo<hi):
  mid=lo+(hi-lo)//2
  w=mid.astype(np.uint64).view(np.float64)
  with np.errstate(over='ignore',under='ignore'):pred=np.ldexp(a+w*b,ex)
  pred=np.where(positive,pred,-pred);target=np.where(positive,t,-t)
  ok=(pred>target) if strict else (pred>=target)
  hi=np.where(ok,mid,hi);lo=np.where(ok,lo,mid+1)
 return lo
low=bound(False);upper=bound(True)-1
lower_id=int(ix[np.argmax(low)]);upper_id=int(ix[np.argmin(upper)])
print('local_witnesses',lower_id,upper_id, 'empty',int(low.max())>int(upper.min()),flush=True)
# Exact exhaustive binary64 parameter-domain interval search for the small
# witness set. Division.predict takes copysign of the magnitude, so both raw
# positive and raw negative mantissa-output branches MUST be considered.
MASK=(1<<64)-1;SIGN=1<<63
MIN_RANK=0x0010000000000000;MAX_RANK=0xffefffffffffffff
def val(rank):
 bits=(rank^SIGN) if rank&SIGN else ((~rank)&MASK)
 return np.array(bits,dtype=np.uint64).view(np.float64).item()
def full_interval(i,sgn,w0=1.):
 aa,bb=map(float,f[i]);ee=int(e[i]);target=float(y[i])*sgn;inc=bb>0
 if not inc:target=-target
 def predicate(rank,strict):
  w=val(rank)
  with np.errstate(over='ignore',under='ignore',invalid='ignore'):
   q=np.float64(w0)*np.float64(aa)+np.float64(w)*np.float64(bb)
   out=np.ldexp(q,ee).item()
  if not inc:out=-out
  return out>target if strict else out>=target
 def search(strict):
  l,h=MIN_RANK,MAX_RANK+1
  while l<h:
   m=l+(h-l)//2
   if predicate(m,strict):h=m
   else:l=m+1
  return l
 l=search(False);u=search(True)-1
 if l>u or l>MAX_RANK or u<MIN_RANK:return None
 return (l,u)
def intersect(left,right):
 return [(max(a,c),min(b,d)) for a,b in left for c,d in right if max(a,c)<=min(b,d)]
chosen=[];feasible=[(MIN_RANK,MAX_RANK)]
for i in [lower_id,upper_id]:
 ints=[q for s in (1,-1) if (q:=full_interval(i,s)) is not None]
 feasible=intersect(feasible,ints)
 chosen.append((i,ints))
# If the two local-conflict witnesses share a far reflected branch, add
# ordinary retention constraints. This does not fit or change parameters.
if feasible:
 for i in ix[::max(1,len(ix)//100)]:
  i=int(i);ints=[q for s in (1,-1) if (q:=full_interval(i,s)) is not None]
  new=intersect(feasible,ints)
  if new!=feasible:chosen.append((i,ints));feasible=new
  if not feasible:break
rows=[]
for i,ints in chosen:
 rows.append(dict(index=i,numerator=float(x[i,0]),denominator=float(x[i,1]),teacher=float(z['teacher'][i]),q0=float(f[i,0]),correction_feature=float(f[i,1]),exponent=int(e[i]),
  intervals=[dict(lower_rank=str(l),upper_rank=str(u),lower=val(l),upper=val(u),lower_hex=val(l).hex(),upper_hex=val(u).hex()) for l,u in ints]))
negative_feasible=[(MIN_RANK,MAX_RANK)]
negative_rows=[]
for i,_ in chosen:
 ints=[q for s in (1,-1) if (q:=full_interval(i,s,-1.)) is not None]
 negative_feasible=intersect(negative_feasible,ints)
 negative_rows.append(dict(index=i,intervals=[dict(lower=val(l),upper=val(u),lower_hex=val(l).hex(),upper_hex=val(u).hex()) for l,u in ints]))
# Check interval edges against the same eager Torch arithmetic as inference.
import torch
edge_checks=0
for w0 in (1.,-1.):
 for i,_ in chosen:
  for s in (1,-1):
   interval=full_interval(i,s,w0)
   if interval is None:continue
   lo,hi=interval
   for rank,expected in ((lo,True),(hi,True),(lo-1,False),(hi+1,False)):
    if not MIN_RANK<=rank<=MAX_RANK:continue
    aa,bb=torch.tensor(f[i],dtype=torch.float64)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
     actual=np.ldexp((torch.tensor(w0,dtype=torch.float64)*aa+torch.tensor(val(rank),dtype=torch.float64)*bb).item(),int(e[i]))
    assert (actual==float(y[i])*s)==expected
    edge_checks+=1
result=dict(method='Exact monotone binary search of every finite binary64 w1, for both possible w0 values, including both positive/negative raw-output branches before copysign',
 w0_forced_by_exact_retention=dict(index=int(anchor),inputs=[1.,1.],features=f[anchor].tolist(),exponent=int(e[anchor]),target=1.,conclusion='Exact retention requires abs(w0)=1.0 because predict applies copysign; both +1 and -1 are audited. This is an analytical constraint, not a parameter assignment.'),
 witnesses=rows,intersection_empty=not feasible,remaining_intervals=[(str(a),str(b)) for a,b in feasible],
 negative_w0_witnesses=negative_rows,negative_w0_intersection_empty=not negative_feasible,
 all_finite_weight_pairs_infeasible=not feasible and not negative_feasible,torch_interval_edge_checks=edge_checks,
 explanation='For each possible w0, the raw output is monotone in w1. Each target allows up to two intervals after copysign. Empty intersections for both w0 values prove no pair of finite learned weights can satisfy the anchor and witness cases simultaneously.',
 training_performed=False,parameters_modified=False)
old_count=json.loads((R/'results.json').read_text())['prior_validation_cases']
original=ix<old_count
original_ids=[int(ix[original][np.argmax(low[original])]),int(ix[original][np.argmin(upper[original])])]
original_cases=[]
for i in original_ids:
 original_cases.append(dict(index=i,numerator=float(x[i,0]),denominator=float(x[i,1]),teacher=float(z['teacher'][i]),
  previously_exact=bool(z['before'][i].view('u8')==z['teacher'][i].view('u8')),
  q0=float(f[i,0]),correction_feature=float(f[i,1]),exponent=int(e[i])))
original_results=[]
for w0 in (1.,-1.):
 possible=[(MIN_RANK,MAX_RANK)];constraints=[]
 for i in original_ids:
  ints=[v for s in (1,-1) if (v:=full_interval(i,s,w0)) is not None]
  possible=intersect(possible,ints)
  constraints.append(dict(index=i,intervals=[dict(lower=val(a),upper=val(b),lower_hex=val(a).hex(),upper_hex=val(b).hex()) for a,b in ints]))
 original_results.append(dict(w0=w0,constraints=constraints,intersection_empty=not possible))
result['original_validation_only_witnesses']=original_cases
result['original_validation_only_feasibility']=original_results
result['original_validation_already_infeasible']=all(v['intersection_empty'] for v in original_results)
(R/'coefficient_feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
