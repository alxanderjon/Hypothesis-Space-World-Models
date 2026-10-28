"""Read-only fixed-allocation interval and score-branch audit. No training/search."""
import json,time,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq
from experiments.sho.audit_deterministic_cuts import ROOT,SOURCE,CUTS,bucket,local_minima
from experiments.sho.train_recursive_region_splitter_poc import LockedAllocation
from experiments.sho.train_finite_split_score_smoke import load_frozen,FiniteSplitScore,upstream,predict
from experiments.sho.train_split_validity_smoke import ValidityMargin,ValidityThreshold,route
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv
OUT=ROOT/'fixed_allocation_interval_audit'

class Audit:
 def __init__(self):
  self.models,self.execution,_=load_frozen();self.allocator=LockedAllocation()
  lock=torch.load(ROOT/'split_validity_smoke/split_score_operator_locked.pt',weights_only=False)
  self.finite=FiniteSplitScore();self.finite.load_state_dict(torch.load(lock['components']['finite_score']['checkpoint'],weights_only=False)['state_dict'])
  self.gate=nn.ModuleDict(dict(margin=ValidityMargin(),threshold=ValidityThreshold()));self.gate.load_state_dict(torch.load(lock['components']['validity']['checkpoint'],weights_only=False)['state_dict'])
  self.finite.eval().requires_grad_(False);self.gate.eval().requires_grad_(False)
  self.score_count=0;self.max_formula_error=0
 def setup(self,r,axis):
  self.r=r;self.axis=axis;self.lo=np.array(r['lower']);self.hi=np.array(r['upper']);self.s=self.hi-self.lo;self.n=r['n'];self.d=r['d']
  self.a=self.s[axis];others=np.delete(self.s,axis);self.B=float(np.sum(others**2));self.h=float(np.exp((np.log(np.prod(self.s))-np.log(self.n))/self.d))
  self.omax=float(others.max()) if len(others) else 0;self.omin=float(others.min()) if len(others) else np.inf
 def geometry(self,c):
  c=np.atleast_1d(c);ell=np.broadcast_to(self.s,(len(c),2,self.d)).copy();coord=self.lo[self.axis]+c*self.a
  ell[:,0,self.axis]=coord-self.lo[self.axis];ell[:,1,self.axis]=self.hi[self.axis]-coord
  return ell
 def counts(self,c):
  ell=self.geometry(c);return self.allocator.allocate(ell/self.s,np.full(len(ell),self.n))[0]
 def scores(self,c):
  c=np.atleast_1d(c);ell=self.geometry(c);cc=self.allocator.allocate(ell/self.s,np.full(len(c),self.n))[0]
  _,_,_,valid=route(self.execution,self.gate['margin'],self.gate['threshold'],cc)
  out=np.full(len(c),np.inf)
  if valid.any():
   out[valid]=predict(self.finite,upstream(self.models,self.execution,np.tile(self.s,(valid.sum(),1)),np.full(valid.sum(),self.n),ell[valid],cc[valid]))
   self.score_count+=int(valid.sum())
  return out,cc
 def formula(self,c,k):
  c=np.atleast_1d(c);t=np.stack((c,1-c),1);counts=np.array([k,self.n-k]);roots=counts**(1/self.d)
  q=np.sqrt(self.B+(self.a*t)**2)/roots
  if self.d==1:logs=np.zeros_like(t);sg=np.zeros_like(t)
  else:
   v=self.a*t;logs=np.log(np.maximum(v,self.omax)/np.minimum(v,self.omin));sg=np.where(v>self.omax,1.,np.where(v<self.omin,-1.,0.))
  iq=q.argmax(1);ia=logs.argmax(1);idx=np.arange(len(c));sign=np.array([1.,-1.]);tj=t[idx,iq]
  dq=sign[iq]*self.a**2*tj/(roots[iq]*np.sqrt(self.B+(self.a*tj)**2))/self.h
  ddq=self.a**2*self.B/(roots[iq]*(self.B+(self.a*tj)**2)**1.5)/self.h
  da=sg[idx,ia]*sign[ia]/t[idx,ia];dda=-sg[idx,ia]/t[idx,ia]**2
  return q[idx,iq]/self.h+.05*logs[idx,ia],dq+.05*da,ddq+.05*dda,iq,ia,sg, q,logs
 def breaks(self,l,u,k):
  pts=[l,u]
  if l<.5<u:pts.append(.5)
  if self.d>1:
   for v in (self.omin/self.a,self.omax/self.a):
    for x in (v,1-v):
     if l<x<u:pts.append(x)
  base=sorted(set(pts));extra=[]
  for kind in (6,7):
   for a,b in zip(base[:-1],base[1:]):
    def f(x):
     val=self.formula(x,k)[kind][0];return val[0]-val[1]
    fa,fb=f(a),f(b)
    if fa*fb<0:extra.append(brentq(f,a,b,xtol=5e-324,rtol=1e-15))
  return np.array(sorted(set(base+extra)))

def run():
 start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
 protected={str(p):sha(p) for p in ROOT.glob('*/*.pt')};audit=Audit()
 records=[]
 for group in ('train','heldout'):
  for i,r in enumerate(json.loads((SOURCE/(group+'_regions.json')).read_text())):records.append(dict(r,source=group,source_case=i,reason=''))
 old=np.load(ROOT/'split_quality_smoke/previous_n4_failure.npz');records.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist(),source='n4_failure',source_case=0,reason='previous n=4 failure'))
 n3=json.loads((SOURCE/'worst_case.json').read_text())['result']['case']+1280;records[n3]['reason']='previous n=3 failure'
 ids={n3,len(records)-1}
 for d in range(1,6):
  for n in list(range(2,21))+[29,73]:
   found=[i for i,r in enumerate(records) if r['source']=='heldout' and r['d']==d and r['n']==n]
   if found:ids.add(found[0])
 saved=np.load(ROOT/'deterministic_cut_audit/audited_dense_menus.npz')['scores']
 intervals=[];pieces=[];recover=[];transitions=[];axisrows=[];checks=[]
 for pi,ri in enumerate(sorted(ids)):
  r=records[ri];d=r['d'];n=r['n'];dense=saved[ri,:d*199];ref=float(dense.min());tol=1e-12*max(1,ref)
  all_cont=[];all_grid=[];requests=0;finite_requests=0;count_start=audit.allocator.calls
  for axis in range(d):
   audit.setup(r,axis);meta=dict(parent=ri,dimension=d,budget=n,budget_bucket=bucket(n),axis=axis,reason=r['reason'],aspect=float(audit.s.max()/audit.s.min()))
   # At machine endpoints, a physical child may round to zero. Work only on representably positive boxes.
   e=max(1e-15,4*np.spacing(abs(audit.lo[axis])+abs(audit.hi[axis]))/audit.a);edge=e;end=1-e
   base=np.linspace(edge,end,max(2049,32*n+1));bc=audit.counts(base)[:,0];assert np.all(np.diff(bc)>=0)
   assert bc[0]==0 and bc[-1]==n
   low=np.array([base[np.flatnonzero(bc<k)[-1]] for k in range(1,n+1)]);high=np.array([base[np.flatnonzero(bc>=k)[0]] for k in range(1,n+1)])
   for _ in range(60):
    mid=(low+high)/2;val=audit.counts(mid)[:,0];left=val<np.arange(1,n+1);low=np.where(left,mid,low);high=np.where(left,high,mid)
   # Audit a float neighborhood for threshold reversals hidden by bisection.
   probes=[]
   for lo,hi in zip(low,high):
    below=lo;above=hi
    for _ in range(16):below=np.nextafter(below,-np.inf);above=np.nextafter(above,np.inf);probes.extend([below,above])
   probes=np.unique(probes);pc=audit.counts(probes)[:,0];float_reversals=int(np.sum(np.diff(pc)<0))
   for j in range(n):transitions.append(dict(**meta,left_count=j,right_count=j+1,last_lower_count_cut=float(low[j]),first_upper_count_cut=float(high[j]),bracket_width=float(high[j]-low[j]),mirror_boundary_error=float(abs(high[j]-(1-low[n-j-1])))))
   # dense exact locked scores across every VALID fixed-count interval in one batched call per axis
   grids=[];ir=[];offset=0
   for k in range(n+1):
    l=edge if k==0 else high[k-1];u=end if k==n else low[k]
    row=dict(**meta,interval=k,count_left=k,count_right=n-k,lower=float(l),upper=float(u),width=float(u-l),valid=bool(0<k<n),mirror_width_error=float(abs((u-l)-((end if n-k==n else low[n-k])-(edge if n-k==0 else high[n-k-1])))),float_boundary_reversals=float_reversals)
    if k==0 or k==n:intervals.append(row);continue
    # Keep actual boundary floats: bisection already checked their count ownership.
    grid=np.linspace(l,u,513);br=audit.breaks(l,u,k);grid=np.unique(np.r_[grid,br]);grids.extend(grid);ir.append((row,k,l,u,br,grid,offset));offset+=len(grid)
   gs=np.asarray(grids);scores,cc=audit.scores(gs);expected=np.concatenate([np.full(len(t[5]),t[1]) for t in ir]);bad=cc[:,0]!=expected
   if bad.any():
    # retain and explicitly report micro-boundary exceptions, rather than feeding wrong counts to a fixed-count claim.
    checks.append(dict(**meta,boundary_sample_count_mismatches=int(bad.sum())))
   opt_candidates=[];grid_candidates=[]
   for row,k,l,u,br,grid,offset in ir:
    vals=scores[offset:offset+len(grid)];valid=cc[offset:offset+len(grid),0]==k
    formula=audit.formula(grid,k);err=abs(vals[valid]-formula[0][valid]);audit.max_formula_error=max(audit.max_formula_error,float(err.max(initial=0)))
    v=vals[valid];x=grid[valid];diff=np.diff(v);dtol=1e-12*max(1,float(np.min(v)));sg=np.sign(np.where(abs(diff)>dtol,diff,0));nz=sg[sg!=0]
    cls='flat' if len(nz)==0 else 'increasing' if np.all(nz>0) else 'decreasing' if np.all(nz<0) else 'unimodal' if np.all(np.diff(nz)>=0) else 'non-unimodal'
    lm=local_minima(v,dtol);loc=int(np.argmin(v));bestcut=float(x[loc]);qchanges=np.flatnonzero(np.diff(formula[3])!=0);achanges=np.flatnonzero(np.diff(formula[4])!=0)
    row.update(samples=len(grid),count_mismatch_samples=int((~valid).sum()),local_minima=len(lm),classification=cls,sampled_min_cut=bestcut,sampled_min_score=float(v[loc]),minimum_location='lower_endpoint' if loc==0 else 'upper_endpoint' if loc==len(v)-1 else 'interior',radius_active_children=json.dumps(np.unique(formula[3]).tolist()),aspect_active_children=json.dumps(np.unique(formula[4]).tolist()),radius_change_brackets=json.dumps([[float(grid[j]),float(grid[j+1])] for j in qchanges]),aspect_change_brackets=json.dumps([[float(grid[j]),float(grid[j+1])] for j in achanges]),analytic_breakpoints=json.dumps(br.tolist()),max_locked_vs_analytic_error=float(err.max(initial=0)))
    candidates=list(br);roots_all=[]
    for si,(a,b) in enumerate(zip(br[:-1],br[1:])):
     xx=np.linspace(np.nextafter(a,b),np.nextafter(b,a),129);f=audit.formula(xx,k);der=f[1];roots=[]
     for j in np.flatnonzero(der[:-1]*der[1:]<0):roots.append(brentq(lambda z:float(audit.formula(z,k)[1][0]),xx[j],xx[j+1],xtol=5e-324,rtol=1e-15))
     roots_all.extend(roots);candidates.extend(roots)
     signs=np.sign(der[abs(der)>1e-10]);quasi=bool(len(signs)==0 or np.all(np.diff(signs)>=0));negpos=int(np.sum((der[:-1]<0)&(der[1:]>0)));posneg=int(np.sum((der[:-1]>0)&(der[1:]<0)))
     tiny=min(1e-9,(b-a)*1e-5);left=float(audit.formula(a+tiny,k)[0][0]);right=float(audit.formula(b-tiny,k)[0][0])
     pieces.append(dict(**meta,interval=k,piece=si,lower=float(a),upper=float(b),radius_active_child=int(f[3][len(xx)//2]),aspect_active_child=int(f[4][len(xx)//2]),aspect_log_power=int(f[5][len(xx)//2,f[4][len(xx)//2]]),derivative_min=float(der.min()),derivative_max=float(der.max()),second_derivative_min=float(f[2].min()),second_derivative_max=float(f[2].max()),sampled_convex=bool(f[2].min()>=-1e-10),sampled_quasiconvex=quasi,stationary_points=json.dumps(roots),interior_minima=negpos,interior_maxima=posneg,derivative_samples=129))
    # Every extremum and piece endpoint, then adjacent discrete reference cuts. No cached score is inspected.
    ca=np.unique(candidates);opt_candidates.extend(ca)
    for c in ca:
     for j in (math.floor(c*200),math.ceil(c*200)):
      if 1<=j<=199 and l<=j/200<=u:grid_candidates.append(j/200)
    row['stationary_points']=json.dumps(roots_all);intervals.append(row)
   oc=np.unique(opt_candidates);gc=np.unique(grid_candidates)
   vs,cs=audit.scores(oc);vg,cg=audit.scores(gc);requests+=len(oc)+len(gc);finite_requests+=int(np.isfinite(vs).sum()+np.isfinite(vg).sum());all_cont.extend((float(s),axis,float(c)) for s,c in zip(vs,oc));all_grid.extend((float(s),axis,float(c)) for s,c in zip(vg,gc))
   axisrows.append(dict(**meta,intervals=n+1,valid_intervals=n-1,allocator_monotone_on_scan=True,boundary_float_reversals=float_reversals,endpoint_excluded_width=edge,continuous_candidates=len(oc),reference_grid_candidates=len(gc)))
  cmin=min(all_cont);gmin=min(all_grid);good=[v for v in all_grid if v[0]<=gmin[0]+1e-12*max(1,gmin[0])];canonical=min(good,key=lambda v:(v[1],v[2]));densecanonical=int(np.flatnonzero(dense<=ref+tol)[0])
  recover.append(dict(parent=ri,dimension=d,budget=n,budget_bucket=bucket(n),reason=r['reason'],dense_best=ref,continuous_best=cmin[0],continuous_axis=cmin[1],continuous_cut=cmin[2],grid_recovered_best=gmin[0],grid_axis=canonical[1],grid_cut=canonical[2],equivalent_best=bool(gmin[0]<=ref+tol),canonical_recovered=bool(canonical[1]==densecanonical//199 and canonical[2]==CUTS[densecanonical%199]),relative_penalty=max(0.,gmin[0]/ref-1),continuous_relative_gain=1-cmin[0]/ref,optimization_score_requests=requests,optimization_finite_score_evaluations=finite_requests,continuous_score_requests=sum(x['continuous_candidates'] for x in axisrows[-d:]),grid_score_requests=sum(x['reference_grid_candidates'] for x in axisrows[-d:]),dense_reference_requests=d*199,allocator_calls_including_audit=audit.allocator.calls-count_start))
  if pi%5==0:print(json.dumps(dict(parents=pi+1,total=len(ids),intervals=len(intervals),seconds=time.monotonic()-start)),flush=True)
 write_csv(OUT/'fixed_allocation_interval_audit.csv',intervals);write_csv(OUT/'fixed_interval_score_structure.csv',pieces);write_csv(OUT/'optimum_recovery.csv',recover);write_csv(OUT/'allocation_transition_brackets.csv',transitions);write_csv(OUT/'axis_summary.csv',axisrows)
 write_csv(OUT/'boundary_exceptions.csv',checks or [dict(boundary_sample_count_mismatches=0)])
 assert all(sha(p)==h for p,h in protected.items())
 valid=[r for r in intervals if r['valid']];summary=dict(parents=len(ids),axes=len(axisrows),intervals=len(intervals),valid_intervals=len(valid),pieces=len(pieces),classification={s:sum(r['classification']==s for r in valid) for s in ('flat','increasing','decreasing','unimodal','non-unimodal')},max_interval_minima=max(r['local_minima'] for r in valid),nonconvex_pieces=sum(not r['sampled_convex'] for r in pieces),nonquasiconvex_pieces=sum(not r['sampled_quasiconvex'] for r in pieces),multiple_interior_minima_pieces=sum(r['interior_minima']>1 for r in pieces),boundary_float_reversals=sum(r['boundary_float_reversals'] for r in axisrows),boundary_sample_mismatches=sum(r.get('boundary_sample_count_mismatches',0) for r in checks),max_locked_vs_analytic_error=audit.max_formula_error,equivalent_recovery=float(np.mean([r['equivalent_best'] for r in recover])),canonical_recovery=float(np.mean([r['canonical_recovered'] for r in recover])),worst_penalty=max(r['relative_penalty'] for r in recover),average_continuous_score_requests=float(np.mean([r['continuous_score_requests'] for r in recover])),average_grid_score_requests=float(np.mean([r['grid_score_requests'] for r in recover])),average_dense_requests=float(np.mean([r['dense_reference_requests'] for r in recover])),protected_hashes_unchanged=True,seconds=time.monotonic()-start)
 (OUT/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
