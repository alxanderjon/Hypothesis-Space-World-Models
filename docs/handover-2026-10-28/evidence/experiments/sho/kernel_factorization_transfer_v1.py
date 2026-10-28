#!/usr/bin/env python3
import argparse, hashlib, json, os, resource, sys, time
from pathlib import Path
import numpy as np

DT=.05; CANDS=('joint','axis_additive','rotated_additive','pairwise')
def sha(p): h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def write(p,x): p=Path(p); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)); os.replace(t,p)
class Guard:
 def __init__(self,c): self.c=c; self.n={k:0 for k in c if k not in ('wall_seconds','peak_RSS_bytes','artifact_bytes')}; self.t=time.time()
 def charge(self,k,v=1):
  if time.time()-self.t>self.c['wall_seconds']: raise RuntimeError('WALL_CAP')
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
  if rss>self.c['peak_RSS_bytes']: raise RuntimeError('RSS_CAP')
  if self.n[k]+v>self.c[k]: raise RuntimeError(f'CAP:{k}:{self.n[k]}+{v}>{self.c[k]}')
  self.n[k]+=v
def states(r,n): return r.uniform(-1,1,(n,4))
def leg3(x): return (5*x**3-3*x)/2
def leg5(x): return (63*x**5-70*x**3+15*x)/8
def stable_A(r):
 a=r.normal(scale=.22,size=(4,4)); a-=np.eye(4)*(.45+.15*r.random()); return a
def orthogonal(seed):
 q,r=np.linalg.qr(np.random.default_rng(seed).normal(size=(4,4))); return q@np.diag(np.sign(np.diag(r)))
Q=orthogonal(2026106401)
def phi(x,kind):
 if kind=='axis_additive': return leg5(x)
 if kind=='rotated_additive': return leg5(x@Q)
 if kind=='pairwise_interaction':
  z=leg3(x); return np.stack([z[:,i]*z[:,j] for i in range(4) for j in range(i+1,4)],1)
 raise ValueError(kind)
def systems(r,kind,n):
 d=4 if kind!='pairwise_interaction' else 6
 out=[]
 for _ in range(n):
  m=r.normal(size=(d,4)); m=m/max(np.linalg.norm(m),1e-12)*2
  out.append((stable_A(r),r.uniform(.6,1.4),m,kind))
 return out
def trans(x,s,g):
 A,a,m,k=s; g.charge('generated_transitions',len(x)); g.charge('generated_rhs_evaluations',len(x)); return x+DT*(x@A.T+a*phi(x,k)@m)
def linear_residual(x,y,g):
 g.charge('coefficient_factorizations'); f=(y-x)/DT; b=np.linalg.solve(x.T@x+1e-6*np.eye(4),x.T@f); return f-x@b,b
def components(x,kind):
 if kind=='joint': return [x]
 if kind=='axis_additive': return [x[:,j:j+1] for j in range(4)]
 if kind=='rotated_additive':
  z=x@Q; return [z[:,j:j+1] for j in range(4)]
 if kind=='pairwise': return [x[:,[i,j]] for i in range(4) for j in range(i+1,4)]
 raise ValueError(kind)
def kmat(a,b,kind):
 ca,cb=components(a,kind),components(b,kind); out=0
 for aa,bb in zip(ca,cb):
  d=aa[:,None,:]-aa[None,:,:]; vv=np.sqrt((d*d).sum(2)); nz=vv[vv>0]; bw=max(float(np.median(nz)),1e-12)
  e=aa[:,None,:]-bb[None,:,:]; out=out+np.exp(-(e*e).sum(2)/(2*bw*bw))
 return out
def fit_predict(x,y,q,kind,g):
 residual,b=linear_residual(x,y,g); kt=kmat(x,x,kind); kq=kmat(x,q,kind); g.charge('kernel_entries',kt.size+kq.size); g.charge('kernel_factorizations'); g.charge('prediction_blocks')
 w=np.linalg.solve(kt+1e-6*np.eye(len(x)),residual); return q+DT*(q@b+kq.T@w)
def mse(a,b,g): g.charge('score_blocks'); return float(np.mean((a-b)**2))
def choose_acquisition(acq,xfit,yfit,xsel,ysel,g):
 scores={}
 for c in CANDS:
  vals=[mse(fit_predict(xfit,yfit[i],xsel,c,g),ysel[i],g) for i in range(len(acq))]; scores[c]=float(np.mean(vals))
 g.charge('structure_selections'); return min(CANDS,key=lambda c:(scores[c],CANDS.index(c))),scores
def choose_scratch(x,y,g):
 scores={c:mse(fit_predict(x[:24],y[:24],x[24:],c,g),y[24:],g) for c in CANDS}; g.charge('structure_selections'); return min(CANDS,key=lambda c:(scores[c],CANDS.index(c)))
def main(a):
 plan=json.load(open(a.plan)); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); g=Guard(plan['ceilings']); write(out/'ledger.json',{'status':'SEALED_PRECALL','plan_sha256':sha(a.plan),'driver_sha256':sha(__file__),'ceilings':plan['ceilings'],'counts':g.n,'rotation':Q.tolist()})
 schedule=plan['world_schedule']; barrier=[]; hidden=[]; pred_arrays={}
 for wi,kind in enumerate(schedule):
  r=np.random.default_rng(2026106500+100*wi); ss=systems(r,kind,16); xf=states(r,48); xs=states(r,32); xq=states(r,64)
  yf=[trans(xf,s,g) for s in ss]; ys=[trans(xs,s,g) for s in ss[:8]]
  selected,scores=choose_acquisition(ss[:8],xf,yf[:8],xs,ys,g); barrier.append({'world':wi,'selected':selected,'training_scores':scores})
  for j in range(8,16):
   scratch=choose_scratch(xf,yf[j],g); random_c=CANDS[int(np.random.default_rng(2026109900+100*wi+j).integers(0,len(CANDS)))]
   for arm,c in [('retained',selected),('joint','joint'),('random',random_c),('scratch',scratch)]: pred_arrays[f'w{wi}_s{j}_{arm}']=fit_predict(xf,yf[j],xq,c,g)
   hidden.append((wi,j,kind,ss[j],xq))
 write(out/'selection_barrier.json',{'status':'STRUCTURES_AND_PREDICTIONS_FROZEN_BEFORE_LATER_QUERY_RESPONSES','selections':barrier,'prediction_keys':sorted(pred_arrays),'counts':g.n.copy()}); np.savez_compressed(out/'predictions_before_query_responses.npz',**pred_arrays)
 rows=[]
 for wi,j,kind,s,xq in hidden:
  yq=trans(xq,s,g)
  for arm in ('retained','joint','random','scratch'): rows.append({'world':wi,'system':j,'hidden_type':kind,'arm':arm,'mse':mse(pred_arrays[f'w{wi}_s{j}_{arm}'],yq,g)})
 def vals(arm,kind=None): return [r['mse'] for r in rows if r['arm']==arm and (kind is None or r['hidden_type']==kind)]
 def comp(a,b,kind=None):
  aa,bb=vals(a,kind),vals(b,kind); return {'ratio':float(sum(aa)/sum(bb)),'wins':int(sum(x<y for x,y in zip(aa,bb))),'n':len(aa)}
 overall={'retained_vs_joint':comp('retained','joint'),'retained_vs_random':comp('retained','random'),'retained_vs_scratch':comp('retained','scratch')}
 types={k:{'retained_vs_joint':comp('retained','joint',k),'retained_vs_random':comp('retained','random',k),'retained_vs_scratch':comp('retained','scratch',k)} for k in sorted(set(schedule))}
 repeat=sum(barrier[i]['selected']==barrier[i+3]['selected'] for i in range(3))
 gates={'retained_vs_joint':overall['retained_vs_joint']['ratio']<=.8 and overall['retained_vs_joint']['wins']>=32,'retained_vs_random':overall['retained_vs_random']['ratio']<=.8 and overall['retained_vs_random']['wins']>=32,'retained_vs_scratch':overall['retained_vs_scratch']['ratio']<=1.1,'type_nonharm':all(v['retained_vs_joint']['ratio']<=1 for v in types.values()),'selection_repeatability':repeat>=2}
 result={'classification':'RETAINED_FACTORIZATION_SUPPORTED_ENGINEERED_DEVELOPMENT' if all(gates.values()) else 'RETAINED_FACTORIZATION_NOT_SUPPORTED','gates':gates,'selections':barrier,'overall':overall,'by_type':types,'rows':rows,'counts':g.n,'qualification':plan['qualification']}
 write(out/'result.json',result); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n})
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--plan'); ap.add_argument('--out'); main(ap.parse_args())
