#!/usr/bin/env python3
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
import numpy as np

DT=.05; WORLD=2026104800
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
def states(seed,n): return np.random.default_rng(seed).uniform(-1,1,(n,4))
def p3(x): return (5*x**3-3*x)/2
def stable_A(r):
 a=r.normal(scale=.22,size=(4,4)); a-=np.eye(4)*(0.45+.15*r.random()); return a
def M(r):
 m=r.normal(size=(4,4)); return m/max(np.linalg.norm(m),1e-12)*2
def trans(x,A,a,m,g):
 g.charge('generated_transitions',len(x)); g.charge('generated_rhs_evaluations',len(x)); return x+DT*(x@A.T+a*(p3(x)@m))
def fit_linear(x,y,g):
 g.charge('coefficient_factorizations'); f=(y-x)/DT; return np.linalg.solve(x.T@x+1e-6*np.eye(4),x.T@f)
def fit_aug(x,y,h,g):
 g.charge('coefficient_factorizations'); f=((y-x)/DT).ravel(); z=np.zeros((len(x)*4,17))
 for i in range(len(x)):
  for j in range(4): z[4*i+j,4*j:4*j+4]=x[i]; z[4*i+j,16]=h[i,j]
 return np.linalg.solve(z.T@z+1e-6*np.eye(17),z.T@f)
def pred_aug(x,h,b): return x+DT*(x@b[:16].reshape(4,4).T+b[16]*h)
class RBF:
 def __init__(self,x,w,bw): self.x=x; self.w=w; self.bw=bw
 def __call__(self,x): return np.exp(-((x[:,None]-self.x[None])**2).sum(2)/(2*self.bw**2))@self.w
def discover(x,resids,shuffle,g):
 z=[]; rr=[]
 for i,r in enumerate(resids):
  q=r.copy()
  if shuffle: q=q[np.random.default_rng(WORLD+991*i).permutation(len(q))]
  rr.append(q); v=q.ravel(); z.append(v/max(np.linalg.norm(v),1e-12))
 z=np.stack(z); sim=np.abs(z@z.T); np.fill_diagonal(sim,np.nan); central=np.nanmedian(sim,1); sel=np.argsort(-central,kind='stable')[:8]
 g.charge('svds'); _,_,vh=np.linalg.svd(z[sel],full_matrices=False); signs=np.sign(z@vh[0]); signs[signs==0]=1
 targets=[]
 for i in sel:
  q=rr[i]; targets.append(signs[i]*q/(np.sqrt(np.mean(q*q))+1e-12))
 target=np.mean(targets,0); d=np.sqrt(((x[:,None]-x[None])**2).sum(2)); bw=float(np.median(d[d>0])); k=np.exp(-(d*d)/(2*bw*bw))
 g.charge('kernel_factorizations'); w=np.linalg.solve(k+1e-6*np.eye(len(k)),target)
 return RBF(x,w,bw),sel,central
def score(a,b,g): g.charge('score_blocks'); return float(np.mean((a-b)**2))
def main(a):
 plan=json.load(open(a.plan)); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); g=Guard(plan['ceilings'])
 write(out/'ledger.json',{'status':'SEALED_PRECALL','plan_sha256':sha(a.plan),'driver_sha256':sha(__file__),'ceilings':plan['ceilings'],'counts':g.n})
 rng=np.random.default_rng(WORLD); shared=M(rng); xc=states(WORLD+1,32); xa=states(WORLD+2,32)
 systems=[]; yc=[]; ya=[]
 for i in range(16):
  reuse=i<8; s=(stable_A(rng),rng.uniform(.6,1.4),shared if reuse else M(rng),reuse); systems.append(s); yc.append(trans(xc,*s[:3],g)); ya.append(trans(xa,*s[:3],g))
 bs=[]; res=[]
 for i in range(16):
  b=fit_linear(xc,yc[i],g); bs.append(b); res.append((yc[i]-xc)/DT-xc@b)
 aligned,sa,ca=discover(xc,res,False,g); shuffled,ss,cs=discover(xc,res,True,g)
 admissions=[]
 for arm,hfun,sel in [('aligned',aligned,sa),('shuffled',shuffled,ss)]:
  hc=hfun(xc); ha=hfun(xa); g.charge('prediction_blocks',2); rows=[]
  for i in sel:
   beta=fit_aug(xc,yc[i],hc,g); mm=score(pred_aug(xa,ha,beta),ya[i],g); m0=score(xa+DT*(xa@bs[i]),ya[i],g); rows.append({'system':int(i),'module_MSE':mm,'linear_MSE':m0})
  ratio=sum(r['module_MSE'] for r in rows)/sum(r['linear_MSE'] for r in rows); wins=sum(r['module_MSE']<r['linear_MSE'] for r in rows); admissions.append({'arm':arm,'ratio':ratio,'wins':wins,'passed':ratio<=.8 and wins>=6,'rows':rows})
 write(out/'admission_decision.json',{'status':'FROZEN_BEFORE_DEVELOPMENT','hidden_labels_read':False,'aligned_selected':[int(i) for i in sa],'shuffled_selected':[int(i) for i in ss],'admissions':admissions,'counts':g.n.copy()})
 if not admissions[0]['passed']:
  r={'classification':'NOT_SUPPORTED_AT_ADMISSION','admissions':admissions,'counts':g.n,'qualification':plan['qualification']}; write(out/'result.json',r); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n}); return
 xd=states(WORLD+101,32); xq=states(WORLD+102,64); dev=[]
 for i in range(16):
  reuse=i<8; s=(stable_A(rng),rng.uniform(.6,1.4),shared if reuse else M(rng),reuse); dev.append((s,trans(xd,*s[:3],g),trans(xq,*s[:3],g)))
 had=aligned(xd); haq=aligned(xq); hsd=shuffled(xd); hsq=shuffled(xq); g.charge('prediction_blocks',4)
 preds={}; rows=[]
 for i,(s,yd,yq) in enumerate(dev):
  ba=fit_aug(xd,yd,had,g); bs2=fit_aug(xd,yd,hsd,g); b0=fit_linear(xd,yd,g)
  preds[f'aligned_{i}']=pred_aug(xq,haq,ba); preds[f'shuffled_{i}']=pred_aug(xq,hsq,bs2); preds[f'linear_{i}']=xq+DT*(xq@b0)
 np.savez_compressed(out/'predictions_before_hidden_labels.npz',**preds); write(out/'prediction_barrier.json',{'status':'PERSISTED_BEFORE_HIDDEN_SCORING','sha256':sha(out/'predictions_before_hidden_labels.npz'),'counts':g.n.copy()})
 for i,(s,yd,yq) in enumerate(dev):
  for arm in ['aligned','shuffled','linear']: rows.append({'system':i,'group':'reuse' if s[3] else 'unrelated','arm':arm,'MSE':score(preds[f'{arm}_{i}'],yq,g)})
 def vv(group,arm): return [r['MSE'] for r in rows if r['group']==group and r['arm']==arm]
 ar,sh,li=vv('reuse','aligned'),vv('reuse','shuffled'),vv('reuse','linear'); au,lu=vv('unrelated','aligned'),vv('unrelated','linear')
 ratio=lambda a,b:sum(a)/sum(b)
 c={'reuse_aligned_over_linear':ratio(ar,li),'reuse_wins_linear':sum(a<b for a,b in zip(ar,li)),'reuse_aligned_over_shuffled':ratio(ar,sh),'reuse_wins_shuffled':sum(a<b for a,b in zip(ar,sh)),'unrelated_aligned_over_linear':ratio(au,lu)}
 c['passed']=c['reuse_aligned_over_linear']<=.8 and c['reuse_wins_linear']>=6 and c['reuse_aligned_over_shuffled']<=.85 and c['reuse_wins_shuffled']>=6 and c['unrelated_aligned_over_linear']<=1.1
 r={'classification':'SUPPORTED_DEVELOPMENT_ONLY' if c['passed'] else 'NOT_SUPPORTED','admissions':admissions,'comparison':c,'scores':rows,'counts':g.n,'qualification':plan['qualification']}; write(out/'result.json',r); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n})
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--plan'); ap.add_argument('--out'); main(ap.parse_args())
