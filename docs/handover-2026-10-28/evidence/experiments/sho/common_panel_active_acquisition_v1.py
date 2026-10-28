#!/usr/bin/env python3
import argparse,hashlib,importlib.util,json,os,resource,sys,time
from pathlib import Path
import numpy as np

WORLD=2026105000; DT=.05
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
def trans(x,s,g):
 A,a,m,_=s; g.charge('generated_transitions',len(x)); g.charge('generated_rhs_evaluations',len(x)); return x+DT*(x@A.T+a*(p3(x)@m))
def fit_linear(x,y,g):
 g.charge('coefficient_factorizations'); f=(y-x)/DT; return np.linalg.solve(x.T@x+1e-6*np.eye(4),x.T@f)
def quotient(x,ys,g):
 bs=[]; rr=[]
 for y in ys:
  b=fit_linear(x,y,g); bs.append(b); rr.append((y-x)/DT-x@b)
 return bs,np.stack(rr)
def signature_info(rr,g):
 z=[]
 for r in rr:
  v=r.ravel(); z.append(v/max(np.linalg.norm(v),1e-12))
 z=np.stack(z); sim=np.abs(z@z.T); np.fill_diagonal(sim,np.nan); central=np.nanmedian(sim,1); sel=np.argsort(-central,kind='stable')[:8]
 g.charge('svds'); _,_,vh=np.linalg.svd(z[sel],full_matrices=False); signs=np.sign(z@vh[0]); signs[signs==0]=1
 return sel,signs,central
class RBF:
 def __init__(self,x,w,bw): self.x=x; self.w=w; self.bw=bw
 def __call__(self,x): return np.exp(-((x[:,None]-self.x[None])**2).sum(2)/(2*self.bw**2))@self.w
def kernel_fit(x,rr,sel,signs,g,leave=None):
 use=[i for i in sel if i!=leave]; targets=[]
 for i in use: targets.append(signs[i]*rr[i]/(np.sqrt(np.mean(rr[i]*rr[i]))+1e-12))
 y=np.mean(targets,0); d=np.sqrt(((x[:,None]-x[None])**2).sum(2)); bw=float(np.median(d[d>0])); k=np.exp(-(d*d)/(2*bw*bw)); g.charge('kernel_factorizations'); w=np.linalg.solve(k+1e-6*np.eye(len(k)),y); return RBF(x,w,bw)
def fit_aug(x,y,h,g):
 g.charge('coefficient_factorizations'); f=((y-x)/DT).ravel(); z=np.zeros((len(x)*4,17))
 for i in range(len(x)):
  for j in range(4): z[4*i+j,4*j:4*j+4]=x[i]; z[4*i+j,16]=h[i,j]
 return np.linalg.solve(z.T@z+1e-6*np.eye(17),z.T@f)
def pred_aug(x,h,b): return x+DT*(x@b[:16].reshape(4,4).T+b[16]*h)
def score(a,b,g): g.charge('score_blocks'); return float(np.mean((a-b)**2))
def main(a):
 plan=json.load(open(a.plan)); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); g=Guard(plan['ceilings']); write(out/'ledger.json',{'status':'SEALED_PRECALL','plan_sha256':sha(a.plan),'driver_sha256':sha(__file__),'ceilings':plan['ceilings'],'counts':g.n})
 rng=np.random.default_rng(WORLD); shared=M(rng); systems=[]
 for i in range(16): systems.append((stable_A(rng),rng.uniform(.6,1.4),shared if i<8 else M(rng),i<8))
 x0=states(WORLD+1,32); xa=states(WORLD+2,32); pool=states(WORLD+3,256); y0=[trans(x0,s,g) for s in systems]
 _,r0=quotient(x0,y0,g); sel0,sign0,cent0=signature_info(r0,g)
 loo=[kernel_fit(x0,r0,sel0,sign0,g,leave=int(j)) for j in sel0]
 pred=np.stack([h(pool) for h in loo]); g.charge('prediction_blocks',len(loo)); g.charge('candidate_states_scored',len(loo)*len(pool))
 disagreement=np.mean(np.var(pred,axis=0),axis=1); chosen=[]; mind=np.min(((pool[:,None]-x0[None])**2).sum(2),axis=1)**.5
 for _ in range(16):
  idx=int(np.argmax(disagreement*np.maximum(mind,1e-12))); chosen.append(idx); d=np.sqrt(((pool-pool[idx])**2).sum(1)); mind=np.minimum(mind,d); disagreement[idx]=-np.inf
 active_idx=np.array(chosen); random_idx=np.random.default_rng(WORLD+4).choice(np.setdiff1d(np.arange(256),active_idx),16,replace=False)
 write(out/'selection_barrier.json',{'status':'INDICES_PERSISTED_BEFORE_EXTRA_RESPONSES','candidate_pool_sha256':hashlib.sha256(pool.tobytes()).hexdigest(),'active_indices':active_idx.tolist(),'random_indices':random_idx.tolist(),'selected_initial_systems':[int(i) for i in sel0],'hidden_labels_read':False,'counts':g.n.copy()})
 xr=pool[random_idx]; xx=pool[active_idx]; yr=[trans(xr,s,g) for s in systems]; yx=[trans(xx,s,g) for s in systems]; ya=[trans(xa,s,g) for s in systems]
 arms={}
 for name,x,ys in [('original32',x0,y0),('random48',np.concatenate([x0,xr]),[np.concatenate([y0[i],yr[i]]) for i in range(16)]),('adaptive48',np.concatenate([x0,xx]),[np.concatenate([y0[i],yx[i]]) for i in range(16)])]:
  bs,rr=quotient(x,ys,g) if name!='original32' else ([None]*16,r0); sel,sign,central=signature_info(rr,g) if name!='original32' else (sel0,sign0,cent0); h=kernel_fit(x,rr,sel,sign,g); arms[name]={'x':x,'ys':ys,'sel':sel,'h':h,'central':central}
 admissions=[]
 for name,d in arms.items():
  hc=d['h'](d['x']); ha=d['h'](xa); g.charge('prediction_blocks',2); rows=[]
  for i in d['sel']:
   beta=fit_aug(d['x'],d['ys'][i],hc,g); mm=score(pred_aug(xa,ha,beta),ya[i],g); b=fit_linear(d['x'],d['ys'][i],g); m0=score(xa+DT*(xa@b),ya[i],g); rows.append({'system':int(i),'module_MSE':mm,'linear_MSE':m0})
  ratio=sum(r['module_MSE'] for r in rows)/sum(r['linear_MSE'] for r in rows); wins=sum(r['module_MSE']<r['linear_MSE'] for r in rows); admissions.append({'arm':name,'ratio':ratio,'wins':wins,'passed':ratio<=.8 and wins>=6,'rows':rows})
 write(out/'admission_decision.json',{'status':'FROZEN_BEFORE_TARGET_GENERATION','hidden_labels_read':False,'admissions':admissions,'counts':g.n.copy()})
 if not any(r['passed'] for r in admissions[1:]):
  res={'classification':'NOT_SUPPORTED_AT_ACQUISITION_ADMISSION','admissions':admissions,'counts':g.n,'qualification':plan['qualification']}; write(out/'result.json',res); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n}); return
 xd=states(WORLD+101,32); xq=states(WORLD+102,64); dev=[]
 for i in range(16):
  s=(stable_A(rng),rng.uniform(.6,1.4),shared if i<8 else M(rng),i<8); dev.append((s,trans(xd,s,g),trans(xq,s,g)))
 preds={}
 for name,d in arms.items():
  hd=d['h'](xd); hq=d['h'](xq); g.charge('prediction_blocks',2)
  for i,(s,yd,yq) in enumerate(dev): preds[f'{name}_{i}']=pred_aug(xq,hq,fit_aug(xd,yd,hd,g))
 for i,(s,yd,yq) in enumerate(dev): preds[f'linear_{i}']=xq+DT*(xq@fit_linear(xd,yd,g))
 np.savez_compressed(out/'predictions_before_hidden_labels.npz',**preds); write(out/'prediction_barrier.json',{'status':'PERSISTED_BEFORE_HIDDEN_SCORING','sha256':sha(out/'predictions_before_hidden_labels.npz'),'counts':g.n.copy()})
 rows=[]
 for i,(s,yd,yq) in enumerate(dev):
  for arm in ['original32','random48','adaptive48','linear']: rows.append({'system':i,'group':'reuse' if s[3] else 'unrelated','arm':arm,'MSE':score(preds[f'{arm}_{i}'],yq,g)})
 def v(group,arm): return [r['MSE'] for r in rows if r['group']==group and r['arm']==arm]
 o,ra,ac,li=v('reuse','original32'),v('reuse','random48'),v('reuse','adaptive48'),v('reuse','linear'); au,lu=v('unrelated','adaptive48'),v('unrelated','linear'); rat=lambda a,b:sum(a)/sum(b)
 c={'random_over_original':rat(ra,o),'random_wins_original':sum(a<b for a,b in zip(ra,o)),'adaptive_over_random':rat(ac,ra),'adaptive_wins_random':sum(a<b for a,b in zip(ac,ra)),'adaptive_over_original':rat(ac,o),'adaptive_wins_original':sum(a<b for a,b in zip(ac,o)),'adaptive_unrelated_over_linear':rat(au,lu)}
 more=c['random_over_original']<=.8 and c['random_wins_original']>=6; intel=c['adaptive_over_random']<=.85 and c['adaptive_wins_random']>=6; absolute=c['adaptive_over_original']<=.8 and c['adaptive_wins_original']>=6 and c['adaptive_unrelated_over_linear']<=1.1
 cls='ACTIVE_ACQUISITION_SUPPORTED' if more and intel and absolute else ('MORE_DATA_ONLY' if more and not intel else 'NOT_SUPPORTED')
 res={'classification':cls,'admissions':admissions,'comparison':c,'scores':rows,'counts':g.n,'qualification':plan['qualification']}; write(out/'result.json',res); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n})
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--plan'); ap.add_argument('--out'); main(ap.parse_args())
