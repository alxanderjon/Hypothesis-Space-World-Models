#!/usr/bin/env python3
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
import numpy as np
DT=.05
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
def leg(x,d):
 if d==3:return (5*x**3-3*x)/2
 if d==5:return (63*x**5-70*x**3+15*x)/8
 if d==7:return (429*x**7-693*x**5+315*x**3-35*x)/16
 raise ValueError(d)
def stable_A(r):
 a=r.normal(scale=.22,size=(4,4)); a-=np.eye(4)*(0.45+.15*r.random()); return a
def M(r):
 m=r.normal(size=(4,4)); return m/max(np.linalg.norm(m),1e-12)*2
def make_systems(seed,degree):
 r=np.random.default_rng(seed); shared=M(r); return [(stable_A(r),r.uniform(.6,1.4),shared if i<8 else M(r),i<8,degree) for i in range(16)]
def trans(x,s,g):
 A,a,m,_,d=s; g.charge('generated_transitions',len(x)); g.charge('generated_rhs_evaluations',len(x)); return x+DT*(x@A.T+a*(leg(x,d)@m))
def fit_linear(x,y,g):
 g.charge('coefficient_factorizations'); f=(y-x)/DT; return np.linalg.solve(x.T@x+1e-6*np.eye(4),x.T@f)
def quotient(x,ys,g):
 rr=[]
 for y in ys:
  b=fit_linear(x,y,g); rr.append((y-x)/DT-x@b)
 return np.stack(rr)
def signature(rr,g):
 z=np.stack([r.ravel()/max(np.linalg.norm(r),1e-12) for r in rr]); sim=np.abs(z@z.T); np.fill_diagonal(sim,np.nan); central=np.nanmedian(sim,1); sel=np.argsort(-central,kind='stable')[:8]; g.charge('svds'); _,_,vh=np.linalg.svd(z[sel],full_matrices=False); signs=np.sign(z@vh[0]); signs[signs==0]=1; return sel,signs
class RBF:
 def __init__(self,x,w,bw): self.x=x; self.w=w; self.bw=bw
 def __call__(self,x): return np.exp(-((x[:,None]-self.x[None])**2).sum(2)/(2*self.bw**2))@self.w
def kernel(x,rr,sel,signs,g,leave=None):
 use=[i for i in sel if i!=leave]; target=np.mean([signs[i]*rr[i]/(np.sqrt(np.mean(rr[i]**2))+1e-12) for i in use],0); d=np.sqrt(((x[:,None]-x[None])**2).sum(2)); bw=float(np.median(d[d>0])); k=np.exp(-(d*d)/(2*bw*bw)); g.charge('kernel_factorizations'); return RBF(x,np.linalg.solve(k+1e-6*np.eye(len(k)),target),bw)
def fit_aug(x,y,h,g):
 g.charge('coefficient_factorizations'); f=((y-x)/DT).ravel(); z=np.zeros((len(x)*4,17))
 for i in range(len(x)):
  for j in range(4): z[4*i+j,4*j:4*j+4]=x[i]; z[4*i+j,16]=h[i,j]
 return np.linalg.solve(z.T@z+1e-6*np.eye(17),z.T@f)
def pred_aug(x,h,b): return x+DT*(x@b[:16].reshape(4,4).T+b[16]*h)
def score(a,b,g): g.charge('score_blocks'); return float(np.mean((a-b)**2))
def features(x0,pool,loos):
 pr=np.stack([h(pool) for h in loos]); mean=pr.mean(0); var=pr.var(0); dist=np.sqrt(((pool[:,None]-x0[None])**2).sum(2)).min(1); base=np.concatenate([pool,pool**2,np.linalg.norm(pool,axis=1,keepdims=True)],1); state=base; context=np.concatenate([base,dist[:,None],mean,var],1); return state,context,var.mean(1),dist
def policy_fit(x,y,g):
 mu=x.mean(0); sd=x.std(0); sd[sd<1e-12]=1; z=(x-mu)/sd; zz=np.concatenate([np.ones((len(z),1)),z],1); g.charge('policy_factorizations'); w=np.linalg.solve(zz.T@zz+1e-3*np.eye(zz.shape[1]),zz.T@y); return mu,sd,w
def policy_score(model,x):
 mu,sd,w=model; z=(x-mu)/sd; return np.concatenate([np.ones((len(z),1)),z],1)@w
def admission(x,ys,xv,yv,h,sel,g):
 hx=h(x); hv=h(xv); g.charge('prediction_blocks',2); vals=[]
 for i in sel: vals.append(score(pred_aug(xv,hv,fit_aug(x,ys[i],hx,g)),yv[i],g))
 return float(np.mean(vals)),vals
def training_world(seed,degree,g):
 systems=make_systems(seed,degree); x0=states(seed+1,32); xv=states(seed+2,32); pool=states(seed+3,32); y0=[trans(x0,s,g) for s in systems]; yv=[trans(xv,s,g) for s in systems]; yp=[trans(pool,s,g) for s in systems]
 r0=quotient(x0,y0,g); sel,signs=signature(r0,g); h0=kernel(x0,r0,sel,signs,g); base,_=admission(x0,y0,xv,yv,h0,sel,g); loos=[kernel(x0,r0,sel,signs,g,leave=int(i)) for i in sel]; st,ct,dis,dist=features(x0,pool,loos); g.charge('prediction_blocks',len(loos)); g.charge('candidate_states_scored',len(loos)*len(pool)); rewards=[]
 for j in range(len(pool)):
  x=np.concatenate([x0,pool[j:j+1]]); ys=[np.concatenate([y0[i],yp[i][j:j+1]]) for i in range(16)]; rr=quotient(x,ys,g); h=kernel(x,rr,sel,signs,g); val,_=admission(x,ys,xv,yv,h,sel,g); rewards.append(np.log(max(base,1e-30)/max(val,1e-30))); g.charge('reward_rows')
 return st,ct,np.array(rewards)
def main(a):
 plan=json.load(open(a.plan)); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); g=Guard(plan['ceilings']); write(out/'ledger.json',{'status':'SEALED_PRECALL','plan_sha256':sha(a.plan),'driver_sha256':sha(__file__),'ceilings':plan['ceilings'],'counts':g.n})
 st=[];ct=[];rw=[]
 for k,(seed,d) in enumerate([(2026105200,3),(2026105300,5),(2026105400,3),(2026105500,5),(2026105600,3),(2026105700,5)]):
  s,c,r=training_world(seed,d,g); st.append(s);ct.append(c);rw.append(r)
 state_model=policy_fit(np.concatenate(st),np.concatenate(rw),g); context_model=policy_fit(np.concatenate(ct),np.concatenate(rw),g)
 write(out/'policy_freeze.json',{'status':'POLICIES_FROZEN_BEFORE_RESERVED_DISTRIBUTION','state_model':[q.tolist() for q in state_model],'context_model':[q.tolist() for q in context_model],'counts':g.n.copy()})
 seed=2026105800; systems=make_systems(seed,7); x0=states(seed+1,32); xv=states(seed+2,32); pool=states(seed+3,32); y0=[trans(x0,s,g) for s in systems]; yv=[trans(xv,s,g) for s in systems]
 r0=quotient(x0,y0,g); sel,signs=signature(r0,g); loos=[kernel(x0,r0,sel,signs,g,leave=int(i)) for i in sel]; sf,cf,dis,dist=features(x0,pool,loos); g.charge('prediction_blocks',len(loos)); g.charge('candidate_states_scored',len(loos)*len(pool))
 idx={}; idx['random48']=np.random.default_rng(seed+4).choice(32,16,replace=False); idx['learned_state_only48']=np.argsort(-policy_score(state_model,sf),kind='stable')[:16]; idx['learned_context48']=np.argsort(-policy_score(context_model,cf),kind='stable')[:16]
 remaining=np.ones(32,bool); md=dist.copy(); chosen=[]
 for _ in range(16):
  sc=dis*np.maximum(md,1e-12); sc[~remaining]=-np.inf; j=int(np.argmax(sc)); chosen.append(j); remaining[j]=False; md=np.minimum(md,np.sqrt(((pool-pool[j])**2).sum(1)))
 idx['frozen_disagreement48']=np.array(chosen)
 write(out/'selection_barrier.json',{'status':'INDICES_PERSISTED_BEFORE_RESERVED_CANDIDATE_RESPONSES','indices':{k:v.tolist() for k,v in idx.items()},'selected_systems':sel.tolist(),'hidden_labels_read':False,'counts':g.n.copy()})
 arms={'original32':(x0,y0)}
 for name,ii in idx.items():
  ys=[trans(pool[ii],s,g) for s in systems]; arms[name]=(np.concatenate([x0,pool[ii]]),[np.concatenate([y0[i],ys[i]]) for i in range(16)])
 admissions=[]; funcs={}
 for name,(x,ys) in arms.items():
  rr=quotient(x,ys,g) if name!='original32' else r0; h=kernel(x,rr,sel,signs,g); funcs[name]=h; val,vals=admission(x,ys,xv,yv,h,sel,g); lin=[]
  for i in sel: lin.append(score(xv+DT*(xv@fit_linear(x,ys[i],g)),yv[i],g))
  ratio=val/np.mean(lin); wins=sum(a<b for a,b in zip(vals,lin)); admissions.append({'arm':name,'ratio':ratio,'wins':wins,'passed':ratio<=.8 and wins>=6})
 write(out/'admission_decision.json',{'status':'FROZEN_BEFORE_TARGETS','admissions':admissions,'counts':g.n.copy()})
 if not [r for r in admissions if r['arm']=='learned_context48'][0]['passed']:
  res={'classification':'CONTEXT_POLICY_NOT_SUPPORTED_AT_ADMISSION','admissions':admissions,'counts':g.n,'qualification':plan['qualification']}; write(out/'result.json',res); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n}); return
 xd=states(seed+101,32); xq=states(seed+102,64)
 # New target systems share the acquisition world's hidden P7 callable but have new linear maps and coefficients.
 rt=np.random.default_rng(seed+1000); shared=systems[0][2]; target=[(stable_A(rt),rt.uniform(.6,1.4),shared if i<8 else M(rt),i<8,7) for i in range(16)]; dev=[(s,trans(xd,s,g),trans(xq,s,g)) for s in target]
 preds={}
 for name,h in funcs.items():
  hd=h(xd); hq=h(xq); g.charge('prediction_blocks',2)
  for i,(s,yd,yq) in enumerate(dev): preds[f'{name}_{i}']=pred_aug(xq,hq,fit_aug(xd,yd,hd,g))
 for i,(s,yd,yq) in enumerate(dev): preds[f'linear_{i}']=xq+DT*(xq@fit_linear(xd,yd,g))
 np.savez_compressed(out/'predictions_before_hidden_labels.npz',**preds); write(out/'prediction_barrier.json',{'status':'PERSISTED_BEFORE_HIDDEN_SCORING','sha256':sha(out/'predictions_before_hidden_labels.npz'),'counts':g.n.copy()})
 rows=[]
 for i,(s,yd,yq) in enumerate(dev):
  for arm in list(arms)+['linear']: rows.append({'system':i,'group':'reuse' if s[3] else 'unrelated','arm':arm,'MSE':score(preds[f'{arm}_{i}'],yq,g)})
 def v(group,arm):return [r['MSE'] for r in rows if r['group']==group and r['arm']==arm]
 cc=v('reuse','learned_context48'); rr=v('reuse','random48'); ss=v('reuse','learned_state_only48'); hh=v('reuse','frozen_disagreement48'); oo=v('reuse','original32'); au=v('unrelated','learned_context48'); ll=v('unrelated','linear'); rat=lambda a,b:sum(a)/sum(b)
 comp={'context_over_random':rat(cc,rr),'context_wins_random':sum(a<b for a,b in zip(cc,rr)),'context_over_state_only':rat(cc,ss),'context_wins_state_only':sum(a<b for a,b in zip(cc,ss)),'context_over_heuristic':rat(cc,hh),'context_wins_heuristic':sum(a<b for a,b in zip(cc,hh)),'context_over_original':rat(cc,oo),'context_wins_original':sum(a<b for a,b in zip(cc,oo)),'context_unrelated_over_linear':rat(au,ll)}
 passed=comp['context_over_random']<=.85 and comp['context_wins_random']>=6 and comp['context_over_state_only']<=.85 and comp['context_wins_state_only']>=6 and comp['context_over_heuristic']<=.85 and comp['context_wins_heuristic']>=6 and comp['context_over_original']<=.8 and comp['context_wins_original']>=6 and comp['context_unrelated_over_linear']<=1.1
 res={'classification':'LEARNED_CONTEXT_ACQUISITION_SUPPORTED_DEVELOPMENT_ONLY' if passed else 'NOT_SUPPORTED','admissions':admissions,'comparison':comp,'scores':rows,'counts':g.n,'qualification':plan['qualification']}; write(out/'result.json',res); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n})
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--plan');ap.add_argument('--out');main(ap.parse_args())
