#!/usr/bin/env python3
import argparse, hashlib, json, math, os, resource, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

DT=.05
BASE_OPS=('x0','x1','x2','x3','sin','tanh','relu','add','mul')
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
def s(a):
 if a[0]=='x': return f'x{a[1]}'
 if a[0] in ('sin','tanh','relu'): return f'{a[0]}({s(a[1])})'
 return f'{a[0]}({s(a[1])},{s(a[2])})'
def size(a): return 1 if a[0]=='x' else 1+sum(size(z) for z in a[1:])
def depth(a): return 0 if a[0]=='x' else 1+max(depth(z) for z in a[1:])
def canon(op,a,b): return (op,a,b) if s(a)<=s(b) else (op,b,a)
def ev(a,x):
 if a[0]=='x': return x[:,a[1]]
 if a[0]=='sin': return np.sin(ev(a[1],x))
 if a[0]=='tanh': return np.tanh(ev(a[1],x))
 if a[0]=='relu': return np.maximum(ev(a[1],x),0)
 if a[0]=='add': return ev(a[1],x)+ev(a[2],x)
 if a[0]=='mul': return ev(a[1],x)*ev(a[2],x)
 raise ValueError(a[0])
def subtrees(a):
 out=[a]
 for z in a[1:]:
  if isinstance(z,tuple): out.extend(subtrees(z))
 return out
def contains(a,b): return any(s(z)==s(b) for z in subtrees(a))
def random_ast(r,d):
 if d<=0 or r.random()<.18: return ('x',int(r.integers(0,4)))
 op=('sin','tanh','relu','add','mul')[int(r.integers(0,5))]
 if op in ('sin','tanh','relu'): return (op,random_ast(r,d-1))
 return canon(op,random_ast(r,d-1),random_ast(r,d-1))
def bank(r,probe,g):
 xs={}; base=[('x',i) for i in range(4)]
 for a in base: xs[hashlib.sha256(np.round(ev(a,probe),10).tobytes()).hexdigest()]=a
 attempts=0
 while len(xs)<2048 and attempts<20000:
  g.charge('bank_generation_attempts'); attempts+=1; a=random_ast(r,4)
  v=ev(a,probe)
  if np.all(np.isfinite(v)) and np.max(np.abs(v))<100:
   k=hashlib.sha256(np.round(v,10).tobytes()).hexdigest(); xs.setdefault(k,a)
 if len(xs)<2048: raise RuntimeError(f'BANK_SHORT:{len(xs)}')
 return sorted(xs.values(),key=lambda a:(size(a),s(a)))[:2048]
def productions(a,macros):
 counts=defaultdict(float)
 def rec(z):
  for name,m in macros.items():
   if s(z)==s(m): counts[name]+=1; return
  counts[f'x{z[1]}' if z[0]=='x' else z[0]]+=1
  for q in z[1:]:
   if isinstance(q,tuple): rec(q)
 rec(a); return counts
def csize(a,macros): return int(sum(productions(a,macros).values()))
def stable_A(r):
 a=r.normal(scale=.18,size=(4,4)); a-=np.eye(4)*(.4+.1*r.random()); return a
def system(r,fn,kind='ast'):
 c=r.normal(size=4); c=c/max(np.linalg.norm(c),1e-12)*r.uniform(.5,1.5); return {'A':stable_A(r),'c':c,'fn':fn,'kind':kind}
def fval(sysm,x):
 if sysm['kind']=='ast': return ev(sysm['fn'],x)
 centers,w,bw=sysm['fn']; d=x[:,None,:]-centers[None,:,:]; return np.exp(-(d*d).sum(2)/(2*bw*bw))@w
def trans(sysm,x,g): g.charge('generated_transitions',len(x));g.charge('generated_rhs_evaluations',len(x));return x+DT*(x@sysm['A'].T+fval(sysm,x)[:,None]*sysm['c'][None,:])
def fit_candidate(x,y,v,g,fit_rows):
 g.charge('candidate_evaluations');g.charge('coefficient_factorizations');g.charge('candidate_fit_rows',fit_rows)
 z=np.concatenate([x,v[:,None]],1); target=(y-x)/DT; return np.linalg.solve(z.T@z+1e-8*np.eye(5),z.T@target)
def predict(x,v,b): return x+DT*(np.concatenate([x,v[:,None]],1)@b)
def search(xf,yf,xs,ys,pool,budget,g,guide=None,features=None,macros=None):
 macros=macros or {}; records=[]
 if guide is None: order=sorted(range(len(pool)),key=lambda i:(csize(pool[i],macros),s(pool[i])))[:budget]
 else:
  ops,W=guide; logits=features@W; lp=logits-np.logaddexp.reduce(logits)
  def sc(i):
   pc=productions(pool[i],macros); return sum(pc.get(o,0)*lp[j] for j,o in enumerate(ops))-0.01*csize(pool[i],macros)
  order=sorted(range(len(pool)),key=lambda i:(-sc(i),csize(pool[i],macros),s(pool[i])))[:budget]
 for i in order:
  a=pool[i]; b=fit_candidate(xf,yf,ev(a,xf),g,len(xf)); pred=predict(xs,ev(a,xs),b); g.charge('candidate_selection_rows',len(xs)); e=float(np.mean((pred-ys)**2)); records.append((e,a,b))
 records.sort(key=lambda z:(z[0],size(z[1]),s(z[1]))); best=records[0][0]; energy=float(np.mean(((ys-xs)/DT)**2)); tol=max(1e-10,.01*energy); hyps=[z[1] for z in records if z[0]<=best+tol][:4]
 return records[0],hyps
def obs_features(x,y):
 target=(y-x)/DT; b=np.linalg.solve(x.T@x+1e-8*np.eye(4),x.T@target); r=target-x@b; u,sv,vh=np.linalg.svd(r,full_matrices=False); z=u[:,0]*sv[0]; j=int(np.argmax(np.abs(vh[0]))); z*=1 if vh[0,j]>=0 else -1
 funcs=[x[:,i] for i in range(4)]+[q(x[:,i]) for q in (np.sin,np.tanh,lambda a:np.maximum(a,0)) for i in range(4)]
 corr=[float(np.dot(z,v)/(np.linalg.norm(z)*np.linalg.norm(v)+1e-12)) for v in funcs]
 return np.array(corr+[float(np.mean(r*r)),float(sv[0]/(sv.sum()+1e-12)),1.0])
def macro_pool(base,macros):
 out={s(a):a for a in base}; ms=list(macros.values()); atoms=ms+[('x',i) for i in range(4)]
 for a in ms:
  for op in ('sin','tanh','relu'): out.setdefault(s((op,a)),(op,a))
  for b in atoms:
   for op in ('add','mul'):
    z=canon(op,a,b); out.setdefault(s(z),z)
 for i,a in enumerate(ms):
  for b in ms[i+1:]:
   for op in ('add','mul'):
    z=canon(op,a,b); out.setdefault(s(z),z)
 return list(out.values())
def main(a):
 plan=json.load(open(a.plan));out=Path(a.out);out.mkdir(parents=True,exist_ok=False);g=Guard(plan['ceilings']);write(out/'ledger.json',{'status':'SEALED_PRECALL','plan_sha256':sha(a.plan),'driver_sha256':sha(__file__),'ceilings':plan['ceilings'],'counts':g.n})
 r=np.random.default_rng(2026107001); probe=r.uniform(-1,1,(64,4)); base=bank(r,probe,g)
 freq=defaultdict(set); ast_by={}
 for i,a0 in enumerate(base):
  for z in subtrees(a0):
   if 3<=size(z)<=7: freq[s(z)].add(i);ast_by[s(z)]=z
 eligible=sorted([k for k,v in freq.items() if len(v)>=10],key=lambda k:(-len(freq[k]),k))
 if len(eligible)<2: raise RuntimeError('MOTIF_PREFLIGHT')
 m0=ast_by[eligible[0]]; m1=next(ast_by[k] for k in eligible[1:] if not contains(m0,ast_by[k]) and not contains(ast_by[k],m0))
 used=set(); acq_prog=[]
 for m in (m0,m1):
  choices=[base[i] for i in sorted(freq[s(m)]) if s(base[i]) not in used]
  if len(choices)<8: raise RuntimeError('PROGRAM_PREFLIGHT')
  acq_prog.extend(choices[:8]);used.update(s(z) for z in choices[:8])
 reuse=[canon(op,a0,b0) for op,a0,b0 in [('add',m0,m1),('mul',m0,m1),('add',('sin',m0),m1),('add',m0,('tanh',m1)),('mul',('sin',m0),m1),('mul',m0,('relu',m1)),('add',('relu',m0),('sin',m1)),('mul',('tanh',m0),('sin',m1))]]
 nonreuse=[z for z in base if not contains(z,m0) and not contains(z,m1) and size(z)>=4][:8]
 hidden_strings=[s(z) for z in acq_prog+reuse+nonreuse]; write(out/'generator_seal.json',{'status':'FROZEN_BEFORE_RESPONSES','base_bank_sha256':hashlib.sha256('\n'.join(s(z) for z in base).encode()).hexdigest(),'hidden_programs_sha256':hashlib.sha256('\n'.join(hidden_strings).encode()).hexdigest(),'motif_hashes':[hashlib.sha256(s(z).encode()).hexdigest() for z in (m0,m1)],'schedule_counts':{'acquisition':16,'reuse':8,'nonreuse':8,'outside_finite_bank':8}})
 xf=r.uniform(-1,1,(32,4));xs=r.uniform(-1,1,(16,4));xc=r.uniform(-1,1,(32,4));xq=r.uniform(-1,1,(64,4))
 acq=[system(r,z) for z in acq_prog]; ayf=[trans(z,xf,g) for z in acq];ays=[trans(z,xs,g) for z in acq]
 solved=[]
 for i in range(16):
  best,hyps=search(xf,ayf[i],xs,ays[i],base,2048,g);solved.append({'best':best[0],'hyps':hyps,'features':obs_features(xf,ayf[i])})
 cand=defaultdict(lambda:{'tasks':set(),'weight':0.0,'ast':None})
 for ti in range(12):
  for h in solved[ti]['hyps']:
   for z in set(subtrees(h)):
    if 3<=size(z)<size(h):
     k=s(z);cand[k]['tasks'].add(ti);cand[k]['weight']+=(size(z)-1)/len(solved[ti]['hyps']);cand[k]['ast']=z;g.charge('compression_subtrees')
 proposals=[]
 for k,v in cand.items():
  saving=v['weight']-size(v['ast'])
  if len(v['tasks'])>=3 and saving>0:proposals.append((saving,k,v['ast']))
 proposals.sort(key=lambda z:(-z[0],z[1])); proposals=proposals[:2]
 macros={};admission=[]
 current_pool=base
 for pi,(_,_,proposal) in enumerate(proposals):
  trial=dict(macros);trial[f'M{pi}']=proposal;newpool=macro_pool(base,trial); old=[];new=[]
  for i in range(12,16):
   old.append(search(xf,ayf[i],xs,ays[i],current_pool,512,g,macros=macros)[0][0]);new.append(search(xf,ayf[i],xs,ays[i],newpool,512,g,macros=trial)[0][0])
  accept=sum(new)<=.8*sum(old) and sum(e<=max(1e-8,10*np.quantile([z['best'] for z in solved[:12]],.95)) for e in new)>=sum(e<=max(1e-8,10*np.quantile([z['best'] for z in solved[:12]],.95)) for e in old)
  admission.append({'proposal_hash':hashlib.sha256(s(proposal).encode()).hexdigest(),'old_error':sum(old),'new_error':sum(new),'accepted':bool(accept)})
  if accept: macros=trial;current_pool=newpool
 threshold=max(1e-8,10*float(np.quantile([z['best'] for z in solved],.95)))
 ops=list(BASE_OPS)+list(macros);X=np.stack([z['features'] for z in solved]);Y=[]
 for z in solved:
  q=np.zeros(len(ops))
  for h in z['hyps']:
   pc=productions(h,macros)
   for j,o in enumerate(ops):q[j]+=pc.get(o,0)/len(z['hyps'])
  Y.append(q)
 Y=np.stack(Y);g.charge('guide_fits');W=np.linalg.solve(X.T@X+1e-3*np.eye(X.shape[1]),X.T@Y);guide=(ops,W)
 write(out/'learning_barrier.json',{'status':'LIBRARY_AND_GUIDE_FROZEN_BEFORE_LATER_RESPONSES','admission':admission,'admitted_macros':{k:s(v) for k,v in macros.items()},'threshold':threshold,'guide_shape':list(W.shape),'counts':g.n.copy()})
 later=[system(r,z) for z in reuse+nonreuse]
 for _ in range(8):
  centers=r.uniform(-1,1,(8,4));w=r.normal(size=8);later.append(system(r,(centers,w,.65),'rbf'))
 lyc=[trans(z,xc,g) for z in later];preds={};meta=[];libpool=macro_pool(base,macros)
 for i,z in enumerate(later):
  feat=obs_features(xc,lyc[i]);
  arms={'neither':(base,None,{}),'library_only':(libpool,None,macros),'guide_only':(base,guide,{}),'both':(libpool,guide,macros)}
  for arm,(pool,gd,mm) in arms.items():
   best,_=search(xc[:16],lyc[i][:16],xc[16:],lyc[i][16:],pool,512,g,guide=gd,features=feat,macros=mm); b=fit_candidate(xc,lyc[i],ev(best[1],xc),g,len(xc));preds[f'{i}_{arm}']=predict(xq,ev(best[1],xq),b);meta.append({'system':i,'arm':arm,'best_context_error':best[0],'abstained':bool(best[0]>threshold),'program':s(best[1])})
 np.savez_compressed(out/'predictions_before_query_responses.npz',**preds);write(out/'prediction_barrier.json',{'status':'PERSISTED_BEFORE_QUERY_RESPONSES_AND_HIDDEN_GROUPS','sha256':sha(out/'predictions_before_query_responses.npz'),'metadata':meta,'counts':g.n.copy()})
 rows=[]
 for i,z in enumerate(later):
  yq=trans(z,xq,g);group='reuse' if i<8 else 'nonreuse' if i<16 else 'outside_bank'
  for arm in ('neither','library_only','guide_only','both'):
   g.charge('query_prediction_blocks');g.charge('query_prediction_rows',len(xq));g.charge('score_blocks');rows.append({'system':i,'group':group,'arm':arm,'mse':float(np.mean((preds[f'{i}_{arm}']-yq)**2)),'abstained':next(q['abstained'] for q in meta if q['system']==i and q['arm']==arm)})
 def vals(group,arm):return[q['mse'] for q in rows if q['group']==group and q['arm']==arm]
 def cmp(group,a,b):
  aa,bb=vals(group,a),vals(group,b);return{'ratio':float(sum(aa)/sum(bb)),'wins':int(sum(x<y for x,y in zip(aa,bb))),'n':len(aa)}
 comps={'reuse_both_neither':cmp('reuse','both','neither'),'reuse_both_guide':cmp('reuse','both','guide_only'),'reuse_both_library':cmp('reuse','both','library_only'),'nonreuse_both_neither':cmp('nonreuse','both','neither')}
 out_abs=sum(q['abstained'] for q in rows if q['group']=='outside_bank' and q['arm']=='both');exp_abs=sum(q['abstained'] for q in rows if q['group']!='outside_bank' and q['arm']=='both')
 gates={'joint_reuse_benefit':comps['reuse_both_neither']['ratio']<=.8 and comps['reuse_both_neither']['wins']>=6,'library_increment':comps['reuse_both_guide']['ratio']<=.85 and comps['reuse_both_guide']['wins']>=6,'guide_increment':comps['reuse_both_library']['ratio']<=.85 and comps['reuse_both_library']['wins']>=6,'nonreuse_nonharm':comps['nonreuse_both_neither']['ratio']<=1.1,'abstention':out_abs>=6 and exp_abs<=2,'library_admission':len(macros)>=1}
 result={'classification':'WAKE_COMPRESS_GUIDE_SUPPORTED_DEVELOPMENT_ONLY' if all(gates.values()) else 'WAKE_COMPRESS_GUIDE_NOT_SUPPORTED','gates':gates,'comparisons':comps,'admission':admission,'admitted_macros':{k:s(v) for k,v in macros.items()},'abstention':{'outside_bank':out_abs,'expressible':exp_abs},'rows':rows,'counts':g.n,'qualification':plan['qualification'],'hidden_generator_reveal_after_scoring':{'motifs':[s(m0),s(m1)],'acquisition':[s(z) for z in acq_prog],'reuse':[s(z) for z in reuse],'nonreuse':[s(z) for z in nonreuse]}}
 write(out/'result.json',result);write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--out');main(ap.parse_args())
