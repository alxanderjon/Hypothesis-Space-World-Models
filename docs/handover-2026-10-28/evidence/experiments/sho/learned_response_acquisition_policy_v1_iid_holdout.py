#!/usr/bin/env python3
import argparse,importlib.util,json,hashlib,os
from pathlib import Path
import numpy as np
def sha(p): h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def write(p,x):p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True));os.replace(t,p)
def load_model(x): return tuple(np.array(q,dtype=float) for q in x)
def main(a):
 plan=json.load(open(a.plan)); pol=json.load(open(a.policy)); spec=importlib.util.spec_from_file_location('orig',a.driver);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);g=m.Guard(plan['ceilings']);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);write(out/'ledger.json',{'status':'SEALED_PRECALL','plan_sha256':sha(a.plan),'driver_sha256':sha(__file__),'frozen_policy_sha256':sha(a.policy),'ceilings':plan['ceilings'],'counts':g.n})
 state_model=load_model(pol['state_model']); context_model=load_model(pol['context_model']);seed=2026105900;systems=m.make_systems(seed,5);x0=m.states(seed+1,32);xv=m.states(seed+2,32);pool=m.states(seed+3,32);y0=[m.trans(x0,s,g) for s in systems];yv=[m.trans(xv,s,g) for s in systems]
 r0=m.quotient(x0,y0,g);sel,signs=m.signature(r0,g);loos=[m.kernel(x0,r0,sel,signs,g,leave=int(i)) for i in sel];sf,cf,dis,dist=m.features(x0,pool,loos);g.charge('prediction_blocks',len(loos));g.charge('candidate_states_scored',len(loos)*len(pool))
 idx={};idx['random48']=np.random.default_rng(seed+4).choice(32,16,replace=False);idx['learned_state_only48']=np.argsort(-m.policy_score(state_model,sf),kind='stable')[:16];idx['learned_context48']=np.argsort(-m.policy_score(context_model,cf),kind='stable')[:16];remaining=np.ones(32,bool);md=dist.copy();chosen=[]
 for _ in range(16):
  sc=dis*np.maximum(md,1e-12);sc[~remaining]=-np.inf;j=int(np.argmax(sc));chosen.append(j);remaining[j]=False;md=np.minimum(md,np.sqrt(((pool-pool[j])**2).sum(1)))
 idx['frozen_disagreement48']=np.array(chosen);write(out/'selection_barrier.json',{'status':'INDICES_PERSISTED_BEFORE_RESPONSES','indices':{k:v.tolist() for k,v in idx.items()},'selected_systems':sel.tolist(),'counts':g.n.copy()})
 arms={'original32':(x0,y0)}
 for name,ii in idx.items():
  ys=[m.trans(pool[ii],s,g) for s in systems];arms[name]=(np.concatenate([x0,pool[ii]]),[np.concatenate([y0[i],ys[i]]) for i in range(16)])
 admissions=[];funcs={}
 for name,(x,ys) in arms.items():
  rr=m.quotient(x,ys,g) if name!='original32' else r0;h=m.kernel(x,rr,sel,signs,g);funcs[name]=h;val,vals=m.admission(x,ys,xv,yv,h,sel,g);lin=[]
  for i in sel:lin.append(m.score(xv+m.DT*(xv@m.fit_linear(x,ys[i],g)),yv[i],g))
  ratio=float(val/np.mean(lin));wins=int(sum(q<w for q,w in zip(vals,lin)));admissions.append({'arm':name,'ratio':ratio,'wins':wins,'passed':bool(ratio<=.8 and wins>=6)})
 write(out/'admission_decision.json',{'status':'FROZEN_BEFORE_TARGETS','admissions':admissions,'counts':g.n.copy()});context=[r for r in admissions if r['arm']=='learned_context48'][0]
 if not context['passed']:
  res={'classification':'FROZEN_CONTEXT_POLICY_NOT_SUPPORTED_IID','admissions':admissions,'counts':g.n,'qualification':plan['qualification']};write(out/'result.json',res);write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n});return
 xd=m.states(seed+101,32);xq=m.states(seed+102,64);rt=np.random.default_rng(seed+1000);shared=systems[0][2];target=[(m.stable_A(rt),rt.uniform(.6,1.4),shared if i<8 else m.M(rt),i<8,5) for i in range(16)];dev=[(s,m.trans(xd,s,g),m.trans(xq,s,g)) for s in target];preds={}
 for name,h in funcs.items():
  hd=h(xd);hq=h(xq);g.charge('prediction_blocks',2)
  for i,(s,yd,yq) in enumerate(dev):preds[f'{name}_{i}']=m.pred_aug(xq,hq,m.fit_aug(xd,yd,hd,g))
 for i,(s,yd,yq) in enumerate(dev):preds[f'linear_{i}']=xq+m.DT*(xq@m.fit_linear(xd,yd,g))
 np.savez_compressed(out/'predictions_before_hidden_labels.npz',**preds);write(out/'prediction_barrier.json',{'status':'PERSISTED_BEFORE_HIDDEN_SCORING','sha256':sha(out/'predictions_before_hidden_labels.npz'),'counts':g.n.copy()});rows=[]
 for i,(s,yd,yq) in enumerate(dev):
  for arm in list(arms)+['linear']:rows.append({'system':i,'group':'reuse' if s[3] else 'unrelated','arm':arm,'MSE':m.score(preds[f'{arm}_{i}'],yq,g)})
 def v(group,arm):return[r['MSE'] for r in rows if r['group']==group and r['arm']==arm]
 cc=v('reuse','learned_context48');rr=v('reuse','random48');ss=v('reuse','learned_state_only48');hh=v('reuse','frozen_disagreement48');oo=v('reuse','original32');au=v('unrelated','learned_context48');ll=v('unrelated','linear');rat=lambda x,y:sum(x)/sum(y)
 comp={'context_over_random':rat(cc,rr),'context_wins_random':sum(x<y for x,y in zip(cc,rr)),'context_over_state_only':rat(cc,ss),'context_wins_state_only':sum(x<y for x,y in zip(cc,ss)),'context_over_heuristic':rat(cc,hh),'context_wins_heuristic':sum(x<y for x,y in zip(cc,hh)),'context_over_original':rat(cc,oo),'context_wins_original':sum(x<y for x,y in zip(cc,oo)),'context_unrelated_over_linear':rat(au,ll)}
 passed=comp['context_over_random']<=.85 and comp['context_wins_random']>=6 and comp['context_over_state_only']<=.85 and comp['context_wins_state_only']>=6 and comp['context_over_heuristic']<=.85 and comp['context_wins_heuristic']>=6 and comp['context_over_original']<=.8 and comp['context_wins_original']>=6 and comp['context_unrelated_over_linear']<=1.1
 res={'classification':'FROZEN_CONTEXT_POLICY_SUPPORTED_IID_DEVELOPMENT_ONLY' if passed else 'NOT_SUPPORTED','admissions':admissions,'comparison':comp,'scores':rows,'counts':g.n,'qualification':plan['qualification']};write(out/'result.json',res);write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--driver');ap.add_argument('--policy');ap.add_argument('--plan');ap.add_argument('--out');main(ap.parse_args())
