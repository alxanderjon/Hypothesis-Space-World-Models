#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def write(p,x):
    p=Path(p); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)); os.replace(t,p)

class Kernel:
    def __init__(self,x,w,bw,additive): self.x=x; self.w=w; self.bw=bw; self.additive=additive
    def __call__(self,q):
        d=q[:,None,:]-self.x[None,:,:]
        if self.additive:
            k=np.exp(-(d*d)/(2*self.bw[None,None,:]**2)).sum(2)
        else:
            k=np.exp(-(d*d).sum(2)/(2*self.bw**2))
        return k@self.w

def make_kernel(x,target,g,additive):
    d=x[:,None,:]-x[None,:,:]
    if additive:
        bw=[]
        for j in range(x.shape[1]):
            v=np.abs(d[:,:,j]); nz=v[v>0]; bw.append(float(np.median(nz)))
        bw=np.maximum(np.array(bw),1e-12); k=np.exp(-(d*d)/(2*bw[None,None,:]**2)).sum(2)
    else:
        v=np.sqrt((d*d).sum(2)); bw=float(np.median(v[v>0])); k=np.exp(-(d*d).sum(2)/(2*bw*bw))
    g.charge('kernel_factorizations')
    return Kernel(x,np.linalg.solve(k+1e-6*np.eye(len(x)),target),bw,additive)

def run_world(m,g,seed,degree):
    systems=m.make_systems(seed,degree); x=m.states(seed+1,48); xv=m.states(seed+2,32)
    ys=[m.trans(x,s,g) for s in systems]; yv=[m.trans(xv,s,g) for s in systems]
    rr=m.quotient(x,ys,g); sel,signs=m.signature(rr,g)
    target=np.mean([signs[i]*rr[i]/(np.sqrt(np.mean(rr[i]**2))+1e-12) for i in sel],0)
    broken=x.copy()
    for j in range(4): broken[:,j]=np.roll(broken[:,j],1+3*j)
    funcs={
        'joint_rbf':make_kernel(x,target,g,False),
        'additive_rbf':make_kernel(x,target,g,True),
        'pairing_broken_additive':make_kernel(broken,target,g,True),
    }
    rows=[]
    for arm,h in funcs.items():
        hx=h(x if arm!='pairing_broken_additive' else broken); hv=h(xv); g.charge('prediction_blocks',2)
        for i in sel:
            pred=m.pred_aug(xv,hv,m.fit_aug(x,ys[i],hx,g))
            rows.append({'arm':arm,'system':int(i),'mse':m.score(pred,yv[i],g)})
    linear=[]
    for i in sel:
        pred=xv+m.DT*(xv@m.fit_linear(x,ys[i],g)); linear.append(m.score(pred,yv[i],g))
    def vals(a): return [r['mse'] for r in rows if r['arm']==a]
    def cmp(a,b):
        aa=vals(a); bb=linear if b=='linear' else vals(b)
        return {'ratio':float(sum(aa)/sum(bb)),'wins':int(sum(u<v for u,v in zip(aa,bb))),'n':len(aa)}
    return {'degree':degree,'selected_systems':[int(i) for i in sel],'rows':rows,'linear_mse':linear,
            'comparisons':{'additive_vs_joint':cmp('additive_rbf','joint_rbf'),
                           'additive_vs_broken':cmp('additive_rbf','pairing_broken_additive'),
                           'additive_vs_linear':cmp('additive_rbf','linear'),
                           'joint_vs_linear':cmp('joint_rbf','linear')}}

def main(a):
    plan=json.load(open(a.plan)); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('base',a.base); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    g=m.Guard(plan['ceilings']); write(out/'ledger.json',{'status':'SEALED_PRECALL','plan_sha256':sha(a.plan),'driver_sha256':sha(__file__),'base_sha256':sha(a.base),'ceilings':plan['ceilings'],'counts':g.n})
    worlds=[run_world(m,g,2026106100,3),run_world(m,g,2026106200,5),run_world(m,g,2026106300,7)]
    by={w['degree']:w for w in worlds}
    gates={
      'p5_additive_vs_joint':by[5]['comparisons']['additive_vs_joint']['ratio']<=.8 and by[5]['comparisons']['additive_vs_joint']['wins']>=6,
      'p7_additive_vs_joint':by[7]['comparisons']['additive_vs_joint']['ratio']<=.8 and by[7]['comparisons']['additive_vs_joint']['wins']>=6,
      'p7_additive_vs_linear':by[7]['comparisons']['additive_vs_linear']['ratio']<=.8 and by[7]['comparisons']['additive_vs_linear']['wins']>=6,
      'p7_pairing_control':by[7]['comparisons']['additive_vs_broken']['ratio']<=.8 and by[7]['comparisons']['additive_vs_broken']['wins']>=6,
      'p3_nonharm':by[3]['comparisons']['additive_vs_joint']['ratio']<=1.1,
    }
    res={'classification':'ADDITIVE_REPRESENTATION_SUPPORTED_DEVELOPMENT_ONLY' if all(gates.values()) else 'ADDITIVE_REPRESENTATION_NOT_SUPPORTED','gates':gates,'worlds':worlds,'counts':g.n,'qualification':plan['qualification']}
    write(out/'result.json',res); write(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':g.n})

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base'); ap.add_argument('--plan'); ap.add_argument('--out'); main(ap.parse_args())
