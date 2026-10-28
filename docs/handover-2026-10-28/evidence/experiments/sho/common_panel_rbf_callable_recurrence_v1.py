#!/usr/bin/env python3
import argparse, importlib.util, numpy as np, torch

class KernelCallable:
    def __init__(self,x,w,bw): self.x=x.copy(); self.w=w.copy(); self.bw=float(bw)
    def predict(self,x):
        d=((x[:,None,:]-self.x[None,:,:])**2).sum(2)
        return np.exp(-d/(2*self.bw*self.bw))@self.w

def gen_p3(rng):
    m=rng.normal(size=(4,4)); return m/max(np.linalg.norm(m),1e-12)*2.0

def apply_p3(x,m): return ((5.0*x**3-3.0*x)/2.0)@m

def train_kernel(xtrain,residuals,selected,signs,shuffled,seed,guard):
    ys=[]
    for i in selected:
        r=residuals[i].copy()
        if shuffled:
            p=np.random.default_rng(seed+2003*i).permutation(len(xtrain)); r=r[p]
        scale=np.sqrt(np.mean(r*r))+1e-12; ys.append(signs[i]*r/scale)
    y=np.mean(np.stack(ys),axis=0)
    d=np.sqrt(((xtrain[:,None,:]-xtrain[None,:,:])**2).sum(2)); bw=float(np.median(d[d>0]))
    k=np.exp(-(d*d)/(2*bw*bw)); guard.charge('shared_fits'); guard.charge('coefficient_factorizations')
    w=np.linalg.solve(k+1e-6*np.eye(len(k)),y)
    return KernelCallable(xtrain,w,bw),[[1,float(np.mean((k@w-y)**2))]]

def module_np_factory(base):
    def module_np(m,x,guard):
        guard.charge('prediction_blocks')
        if isinstance(m,KernelCallable): return m.predict(x)
        with torch.no_grad(): return m(torch.tensor(x,dtype=torch.float64)).numpy()
    return module_np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--driver',required=True); ap.add_argument('--plan',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    spec=importlib.util.spec_from_file_location('frozen_common_panel_driver',a.driver); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.WORLD_SEED=2026104600; m.SEEDS=[2026104701]; m.gen_net=gen_p3; m.apply_net=apply_p3; m.train_module=train_kernel; m.module_np=module_np_factory(m)
    m.main(argparse.Namespace(plan=a.plan,out=a.out))

if __name__=='__main__': main()
