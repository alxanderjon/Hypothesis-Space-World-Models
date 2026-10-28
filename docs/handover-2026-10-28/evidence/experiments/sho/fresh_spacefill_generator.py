"""Geometry-only conditioned generator; existing partition/face scaffolding."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from experiments.sho.run_stage1b_spacefill_generator_smoke import partitions,faces

NOISE_SEED=27092026
class FreshGenerator(nn.Module):
    def __init__(self,dimension=2):
        super().__init__();self.dimension=dimension
        self.net=nn.Sequential(nn.Linear(3*dimension+4,24),nn.Tanh(),nn.Linear(24,dimension)).double()
        self.log_amplitude=nn.Parameter(torch.tensor(math.log(.8),dtype=torch.float64))
        self._partition_cache={}

    def inputs(self,n,iteration,pool_type):
        if pool_type not in ('cell','greedy') or iteration<0:raise ValueError('invalid condition')
        if n not in self._partition_cache:
            c,w=partitions(n,self.dimension);self._partition_cache[n]=(torch.from_numpy(c),torch.from_numpy(w))
        c,w=self._partition_cache[n]
        key=f'{NOISE_SEED}:{self.dimension}:{n}:{int(iteration)}:{pool_type}'.encode()
        seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
        noise=torch.from_numpy(np.random.Generator(np.random.PCG64(seed)).uniform(-1.7,1.7,(n,self.dimension)))
        context=torch.tensor([math.log(n)/12,math.sin(iteration),math.cos(iteration),float(pool_type=='greedy')],dtype=torch.float64).expand(n,-1)
        feature=torch.cat((2*c-1,w*n**(1/self.dimension)-1,noise,context),dim=1)
        return c,w,noise,feature

    def interior(self,n,iteration,pool_type):
        c,w,z,f=self.inputs(n,iteration,pool_type)
        return c+.49*w*torch.tanh(self.log_amplitude.exp()*z+.35*self.net(f))

    def generate(self,n,iteration,pool_type,lower=(0.,0.),upper=(1.,1.)):
        lo,hi=np.asarray(lower),np.asarray(upper)
        assert len(lo)==self.dimension and np.all(hi>lo)
        with torch.no_grad():interior=self.interior(n,iteration,pool_type).numpy()
        boundary,_=faces(n,self.dimension)
        points=np.unique(np.concatenate((interior,boundary)),axis=0)
        assert len(points)==n+len(boundary)
        return lo+(hi-lo)*points,lo+(hi-lo)*interior
