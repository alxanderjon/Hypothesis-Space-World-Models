"""Inference-only execution of the locked float64 nearest operator.

No training, reference-distance validation, or non-tied KD-tree selection.
Exact-tie routing remains TEMPORARY DETERMINISTIC SCAFFOLDING.
"""
import math
import time
import numpy as np
import torch
from scipy.spatial import cKDTree


class FastNearestArgmin:
    def __init__(self, distance, margin, threshold, half, pair_block=262144, query_block=1024):
        self.distance,self.margin,self.threshold,self.half=distance,margin,threshold,half
        self.pair_block=int(pair_block);self.query_block=int(query_block)
        self.w=margin.weight.detach().cpu().numpy()[0].copy()
        self.theta=float(threshold.theta());self.alpha=float(threshold.alpha())

    def logits(self, current, candidate):
        scale=np.maximum(current,candidate)
        # Preserve the locked forward's two products and sum, without FMA/weight tying.
        margin=self.w[0]*(current/scale)+self.w[1]*(candidate/scale)
        return self.alpha*(margin-self.theta)

    def nearest(self, sites, probes):
        start=time.perf_counter();n,p=len(sites),len(probes)
        if n<2:raise ValueError('nearest-other-site requires at least two sites')
        queries=np.concatenate((probes,sites));q=len(queries)
        tree=cKDTree(sites)
        pd,_=tree.query(probes,k=2)
        sd,_=tree.query(sites,k=3)
        ptie=pd[:,0]==pd[:,1];stie=sd[:,1]==sd[:,2]
        tied=np.r_[ptie,stie]
        owners=np.full(q,-1,dtype=np.int64);winner_distance=np.empty(q,dtype=np.float64)
        tree_calls=2
        # k-dependent heap tie ordering cannot replace production ownership.
        # These conditional queries resolve only exact ties, all at once.
        if np.any(ptie):owners[np.flatnonzero(ptie)]=tree.query(probes[ptie],k=1)[1];tree_calls+=1
        if np.any(stie):owners[p+np.flatnonzero(stie)]=tree.query(sites[stie],k=2)[1][:,1];tree_calls+=1
        stats=dict(tie_precheck_ms=1000*(time.perf_counter()-start),kd_query_calls=tree_calls,
            probe_ties=int(ptie.sum()),spacing_ties=int(stie.sum()),distance_pairs=0,
            tournament_comparisons=0,tournament_round_calls=0,distance_calls=0,
            tournament_depth=math.ceil(math.log2(n)),distance_ms=0.,tournament_ms=0.,
            pair_block=self.pair_block,query_block=min(self.query_block,max(1,self.pair_block//n)))
        active=np.flatnonzero(~tied);block=stats['query_block']
        for begin in range(0,len(active),block):
            ids=active[begin:begin+block];t=time.perf_counter()
            delta=(queries[ids,None,:]-sites[None,:,:]).reshape(-1,sites.shape[1])
            parts=[]
            with torch.no_grad():
                for offset in range(0,len(delta),self.pair_block):
                    parts.append(self.distance(torch.from_numpy(delta[offset:offset+self.pair_block])).numpy())
                    stats['distance_calls']+=1
            d=np.concatenate(parts).reshape(len(ids),n)
            stats['distance_pairs']+=len(delta)
            stats['distance_ms']+=1000*(time.perf_counter()-t)
            t=time.perf_counter()
            spacing=ids>=p
            d[np.flatnonzero(spacing),ids[spacing]-p]=np.inf
            index=np.broadcast_to(np.arange(n,dtype=np.int64),d.shape)
            while d.shape[1]>1:
                pairs=d.shape[1]//2
                left=d[:,:2*pairs:2];right=d[:,1:2*pairs:2]
                li=index[:,:2*pairs:2];ri=index[:,1:2*pairs:2]
                both=np.isfinite(left)&np.isfinite(right)
                compare=both&(left!=right)
                replace=~np.isfinite(left)&np.isfinite(right)
                if np.any(compare):replace[compare]=self.logits(left[compare],right[compare])>0
                stats['tournament_comparisons']+=int(both.sum())
                stats['tournament_round_calls']+=1
                nd=np.where(replace,right,left);ni=np.where(replace,ri,li)
                if d.shape[1]%2:
                    nd=np.concatenate((nd,d[:,-1:]),axis=1)
                    ni=np.concatenate((ni,index[:,-1:]),axis=1)
                d,index=nd,ni
            owners[ids]=index[:,0];winner_distance[ids]=d[:,0]
            stats['tournament_ms']+=1000*(time.perf_counter()-t)
        ids=np.flatnonzero(tied)
        if len(ids):
            t=time.perf_counter()
            with torch.no_grad():winner_distance[ids]=self.distance(torch.from_numpy(queries[ids]-sites[owners[ids]])).numpy()
            stats['distance_pairs']+=len(ids);stats['distance_calls']+=1
            stats['distance_ms']+=1000*(time.perf_counter()-t)
        stats['nearest_total_ms']=1000*(time.perf_counter()-start)
        stats['comparisons_per_second']=stats['tournament_comparisons']/max(stats['tournament_ms']/1000,1e-30)
        return dict(probe_owner=owners[:p],spacing_owner=owners[p:],probe_distance=winner_distance[:p],spacing=winner_distance[p:],stats=stats)

    def larger(self,current,candidate):
        active=current!=candidate
        replace=np.zeros(len(current),dtype=bool)
        if np.any(active):replace[active]=self.logits(current[active],candidate[active])<0
        return replace

    def cells(self,sites,probes):
        start=time.perf_counter();out=self.nearest(sites,probes);n=len(sites)
        t=time.perf_counter();owners=out['probe_owner'];distances=out['probe_distance']
        counts=np.bincount(owners,minlength=n);order=np.argsort(owners,kind='stable')
        offsets=np.r_[0,np.cumsum(counts)];radius=np.zeros(n);witness=np.full(n,-1,dtype=np.int64)
        nonempty=np.flatnonzero(counts);first=order[offsets[nonempty]]
        radius[nonempty]=distances[first];witness[nonempty]=first
        for position in range(1,int(counts.max(initial=0))):
            active=np.flatnonzero(counts>position);incoming=order[offsets[active]+position]
            replace=self.larger(radius[active],distances[incoming]);chosen=active[replace]
            radius[chosen]=distances[incoming[replace]];witness[chosen]=incoming[replace]
        out['stats']['assigned_radius_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter()
        with torch.no_grad():half=self.half(torch.from_numpy(out['spacing'][:,None])).numpy()[:,0]
        replace=self.larger(radius,half);sizes=np.where(replace,half,radius)
        out['stats']['remaining_cell_size_ms']=1000*(time.perf_counter()-t)
        out['stats']['geometry_total_ms']=1000*(time.perf_counter()-start)
        out.update(radius=radius,radius_witness=witness,half=half,sizes=sizes)
        return out
