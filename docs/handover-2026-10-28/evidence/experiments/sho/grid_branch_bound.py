"""Deterministic grid branch-and-bound scaffold with locked learned comparisons."""
import math,time
import numpy as np
import torch
from scipy.spatial import cKDTree


class GridBranchBound:
    def __init__(self,distance,margin,threshold,pair_block=262144):
        self.distance=distance;self.w=margin.weight.detach().numpy()[0].copy()
        self.theta=float(threshold.theta());self.alpha=float(threshold.alpha());self.pair_block=pair_block
        # Euclidean box bounds require this exact audited metric, not a general PSD metric.
        if not np.array_equal(distance.metric_matrix().detach().numpy(),np.eye(2)):
            raise ValueError('Euclidean branch bound requires the locked identity metric')

    def replace(self,old,new,query,new_site,tie_owner):
        equal=old==new;result=np.zeros(len(old),dtype=bool);active=~equal
        if np.any(active):
            scale=np.maximum(old[active],new[active])
            margin=self.w[0]*(old[active]/scale)+self.w[1]*(new[active]/scale)
            result[active]=self.alpha*(margin-self.theta)>0
        result[equal]=tie_owner[query[equal]]==new_site[equal]
        return result

    def reduce(self,qid,site,distance,tie_owner):
        # qid is sorted; each query has a contiguous, variable-length candidate group.
        while len(qid)>1 and np.any(qid[1:]==qid[:-1]):
            starts=np.flatnonzero(np.r_[True,qid[1:]!=qid[:-1]])
            count=np.diff(np.r_[starts,len(qid)])
            position=np.arange(len(qid))-np.repeat(starts,count)
            left=np.flatnonzero(position%2==0)
            paired=(left+1<len(qid))
            paired[paired]&=qid[left[paired]]==qid[left[paired]+1]
            selected=left.copy();li=left[paired];ri=li+1
            replace=self.replace(distance[li],distance[ri],qid[li],site[ri],tie_owner)
            selected[np.flatnonzero(paired)[replace]]=ri[replace]
            qid,site,distance=qid[selected],site[selected],distance[selected]
        return qid,site,distance

    def search(self,sites,probes):
        started=time.perf_counter();n,p=len(sites),len(probes)
        if n<2:raise ValueError('at least two distinct evaluated sites required')
        g=math.ceil(math.sqrt(n));edges=np.arange(g+1,dtype=np.float64)/g
        queries=np.concatenate((probes,sites));q=len(queries)
        sb=np.minimum(np.searchsorted(edges,sites,side='right')-1,g-1)
        qb=np.minimum(np.searchsorted(edges,queries,side='right')-1,g-1)
        site_bin=sb[:,0]*g+sb[:,1];query_bin=qb[:,0]*g+qb[:,1]
        bin_sites=[np.flatnonzero(site_bin==j) for j in range(g*g)]
        tree=cKDTree(sites);pd,_=tree.query(probes,k=2);sd,_=tree.query(sites,k=3)
        tied=np.r_[pd[:,0]==pd[:,1],sd[:,1]==sd[:,2]]
        tie_owner=np.full(q,-1,dtype=np.int64);pt=np.flatnonzero(tied[:p]);st=np.flatnonzero(tied[p:])
        if len(pt):tie_owner[pt]=tree.query(probes[pt],k=1)[1]
        if len(st):tie_owner[p+st]=tree.query(sites[st],k=2)[1][:,1]
        setup=time.perf_counter()-started
        self_id=np.r_[np.full(p,-1,dtype=np.int64),np.arange(n)]
        owner=np.full(q,-1,dtype=np.int64);best=np.full(q,np.inf)
        stop_ring=np.full(q,-1,dtype=np.int64);bins_searched=np.zeros(q,dtype=np.int64)
        evaluated=np.zeros(q,dtype=np.int64);stop_bound=np.full(q,np.nan)
        active=np.arange(q);distance_seconds=0.;neural_calls=0;equal_bound_continuations=0
        for ring in range(g):
            if not len(active):break
            occupied=np.unique(query_bin[active]);counts=np.zeros(g*g,dtype=np.int64)
            starts=np.zeros(g*g,dtype=np.int64);parts=[];offset=0
            if ring==0:delta=np.array([[0,0]],dtype=np.int64)
            else:delta=np.array([(x,y) for x in range(-ring,ring+1) for y in (-ring,ring)]+
                [(x,y) for x in (-ring,ring) for y in range(-ring+1,ring)],dtype=np.int64)
            for b in occupied:
                neighbors=np.array([b//g,b%g])+delta
                neighbors=neighbors[np.all((neighbors>=0)&(neighbors<g),axis=1)]
                incoming=np.concatenate([bin_sites[x*g+y] for x,y in neighbors]) if len(neighbors) else np.empty(0,dtype=np.int64)
                starts[b]=offset;counts[b]=len(incoming);offset+=len(incoming);parts.append(incoming)
            flat=np.concatenate(parts) if parts else np.empty(0,dtype=np.int64)
            per_query=counts[query_bin[active]]
            qid=np.repeat(active,per_query)
            if len(qid):
                offsets=np.cumsum(per_query)-per_query
                site=flat[np.repeat(starts[query_bin[active]],per_query)+np.arange(len(qid))-np.repeat(offsets,per_query)]
                eligible=site!=self_id[qid];qid=qid[eligible];site=site[eligible]
                evaluated+=np.bincount(qid,minlength=q)
                t=time.perf_counter();dist=[]
                with torch.no_grad():
                    for begin in range(0,len(qid),self.pair_block):
                        ids=qid[begin:begin+self.pair_block];ss=site[begin:begin+self.pair_block]
                        dist.append(self.distance(torch.from_numpy(queries[ids]-sites[ss])).numpy());neural_calls+=1
                distance_seconds+=time.perf_counter()-t
                if len(qid):
                    incoming_q,incoming_site,incoming_d=self.reduce(qid,site,np.concatenate(dist),tie_owner)
                    initialize=owner[incoming_q]<0;replace=initialize.copy()
                    ix=~initialize
                    replace[ix]=self.replace(best[incoming_q[ix]],incoming_d[ix],incoming_q[ix],incoming_site[ix],tie_owner)
                    chosen=incoming_q[replace];owner[chosen]=incoming_site[replace];best[chosen]=incoming_d[replace]
            low=np.maximum(qb[active]-ring,0);high=np.minimum(qb[active]+ring+1,g)
            # Complement of the clipped searched rectangle: minimum axis gap to
            # any exposed side. Full-domain sides have no unsearched region.
            lower=np.where(low>0,queries[active]-edges[low],np.inf)
            upper=np.where(high<g,edges[high]-queries[active],np.inf)
            unseen=np.minimum(lower.min(1),upper.min(1))
            finite=np.isfinite(unseen)
            # Directed float64 rounding, not an arbitrary tolerance. This can
            # only postpone stopping and includes potential equal-distance ties.
            unseen[finite]=np.maximum(0,np.nextafter(np.nextafter(unseen[finite],-np.inf),-np.inf))
            equal_bound_continuations+=int(np.sum((owner[active]>=0)&(unseen==best[active])))
            stop=(owner[active]>=0)&(unseen>best[active])
            ids=active[stop];stop_ring[ids]=ring;stop_bound[ids]=unseen[stop]
            bins_searched[ids]=np.prod(high[stop]-low[stop],axis=1)
            active=active[~stop]
        if len(active):raise RuntimeError('grid exhausted without completing every query')
        return dict(owner=owner,distance=best,sites_evaluated=evaluated,bins_searched=bins_searched,
            stop_ring=stop_ring,stop_bound=stop_bound,tied=tied,grid_bins=g,
            runtime_seconds=time.perf_counter()-started,setup_and_tie_seconds=setup,
            learned_distance_seconds=distance_seconds,distance_forward_calls=neural_calls,
            equality_bound_continuations=equal_bound_continuations)
