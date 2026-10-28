"""Recover original Sobol distribution metrics; compare unchanged placement.

No training, checkpoint modification, objective evaluation, or N1 rollout.
"""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc

from experiments.sho import conditioned_leaf_placement_screen as placement
from experiments.sho import run_stage1b_edirect_vs_direct as ed
from experiments.sho.evaluate_fresh_spacefill_smoke import metrics, overlaps
from experiments.sho.train_anchored_discrepancy_curriculum import hard_metrics

OUT=placement.ROOT/'sobol_placement_target_audit'
CP=placement.ROOT/'conditioned_leaf_placement_refinement/best_refined_diagnostic.pt'
SCALES=(2,4,8,16,32,64,128)


def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def frozen_leaves():
    selector=placement.load_partition_runtime()
    root=dict(d=2,n=1000,lower=[0.,0.],upper=[1.,1.])
    stack=[('r',root)];leaves=[];nodes=[];start=time.monotonic()
    while stack:
        path,p=stack.pop()
        if p['n']==1:
            leaves.append(dict(path=path,parent=p));continue
        _,got,_=selector.select(p,'sobol_target_n1000:'+path)
        assert sum(got['counts'])==p['n'] and min(got['counts'])>0
        nodes.append(dict(path=path,n=p['n'],axis=got['axis'],cut=got['cut'],counts=got['counts'],identity=got['identity']))
        for j in (1,0):
            stack.append((path+str(j),dict(d=2,n=got['counts'][j],lower=got['child_lower'][j],upper=got['child_upper'][j])))
        if len(nodes)==1 or len(nodes)%25==0:
            print(json.dumps(dict(event='frozen_partition',splits=len(nodes),leaves=len(leaves),pending=len(stack),seconds=time.monotonic()-start)),flush=True)
    assert len(leaves)==1000
    dump('continuous_n1000_leaves.json',leaves);dump('continuous_n1000_splits.json',nodes)
    return leaves


def multiscale(p,n,name,it,pool):
    result=[]
    hard=hard_metrics(p,SCALES)
    for bins in SCALES:
        for shift in (0.,.25,.5,.75):
            shifted=(p+shift/bins)%1.
            count=np.histogramdd(shifted,bins=(np.linspace(0,1,bins+1),)*2)[0]
            edge=np.arange(1,bins+1)/bins
            error=count.cumsum(0).cumsum(1)/n-edge[:,None]*edge[None,:]
            result.append(dict(generator=name,N=n,iteration=it,pool=pool,scale=bins,shift=shift,
                grid_anchored_max=float(abs(error).max()),grid_anchored_rms=float(np.sqrt(np.mean(error**2))),
                count_variance=float(count.var()),count_fano=float(count.var()/count.mean()),empty_fraction=float(np.mean(count==0)),
                inclusive_unshifted_max=hard[bins]['max'],inclusive_unshifted_rms=hard[bins]['rms']))
    return result


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False);start=time.monotonic()
    protected={str(p):placement.sha(p) for p in placement.ROOT.rglob('*.pt')}
    protected[str(CP)]=placement.sha(CP)
    dump('protected_hashes.json',protected)
    dump('plan.json',dict(N=1000,sites=10,generator_id=200136,conditions=[[i,'cell'] for i in range(100,104)]+[[100,'greedy']],
        model=str(CP),checkpoint_sha256=placement.sha(CP),train=False,
        reference='Original fresh-spacefill recipe:512x512 midpoint grid plus262144 independent uniform points; seed27093001',
        metrics_authority=['evaluate_fresh_spacefill_smoke.metrics,overlaps','train_anchored_discrepancy_curriculum.hard_metrics',
            'train_dyadic_spacefill_curriculum sampled pairwise p01/p05','run_stage1b_edirect_vs_direct cell radius + half spacing'],
        new_acceptance_metric=False,scope='Static matched N1 pool comparison, not objective/search or learned partition acceptance extension'))
    try:
        leaves=frozen_leaves()
    except Exception as ex:
        dump('results.json',dict(status='STOPPED_FROZEN_PARTITION_DEPENDENCY',error=repr(ex),training_run=False))
        raise
    c=placement.case_from_leaves('N1_generator_200136_N1000',leaves,'audit')
    model=placement.Placement(8);model.load_state_dict(torch.load(CP,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    sites=ed._initial_design();center=(c['lo']+c['hi'])/2;face=placement.edges(1000,2)
    t=(np.arange(512)+.5)/512
    ref=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    cond=[(i,'cell') for i in range(100,104)]+[(100,'greedy')]
    arrays={};geo=[];ms=[];fr=[];cells=[];sets={name:[] for name in ('Sobol','centers','diagnostic')}
    tree=cKDTree(sites);half=.5*tree.query(sites,k=2)[0][:,1]
    dense_radius=np.zeros(len(sites));dd,oo=tree.query(ref)
    np.maximum.at(dense_radius,oo,dd)
    bd,bo=tree.query(face);np.maximum.at(dense_radius,bo,bd)
    rng=np.random.default_rng(2026092909)
    ia=rng.integers(0,1000,131072);ib=rng.integers(0,999,131072);ib+=ib>=ia
    for it,pool in cond:
        sobol=ed._candidate_pool(10,200136,it+(1000000 if pool=='greedy' else 0))
        interior_sobol=sobol[~np.any((sobol==0)|(sobol==1),axis=1)]
        with torch.no_grad():learned=placement.placed(model,c,it,int(pool=='greedy'),sites).numpy()
        for name,interior in [('Sobol',interior_sobol),('centers',center),('diagnostic',learned)]:
            points=np.unique(np.concatenate((interior,face)),axis=0)
            assert len(interior)==1000 and len(points)==1124
            if name=='Sobol':assert np.array_equal(points,sobol)
            assert np.array_equal(points[np.any((points==0)|(points==1),axis=1)],face)
            m=metrics(points,interior,ref,1000)
            pair=np.linalg.norm(interior[ia]-interior[ib],axis=1)
            near=cKDTree(interior).query(interior,k=2)[0][:,1]
            row=dict(generator=name,N=1000,iteration=it,pool=pool,**m,
                     centered_discrepancy=float(qmc.discrepancy(interior,method='CD')),
                     pairwise_p01=float(np.quantile(pair,.01)),pairwise_p05=float(np.quantile(pair,.05)),
                     nn_p01=float(np.quantile(near,.01)),nn_p05=float(np.quantile(near,.05)),
                     stale_history=int(np.sum(tree.query(interior)[0]<=1e-12)))
            geo.append(row);ms.extend(multiscale(interior,1000,name,it,pool));sets[name].append((points,interior))
            arrays[f'{name}_{it}_{pool}']=interior
            dist,owner=tree.query(points);radius=np.zeros(10);np.maximum.at(radius,owner,dist)
            sizes=np.maximum(radius,half);refsize=np.maximum(dense_radius,half)
            for j in range(10):
                cells.append(dict(generator=name,iteration=it,pool=pool,site=j,assigned_count=int(np.sum(owner==j)),
                                  assigned_radius=float(radius[j]),half_spacing=float(half[j]),cell_size=float(sizes[j]),
                                  dense_reference_radius=float(dense_radius[j]),radius_error=float(radius[j]-dense_radius[j]),
                                  dense_reference_cell_size=float(refsize[j]),cell_size_error=float(sizes[j]-refsize[j])))
        print(json.dumps(dict(event='matched_condition',iteration=it,pool=pool,seconds=time.monotonic()-start)),flush=True)
    for name,pools in sets.items():
        for i,j,label in ((0,1,'cell_t_to_t1'),(1,2,'cell_t1_to_t2'),(2,3,'cell_t2_to_t3'),(0,4,'cell_to_greedy')):
            fr.append(dict(generator=name,kind='overlap',comparison=label,**overlaps(pools[i][1],pools[j][1],1000)))
        union=np.unique(np.concatenate([x[0] for x in pools[:4]]),axis=0);d=cKDTree(union).query(ref)[0]
        single=[r['complete_cover'] for r in geo if r['generator']==name and r['pool']=='cell']
        fr.append(dict(generator=name,kind='union',unique_count=len(union),union_complete_cover=float(d.max()),
                       union_gain_vs_mean_single=1-float(d.max())/np.mean(single),union_large_holes=int(np.sum(d>1/math.sqrt(4000)))))
    # Feasibility diagnostic uses the SAME existing inclusive anchor errors:
    # at each anchor, one point/leaf can only choose a count in [forced,possible].
    feasibility=[]
    for k in SCALES:
        a=np.arange(1,k+1)/k;anchors=np.array(np.meshgrid(a,a)).reshape(2,-1).T
        forced=np.sum(np.all(c['hi'][None]<=anchors[:,None],axis=2),axis=1)
        possible=np.sum(np.all(c['lo'][None]<anchors[:,None],axis=2),axis=1)
        target=anchors.prod(1)*1000
        lb=np.maximum(np.maximum(forced-target,target-possible),0)/1000
        feasibility.append(dict(scale=k,forced_count_bound_max=float(lb.max()),unreachable_uniform_anchors=int(np.sum(lb>0))))
    write('geometry.csv',geo);write('multiscale.csv',ms);write('freshness_union.csv',fr);write('cell_sizes.csv',cells);write('leaf_feasibility.csv',feasibility)
    np.savez_compressed(OUT/'matched_pools.npz',sites=sites,boundary=face,**arrays)
    assert all(placement.sha(p)==h for p,h in protected.items())
    dump('results.json',dict(status='AUDIT_COMPLETE',training_run=False,conditions=5,N=1000,
                             protected_files=len(protected),protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
