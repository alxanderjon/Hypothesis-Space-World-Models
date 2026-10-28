"""Bounded recursion audit; deterministic proposals/centers are scaffolding only."""
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.sho.train_recursive_region_splitter_poc import LockedAllocation,CUTS,children,quality,ALLOC,ALLOC_SHA
from experiments.sho.train_finite_split_score_smoke import load_frozen,FiniteSplitScore,upstream,predict,synthetic_teacher
from experiments.sho.train_split_validity_smoke import ValidityMargin,ValidityThreshold,route
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv

ROOT=Path('outputs/sho/stage1B');OUT=ROOT/'recursive_split_assembly'
SEED=27262026


def canonical(score):
    best=float(np.min(score))
    if not np.isfinite(best):raise RuntimeError('No finite split proposal')
    return int(np.flatnonzero(score<=best+1e-12*max(1.,best))[0])


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    # Leave room to serialize partial results well before the total ten-minute cap.
    def deadline():
        if time.monotonic()-started>300:raise TimeoutError('Five-minute execution budget reached; saving partial results.')
    lockpath=ROOT/'split_validity_smoke/split_score_operator_locked.pt'
    lm=json.loads((ROOT/'split_validity_smoke/locked_manifest.json').read_text())
    assert sha(lockpath)==lm['split_score_sha256']
    lock=torch.load(lockpath,weights_only=False);assert lock['status']=='LOCKED'
    for c in lock['components'].values():assert sha(c['checkpoint'])==c['sha256']
    assert sha(ALLOC)==ALLOC_SHA
    protected={str(p):sha(p) for p in ROOT.glob('*/*.pt')}
    models,execution,_=load_frozen()
    finite=FiniteSplitScore();finite.load_state_dict(torch.load(lock['components']['finite_score']['checkpoint'],weights_only=False)['state_dict'])
    gate=torch.nn.ModuleDict(dict(margin=ValidityMargin(),threshold=ValidityThreshold()))
    gate.load_state_dict(torch.load(lock['components']['validity']['checkpoint'],weights_only=False)['state_dict'])
    for m in (finite,gate):m.eval().requires_grad_(False)
    allocators={mode:LockedAllocation() for mode in ('teacher','learned')}
    config=dict(budgets=[32,64,128,256],dimensions=[2,3,4],boxes=['unit','moderate_aspect'],cuts=CUTS.tolist(),
        proposal_order='axis ascending then cut ascending',canonical_rule='first score <= best + 1e-12*max(1,best)',
        teacher_allocator='same frozen learned proportional allocator and audited deterministic largest remainder',
        teacher_score='original 2D quality(); dimension-general audited extension for d=3,4',
        center_rule='TEMPORARY DIAGNOSTIC SCAFFOLDING: (lower+upper)/2',
        reference='16384 independent uniform interior points + 2048 points on each of 2d faces + all corners',
        large_hole_threshold='(root_volume/N)**(1/d), matching prior spacing-scale diagnostic',
        pathology_gates=dict(max_depth=64,max_leaf_aspect=16,max_normalized_cover=3,min_leaf_volume_positive=True),
        protected_hashes=protected,seed=SEED)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    fidelity=[];geometry=[];traces={};failure=None;candidate_evaluations=0

    def recurse(extent,n,mode,reference_tree=None,case=''):
        nonlocal candidate_evaluations
        allocator=allocators[mode];d=len(extent);active=[('r',np.zeros(d),extent.copy(),n,0)];nodes={};leaves={}
        while active:
            deadline();path,lo,hi,budget,depth=active.pop()
            if depth>=64:raise RuntimeError('Depth safeguard exceeded')
            if budget==1:
                leaves[path]=dict(lower=lo.tolist(),upper=hi.tolist(),budget=1,depth=depth);continue
            axes=np.repeat(np.arange(d),len(CUTS));cuts=np.tile(CUTS,d);k=len(cuts)
            proposal=[children(lo,hi,int(a),float(c)) for a,c in zip(axes,cuts)]
            cl=np.array([p[0] for p in proposal]);ch=np.array([p[1] for p in proposal]);lengths=ch-cl
            parent=np.tile(hi-lo,(k,1));budgets=np.full(k,budget,dtype=np.int64)
            counts,_,_=allocator.allocate(lengths/parent[:,None,:],budgets)
            if np.any(counts.sum(1)!=budget):raise RuntimeError('Candidate budgets not conserved')
            candidate_evaluations+=k
            if mode=='teacher':
                valid=counts.min(1)>=1;score=np.full(k,np.inf)
                if d==2:score=np.array([quality(ell,c,hi-lo,budget) for ell,c in zip(lengths,counts)])
                elif valid.any():score[valid]=synthetic_teacher(parent[valid],budgets[valid],lengths[valid],counts[valid])[2]
            else:
                _,_,_,valid=route(execution,gate['margin'],gate['threshold'],counts)
                score=np.full(k,np.inf)
                if valid.any():score[valid]=predict(finite,upstream(models,execution,parent[valid],budgets[valid],lengths[valid],counts[valid]))
            choice=canonical(score)
            # Independently reapply the allocator to the selected geometry, as in
            # the audited recursion; proposal allocation cannot silently differ.
            child_counts,_,_=allocator.allocate((lengths[choice]/(hi-lo))[None],np.array([budget]))
            cc=child_counts[0]
            if not np.array_equal(cc,counts[choice]) or cc.sum()!=budget or np.any(cc<1) or np.any(cc>=budget):
                raise RuntimeError('Selected child allocation invalid or inconsistent')
            node=dict(lower=lo.tolist(),upper=hi.tolist(),budget=budget,depth=depth,axis=int(axes[choice]),
                cut=float(cuts[choice]),candidate=choice,child_counts=cc.tolist(),score=float(score[choice]),
                valid_mask=valid.tolist(),all_counts=counts.tolist(),candidate_scores=score.tolist())
            nodes[path]=node
            if reference_tree is not None:
                teacher=reference_tree.get(path)
                keys=('lower','upper','budget','depth','axis','cut','candidate','child_counts','valid_mask','all_counts')
                match=teacher is not None and all(node[key]==teacher[key] for key in keys)
                fidelity.append(dict(case=case,record_type='node',path=path,dimension=d,N=n,depth=depth,budget=budget,
                    teacher_axis=teacher['axis'] if teacher else '',learned_axis=node['axis'],
                    teacher_cut=teacher['cut'] if teacher else '',learned_cut=node['cut'],
                    teacher_counts=json.dumps(teacher['child_counts']) if teacher else '',learned_counts=json.dumps(cc.tolist()),
                    teacher_score=teacher['score'] if teacher else '',learned_score=node['score'],exact_decision_match=match,
                    allocation_match=bool(teacher and teacher['all_counts']==node['all_counts']),
                    validity_match=bool(teacher and teacher['valid_mask']==node['valid_mask'])))
                if not match:
                    (OUT/'first_divergence.json').write_text(json.dumps(dict(case=case,path=path,teacher=teacher,learned=node),indent=2))
                    raise RuntimeError(f'First recursive divergence: {case} / {path}')
            for j in (1,0):active.append((path+str(j),cl[choice,j],ch[choice,j],int(cc[j]),depth+1))
        if len(leaves)!=n or len(nodes)!=n-1:raise RuntimeError('Incorrect terminal tree count')
        return nodes,leaves

    references={}
    for d in (2,3,4):
        rng=np.random.default_rng(SEED+d);parts=[rng.random((16384,d))]
        for axis in range(d):
            for face in (0.,1.):
                x=rng.random((2048,d));x[:,axis]=face;parts.append(x)
        parts.append(np.array(list(itertools.product((0.,1.),repeat=d))))
        references[d]=np.concatenate(parts)
    try:
        for d in (2,3,4):
            for kind in ('unit','moderate_aspect'):
                extent=np.ones(d) if kind=='unit' else np.geomspace(.5,2.,d)
                for n in (32,64,128,256):
                    deadline();case=f'd{d}_{kind}_N{n}';t=time.monotonic()
                    tn,tl=recurse(extent,n,'teacher',case=case)
                    ln,ll=recurse(extent,n,'learned',tn,case)
                    if set(tn)!=set(ln) or tl!=ll:raise RuntimeError(f'Complete tree or leaf mismatch: {case}')
                    traces[case]=dict(teacher_nodes=tn,learned_nodes=ln,leaves=ll)
                    leaves=list(ll.values());lo=np.array([r['lower'] for r in leaves]);hi=np.array([r['upper'] for r in leaves])
                    centres=(lo+hi)/2;length=hi-lo;vol=length.prod(1);aspect=length.max(1)/length.min(1)
                    spacing=float((extent.prod()/n)**(1/d));reference=references[d]*extent
                    dist=cKDTree(centres).query(reference)[0];sep=cKDTree(centres).query(centres,k=2)[0][:,1]
                    volume_error=abs(vol.sum()-extent.prod())
                    if np.any(length<=0) or len(np.unique(centres,axis=0))!=n or aspect.max()>16 or dist.max()/spacing>3:
                        raise RuntimeError(f'Pathological leaf geometry: {case}')
                    if volume_error>256*np.finfo(float).eps*extent.prod():raise RuntimeError('Leaf volumes do not conserve domain volume')
                    # Prove disjoint interiors pairwise; shared faces are allowed.
                    overlap=np.minimum(hi[:,None,:],hi[None,:,:])-np.maximum(lo[:,None,:],lo[None,:,:])
                    overlapping=np.all(overlap>0,axis=2);np.fill_diagonal(overlapping,False)
                    if overlapping.any():raise RuntimeError('Overlapping leaf interiors')
                    row=dict(case=case,dimension=d,box=kind,N=n,total_points=len(centres),internal_nodes=len(ln),
                        max_depth=max(r['depth'] for r in leaves),min_depth=min(r['depth'] for r in leaves),
                        mean_depth=float(np.mean([r['depth'] for r in leaves])),extent=json.dumps(extent.tolist()),
                        leaf_volume_min=float(vol.min()),leaf_volume_p05=float(np.quantile(vol,.05)),leaf_volume_median=float(np.median(vol)),
                        leaf_volume_p95=float(np.quantile(vol,.95)),leaf_volume_max=float(vol.max()),leaf_volume_cv=float(vol.std()/vol.mean()),
                        total_leaf_volume=float(vol.sum()),volume_conservation_error=float(volume_error),
                        leaf_aspect_min=float(aspect.min()),leaf_aspect_median=float(np.median(aspect)),leaf_aspect_p95=float(np.quantile(aspect,.95)),
                        leaf_aspect_max=float(aspect.max()),covering_radius=float(dist.max()),covering_radius_interior=float(dist[:16384].max()),
                        normalized_cover=float(dist.max()/spacing),large_hole_threshold=spacing,large_hole_count=int(np.sum(dist>spacing)),
                        interior_large_hole_count=int(np.sum(dist[:16384]>spacing)),reference_count=len(dist),
                        nearest_neighbor_cv=float(sep.std()/sep.mean()),minimum_separation=float(sep.min()),
                        overlapping_leaf_pairs=0,duplicates=0,pathological=False,elapsed_seconds=time.monotonic()-t)
                    normalized=centres/extent
                    for scale in (2,4,8):
                        bins=np.minimum((normalized*scale).astype(int),scale-1)
                        hist=np.bincount(np.ravel_multi_index(bins.T,(scale,)*d),minlength=scale**d)
                        err=hist-n/(scale**d)
                        row[f'regional_count_rmse_{scale}']=float(np.sqrt(np.mean(err**2)))
                        row[f'regional_count_max_error_{scale}']=float(abs(err).max())
                        for axis in range(d):
                            pe=np.bincount(bins[:,axis],minlength=scale)-n/scale
                            row[f'projection_max_error_axis{axis}_scale{scale}']=float(abs(pe).max())
                    geometry.append(row)
                    fidelity.append(dict(case=case,record_type='tree',dimension=d,N=n,exact_decision_match=True,
                        complete_tree_exact=True,leaf_boxes_exact=True,terminal_leaves=len(ll),budget_conserved=True))
                    np.savez_compressed(OUT/(case+'_leaves.npz'),lower=lo,upper=hi,centres=centres)
                    write_csv(OUT/'recursive_split_fidelity.csv',fidelity);write_csv(OUT/'recursive_split_geometry.csv',geometry)
                    print(json.dumps(dict(event='case_complete',case=case,nodes=len(ln),depth=row['max_depth'],max_aspect=row['leaf_aspect_max'],
                        cover=row['normalized_cover'],holes=row['large_hole_count'],elapsed=time.monotonic()-started)),flush=True)
    except (RuntimeError,TimeoutError) as e:
        failure=str(e)
    unchanged=all(sha(p)==h for p,h in protected.items())
    if not unchanged:failure='Protected checkpoint hash changed'
    if fidelity:write_csv(OUT/'recursive_split_fidelity.csv',fidelity)
    if geometry:write_csv(OUT/'recursive_split_geometry.csv',geometry)
    (OUT/'recursive_trees.json').write_text(json.dumps(traces))
    result=dict(status='VALIDATED_SCAFFOLD' if failure is None and len(geometry)==24 else 'STOPPED',
        cases_completed=len(geometry),cases_planned=24,split_decisions_checked=sum(r['record_type']=='node' for r in fidelity),
        leaves=sum(r['total_points'] for r in geometry),candidate_evaluations_both_arms=candidate_evaluations,
        failure=failure,all_protected_hashes_unchanged=unchanged,allocator_stats={k:v.stats() for k,v in allocators.items()},
        elapsed_seconds=time.monotonic()-started,point_placement='TEMPORARY leaf-center diagnostic only; not solved',
        scaffolding=['candidate cut proposals','canonical teacher-equivalence bookkeeping','integer-conservation rounding','leaf centers'],
        reference_note='Covering radius and hole counts are independent-sample estimates; boundary faces and corners included.')
    if result['status']=='VALIDATED_SCAFFOLD':
        manifest=dict(**result,execution_script=__file__,execution_sha256=sha(__file__),
            split_score_checkpoint=str(lockpath),split_score_sha256=sha(lockpath),allocator_checkpoint=str(ALLOC),allocator_sha256=sha(ALLOC),
            config=config)
        (OUT/'validated_scaffold_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    lines=['Recursive split assembly: '+result['status'],json.dumps(result,indent=2),
        'dimension | box | N | points | depth | max aspect | estimated cover / spacing | holes / reference points']
    for r in geometry:lines.append(f"{r['dimension']} | {r['box']} | {r['N']} | {r['total_points']} | {r['max_depth']} | {r['leaf_aspect_max']:.6g} | {r['normalized_cover']:.6g} | {r['large_hole_count']} / {r['reference_count']}")
    lines+=['Both recursions independently propose, allocate and traverse; only learned decisions drive the learned tree.',
        'Teacher uses the audited locked allocator; differences from ideal-volume integer rounding are reported separately, not substituted.',
        'No training, checkpoint changes, SHO search, or N=8600 scaling. Leaf centers are temporary scaffolding.']
    (OUT/'recursive_split_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
