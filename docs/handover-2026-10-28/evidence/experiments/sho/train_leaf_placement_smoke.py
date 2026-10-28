"""Geometry-only complete-set training inside immutable recursive leaves."""
import copy
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv

ROOT=Path('outputs/sho/stage1B'); LEAVES=ROOT/'recursive_split_assembly'; OUT=ROOT/'leaf_placement_smoke'
SEED=27272026; UPDATES=384; SCALES=(2,4,8,16)
GATES=dict(cover_ratio=1.10,large_hole_fraction_increase=.001,near_overlap=.05,
    near_radius_in_spacing=.01,min_separation_ratio=.5,mean_anchored_rms_ratio=.95,
    worst_anchored_rms_ratio=1.05,worst_anchored_max_ratio=1.10)


class LeafPlacement(nn.Module):
    """Shared coordinate network; no fixed dimensionality in learned weights.

    Fixed Fourier features encode position and iteration, not target points.
    Each scalar output is a sigmoid leaf coordinate. No set is stored in weights.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('space_frequency',3+37*torch.rand(4,dtype=torch.float64))
        self.register_buffer('iteration_frequency',.5+2.5*torch.rand(4,dtype=torch.float64))
        self.register_buffer('pool_phase',6.28*torch.rand(4,dtype=torch.float64))
        self.net=nn.Sequential(nn.Linear(25,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1)).double()
        with torch.no_grad():self.net[-1].weight.mul_(.15);self.net[-1].bias.mul_(.15)

    def forward(self,lower,upper,extent,iteration,pool):
        l=lower/extent;u=upper/extent;c=(l+u)/2;s=u-l;n,d=l.shape
        axis=torch.arange(d,dtype=torch.float64)[None,:].expand(n,d)
        f=[l,u,s,c,torch.log(s),torch.full_like(l,d/4),axis/d,torch.full_like(l,math.log2(n)/8),torch.full_like(l,float(pool))]
        for j in range(4):
            phase=iteration*self.iteration_frequency[j]+pool*self.pool_phase[j]
            f.extend([torch.ones_like(l)*torch.sin(phase),torch.ones_like(l)*torch.cos(phase)])
            spatial=self.space_frequency[j]*c+phase+axis*(j+1)*.73
            f.extend([torch.sin(spatial),torch.cos(spatial)])
        z=torch.sigmoid(self.net(torch.stack(f,dim=-1)).squeeze(-1))
        return lower+z*(upper-lower),z


def anchors(d,seed,training):
    rng=np.random.default_rng(seed);groups=[]
    for scale in SCALES:
        count=96 if training else 1024
        # Anchored boxes at randomized coarse-to-fine locations; no target points.
        a=(rng.integers(scale,size=(count,d))+rng.random((count,d)))/scale
        for i in range(count):
            active=1+i%d
            if active<d:a[i,rng.choice(d,d-active,replace=False)]=1.
        groups.append(a)
    return groups


def hard_discrepancy(points,groups):
    values=[]
    for a in groups:
        fraction=np.all(points[None,:,:]<=a[:,None,:],axis=2).mean(1)
        error=fraction-np.prod(a,axis=1)
        values.append((float(np.sqrt(np.mean(error**2))),float(abs(error).max())))
    return values


def overlap(a,b,radius):
    distance=cKDTree(b).query(a)[0]
    return dict(exact_overlap=int(np.sum(distance==0)),exact_overlap_fraction=float(np.mean(distance==0)),
        near_overlap=int(np.sum(distance<=radius)),near_overlap_fraction=float(np.mean(distance<=radius)),
        minimum_cross_distance=float(distance.min()),mean_cross_distance=float(distance.mean()))


def soft_discrepancy(points,groups,tau):
    losses=[]
    for a in groups:
        members=torch.sigmoid((a[:,None,:]-points[None,:,:])/tau)
        members=torch.where(a[:,None,:]==1,torch.ones_like(members),members)
        error=members.prod(2).mean(1)-a.prod(1)
        losses.append((error*error).mean())
    return torch.stack(losses)


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in list(ROOT.glob('*/*.pt'))+list(LEAVES.iterdir()) if p.is_file()}
    scaffold=json.loads((LEAVES/'validated_scaffold_manifest.json').read_text());assert scaffold['status']=='VALIDATED_SCAFFOLD'
    assert sha(scaffold['execution_script'])==scaffold['execution_sha256']
    config=dict(seed=SEED,updates=UPDATES,dtype='float64',architecture='shared 25->32 tanh->32 tanh->1 sigmoid coordinate MLP',
        dimensions=[2,3,4],budgets=[32,64,128,256],training_iterations=list(range(9)),evaluation_iterations=[16,17,18,19],
        pools=['cell','greedy'],primary='worst-scale normalized soft anchored squared discrepancy, plus mean across scales',
        safeguards='coverage hinge, excess holes, low-separation hinge, within-leaf cross-condition separation',
        sigmoid_containment=True,training_sources='saved recursive leaf geometry and independent uniform reference/anchor samples only',
        sobol='external scrambled scipy.stats.qmc.Sobol geometry baseline after checkpoint persistence only; exactly N points',
        boundary_note='No extra generated boundary points: one per leaf and exact N; covering reference explicitly includes every face and corners.',
        gates=GATES,protected_hashes=protected)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    cases=[]
    for path in sorted(LEAVES.glob('*_leaves.npz')):
        a=np.load(path);lo=a['lower'];hi=a['upper'];center=a['centres'];n,d=lo.shape;extent=hi.max(0)
        assert np.array_equal(center,(lo+hi)/2)
        rng=np.random.default_rng(SEED+len(cases));refs=[rng.random((768,d))]
        for axis in range(d):
            for face in (0.,1.):
                r=rng.random((64,d));r[:,axis]=face;refs.append(r)
        refs.append(np.array(list(itertools.product((0.,1.),repeat=d))))
        ref=np.concatenate(refs)*extent;near=cKDTree(center).query(ref)[0]
        h=float((extent.prod()/n)**(1/d));sep=cKDTree(center).query(center,k=2)[0][:,1]
        group=anchors(d,SEED+100+len(cases),True)
        cases.append(dict(name=path.stem.replace('_leaves',''),n=n,d=d,lo=torch.from_numpy(lo),hi=torch.from_numpy(hi),
            extent=torch.from_numpy(extent),center=center,h=h,train_ref=torch.from_numpy(ref),base_cover=float(near.max()),
            base_holes=float(np.mean(near>h)),base_hole_loss=float(np.mean(np.maximum(near/h-1,0)**2)),
            base_sep=float(np.quantile(sep,.01)),groups=[torch.from_numpy(g) for g in group]))
    assert len(cases)==24
    model=LeafPlacement();optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};history=[];best=[None,None,None]

    def objective(c,t,progress,grad=True):
        samples=[model(c['lo'],c['hi'],c['extent'],t,0)[0],model(c['lo'],c['hi'],c['extent'],t+1,0)[0],
                 model(c['lo'],c['hi'],c['extent'],t,1)[0]]
        tau=float(torch.median((c['hi']-c['lo'])/c['extent']))*(.20-.14*progress)
        baseline=soft_discrepancy(torch.from_numpy(c['center'])/c['extent'],c['groups'],tau).detach().clamp_min(1e-7)
        ds=[];covers=[];holes=[];separation=[]
        for p in samples:
            rel=soft_discrepancy(p/c['extent'],c['groups'],tau)/baseline
            ds.append(rel.max()+.25*rel.mean())
            nearest=torch.cdist(c['train_ref'],p).min(1).values
            covers.append(torch.relu(nearest.max()/c['base_cover']-1.05)**2)
            holes.append(torch.relu(torch.relu(nearest/c['h']-1).square().mean()-c['base_hole_loss']))
            pair=torch.cdist(p,p);pair=pair.masked_fill(torch.eye(c['n'],dtype=torch.bool),torch.inf)
            separation.append(torch.relu(.6-pair.min(1).values/c['base_sep']).square().mean())
        fresh=[]
        for j in (1,2):
            delta=torch.linalg.vector_norm(samples[0]-samples[j],dim=1)/c['h']
            fresh.append(torch.relu(1-delta/.05).square().mean())
        disc=torch.stack(ds).mean();cov=torch.stack(covers).mean();hole=torch.stack(holes).mean()
        sep=torch.stack(separation).mean();fr=torch.stack(fresh).mean()
        loss=disc+80*cov+100*hole+10*sep+2*fr
        return loss,dict(discrepancy=float(disc.detach()),coverage_loss=float(cov.detach()),hole_loss=float(hole.detach()),
            separation_loss=float(sep.detach()),freshness_loss=float(fr.detach()))

    def monitor(step):
        vals=[];rms=[];cover_ratios=[];hole_changes=[];near=[]
        with torch.no_grad():
            for c in cases:
                loss,terms=objective(c,0,1.,False);vals.append(float(loss));rms.append(terms['discrepancy'])
                pts=[model(c['lo'],c['hi'],c['extent'],t,p)[0].numpy() for t,p in ((0,0),(1,0),(0,1))]
                for p in pts:
                    distance=cKDTree(p).query(c['train_ref'].numpy())[0]
                    cover_ratios.append(float(distance.max()/c['base_cover']));hole_changes.append(float(np.mean(distance>c['h'])-c['base_holes']))
                for j in (1,2):near.append(overlap(pts[0],pts[j],.01*c['h'])['near_overlap_fraction'])
        violation=max(0,max(cover_ratios)-GATES['cover_ratio'])+max(0,max(hole_changes)-GATES['large_hole_fraction_increase'])+max(0,max(near)-GATES['near_overlap'])
        key=(violation,float(np.mean(rms)))
        row=dict(record_type='monitor',update=step,loss=float(np.mean(vals)),primary=float(np.mean(rms)),
            worst_cover_ratio=max(cover_ratios),worst_hole_fraction_increase=max(hole_changes),worst_near_overlap=max(near),constraint_violation=violation)
        history.append(row)
        if best[0] is None or key<best[0]:best[:]=[key,copy.deepcopy(model.state_dict()),step]
        print(json.dumps(dict(event='training_monitor',**row)),flush=True)

    monitor(0)
    order=np.random.default_rng(SEED+300)
    for step in range(1,UPDATES+1):
        c=cases[int(order.integers(len(cases)))];iteration=int(order.integers(8))
        optimizer.zero_grad();loss,terms=objective(c,iteration,(step-1)/(UPDATES-1));loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        history.append(dict(record_type='update',update=step,case=c['name'],iteration=iteration,loss=float(loss.detach()),**terms))
        if step%64==0:monitor(step)
    # Select by training-only safeguard violation first, then primary discrepancy.
    model.load_state_dict(best[1]);cp=OUT/'leaf_placement_before_evaluation.pt'
    digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),initial_state_dict=initial,config=config,
        selected_update=best[2],training_selection=best[0],status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'leaf_placement_training.csv',history)
    print(json.dumps(dict(event='checkpoint_saved',selected_update=best[2],sha256=digest)),flush=True)
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)

    # Sobol is introduced only now, outside every training/selection operation.
    from scipy.stats import qmc
    geometry=[];freshness=[];gate_rows=[];coordinates={}
    conditions=[(t,p) for t in (16,17,18,19) for p in (0,1)]
    for ci,c in enumerate(cases):
        d,n=c['d'],c['n'];extent=c['extent'].numpy();rng=np.random.default_rng(SEED+10000+ci)
        refs=[rng.random((16384,d))]
        for axis in range(d):
            for face in (0.,1.):
                z=rng.random((2048,d));z[:,axis]=face;refs.append(z)
        refs.append(np.array(list(itertools.product((0.,1.),repeat=d))))
        reference=np.concatenate(refs)*extent;groups=anchors(d,SEED+20000+ci,False)
        sets={name:{} for name in ('centers','learned','sobol')};rows={name:[] for name in sets}
        for t,pool in conditions:
            with torch.no_grad():placed,z=model(c['lo'],c['hi'],c['extent'],t,pool)
            sets['centers'][t,pool]=c['center']
            sets['learned'][t,pool]=placed.numpy()
            sets['sobol'][t,pool]=qmc.Sobol(d,scramble=True,seed=SEED+30000+ci*100+t*2+pool).random_base2(int(math.log2(n)))*extent
            assert placed.shape==(n,d)
            for name in sets:
                points=sets[name][t,pool];dist=cKDTree(points).query(reference)[0]
                nn_dist=cKDTree(points).query(points,k=2)[0][:,1]
                disc=hard_discrepancy(points/extent,groups)
                outside=np.any((points<c['lo'].numpy())|(points>c['hi'].numpy()),axis=1) if name!='sobol' else None
                row=dict(case=c['name'],generator=name,dimension=d,N=n,iteration=t,pool='cell' if pool==0 else 'greedy',
                    points=len(points),covering_radius=float(dist.max()),interior_covering_radius=float(dist[:16384].max()),
                    large_hole_threshold=c['h'],large_hole_count=int(np.sum(dist>c['h'])),large_hole_fraction=float(np.mean(dist>c['h'])),
                    interior_large_holes=int(np.sum(dist[:16384]>c['h'])),reference_count=len(reference),
                    nn_cv=float(nn_dist.std()/nn_dist.mean()),min_separation=float(nn_dist.min()),p01_separation=float(np.quantile(nn_dist,.01)),
                    outside_leaves=int(outside.sum()) if outside is not None else '',outside_fraction=float(outside.mean()) if outside is not None else '',
                    one_point_per_leaf=bool(not outside.any()) if outside is not None else '',
                    anchored_rms=float(np.sqrt(np.mean([v[0]**2 for v in disc]))),anchored_max=max(v[1] for v in disc),
                    centered_discrepancy=float(qmc.discrepancy(points/extent,method='CD')))
                for scale,(rms,mx) in zip(SCALES,disc):
                    row[f'anchored_rms_{scale}']=rms;row[f'anchored_max_{scale}']=mx
                    bins=np.minimum((points/extent*scale).astype(int),scale-1)
                    hist=np.bincount(np.ravel_multi_index(bins.T,(scale,)*d),minlength=scale**d)
                    err=hist/n-1/scale**d
                    row[f'occupancy_rms_{scale}']=float(np.sqrt(np.mean(err**2)));row[f'occupancy_max_{scale}']=float(abs(err).max())
                rows[name].append(row);geometry.append(row)
        for name,points in sets.items():
            for t in (16,17,18):
                for pool in (0,1):
                    freshness.append(dict(case=c['name'],generator=name,dimension=d,N=n,kind='successive_iterations',iteration=t,pool=pool,
                        **overlap(points[t,pool],points[t+1,pool],.01*c['h'])))
            for t in (16,17,18,19):
                freshness.append(dict(case=c['name'],generator=name,dimension=d,N=n,kind='cell_vs_greedy',iteration=t,pool='both',
                    **overlap(points[t,0],points[t,1],.01*c['h'])))
        baseline=rows['centers'][0];lr=rows['learned'];fr=[r for r in freshness if r['case']==c['name'] and r['generator']=='learned']
        cover=max(r['covering_radius']/baseline['covering_radius'] for r in lr)
        holes=max(r['large_hole_fraction']-baseline['large_hole_fraction'] for r in lr)
        rmsratios=[r[f'anchored_rms_{s}']/baseline[f'anchored_rms_{s}'] for r in lr for s in SCALES]
        maxratio=max(r['anchored_max']/baseline['anchored_max'] for r in lr)
        row=dict(case=c['name'],dimension=d,N=n,worst_cover_ratio=cover,worst_hole_fraction_increase=holes,
            mean_anchored_rms_ratio=float(np.mean(rmsratios)),worst_scale_anchored_rms_ratio=max(rmsratios),worst_anchored_max_ratio=maxratio,
            worst_near_overlap=max(r['near_overlap_fraction'] for r in fr),exact_overlap=sum(r['exact_overlap'] for r in fr),
            outside_points=sum(r['outside_leaves'] for r in lr),minimum_separation_ratio=min(r['p01_separation']/baseline['p01_separation'] for r in lr))
        row['coverage_pass']=cover<=GATES['cover_ratio'];row['holes_pass']=holes<=GATES['large_hole_fraction_increase']
        row['discrepancy_pass']=row['mean_anchored_rms_ratio']<=GATES['mean_anchored_rms_ratio'] and max(rmsratios)<=GATES['worst_anchored_rms_ratio'] and maxratio<=GATES['worst_anchored_max_ratio']
        row['freshness_pass']=row['exact_overlap']==0 and row['worst_near_overlap']<=GATES['near_overlap']
        row['separation_pass']=row['minimum_separation_ratio']>=GATES['min_separation_ratio']
        row['passed']=all(row[k] for k in ('coverage_pass','holes_pass','discrepancy_pass','freshness_pass','separation_pass')) and row['outside_points']==0
        gate_rows.append(row)
        for name in sets:
            for (t,pool),points in sets[name].items():coordinates[f"{c['name']}_{name}_t{t}_p{pool}"]=points
        print(json.dumps(dict(event='case_evaluated',**row)),flush=True)
    np.savez_compressed(OUT/'evaluated_coordinates.npz',**coordinates)
    write_csv(OUT/'leaf_placement_geometry.csv',geometry);write_csv(OUT/'leaf_placement_freshness.csv',freshness)
    write_csv(OUT/'leaf_placement_gates.csv',gate_rows)
    unchanged=all(sha(p)==h for p,h in protected.items())
    passed=all(r['passed'] for r in gate_rows) and unchanged and best[2]>0
    monitors=[r for r in history if r['record_type']=='monitor']
    result=dict(status='LEAF_PLACEMENT_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY',passed=bool(passed),
        checkpoint=str(cp),sha256=digest,selected_update=best[2],updates_run=UPDATES,initial_training_monitor=monitors[0],
        selected_training_monitor=next(r for r in monitors if r['update']==best[2]),cases=24,conditions_per_case=8,
        gate_pass_counts={k:sum(r[k] for r in gate_rows) for k in ('coverage_pass','holes_pass','discrepancy_pass','freshness_pass','separation_pass','passed')},
        worst_cover_ratio=max(r['worst_cover_ratio'] for r in gate_rows),worst_hole_fraction_increase=max(r['worst_hole_fraction_increase'] for r in gate_rows),
        mean_anchored_rms_ratio=float(np.mean([r['mean_anchored_rms_ratio'] for r in gate_rows])),
        worst_scale_anchored_rms_ratio=max(r['worst_scale_anchored_rms_ratio'] for r in gate_rows),
        exact_overlap=sum(r['exact_overlap'] for r in gate_rows),worst_near_overlap=max(r['worst_near_overlap'] for r in gate_rows),
        outside_points=sum(r['outside_points'] for r in gate_rows),exact_N=True,one_point_per_leaf=True,
        all_protected_hashes_unchanged=unchanged,recursive_partition_unchanged=unchanged,elapsed_seconds=time.monotonic()-started,
        notes='Sobol evaluation only. No split/proposal/allocator training, recursion changes, SHO labels, or search. No automatic continuation after failed gates.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2))
    text=['Within-leaf placement: '+result['status'],json.dumps(result,indent=2),
        'Gates were fixed before training: '+json.dumps(GATES),
        'Anchored metrics are hard indicators on independent anchor boxes. Training uses soft indicators only.',
        'Covering radius is estimated on independent interior, every boundary face, and corner reference points.',
        'Near-overlap means nearest cross-set distance <= 0.01 times characteristic spacing.',
        'Sobol uses exactly N scrambled points, with independent deterministic condition seeds; no added boundary probes.',
        'Geometry/freshness CSVs include all three baselines, each dimension, budget, box, and held-out condition.',
        'Failure diagnosis: '+', '.join(k for k,v in result['gate_pass_counts'].items() if k!='passed' and v<24),
        'Stop here; preserved diagnostic weights are not promoted if any gate fails.']
    (OUT/'leaf_placement_summary.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
