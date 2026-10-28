"""Learn two-child fractions; exact conservation; frozen local placement.

Only 1D halves/quarters are allocated. X/y are separate experiments, never a
Cartesian 2D partition. No downstream objectives or coordinate supervision.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator,boundary_faces
from experiments.sho.train_progressive_spacefill_smoke import sha,write_csv
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics,overlap
from experiments.sho.smoke_artifacts import save_checkpoint

N=8600
OUT=Path('outputs/sho/stage1B/hierarchical_allocation_poc')
LOCAL=Path('outputs/sho/stage1B/progressive_scale_curriculum/stage_1_8x8_best.pt')
GEOMETRY_CONDITIONS=[(2000+j,'cell') for j in range(4)]+[(2000,'greedy')]
SEED=27142026


def children(lo,hi,axis):
    mid=(lo[axis]+hi[axis])/2
    ahi=hi.copy();ahi[axis]=mid
    blo=lo.copy();blo[axis]=mid
    return ((lo.copy(),ahi),(blo,hi.copy()))


def features(n,lo,hi,axis,iteration,pool):
    regions=children(lo,hi,axis)
    volumes=np.array([np.prod(b-a) for a,b in regions])
    centres=np.concatenate([(a+b)/2 for a,b in regions])
    context=[math.sin(iteration),math.cos(iteration),math.sin(iteration*.01),math.cos(iteration*.01),float(pool=='greedy')]
    x=np.concatenate(([math.log1p(n)/math.log1p(N)],volumes,lo,hi-lo,centres,context,np.eye(2)[axis]))
    return x,volumes/volumes.sum(),regions


class Allocator(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(18,2,dtype=torch.float64)
    def forward(self,x):return torch.softmax(self.linear(x),dim=-1)


def conserve(n,fractions):
    """Largest remainder, child-index tie break. No equal-split special case."""
    f=np.asarray(fractions,dtype=np.float64)
    assert np.all(f>=0) and np.isfinite(f).all()
    f=f/f.sum();quotas=n*f
    count=np.floor(quotas).astype(np.int64);remaining=int(n-count.sum())
    assert 0<=remaining<=len(count)
    order=np.lexsort((np.arange(len(count)),-(quotas-count)))
    count[order[:remaining]]+=1
    assert count.sum()==n and np.all(count>=0)
    return count


def allocate(model,n,lo,hi,axis,iteration,pool):
    x,target,regions=features(n,lo,hi,axis,iteration,pool)
    with torch.no_grad():f=model(torch.from_numpy(x)).numpy()
    return conserve(n,f),f,target,regions


def allocation_tree(model,axis,iteration,pool,depth,records=None,stage=''):
    current=[(N,np.zeros(2),np.ones(2),'root')]
    for level in range(depth):
        following=[]
        for n,lo,hi,path in current:
            count,f,target,regions=allocate(model,n,lo,hi,axis,iteration,pool)
            error=count-n*target
            if records is not None:
                records.append(dict(record_type='allocated_split',stage=stage,N=N,iteration=iteration,pool_type=pool,
                    axis='xy'[axis],level=level+1,parent_path=path,parent_budget=n,lower=json.dumps(lo.tolist()),upper=json.dumps(hi.tolist()),
                    fraction_left=float(f[0]),fraction_right=float(f[1]),ideal_left=float(n*target[0]),ideal_right=float(n*target[1]),
                    left=int(count[0]),right=int(count[1]),left_error=float(error[0]),right_error=float(error[1]),
                    exact_conservation=bool(count.sum()==n),max_fraction_error=float(np.abs(f-target).max())))
            for j,(a,b) in enumerate(regions):following.append((int(count[j]),a,b,path+str(j)))
        current=following
    assert sum(v[0] for v in current)==N
    return current


def hard_projection(p,axis,k):
    return np.bincount(np.minimum((p[:,axis]*k).astype(int),k-1),minlength=k)


def placed(model,local,axis,iteration,pool,depth=2):
    """Round-robin blocks in allocated 1D strips; original frozen local policy.

    Candidates, distance/separation, occupancy and NN features use GLOBAL
    coordinates and the unchanged total N. No local square is stretched.
    """
    leaves=allocation_tree(model,axis,iteration,pool,depth)
    remaining=np.array([v[0] for v in leaves]);p=np.empty((0,2))
    rngs=[]
    for _,lo,hi,path in leaves:
        key=f'hierarchical-local-v1:{SEED}:{N}:{iteration}:{pool}:{axis}:{path}'.encode()
        rngs.append(np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little')))
    calls=0
    while remaining.sum():
        for region,(_,lo,hi,path) in enumerate(leaves):
            if remaining[region]==0:continue
            b=min(128,int(remaining[region]));rng=rngs[region]
            q=lo+(hi-lo)*rng.random((max(256,8*b),2))
            feature,_,nearest=local.make_context(p,q,N,iteration,pool)
            with torch.no_grad():logits=local.net(feature).flatten().clamp(-8,8).numpy()
            valid=nearest>=.4/math.sqrt(N)
            logits[~valid]=-100
            score=logits+.7*rng.gumbel(size=len(q));accepted=[]
            for j in np.argsort(-score,kind='stable'):
                if not valid[j]:continue
                if accepted and np.min(np.sum((q[accepted]-q[j])**2,axis=1))<.16/N:continue
                accepted.append(j)
                if len(accepted)==b:break
            if not accepted:raise RuntimeError('no admissible local placement candidates')
            p=np.vstack((p,q[accepted]));remaining[region]-=len(accepted);calls+=1
    assert len(p)==N and len(np.unique(p,axis=0))==N
    # Verify actual coordinates obey learned quotas, not just declared budgets.
    assert np.array_equal(hard_projection(p,axis,2**depth),np.array([v[0] for v in leaves]))
    complete=np.unique(np.concatenate((p,boundary_faces(N,2))),axis=0)
    assert len(complete)==N+len(boundary_faces(N,2))
    return complete,p,calls


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    prior=json.loads(Path('outputs/sho/stage1B/dyadic_spacefill_curriculum/results.json').read_text())
    protected=dict(prior['protected_hashes'])
    protected[str(LOCAL)]=sha(LOCAL)
    for name in ('experiments/sho/progressive_spacefill_generator.py','experiments/sho/fresh_spacefill_generator.py'):
        protected[name]=sha(name)
    config=dict(N=N,seed=SEED,allocator='random-initialized float64 Linear(18,2) -> softmax',
        inputs='parent n, child volumes, parent lower/span, child centres, four bounded iteration codes, pool, axis',
        training='root half splits only; iterations0..63, both pools and axes; volume-ratio targets',
        loss='mean squared fraction error +0.01*sum(weight^2); biases trained without penalty',
        optimizer='LBFGS, strong_wolfe, max_iter250, tolerance_grad1e-12, tolerance_change1e-16',
        integer_rule='floor learned quotas, distribute residual by fractional remainder; child-index tie break',
        quarters='zero-shot recursive use of the SAME saved allocator, no additional training or finer levels',
        local_checkpoint=str(LOCAL),local_sha256=sha(LOCAL),
        placement='original frozen policy, round-robin blocks128 inside allocated 1D strips; global distances/features and original .4/sqrt(N) separation guard',
        geometry_conditions=GEOMETRY_CONDITIONS,
        geometry_gate='cover<=1.10 matched unallocated baseline; holes fraction<=baseline+.001; NN CV>=.08; cluster<=.005; near overlap<=.03; exact overlap0; identical boundary',
        no_Sobol_coordinates=True,no_SHO_or_search_labels=True,no_search=True)
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    x=[];targets=[]
    for it in range(64):
        for pool in ('cell','greedy'):
            for axis in (0,1):
                f,t,_=features(N,np.zeros(2),np.ones(2),axis,it,pool);x.append(f);targets.append(t)
    x=torch.from_numpy(np.array(x));target=torch.from_numpy(np.array(targets))
    model=Allocator()
    with torch.no_grad():initial=model(x).numpy()
    initial_mistakes=sum(not np.array_equal(conserve(N,f),[4300,4300]) for f in initial)
    assert initial_mistakes>0,'random initialization unexpectedly solves all rows'
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=250,line_search_fn='strong_wolfe',
        tolerance_grad=1e-12,tolerance_change=1e-16)
    history=[]
    def closure():
        optimizer.zero_grad();pred=model(x)
        mse=(pred-target).square().mean();penalty=.01*model.linear.weight.square().sum()
        loss=mse+penalty;loss.backward()
        history.append(dict(evaluation=len(history),fraction_MSE=float(mse.detach()),weight_penalty=float(penalty.detach()),loss=float(loss.detach())))
        return loss
    optimizer.step(closure)
    # Persist immediately after training, before heldout or recursive evaluation.
    checkpoint=OUT/'allocator_trained_before_validation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,history=history,status='PENDING_COUNT_GATE'))
    write_csv(OUT/'allocation_training_history.csv',history)
    print(json.dumps(dict(event='allocator_saved',sha256=digest,closure_evaluations=len(history),initial_integer_mistakes=initial_mistakes)),flush=True)
    # Counterfactual checks: conservation does NOT force equal counts.
    opposite=torch.softmax(torch.tensor([1.,-1.],dtype=torch.float64),0).numpy()
    assert not np.array_equal(conserve(N,opposite),[4300,4300])
    assert np.array_equal(conserve(N,opposite),conserve(N,opposite[::-1])[::-1])
    iterations=list(range(1000,1256))+[2000,2001,2002,2003,10000,1000000,1000000000]
    conditions=[(it,pool) for it in iterations for pool in ('cell','greedy')]
    records=[]
    for it,pool in conditions:
        for axis in (0,1):allocation_tree(model,axis,it,pool,1,records,'halves')
    halves=all(r['left_error']==0 and r['right_error']==0 and r['exact_conservation'] for r in records)
    print(json.dumps(dict(event='halves_gate',passed=halves,queries=len(records),max_fraction_error=max(r['max_fraction_error'] for r in records))),flush=True)
    quarters=False
    if halves:
        # Reload persisted learned weights before extending depth.
        restored=Allocator();restored.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model=restored
        for it,pool in conditions:
            for axis in (0,1):
                leaves=allocation_tree(model,axis,it,pool,2,records,'quarters')
                assert len(leaves)==4
        quarters=all(r['left_error']==0 and r['right_error']==0 and r['exact_conservation'] for r in records)
        print(json.dumps(dict(event='quarters_gate',passed=quarters,split_records=len(records))),flush=True)
    write_csv(OUT/'hierarchical_allocation_counts.csv',records)
    geometry=[];geometry_gates=[];coordinates={}
    if halves and quarters:
        local=ProgressiveGenerator();local.load_state_dict(torch.load(LOCAL,weights_only=False)['state_dict']);local.eval().requires_grad_(False)
        t=(np.arange(512)+.5)/512
        reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
        pools={};by_condition={}
        for name,axis in [('frozen_unallocated',None),('x_quarters',0),('y_quarters',1)]:
            pools[name]=[]
            for it,pool in GEOMETRY_CONDITIONS:
                begin=time.monotonic()
                if axis is None:
                    with torch.no_grad():complete,p=local.generate(N,it,pool)
                    calls=math.ceil(N/128)
                else:complete,p,calls=placed(model,local,axis,it,pool)
                seconds=time.monotonic()-begin
                if axis is not None and it==2000 and pool=='cell':
                    assert np.array_equal(p,placed(model,local,axis,it,pool)[1]),'placement is not reproducible'
                m=metrics(complete,p,reference,N)
                row=dict(record_type='pool',generator=name,N=N,iteration=it,pool_type=pool,**m,policy_calls=calls,generation_seconds=seconds)
                geometry.append(row);by_condition[name,it,pool]=row;pools[name].append((complete,p))
                if axis is not None:
                    coordinates[f'{name}_{it}_{pool}']=p
                    for projection_axis in (0,1):
                        for k in (2,4):
                            count=hard_projection(p,projection_axis,k)
                            controlled=projection_axis==axis
                            if controlled:assert np.all(count==N/k)
                            records.append(dict(record_type='placed_projection',stage='local_placement',N=N,iteration=it,pool_type=pool,
                                routing_axis='xy'[axis],axis='xy'[projection_axis],level=1 if k==2 else 2,
                                controlled_projection=controlled,counts=json.dumps(count.tolist()),
                                worst_count_error=float(np.max(np.abs(count-N/k))),exact_conservation=bool(count.sum()==N)))
            for a,b,kind in [(0,1,'iteration_0_1'),(1,2,'iteration_1_2'),(2,3,'iteration_2_3'),(0,4,'cell_greedy')]:
                geometry.append(dict(record_type='overlap',generator=name,N=N,comparison=kind,
                    **overlap(pools[name][a][1],pools[name][b][1],N)))
        for name in ('x_quarters','y_quarters'):
            ratios=[];holes=[]
            for it,pool in GEOMETRY_CONDITIONS:
                a=by_condition[name,it,pool];b=by_condition['frozen_unallocated',it,pool]
                ratios.append(a['complete_cover']/b['complete_cover']);holes.append(a['large_hole_fraction']-b['large_hole_fraction'])
            geom=[r for r in geometry if r['generator']==name and r['record_type']=='pool']
            over=[r for r in geometry if r['generator']==name and r['record_type']=='overlap']
            gates=dict(cover=max(ratios)<=1.10,holes=max(holes)<=.001,
                not_rigid=min(r['nn_CV'] for r in geom)>=.08,not_clustered=max(r['clustered_fraction'] for r in geom)<=.005,
                exact_freshness=all(r['exact_overlap_count']==0 for r in over),near_freshness=max(r['near_overlap_fraction_005_spacing'] for r in over)<=.03,
                boundary=all(by_condition[name,it,pool]['boundary_cover']==by_condition['frozen_unallocated',it,pool]['boundary_cover'] for it,pool in GEOMETRY_CONDITIONS))
            geometry_gates.append(dict(generator=name,passed=all(gates.values()),gates=gates,maximum_matched_cover_ratio=max(ratios),
                maximum_hole_fraction_increase=max(holes)))
        np.savez_compressed(OUT/'generated_interiors.npz',**coordinates)
        (OUT/'generated_interiors.sha256').write_text(sha(OUT/'generated_interiors.npz')+'  generated_interiors.npz\n')
    else:geometry.append(dict(status='NOT_RUN_ALLOCATION_GATE_FAILED'))
    write_csv(OUT/'hierarchical_allocation_counts.csv',records)
    write_csv(OUT/'hierarchical_allocation_geometry.csv',geometry)
    success=halves and quarters and all(g['passed'] for g in geometry_gates)
    candidate=None;candidate_hash=None
    if success:
        candidate=OUT/'HIERARCHICAL_ALLOCATION_POC.pt'
        candidate_hash=save_checkpoint(candidate,dict(state_dict=model.state_dict(),local_placement_state_dict=local.state_dict(),
            config=config,allocator_checkpoint_sha256=digest,local_checkpoint_sha256=sha(LOCAL),
            assembly_source_sha256=sha(__file__),status='HIERARCHICAL_ALLOCATION_POC',
            limitation='equal-volume halves/quarters, separate x/y strip experiments only; not arbitrary unequal allocation or a joint 2D hierarchy'))
    unchanged=all(sha(p)==h for p,h in protected.items());assert unchanged
    result=dict(status='HIERARCHICAL_ALLOCATION_POC' if success else 'STOPPED_DIAGNOSTIC_ONLY',halves_passed=halves,quarters_passed=quarters,
        heldout_conditions=len(conditions),axes=2,split_tests=sum(r['record_type']=='allocated_split' for r in records),
        initial_training_integer_mistakes=initial_mistakes,closure_evaluations=len(history),final_training_fraction_MSE=history[-1]['fraction_MSE'],
        learned_weight_norm=float(model.linear.weight.detach().norm()),counterfactual_unequal_counts=conserve(N,opposite).tolist(),
        geometry_gates=geometry_gates,allocator_checkpoint=str(checkpoint),allocator_sha256=digest,
        candidate_checkpoint=str(candidate) if candidate else None,candidate_sha256=candidate_hash,
        elapsed_seconds=time.monotonic()-start,protected_unchanged=unchanged,protected_hashes=protected,
        no_search=True,no_Sobol_or_SHO_or_search_supervision=True,no_2d_joint_allocation=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['HIERARCHICAL POINT ALLOCATION — '+result['status'],f'N=8600 only; runtime {result["elapsed_seconds"]:.2f} seconds.',
        'Random float64 two-logit allocator learned volume-proportional fraction labels; no coefficient/fraction initialized or assigned to50/50. L2 regularization suppresses spurious dependence on context; equal-volume labels alone do not identify an unequal-volume allocator.',
        'Training ONLY root halves, parent n8600, iteration0..63, both axes/pools. Recursive quarters use the same saved allocator at n4300 with NO further training.',
        'Integer conservation: floor learned quotas, distribute remaining points by largest fractional remainder, child-index ties. This rule does not force equality; unequal-logit counterfactual counts '+str(result['counterfactual_unequal_counts'])+'.',
        f'Halves gate {halves}; quarters gate {quarters}; {result["split_tests"]} split tests across{len(conditions)} unseen iteration/pool conditions and2 axes.',
        'Correct controlled counts: halves4300/4300, quarters2150/2150/2150/2150. Count CSV includes each actual split, fraction error and conservation, and actual coordinate projection counts after placement.',
        'X and Y are SEPARATE1D slab-allocation experiments. The unconstrained projection is measured too; this does not claim one point set has jointly exact x/y quarters.',
        'Frozen existing local policy selects random proposals inside each allocated strip; block scheduling is round-robin. Coordinates and separation/occupancy features stay global. No local weights were trained. Domain boundary support unchanged; no added internal-boundary probes.',
        'Geometry compares to the SAME frozen unallocated policy on matching heldout conditions. Finite reference524288 points. Near overlap thresholds .05/sqrt(N),.10/sqrt(N); boundaries excluded. Large holes reference distance>1/sqrt(N).',
        'No Sobol coordinates, objective/SHO labels, good-candidate coordinates, 2D dyadic allocations, finer levels, or closed-loop search.',
        'All prior artifacts protected/unchanged. Learned allocator saved before validation; candidate bundle saved only if all gates pass.',
        json.dumps({k:v for k,v in result.items() if k!='protected_hashes'},indent=2)]
    (OUT/'hierarchical_allocation_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='protected_hashes'},indent=2),flush=True)


if __name__=='__main__':run()
