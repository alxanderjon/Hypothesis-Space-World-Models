"""Read-only teacher; single-context proposal buffering screen. No training."""
import hashlib,itertools,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import validate_joint_quota_assignment_teacher as fixed
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator,boundary_faces,SEED
from experiments.sho.train_dyadic_spacefill_curriculum import PHASES,health
from experiments.sho.evaluate_progressive_spacefill_smoke import overlap
ROOT=Path('outputs/sho/stage1B');OUT=ROOT/'online_quota_reservoir_audit';N=8600
FAMILIES=[(i,tuple(f)) for i,(_,_,fs) in enumerate(PHASES) for f in fs]
shape=np.max(np.array([f for _,f in FAMILIES]),axis=0)
# Cell centers are incidence representatives ONLY, never generated probes.
cells=np.array(list(itertools.product(*[range(int(v)) for v in shape])))
A0,META,_=fixed.incidence((cells+.5)/shape,FAMILIES)
A,C0,MAP=fixed.aggregate(A0);K=A.shape[1];LOW,HIGH=fixed.bounds(META,N)

def ids(points):
    ix=np.minimum((points*shape).astype(int),shape-1)
    return MAP[np.ravel_multi_index(ix.T,tuple(shape))]

def capacity(points):return np.bincount(ids(points),minlength=K)

def oracle(cap,current=None,canonical=False):
    current=np.zeros(len(META),dtype=np.int64) if current is None else current
    # Current total is determined by one complete family.
    family=[j for j,v in enumerate(META) if v['family']==META[0]['family']]
    remaining=N-int(current[family].sum())
    if canonical:return fixed.teacher(A,cap,current,LOW,HIGH,remaining)[0]
    if np.any(current>HIGH) or cap.sum()<remaining:return None
    return fixed.solve(np.zeros(K),np.vstack((np.ones(K,int),A)),np.r_[remaining,LOW-current],np.r_[remaining,HIGH-current],np.zeros(K,int),cap)

def diagnostic(points,current=None):
    cap=capacity(points);cur=np.zeros(len(META),int) if current is None else current
    missing=np.maximum(0,LOW-cur-A@cap);too_many=np.flatnonzero(cur>HIGH)
    row={'reservoir_size':len(points),'incidence_classes_present':int(np.count_nonzero(cap)),'class_capacity_min':int(cap.min()),'class_capacity_max':int(cap.max()),'capacity':cap.tolist(),'limited_constraints':[dict(id=int(i),**META[i],available=int((A@cap)[i]),current=int(cur[i]),lower=int(LOW[i]),shortfall=int(missing[i])) for i in np.flatnonzero(missing)],'immutable_overfilled_rows':too_many.tolist()}
    row['feasible']=oracle(cap,cur) is not None
    if len(too_many):row['minimum_added_compatible_capacity']=None;row['augmentation_status']='cannot repair immutable prefix by adding proposals';return row
    remaining=N-int(cur[[i for i,v in enumerate(META) if v['family']==META[0]['family']]].sum())
    # DIAGNOSTIC ONLY: min number of new compatible slots at any existing geometric atom.
    # Does not alter the fixed teacher's objective or create coordinates.
    AA=np.concatenate((A,A),axis=1)
    sol=fixed.solve(np.r_[np.zeros(K),np.ones(K)],np.vstack((np.ones(2*K,int),AA)),np.r_[remaining,LOW-cur],np.r_[remaining,HIGH-cur],np.zeros(2*K,int),np.r_[cap,np.full(K,remaining)])
    if sol is None:row['minimum_added_compatible_capacity']=None;row['augmentation_status']='infeasible even with all atom capacities';return row
    extra=sol[K:];row['minimum_added_compatible_capacity']=int(extra.sum());row['minimum_augmentation_by_class']=extra.tolist()
    assert oracle(cap+extra,cur) is not None
    return row

def smallest_prefix(points):
    ii=ids(points);left=N;right=len(points)
    if oracle(np.bincount(ii,minlength=K)) is None:return None
    while left<right:
        mid=(left+right)//2
        if oracle(np.bincount(ii[:mid],minlength=K)) is None:left=mid+1
        else:right=mid
    cap=np.bincount(ii[:left],minlength=K)
    assert oracle(cap,canonical=True) is not None
    if left>N:assert oracle(np.bincount(ii[:left-1],minlength=K)) is None
    return left

def capture(model,iteration,pool):
    key=f'{SEED}:2:{N}:{iteration}:{pool}'.encode();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
    points=np.empty((0,2));all_raw=[];blocks=[]
    with torch.no_grad():
        while len(points)<N:
            m=len(points);b=min(128,N-m);c=rng.random((max(256,8*b),2));all_raw.append(c.copy())
            features,_,near=model.make_context(points,c,N,iteration,pool);logits=model.net(features).flatten().clamp(-8,8)
            valid=near>=.4/math.sqrt(N);logits=logits.masked_fill(torch.from_numpy(~valid),-100)
            score=logits.numpy()+.7*rng.gumbel(size=len(c));accepted=[]
            for j in np.argsort(-score,kind='stable'):
                if not valid[j]:continue
                if accepted and np.min(np.sum((c[accepted]-c[j])**2,axis=1))<.16/N:continue
                accepted.append(j)
                if len(accepted)==b:break
            if not accepted:raise RuntimeError('original generator empty block')
            current=A@capacity(points) if m else np.zeros(len(META),int)
            blocks.append({'prefix_count':m,'raw_count':len(c),'prefix_distance_valid':int(valid.sum()),'prefix_overfilled_constraints':np.flatnonzero(current>HIGH).tolist(),'available_capacity':capacity(c[valid]).tolist()})
            points=np.vstack((points,c[accepted]))
    return points,np.concatenate(all_raw),blocks

def guarded_buffer(base,raw):
    # Existing minimum distance; original final pool retained first. Refills from
    # the SAME conditioned proposal stream, never another iteration/pool.
    radius=.4/math.sqrt(N);squared=.16/N
    candidates=raw[cKDTree(base).query(raw)[0]>=radius];grid={};extra=[]
    offsets=list(itertools.product((-1,0,1),repeat=base.shape[1]))
    for q in candidates:
        key=tuple(np.floor(q/radius).astype(int));okay=True
        for delta in offsets:
            near=grid.get(tuple(k+d for k,d in zip(key,delta)),[])
            if near and min(np.sum((q-v)**2) for v in near)<squared:okay=False;break
        if okay:extra.append(q);grid.setdefault(key,[]).append(q)
    return np.vstack((base,np.asarray(extra)))

def select(buffer):
    cap=capacity(buffer);z=oracle(cap,canonical=True)
    if z is None:return None
    ii=ids(buffer);chosen=np.concatenate([np.flatnonzero(ii==j)[:int(n)] for j,n in enumerate(z)])
    p=buffer[np.sort(chosen)]
    assert len(p)==N and len(np.unique(p,axis=0))==N
    assert np.array_equal(A@capacity(p),A@z)
    return p

def geometric(p,reference):
    complete=np.unique(np.vstack((p,boundary_faces(N,p.shape[1]))),axis=0)
    nn=cKDTree(p).query(p,k=2)[0][:,1];dd=cKDTree(complete).query(reference)[0]
    return {'cover':float(dd.max()),'holes':int(np.count_nonzero(dd>1/math.sqrt(N))),'hole_fraction':float(np.mean(dd>1/math.sqrt(N))),'min_nn_CV':float(nn.std()/nn.mean()),'nn_p01':float(np.quantile(nn,.01)),'clustering':float(np.mean(nn<.1/math.sqrt(N))),'nn_min':float(nn.min()),'finite':bool(np.isfinite(p).all()),'boundary_preserved':bool(np.array_equal(complete[np.any((complete==0)|(complete==1),axis=1)],boundary_faces(N,p.shape[1])))}

def historical_gate_report(report, screen_points, reference):
    """Use the authoritative cohort aggregation and original frozen entry reference.

    Per-condition comparisons to diagnostic outputs are descriptive only.
    """
    from experiments.sho.train_dyadic_spacefill_curriculum import health
    def cohort(points, pairs):
        values=[geometric(p,reference) for p in points]
        ov=[overlap(points[i],points[j],N) for i,j in pairs]
        return dict(cover=max(v['cover'] for v in values),
                    hole_fraction=max(v['hole_fraction'] for v in values),
                    min_nn_CV=min(v['min_nn_CV'] for v in values),
                    clustering=max(v['clustering'] for v in values),
                    nn_p01=min(v['nn_p01'] for v in values),
                    exact_overlap=max(v['exact_overlap_fraction'] for v in ov),
                    near_overlap=max(v['near_overlap_fraction_005_spacing'] for v in ov),
                    boundary=all(v['boundary_preserved'] for v in values),
                    finite=all(v['finite'] for v in values))
    archived_names=[f'iteration{100+i}_cell' for i in range(4)]+['iteration100_greedy']
    fresh_names=['iteration12000_cell','iteration12001_cell','iteration12000_greedy']
    archived_entry=json.load(open(ROOT/'dyadic_spacefill_curriculum/results.json'))['phases'][0]['entry']
    old=ProgressiveGenerator();old.load_state_dict(torch.load(ROOT/'progressive_scale_curriculum/stage_1_8x8_best.pt',weights_only=False)['state_dict']);old.eval().requires_grad_(False)
    with torch.no_grad():fresh_entry_points=[old.generate(N,it,pool)[1] for it,pool in [(12000,'cell'),(12001,'cell'),(12000,'greedy')]]
    groups=[]
    for name,names,pairs,entry in [('archived',archived_names,[(0,1),(1,2),(2,3),(0,4)],archived_entry),('fresh',fresh_names,[(0,1),(0,2)],cohort(fresh_entry_points,[(0,1),(0,2)]))]:
        value=cohort([screen_points[k] for k in names],pairs)
        checks=dict(cover=value['cover']<=1.05*entry['cover'],holes=value['hole_fraction']<=entry['hole_fraction']+.0005,
                    nn_CV=value['min_nn_CV']>=.08,clustering=value['clustering']<=.005,nn_p01=value['nn_p01']>=.95*entry['nn_p01'],
                    exact_freshness=value['exact_overlap']==0,near_freshness=value['near_overlap']<=min(.03,entry['near_overlap']+.005),boundary=value['boundary'],finite=value['finite'])
        assert health(value,entry)==all(v for k,v in checks.items() if k not in ('boundary','finite'))
        groups.append(dict(group=name,entry=entry,screen=value,checks=checks,passed=all(checks.values())))
    return groups

def run():
    start=time.monotonic();OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    files=list(ROOT.rglob('*.pt'))+list((ROOT/'joint_quota_assignment_teacher').glob('*'))+[Path(fixed.__file__)]
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    cp=ROOT/'dyadic_spacefill_curriculum/converged_continuation/phase_1_halves_best.pt'
    model=ProgressiveGenerator();model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    saved=dict(np.load(ROOT/'dyadic_spacefill_curriculum/converged_continuation/phase_1_halves_heldout_interiors.npz'))
    rows=[];prefixes=[];screens=[];streams={};screen_points={};base_points={}
    # Historical alternatives, all filtered to interior-only comparison.
    sources=[('diagnostic_progressive',saved),('original_progressive',dict(np.load(ROOT/'progressive_spacefill_smoke/progressive_heldout_interiors.npz'))),('fresh_v1',dict(np.load(ROOT/'fresh_spacefill_smoke/heldout_generated_probes.npz')))]
    for source,pools in sources:
        for key,p in pools.items():
            if source!='diagnostic_progressive' and 'N8600_' not in key:continue
            p=p[np.all((p>0)&(p<1),axis=1)];assert len(p)==N
            row={'source':source,'condition':key,**diagnostic(p)};rows.append(row)
            print(source,key,'minimum_addition',row['minimum_added_compatible_capacity'],flush=True)
    merged=np.concatenate(list(saved.values()));rows.append({'source':'successful_43k_composition','condition':'five_distinct_historical_contexts',**diagnostic(merged)})
    # Exact original reference set (not new geometry metrics).
    t=(np.arange(512)+.5)/512;reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    conditions=[(100+i,'cell') for i in range(4)]+[(100,'greedy'),(12000,'cell'),(12001,'cell'),(12000,'greedy')]
    for it,pool in conditions:
        name=f'iteration{it}_{pool}';begin=time.monotonic();base,raw,blocks=capture(model,it,pool)
        if name in saved:assert np.array_equal(base,saved[name]),'capture must exactly replay historical generator'
        base_points[name]=base;streams[name+'_base']=base;streams[name+'_raw']=raw
        for m in (0,128,N//2,N):prefixes.append({'condition':name,'prefix':m,**diagnostic(base[m:],A@capacity(base[:m]))})
        row={'source':'captured_raw_single_context','condition':name,**diagnostic(raw)}
        row['minimum_feasible_ordered_prefix']=smallest_prefix(raw);rows.append(row)
        buffer=guarded_buffer(base,raw);minimum=smallest_prefix(buffer)
        bdiag={'source':'guarded_single_context_buffer','condition':name,**diagnostic(buffer),'minimum_feasible_ordered_prefix':minimum};rows.append(bdiag)
        # ONE fixed screen: first feasible prefix, fixed quota teacher, stable
        # within-class first-available dispatch. No tuning against geometry.
        chosen=None if minimum is None else select(buffer[:minimum]);base_geom=geometric(base,reference)
        item={'condition':name,'archived':name in saved,'block_count':len(blocks),'raw_proposals':len(raw),'guarded_buffer_size':len(buffer),'minimum_feasible_buffer_size':minimum,'blocks':blocks,'base_geometry':base_geom,'feasible':chosen is not None,'seconds':time.monotonic()-begin}
        if chosen is not None:
            screen_points[name]=chosen;streams[name+'_screen']=chosen
            g=geometric(chosen,reference);item['screen_geometry']=g
            item['paired_diagnostic_checks']={'cover':g['cover']<=1.05*base_geom['cover'],'holes':g['hole_fraction']<=base_geom['hole_fraction']+.0005,'nn_CV':g['min_nn_CV']>=.08,'clustering':g['clustering']<=.005,'nn_p01':g['nn_p01']>=.95*base_geom['nn_p01'],'boundary':g['boundary_preserved'],'finite':g['finite']}
            item['count_metrics']=fixed.metrics(A@capacity(chosen),META,N)
            item['changed_original_points']=N-len(set(map(tuple,base))&set(map(tuple,chosen)))
        screens.append(item)
        print('screen',name,'buffer',minimum,'gates',item.get('paired_diagnostic_checks'),flush=True)
        (OUT/'progress.json').write_text(json.dumps({'rows':rows,'prefixes':prefixes,'screens':screens}))
    freshness=[]
    pairs=[('iteration100_cell','iteration101_cell'),('iteration101_cell','iteration102_cell'),('iteration102_cell','iteration103_cell'),('iteration100_cell','iteration100_greedy'),('iteration12000_cell','iteration12001_cell'),('iteration12000_cell','iteration12000_greedy')]
    for a,b in pairs:
        if a not in screen_points or b not in screen_points:continue
        old=overlap(base_points[a],base_points[b],N);new=overlap(screen_points[a],screen_points[b],N)
        passed=new['exact_overlap_fraction']==0 and new['near_overlap_fraction_005_spacing']<=min(.03,old['near_overlap_fraction_005_spacing']+.005)
        freshness.append({'pair':[a,b],'baseline':old,'screen':new,'passed':passed})
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items());assert unchanged
    passed=all(s['feasible'] and all(s.get('paired_diagnostic_checks',{}).values()) for s in screens) and all(f['passed'] for f in freshness)
    report={'provisional_paired_diagnostic_check_passed':passed,'status':'PENDING_AUTHORITATIVE_COHORT_GATES','rows':rows,'prefix_states':prefixes,'screens':screens,'freshness':freshness,'protected_files':len(protected),'protected_files_unchanged':unchanged,'runtime_seconds':time.monotonic()-start,'teacher_sha256':protected[str(Path(fixed.__file__))],'note':'Minimum added slots is an incidence-only lower requirement, not a guarantee of geometrically valid locations. Binary-search minima are specific stream prefixes, not universal reservoir ratios.'}
    report['historical_cohort_gates']=historical_gate_report(report,screen_points,reference)
    report['status']='SCREEN_PASSED_HISTORICAL_GATES' if all(s['feasible'] for s in screens) and all(g['passed'] for g in report['historical_cohort_gates']) else 'STOPPED_HISTORICAL_GATE_FAILURE'
    (OUT/'results.json').write_text(json.dumps(report,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2));np.savez_compressed(OUT/'captured_pools.npz',**streams)
    print(report['status'],report['runtime_seconds'],flush=True)

if __name__=='__main__':run()
