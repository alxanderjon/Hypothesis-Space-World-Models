"""Discrepancy-first continuation with independent hard feasibility checks."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.fresh_spacefill_generator import FreshGenerator
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_generator_smoke as old
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/refined_spacefill')
INITIAL=Path('outputs/sho/stage1B/fresh_spacefill_smoke/fresh_spacefill_trained_pending_evaluation.pt')
BUDGETS=(1000,8600,32768);SCALES=(2,4,8,16,32,64,128);SHIFTS=(0.,.25,.5,.75)
CONDITIONS=[(8000+j,'cell') for j in range(4)]+[(8000,'greedy')]

def sobol(n,it,pool):
    sites=10 if n==1000 else 86 if n==8600 else 328
    p=src.edirect._candidate_pool(sites,200136,it+(1000000 if pool=='greedy' else 0))
    return p,p[~np.any((p==0.)|(p==1.),axis=1)]

def hard_discrepancy(points):
    result=[];n=len(points)
    for k in SCALES:
        values=[]
        for shift in SHIFTS:
            hist=np.histogramdd((points+shift/k)%1,bins=(np.linspace(0,1,k+1),)*2)[0]
            edge=np.arange(1,k+1)/k
            values.append(float(np.max(np.abs(hist.cumsum(0).cumsum(1)/n-edge[:,None]*edge[None,:]))))
        result.append(max(values))
    return np.asarray(result)

def straight_discrepancy(points):
    """Exact histogram in forward pass, CIC derivative as a surrogate gradient."""
    n=len(points);result=[]
    for k in SCALES:
        values=[]
        for shift in SHIFTS:
            raw=torch.remainder(points*k+shift,k)
            ids=torch.floor(raw).long();hard=torch.bincount(ids[:,0]*k+ids[:,1],minlength=k*k).to(torch.float64)
            z=torch.remainder(raw-.5,k);base=torch.floor(z).long();f=z-base
            soft=torch.zeros(k*k,dtype=torch.float64)
            for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
                weight=(f[:,0] if dx else 1-f[:,0])*(f[:,1] if dy else 1-f[:,1])
                index=((base[:,0]+dx)%k)*k+(base[:,1]+dy)%k
                soft=soft.scatter_add(0,index,weight)
            hist=(hard+(soft-soft.detach())).reshape(k,k)
            edge=torch.arange(1,k+1,dtype=torch.float64)/k
            values.append((hist.cumsum(0).cumsum(1)/n-edge[:,None]*edge[None,:]).abs().max())
        result.append(torch.stack(values).max())
    return torch.stack(result)

def sample_stats(pools,reference,n):
    cover=[];hole=[];cv=[];cluster=[]
    for points,interior in pools:
        d=cKDTree(points).query(reference)[0];nn=cKDTree(interior).query(interior,k=2)[0][:,1]
        cover.append(float(d.max()));hole.append(float(np.mean(d>1/math.sqrt(n))))
        cv.append(float(nn.std()/nn.mean()));cluster.append(float(np.mean(nn<.1/math.sqrt(n))))
    overlaps=[];exact=[]
    for a,b in ((0,1),(1,2),(2,3),(0,4)):
        dd=cKDTree(pools[b][1]).query(pools[a][1])[0]
        overlaps.append(float(np.mean(dd<.05/math.sqrt(n))));exact.append(int(np.sum(dd==0)))
    return dict(cover=max(cover),holes=max(hole),min_CV=min(cv),clustering=max(cluster),near=max(overlaps),exact=max(exact))

def run():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'refined_spacefill_trained_diagnostic.pt'
    if path.exists():raise FileExistsError('No automatic retraining or run extension')
    torch.set_num_threads(1);started=time.time()
    protected=json.loads(Path('outputs/sho/stage1B/fresh_spacefill_smoke/training_summary.json').read_text())['protected_hashes']
    protected[str(INITIAL)]=src.sha(INITIAL)
    model=FreshGenerator(2);model.load_state_dict(torch.load(INITIAL,weights_only=False)['state_dict'])
    baseline={};reference={};denominator={};history=[]
    for n in BUDGETS:
        t=(np.arange(256)+.5)/256
        reference[n]=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27094000+n).random((65536,2))))
        pools=[sobol(n,*c) for c in CONDITIONS]
        baseline[n]=sample_stats(pools,reference[n],n)
        denominator[n]=np.maximum.reduce([hard_discrepancy(x[1]) for x in pools])
    limits={};latest={}
    def monitor(step,establish=False):
        worst=0.;violations=[]
        for n in BUDGETS:
            pools=[model.generate(n,*c) for c in CONDITIONS]
            st=sample_stats(pools,reference[n],n)
            if establish:
                limits[n]=dict(cover=baseline[n]['cover'],near=st['near']+.005,
                    clustering=max(.001,st['clustering']+.001),holes=baseline[n]['holes']+.001,min_CV=max(.08,.8*st['min_CV']))
            checks=dict(coverage=st['cover']<=limits[n]['cover'],freshness=st['near']<=limits[n]['near'] and st['exact']==0,
                clustering=st['clustering']<=limits[n]['clustering'],holes=st['holes']<=limits[n]['holes'],not_rigid=st['min_CV']>=limits[n]['min_CV'])
            d=np.maximum.reduce([hard_discrepancy(p[1]) for p in pools]);ratio=d/np.maximum(denominator[n],1/n)
            worst=max(worst,float(ratio.max()));latest[n]=dict(stats=st,discrepancy=d.tolist(),operational_ratio=ratio.tolist())
            history.append(dict(record_type='training_monitor',step=step,N=n,worst_normalized_discrepancy=float(ratio.max()),
                discrepancy_by_scale=json.dumps(d.tolist()),normalization_by_scale=json.dumps(denominator[n].tolist()),
                ratios_by_scale=json.dumps(ratio.tolist()),coverage_ratio=st['cover']/baseline[n]['cover'],**st,
                constraints_passed=all(checks.values()),violated=json.dumps([k for k,v in checks.items() if not v])))
            violations += [f'N{n}:{key}' for key,value in checks.items() if not value]
        return worst,violations
    initial,violations=monitor(0,True);best=initial;best_state=copy.deepcopy(model.state_dict());best_step=0
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    accepted=0;attempted=0;stop='40_update_cap';last_improvement=0
    criteria=dict(primary='maximum Sobol-normalized hard discrepancy across budgets/scales/shifts; straight-through CIC derivative',
        count_floor='max(D_Sobol,1/N) ONLY for optimization; report literal ratios including infinity at evaluation',
        updates_cap=40,learning_rate=.001,no_improvement_patience=10,constraints=limits,
        validation_max_operational_ratio=3.,validation_required_ratio_improvement_vs_v1=.20,
        validation_cover_ratio_limit=1.,validation_near_overlap_allowance_absolute=.005)
    (OUT/'criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
    if violations:stop='initial_constraint_failure:'+','.join(violations)
    else:
        for step in range(1,41):
            last_feasible=copy.deepcopy(model.state_dict());optimizer.zero_grad();terms=[]
            for n in BUDGETS:
                denom=torch.from_numpy(np.maximum(denominator[n],1/n))
                # Rotate cell iterations; cell and greedy conditions remain
                # distinct, while the fixed monitor includes every condition.
                for it,pool in (CONDITIONS[(step-1)%4],CONDITIONS[4]):
                    points=model.interior(n,it,pool)
                    terms.append(straight_discrepancy(points)/denom)
            primary=torch.cat(terms).max()
            if not torch.isfinite(primary):stop='nonfinite_primary';break
            primary.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();attempted=step
            current,violations=monitor(step)
            if violations:
                model.load_state_dict(last_feasible);stop='constraint_crossing_rejected:'+','.join(violations);break
            accepted=step
            if current<best:
                best=current;best_state=copy.deepcopy(model.state_dict());best_step=step;last_improvement=step
            src.write_csv(OUT/'refined_training_history.csv',history)
            if step%5==0:print(f'update {step}/40 worst ratio {current:.4g}, best {best:.4g}, initial {initial:.4g}, elapsed {time.time()-started:.1f}s',flush=True)
            if step-last_improvement>=10:stop='no_discrepancy_improvement_for_10_updates';break
            if time.time()-started>180:stop='three_minute_training_cap';break
    model.load_state_dict(best_state);model.eval().requires_grad_(False)
    # First operation after selecting final weights: durable persistence.
    digest=save_checkpoint(path,dict(status='TRAINED_DIAGNOSTIC_PENDING_EVALUATION',state_dict=model.state_dict(),dimension=2,
        source_checkpoint=str(INITIAL),source_sha256=protected[str(INITIAL)],attempted_updates=attempted,accepted_updates=accepted,
        selected_update=best_step,stop_reason=stop,criteria=criteria,training_conditions=CONDITIONS,
        reference_seeds={n:27094000+n for n in BUDGETS},initial_objective=initial,selected_objective=best,
        model_implementation='experiments/sho/fresh_spacefill_generator.py',model_implementation_sha256=src.sha('experiments/sho/fresh_spacefill_generator.py')))
    src.write_csv(OUT/'refined_training_history.csv',history)
    result=dict(attempted_updates=attempted,accepted_updates=accepted,selected_update=best_step,stop_reason=stop,
        initial_worst_discrepancy_ratio=initial,selected_worst_discrepancy_ratio=best,training_seconds=time.time()-started,
        checkpoint=str(path),sha256=digest,protected_hashes=protected,protected_unchanged=all(src.sha(p)==h for p,h in protected.items()))
    (OUT/'training_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='protected_hashes'},indent=2),flush=True)

if __name__=='__main__':run()
