"""Isolate historical global-deficit -> proposal-logit operation; quick screen.

All proposal, feature, loss, hard selection and boundary code is imported unchanged.
Saved online traces lack state arrays: rebuild under their final diagnostic weights,
not a claim to reproduce the original changing-policy training prefixes.
"""
import copy,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator,discrepancy_rows,SCALES,boundary_faces,SEED
from experiments.sho.train_progressive_spacefill_smoke import monitor,sha,write_csv,BUDGETS
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics,overlap,read_baseline
from experiments.sho.train_dyadic_spacefill_curriculum import imbalance,PHASES
from experiments.sho.fresh_spacefill_generator import FreshGenerator
from experiments.sho.run_stage1b_spacefill_generator_smoke import Generator
from experiments.sho.run_stage1b_edirect_vs_direct import _candidate_pool
ROOT=Path('outputs/sho/stage1B');OLD=ROOT/'progressive_spacefill_smoke';OUT=ROOT/'global_deficit_scorer_screen'
CP=OLD/'progressive_best_before_evaluation.pt'

def dump(name,v): (OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def frozen(model,path):
    model.load_state_dict(torch.load(path,weights_only=False)['state_dict']);return model.eval().requires_grad_(False)

class Recorder(ProgressiveGenerator):
    def make_context(self,points,candidates,n,iteration,pool):
        x,g,d=super().make_context(points,candidates,n,iteration,pool)
        self.records.append(dict(m=len(points),q=candidates.copy(),x=x.numpy().copy(),nearest=d.copy()))
        return x,g,d

def collect(model,n,it,pool):
    model.records=[]
    with torch.no_grad():complete,p=model.generate(n,it,pool)
    block=32 if n<=1000 else 128 if n<=8600 else 256
    records=model.records
    for r in records:r['b']=min(block,n-r['m'])
    return dict(n=n,it=it,pool=pool,p=p,records=records)

def save_data(key,data):
    counts=np.array([len(r['q']) for r in data['records']]);offset=np.r_[0,counts.cumsum()]
    path=OUT/'dataset'/f'{key}.npz'
    np.savez_compressed(path,points=data['p'],offsets=offset,
        prefix_counts=[r['m'] for r in data['records']],block_counts=[r['b'] for r in data['records']],
        proposals=np.concatenate([r['q'] for r in data['records']]),
        features=np.concatenate([r['x'] for r in data['records']]),nearest=np.concatenate([r['nearest'] for r in data['records']]))
    return dict(key=key,N=data['n'],iteration=data['it'],pool=data['pool'],states=len(data['records']),
                proposals=int(counts.sum()),path=str(path),sha256=sha(path))

def components(model,data,r):
    _,grids,_=model.make_context(data['p'][:r['m']],r['q'],data['n'],data['it'],data['pool'])
    x=torch.from_numpy(r['x']);near=r['nearest']
    logits=model.net(x).flatten().clamp(-8,8).masked_fill(torch.from_numpy(near<.4/math.sqrt(data['n'])),-100)
    return model.losses(logits,grids,near,data['n'],r['m'],r['b'])

def update(model,optimizer,data,r):
    primary,secondary=components(model,data,r);pars=list(model.net.parameters())
    gp=torch.autograd.grad(primary,pars,retain_graph=True);gs=torch.autograd.grad(secondary,pars)
    dot=sum((a*b).sum() for a,b in zip(gp,gs));norm=sum((a*a).sum() for a in gp)+1e-30
    if dot<0:gs=tuple(b-dot/norm*a for a,b in zip(gp,gs))
    secnorm=sum((b*b).sum() for b in gs)+1e-30;factor=min(1.,float(.1*torch.sqrt(norm/secnorm)))
    optimizer.zero_grad()
    for p,a,b in zip(pars,gp,gs):p.grad=a+factor*b
    torch.nn.utils.clip_grad_norm_(pars,5.);optimizer.step()
    return float(primary.detach()),float(secondary.detach())

def state_metrics(model,data):
    with torch.no_grad():values=[float(components(model,d,r)[0]) for d in data for r in d['records']]
    return dict(states=len(values),mean_primary=float(np.mean(values)),max_primary=max(values))

def evaluate(model,conditions,split,reference,fresh,old):
    geometry=[];multi=[];freshness=[];dyadic=[];pools={};coords={}
    for n in BUDGETS:
        for name in ('Sobol','old_spacefill','fresh_v1','isolated'):
            pools[n,name]=[]
            for it,pool in conditions:
                if name=='Sobol':
                    full=_candidate_pool({1000:10,8600:86,32768:328}[n],200136,it+(1000000 if pool=='greedy' else 0))
                    p=full[~np.any((full==0)|(full==1),axis=1)]
                elif name=='old_spacefill':full,p=old.generate(n,[0.,0.],[1.,1.])
                elif name=='fresh_v1':full,p=fresh.generate(n,it,pool)
                else:
                    with torch.no_grad():full,p=model.generate(n,it,pool)
                    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and len(np.unique(p,axis=0))==n
                    assert np.array_equal(full[np.any((full==0)|(full==1),axis=1)],boundary_faces(n,2))
                    coords[f'N{n}_iteration{it}_{pool}']=p
                geometry.append(dict(split=split,N=n,generator=name,iteration=it,pool_type=pool,**metrics(full,p,reference,n)))
                multi.extend(dict(split=split,N=n,generator=name,iteration=it,pool_type=pool,**r) for r in discrepancy_rows(p))
                for _,_,families in PHASES:
                    for family in families:
                        dyadic.append(dict(split=split,N=n,generator=name,iteration=it,pool_type=pool,family=f'{family[0]}x{family[1]}',**imbalance(p,family)))
                pools[n,name].append((full,p))
            values=pools[n,name]
            for i,j in ((0,1),(1,2),(2,3),(0,4)):
                freshness.append(dict(split=split,N=n,generator=name,kind='overlap',pair=str((i,j)),**overlap(values[i][1],values[j][1],n)))
            dist=cKDTree(np.unique(np.concatenate([v[0] for v in values[:4]]),axis=0)).query(reference)[0]
            freshness.append(dict(split=split,N=n,generator=name,kind='union',cover=float(dist.max())))
        print(json.dumps(dict(event='evaluation',split=split,N=n)),flush=True)
    gates=[];summaries=[];scale_rows=[]
    for n in BUDGETS:
        groups={name:[r for r in geometry if r['N']==n and r['generator']==name] for name in ('Sobol','old_spacefill','fresh_v1','isolated')}
        ds={name:{k:max(r['grid_anchored_discrepancy'] for r in multi if r['N']==n and r['generator']==name and r['bins_per_axis']==k) for k in SCALES} for name in groups}
        ratios={name:max(ds[name][k]/max(ds['Sobol'][k],1/n) for k in SCALES) for name in groups}
        for name,gg in groups.items():
            for k in SCALES:scale_rows.append(dict(split=split,N=n,generator=name,scale=k,discrepancy=ds[name][k],Sobol=ds['Sobol'][k],one_count_ratio=ds[name][k]/max(ds['Sobol'][k],1/n)))
            summaries.append(dict(split=split,N=n,generator=name,ratio=ratios[name],max_cover=max(r['complete_cover'] for r in gg),
                                  max_holes=max(r['large_hole_count'] for r in gg),mean_CV=float(np.mean([r['nn_CV'] for r in gg]))))
        ng=groups['isolated'];sg=groups['Sobol'];fr=[r for r in freshness if r['N']==n and r['generator']=='isolated' and r['kind']=='overlap']
        nu=next(r['cover'] for r in freshness if r['N']==n and r['generator']=='isolated' and r['kind']=='union');su=next(r['cover'] for r in freshness if r['N']==n and r['generator']=='Sobol' and r['kind']=='union')
        paired=max(a['complete_cover']/b['complete_cover'] for a,b in zip(ng,sg))
        g=dict(split=split,N=n,discrepancy_materially_better=all(ratios['isolated']<=.8*ratios[name] for name in ('old_spacefill','fresh_v1')),
               no_scale_regression=all(ds['isolated'][k]<=1.1*ds['fresh_v1'][k] for k in SCALES),
               coverage=max(r['complete_cover'] for r in ng)<=1.1*max(r['complete_cover'] for r in sg) and paired<=1.15,
               boundary=max(r['boundary_cover'] for r in ng)==max(r['boundary_cover'] for r in sg),
               freshness=all(r['exact_overlap_fraction']==0 and r['near_overlap_fraction_005_spacing']<=.03 for r in fr),
               union=nu<=1.25*su,holes=max(r['large_hole_count'] for r in ng)<=max(r['large_hole_count'] for r in sg)+.001*len(reference),
               not_rigid_or_clustered=np.mean([r['nn_CV'] for r in ng])>=.08 and max(r['clustered_fraction'] for r in ng)<=.005)
        g={k:bool(v) if isinstance(v,(bool,np.bool_)) else v for k,v in g.items()};g['passed']=all(v for k,v in g.items() if k not in ('N','split'));gates.append(g)
    for filename,rows in [('geometry',geometry),('multiscale',multi),('freshness',freshness),('dyadic',dyadic),('scale_ratios',scale_rows),('summary',summaries)]:write_csv(OUT/f'{split}_{filename}.csv',rows)
    np.savez_compressed(OUT/f'{split}_interiors.npz',**coords)
    return gates

def main():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=False);(OUT/'dataset').mkdir()
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    for p in ['experiments/sho/progressive_spacefill_generator.py','experiments/sho/hierarchical_allocation_poc.py']:protected[p]=sha(p)
    dump('protected_hashes.json',protected)
    criteria=json.loads((OLD/'evaluation_criteria.json').read_text());dump('historical_gate.json',criteria)
    dump('plan.json',dict(operation='34 global-deficit/context features -> one proposal logit',model='float64 Linear(34,1,bias=False)',parameters=34,
        random_seed=SEED,epochs=3,optimizer='Adam lr=.003',max_training_seconds=180,
        loss_authority='ProgressiveGenerator.losses unchanged; same projected <=10% secondary gradient',
        gate_authority=str(OLD/'evaluation_criteria.json'),unchanged='make_context, IID proposals, Gumbel selection, exclusion, boundaries',
        dataset='Saved historical trace contexts reconstructed under final diagnostic weights; original online prefixes unavailable',
        monitor='Original iteration7000 and original training-only metric/coverage selection',
        heldout=[(100+j,'cell') for j in range(4)]+[(100,'greedy')],fresh=[(11000+j,'cell') for j in range(4)]+[(11000,'greedy')],
        no_coordinate_training=True,no_quota_modification=True,no_N1_search=True))
    trace=list(csv.DictReader((OLD/'progressive_training_history.csv').open()));keys=list(dict.fromkeys((int(r['N']),int(r['iteration']),r['pool_type']) for r in trace))
    recorder=frozen(Recorder(),CP);data=[];manifest=[]
    for n,it,pool in keys:
        d=collect(recorder,n,it,pool);data.append(d);manifest.append(save_data(f'train_N{n}_{it}_{pool}',d));print(json.dumps(dict(event='dataset',N=n,iteration=it,states=len(d['records']))),flush=True)
    held=[];saved=np.load(OLD/'progressive_heldout_interiors.npz')
    for n in BUDGETS:
        d=collect(recorder,n,100,'cell');assert np.array_equal(d['p'],saved[f'N{n}_iteration100_cell'])
        held.append(d);manifest.append(save_data(f'heldout_N{n}_100_cell',d))
    dump('dataset_manifest.json',dict(original_trace_rows=len(trace),trajectories=len(data),training_states=sum(len(d['records']) for d in data),
        heldout_states=sum(len(d['records']) for d in held),archived_coordinate_replay_exact=True,records=manifest))
    model=ProgressiveGenerator();model.net=torch.nn.Linear(34,1,bias=False,dtype=torch.float64)
    opt=torch.optim.Adam(model.net.parameters(),lr=.003)
    reference=np.random.default_rng(SEED+1).random((16384,2));mon=monitor(model,reference,0)
    initial=float(np.mean([r['primary_monitor'] for r in mon]));covers={r['N']:r['complete_cover'] for r in mon}
    before=dict(train=state_metrics(model,data),heldout=state_metrics(model,held));best=initial;beststate=copy.deepcopy(model.state_dict());bestcycle=0;history=[];stale=0;train_start=time.monotonic()
    for epoch in range(1,4):
        for d in data:
            for r in d['records']:
                a,b=update(model,opt,d,r);history.append(dict(epoch=epoch,N=d['n'],iteration=d['it'],pool=d['pool'],prefix=r['m'],primary=a,secondary=b))
        mm=monitor(model,reference,epoch);mon+=mm;value=float(np.mean([r['primary_monitor'] for r in mm]));feasible=all(r['complete_cover']<=1.1*covers[r['N']] for r in mm)
        if feasible and value<best*(1-1e-4):best=value;beststate=copy.deepcopy(model.state_dict());bestcycle=epoch;stale=0
        else:stale+=1
        print(json.dumps(dict(event='training',epoch=epoch,monitor=value,best=best,feasible=feasible,seconds=time.monotonic()-train_start)),flush=True)
        if stale>=2 or time.monotonic()-train_start>180:break
    model.load_state_dict(beststate);model.eval().requires_grad_(False)
    cp=OUT/'deficit_logit_diagnostic.pt';torch.save(dict(state_dict=beststate,status='DIAGNOSTIC_PENDING_EVALUATION',architecture='Linear(34,1,bias=False)',selected_epoch=bestcycle,parameters=34),cp);cp.with_suffix('.sha256').write_text(sha(cp)+'\n')
    after=dict(train=state_metrics(model,data),heldout=state_metrics(model,held))
    write_csv(OUT/'training.csv',history);write_csv(OUT/'monitor.csv',mon)
    dump('training_results.json',dict(updates=len(history),selected_epoch=bestcycle,initial_monitor=initial,best_monitor=best,before=before,after=after,weights=model.net.weight.detach().numpy().tolist(),checkpoint=str(cp),sha256=sha(cp)))
    freshstates=[collect(recorder,n,11000,'cell') for n in BUDGETS]
    dump('fresh_state_loss.json',state_metrics(model,freshstates))
    for d in freshstates:manifest.append(save_data(f'fresh_N{d["n"]}_11000_cell',d))
    dump('dataset_files.json',manifest)
    fresh=frozen(FreshGenerator(2),ROOT/'fresh_spacefill_smoke/fresh_spacefill_trained_pending_evaluation.pt')
    old=frozen(Generator(2),ROOT/'spacefill_downstream_screen/spacefill_generator_candidate.pt')
    t=(np.arange(512)+.5)/512;ref=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    allg=[]
    for split,conditions in [('heldout',[(100+j,'cell') for j in range(4)]+[(100,'greedy')]),('fresh',[(11000+j,'cell') for j in range(4)]+[(11000,'greedy')])]:
        allg.extend(evaluate(model,conditions,split,ref,fresh,old));dump('gates.json',allg)
    assert all(sha(p)==h for p,h in protected.items())
    passed=bestcycle>0 and all(r['passed'] for r in allg)
    if passed:
        candidate=OUT/'global_deficit_logit_candidate.pt';torch.save(dict(state_dict=beststate,status='FROZEN_CANDIDATE',parameters=34,source=str(cp)),candidate);candidate.with_suffix('.sha256').write_text(sha(candidate)+'\n')
    dump('results.json',dict(status='CANDIDATE' if passed else 'STOPPED_HISTORICAL_GATE_FAILURE',passed=passed,parameters=34,updates=len(history),selected_epoch=bestcycle,
        training_states=sum(len(d['records']) for d in data),heldout_state_replay_exact=True,gates=allg,protected_unchanged=True,
        promoted=passed,ready_to_compose=False if not passed else 'Requires unchanged regional-quota composition validation',seconds=time.monotonic()-started))

if __name__=='__main__':main()
