"""Conditional, short unchanged-architecture screen after Sobol target audit.

Uses existing shifted-CDF metric and its existing CIC surrogate derivative.
No leaf-center objective, new metric, frozen model edit, or promotion.
"""
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.sho import audit_sobol_placement_target as audit
from experiments.sho import conditioned_leaf_placement_screen as base
from experiments.sho.refine_fresh_spacefill import hard_discrepancy, straight_discrepancy

OUT=base.ROOT/'sobol_aligned_placement_screen'


def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    torch.set_num_threads(1);start=time.monotonic()
    assert json.loads((audit.OUT/'results.json').read_text())['status']=='AUDIT_COMPLETE'
    # This audit must be read/reviewed before explicitly invoking this screen.
    OUT.mkdir(exist_ok=False)
    protected=json.loads((audit.OUT/'protected_hashes.json').read_text())
    assert all(base.sha(p)==h for p,h in protected.items())
    c=base.case_from_leaves('N1_generator_200136_N1000',json.loads((audit.OUT/'continuous_n1000_leaves.json').read_text()),'screen')
    n=c['n'];h=1/math.sqrt(n);sites=audit.ed._initial_design()
    model=base.Placement(8);model.load_state_dict(torch.load(audit.CP,weights_only=False)['state_dict'])
    original=copy.deepcopy(model.state_dict())
    train=[(8000+i,0) for i in range(4)]+[(8000,1)]
    held=[(9000+i,0) for i in range(4)]+[(9000,1)]
    dump('plan.json',dict(source=str(audit.CP),source_sha256=base.sha(audit.CP),parameters=65,architecture_unchanged=True,
         updates=40,learning_rate=.001,train=train,heldout=held,
         primary='Worst shifted anchored-CDF discrepancy / max(matched Sobol training D,1/N), same metric and CIC surrogate as refine_fresh_spacefill.py',
         secondary='Match existing Sobol NN p10/median/p90; Sobol-relative covering safeguard; current/current cross-pool freshness',
         center_loss=False,center_acceptance=False,
         recovery_gate='Literal heldout hard max and RMS at every scale <= matched Sobol envelope; no one-count-floor relaxation at validation',
         geometry_gates=dict(max_complete_cover_ratio=1.10,max_hole_excess=.001,min_nn_CV=.08,max_cluster=.005,
                             max_near_overlap_005=.10,max_union_cover_ratio=1.25,min_union_gain=.10),
         domain='One frozen N1000 partition, new pool contexts; not new geometry or N1 rollout. No promotion.'))
    teacher={}
    grid=(np.arange(256)+.5)/256
    reference=np.vstack((np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2),np.random.default_rng(27094000+n).random((65536,2))))
    face=base.edges(n,2)
    for it,pool in train:
        complete=audit.ed._candidate_pool(10,200136,it+(1000000 if pool else 0))
        p=complete[~np.any((complete==0)|(complete==1),axis=1)]
        nn=cKDTree(p).query(p,k=2)[0][:,1]
        teacher[it,pool]=dict(D=hard_discrepancy(p),q=np.quantile(nn,[.1,.5,.9]),cover=float(cKDTree(complete).query(reference)[0].max()))
    denom=torch.from_numpy(np.maximum(np.maximum.reduce([v['D'] for v in teacher.values()]),1/n))
    def points(condition):return base.placed(model,c,*condition,sites)
    def monitor():
        with torch.no_grad():
            d=np.maximum.reduce([hard_discrepancy(points(v).numpy()) for v in train])
        return float((d/denom.numpy()).max())
    initial=monitor();best=initial;beststate=copy.deepcopy(original);selected=0;history=[]
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    for step in range(1,41):
        conditions=[train[(step-1)%4],train[-1]]
        ps=[points(v) for v in conditions];primary=torch.cat([straight_discrepancy(p)/denom for p in ps]).max()
        secondary=0.
        for p,condition in zip(ps,conditions):
            detached=p.detach().numpy();nnid=cKDTree(detached).query(detached,k=2)[1][:,1]
            near=torch.linalg.vector_norm(p-p[nnid],dim=1)
            secondary=secondary+.05*((torch.quantile(near,torch.tensor([.1,.5,.9],dtype=torch.float64))-torch.from_numpy(teacher[condition]['q']))/h).square().mean()
            full=torch.cat((p,torch.from_numpy(face)))
            ids=cKDTree(full.detach().numpy()).query(reference)[1]
            dd=torch.linalg.vector_norm(torch.from_numpy(reference)-full[ids],dim=1)
            secondary=secondary+20*torch.relu(dd.max()/teacher[condition]['cover']-1.10).square()
        ids=cKDTree(ps[0].detach().numpy()).query(ps[1].detach().numpy())[1]
        cross=torch.linalg.vector_norm(ps[1]-ps[0][ids],dim=1)
        secondary=secondary+2*torch.relu(1-cross/(.05*h)).square().mean()
        loss=primary+secondary
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        value=monitor()
        if value<best:
            best=value;selected=step;beststate=copy.deepcopy(model.state_dict())
        history.append(dict(step=step,loss=float(loss.detach()),hard_training_ratio=value,best_ratio=best))
        if step%10==0:print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(beststate);model.eval().requires_grad_(False)
    path=OUT/'sobol_objective_diagnostic.pt'
    torch.save(dict(state_dict=beststate,hidden=8,parameters=65,status='DIAGNOSTIC_ONLY',source=str(audit.CP),selected_step=selected),path)
    path.with_suffix('.sha256').write_text(base.sha(path)+'\n')
    def write(name,rows):
        import csv
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    write('training.csv',history)
    t=(np.arange(512)+.5)/512
    ref=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    geo=[];multi=[];sets={k:[] for k in ('Sobol','before','after')}
    for it,pool in held:
        sobol=audit.ed._candidate_pool(10,200136,it+(1000000 if pool else 0))
        sb=sobol[~np.any((sobol==0)|(sobol==1),axis=1)]
        with torch.no_grad():after=points((it,pool)).numpy()
        model.load_state_dict(original)
        with torch.no_grad():before=points((it,pool)).numpy()
        model.load_state_dict(beststate)
        for name,interior in [('Sobol',sb),('before',before),('after',after)]:
            full=np.unique(np.concatenate((interior,face)),axis=0)
            geo.append(dict(generator=name,iteration=it,pool=pool,**audit.metrics(full,interior,ref,n),
                            outside=None if name=='Sobol' else int(np.sum(np.any((interior<c['lo'])|(interior>c['hi']),axis=1))),
                            duplicates=n-len(np.unique(interior,axis=0)),finite=bool(np.isfinite(interior).all())))
            multi.extend(audit.multiscale(interior,n,name,it,str(pool)));sets[name].append((full,interior))
    per_scale=[];distribution_pass=True
    for k in audit.SCALES:
        sb=[r for r in multi if r['generator']=='Sobol' and r['scale']==k]
        for name in ('before','after'):
            rr=[r for r in multi if r['generator']==name and r['scale']==k]
            a=max(r['grid_anchored_max'] for r in rr);b=max(r['grid_anchored_max'] for r in sb)
            ar=max(r['grid_anchored_rms'] for r in rr);br=max(r['grid_anchored_rms'] for r in sb)
            passed=a<=b and ar<=br
            if name=='after':distribution_pass=distribution_pass and passed
            per_scale.append(dict(generator=name,scale=k,max=a,Sobol_max=b,RMS=ar,Sobol_RMS=br,
                                  operational_ratio=a/max(b,1/n),literal_ratio=a/b if b else None,passed=passed))
    freshness=[]
    for name,pools in sets.items():
        for i,j in ((0,1),(1,2),(2,3),(0,4)):
            freshness.append(dict(generator=name,kind='overlap',pair=str((i,j)),**audit.overlaps(pools[i][1],pools[j][1],n)))
        d=cKDTree(np.unique(np.concatenate([p[0] for p in pools[:4]]),axis=0)).query(ref)[0]
        freshness.append(dict(generator=name,kind='union',cover=float(d.max()),gain=1-float(d.max())/np.mean([r['complete_cover'] for r in geo if r['generator']==name and r['pool']==0])))
    write('geometry.csv',geo);write('multiscale.csv',multi);write('scale_comparison.csv',per_scale);write('freshness.csv',freshness)
    a=[r for r in geo if r['generator']=='after'];sb=[r for r in geo if r['generator']=='Sobol'];fr=[r for r in freshness if r['generator']=='after' and r['kind']=='overlap']
    au=next(r for r in freshness if r['generator']=='after' and r['kind']=='union');su=next(r for r in freshness if r['generator']=='Sobol' and r['kind']=='union')
    gates=dict(distribution=distribution_pass,containment=all(r['outside']==0 for r in a),finite=all(r['finite'] for r in a),duplicates=all(r['duplicates']==0 for r in a),
               coverage=max(r['complete_cover'] for r in a)<=1.1*max(r['complete_cover'] for r in sb),
               holes=max(r['large_hole_fraction'] for r in a)<=max(r['large_hole_fraction'] for r in sb)+.001,
               not_rigid=min(r['nn_CV'] for r in a)>=.08,clustering=max(r['clustered_fraction'] for r in a)<=.005,
               freshness=all(r['exact_overlap_count']==0 and r['near_overlap_fraction_005_spacing']<.10 for r in fr),
               union_gain=au['gain']>=.10,union_vs_sobol=au['cover']<=1.25*su['cover'])
    assert all(base.sha(p)==h for p,h in protected.items())
    dump('results.json',dict(status='SCREEN_PASSED_NO_PROMOTION' if all(gates.values()) else 'STOPPED_SOBOL_DISTRIBUTION_GATE_FAILURE',
                             parameters=65,updates=40,selected=selected,initial_training_ratio=initial,best_training_ratio=best,
                             gates={k:bool(v) for k,v in gates.items()},checkpoint=str(path),sha256=base.sha(path),
                             existing_checkpoints_unchanged=True,promoted=False,seconds=time.monotonic()-start))


if __name__=='__main__':main()
