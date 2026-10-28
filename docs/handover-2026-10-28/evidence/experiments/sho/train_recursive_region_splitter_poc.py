"""Supervised synthetic-box splitting only; frozen learned mass allocator."""
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from experiments.sho.train_geometry_allocation_curriculum import GeometryAllocator
from experiments.sho.hierarchical_allocation_poc import conserve
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/recursive_region_splitter_poc')
ALLOC=Path('outputs/sho/stage1B/geometry_allocation_curriculum/stage_4_best.pt')
ALLOC_SHA='046d2c95a8efb02d7f1d0e5dee40d5fd6b729be9e1b3c2eb2f04cf7fdd531c9c'
SEED=27172026
CUTS=np.array([.2,.3,.4,.5,.6,.7,.8])
HELD_BUDGETS=[11,19,32,37,64,73,128,149,251,256]
ASPECT_BANDS=[(1.3,1.5),(2.7,3.3),(6.,8.)]
GATES=dict(axis_accuracy=.99,split_agreement=.90,cut_mae=.03,mean_score_regret=.005,max_score_regret=.05)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


class Splitter(nn.Module):
    """Shared axis network. Seven learned cut logits per axis, plus axis logit."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,8)).double()

    def forward(self,length,n):
        length=length/length.log().mean(1,keepdim=True).exp()
        z=length.log()
        b=n.log()/math.log(257)
        feats=torch.stack((z,z.flip(1),b[:,None].expand(-1,2),
                           (1/n)[:,None].expand(-1,2),torch.full_like(z,2/3)),dim=-1)
        out=self.net(feats)
        return out[:,:,0],out[:,:,1:]

    def choose(self,length,n):
        with torch.no_grad():axis_score,cut_score=self(length,n)
        axis=axis_score.argmax(1)
        cut=cut_score[torch.arange(len(n)),axis].argmax(1)
        return axis.numpy(),cut.numpy()


def children(lo,hi,axis,q):
    lower=np.stack((lo,lo));upper=np.stack((hi,hi))
    cut=lo[axis]+q*(hi[axis]-lo[axis])
    upper[0,axis]=cut;lower[1,axis]=cut
    return lower,upper


class LockedAllocation:
    def __init__(self):
        assert digest(ALLOC)==ALLOC_SHA
        self.net=GeometryAllocator();self.net.load_state_dict(torch.load(ALLOC,weights_only=False)['state_dict'])
        self.net.eval().requires_grad_(False)
        self.calls=0;self.count_conservation_errors=0;self.volume_count_disagreements=0
        self.fraction_max_error=0.;self.fraction_conservation_max=0.

    def allocate(self,relative_lengths,budgets):
        z=np.log(relative_lengths);b=np.log1p(budgets)/math.log1p(32768)
        f=np.stack((z,relative_lengths-1,(relative_lengths-1)**2,z*z,
                    z*b[:,None,None],z*b[:,None,None]**2),axis=-1).sum(2)
        with torch.no_grad():p=self.net(torch.from_numpy(f),torch.ones(f.shape[:2],dtype=torch.bool)).numpy()
        counts=np.stack([conserve(int(n),v) for n,v in zip(budgets,p)])
        target=np.prod(relative_lengths,axis=2)
        exact=np.stack([conserve(int(n),v) for n,v in zip(budgets,target)])
        self.calls+=len(p)
        self.count_conservation_errors+=int(np.sum(counts.sum(1)!=budgets))
        self.volume_count_disagreements+=int(np.sum(np.any(counts!=exact,axis=1)))
        self.fraction_max_error=max(self.fraction_max_error,float(np.abs(p-target).max()))
        self.fraction_conservation_max=max(self.fraction_conservation_max,float(np.abs(p.sum(1)-1).max()))
        return counts,p,exact

    def stats(self):
        return {k:v for k,v in self.__dict__.items() if k!='net'}


def quality(lengths,counts,parent_length,n):
    if np.min(counts)<1:return float('inf')
    reference_spacing=math.sqrt(float(np.prod(parent_length))/n)
    radius_term=np.max(np.linalg.norm(lengths,axis=1)/np.sqrt(counts))/reference_spacing
    aspect=lengths.max(1)/lengths.min(1)
    return float(radius_term+.05*np.log(aspect).max())


def label_boxes(lo,hi,budgets,allocator):
    child_geometry=[];child_bounds=[];bs=[]
    for a,b,n in zip(lo,hi,budgets):
        for axis in range(2):
            for q in CUTS:
                cl,ch=children(a,b,axis,q)
                child_geometry.append((ch-cl)/(b-a));child_bounds.append(ch-cl);bs.append(n)
    cg=np.array(child_geometry);counts,prob,exact=allocator.allocate(cg,np.array(bs))
    score=np.empty((len(lo),2,7));allocations=counts.reshape(-1,2,7,2)
    cursor=0
    for i,(a,b,n) in enumerate(zip(lo,hi,budgets)):
        for axis in range(2):
            for k in range(7):
                score[i,axis,k]=quality(child_bounds[cursor],counts[cursor],b-a,n);cursor+=1
    flat=score.reshape(-1,14)
    # Numerical quality ties are bookkeeping only; no tolerance in allocation.
    optimal=flat.min(1)
    equivalent=flat<=optimal[:,None]+1e-12*np.maximum(1,optimal[:,None])
    selected=equivalent.argmax(1) # deterministic axis, then ascending cut
    return dict(scores=score,axis=selected//7,cut=selected%7,equivalent=equivalent,
                counts=allocations,probabilities=prob.reshape(-1,2,7,2),
                count_vs_volume_mismatch=np.any(counts!=exact,axis=1).reshape(-1,2,7))


def data(count,seed,kind):
    rng=np.random.default_rng(seed);lengths=[];budgets=[]
    train_budget=np.array([n for n in range(2,257) if n not in HELD_BUDGETS])
    while len(lengths)<count:
        aspect=math.exp(rng.uniform(0,math.log(16)))
        reserved=any(a<=aspect<=b for a,b in ASPECT_BANDS)
        if kind=='train' and reserved:continue
        if kind=='aspect' and not reserved:continue
        scale=10**rng.uniform(-1,1)
        if kind in ('scale','joint'):
            scale=10**rng.uniform(-6,-3) if rng.random()<.5 else 10**rng.uniform(3,6)
        if kind=='joint' and not reserved:continue
        length=np.array([math.sqrt(aspect),1/math.sqrt(aspect)])*scale
        if rng.random()<.5:length=length[::-1]
        if kind in ('budget','joint'):n=int(rng.choice(HELD_BUDGETS))
        else:
            weights=1/train_budget.astype(float);weights/=weights.sum()
            n=int(rng.choice(train_budget,p=weights))
        lengths.append(length);budgets.append(n)
    length=np.array(lengths);lo=rng.uniform(-2,2,size=length.shape)*np.sqrt(np.prod(length,axis=1))[:,None]
    return lo,lo+length,np.array(budgets,dtype=np.int64)


def evaluate(model,lo,hi,n,teacher,kind,rows):
    axis,cut=model.choose(torch.from_numpy(hi-lo),torch.from_numpy(n.astype(float)))
    ix=np.arange(len(n));score=teacher['scores'][ix,axis,cut]
    optimum=teacher['scores'].reshape(-1,14).min(1)
    regret=score/optimum-1
    valid=np.isfinite(score)
    agreement=teacher['equivalent'][ix,axis*7+cut]
    axis_equiv=teacher['equivalent'].reshape(-1,2,7).any(2)
    axis_ok=axis_equiv[ix,axis]
    err=abs(CUTS[cut]-CUTS[teacher['cut']])
    equiv_error=np.array([min(abs(CUTS[k]-CUTS[c]) for a in range(2) for k in range(7)
                              if teacher['equivalent'][i,a*7+k]) for i,c in enumerate(cut)])
    for i in range(len(n)):
        rows.append(dict(record_type='single_split',group=kind,case=i,n=int(n[i]),
            length_x=hi[i,0]-lo[i,0],length_y=hi[i,1]-lo[i,1],teacher_axis=int(teacher['axis'][i]),
            predicted_axis=int(axis[i]),teacher_cut=float(CUTS[teacher['cut'][i]]),predicted_cut=float(CUTS[cut[i]]),
            teacher_score=float(optimum[i]),predicted_score=float(score[i]),relative_regret=float(regret[i]),
            axis_correct=bool(axis_ok[i]),teacher_optimal=bool(agreement[i]),cut_error=float(err[i]),
            equivalent_cut_error=float(equiv_error[i]),child_counts=json.dumps(teacher['counts'][i,axis[i],cut[i]].tolist()),
            count_vs_volume_mismatch=bool(teacher['count_vs_volume_mismatch'][i,axis[i],cut[i]])))
    result=dict(group=kind,cases=len(n),axis_accuracy=float(axis_ok.mean()),
                canonical_axis_accuracy=float((axis==teacher['axis']).mean()),
                teacher_split_agreement=float(agreement.mean()),
                canonical_split_agreement=float(((axis==teacher['axis'])&(cut==teacher['cut'])).mean()),
                cut_mae=float(err.mean()),equivalent_cut_mae=float(equiv_error.mean()),cut_max=float(err.max()),
                mean_score_regret=float(regret.mean()),max_score_regret=float(regret.max()),invalid_splits=int((~valid).sum()),
                exact_count_conservation=bool(np.all(teacher['counts'][ix,axis,cut].sum(1)==n)))
    result['passed']=(result['axis_accuracy']>=GATES['axis_accuracy'] and result['teacher_split_agreement']>=GATES['split_agreement']
                      and result['equivalent_cut_mae']<=GATES['cut_mae'] and result['mean_score_regret']<=GATES['mean_score_regret']
                      and result['max_score_regret']<=GATES['max_score_regret'] and not result['invalid_splits'])
    return result


def recurse(model,allocator,n,extent,teacher_mode=False):
    active=[(np.zeros(2),np.array(extent,dtype=float),n,0)];leaves=[];trace=[]
    while active:
        lo,hi,budget,depth=active.pop()
        if budget==1:leaves.append((lo,hi,depth));continue
        assert depth<32
        if teacher_mode:
            lab=label_boxes(lo[None],hi[None],np.array([budget]),allocator)
            axis=int(lab['axis'][0]);ci=int(lab['cut'][0])
        else:
            ax,cu=model.choose(torch.from_numpy((hi-lo)[None]),torch.tensor([float(budget)],dtype=torch.float64))
            axis=int(ax[0]);ci=int(cu[0])
        cl,ch=children(lo,hi,axis,CUTS[ci])
        counts,probs,exact=allocator.allocate(((ch-cl)/(hi-lo))[None],np.array([budget]))
        if np.any(counts<1):raise RuntimeError(f'Zero-budget recursive child n={budget}, axis={axis}, cut={CUTS[ci]}')
        trace.append(dict(n=budget,depth=depth,axis=axis,cut=float(CUTS[ci]),counts=counts[0].tolist(),
                          count_vs_volume_mismatch=bool(np.any(counts!=exact))))
        for i in range(2):active.append((cl[i],ch[i],int(counts[0,i]),depth+1))
    assert len(leaves)==n
    lower=np.array([v[0] for v in leaves]);upper=np.array([v[1] for v in leaves])
    return lower,upper,trace


def recursive_metrics(lower,upper,n,extent,reference,mode):
    centres=(lower+upper)/2
    q=reference*np.array(extent)
    distances=cKDTree(centres).query(q)[0]
    lengths=upper-lower;aspect=lengths.max(1)/lengths.min(1)
    spacing=math.sqrt(np.prod(extent)/n)
    separation=cKDTree(centres).query(centres,k=2)[0][:,1]
    row=dict(mode=mode,n=n,extent=json.dumps(extent),leaves=len(lower),
             covering_radius=float(distances.max()),normalized_cover=float(distances.max()/spacing),
             large_hole_threshold=spacing,large_hole_count=int((distances>spacing).sum()),
             leaf_aspect_mean=float(aspect.mean()),leaf_aspect_p95=float(np.quantile(aspect,.95)),
             leaf_aspect_max=float(aspect.max()),min_leaf_volume=float(np.prod(lengths,axis=1).min()),
             total_leaf_volume=float(np.prod(lengths,axis=1).sum()),nearest_neighbor_cv=float(separation.std()/separation.mean()),
             duplicate_centres=n-len(np.unique(centres,axis=0)))
    normalized=centres/np.array(extent)
    for scale in (2,4,8,16):
        idx=np.minimum((normalized*scale).astype(int),scale-1)
        hist=np.bincount(idx[:,0]*scale+idx[:,1],minlength=scale**2)
        err=hist-n/scale**2
        row[f'count_rmse_{scale}']=float(np.sqrt(np.mean(err**2)))
        row[f'count_max_error_{scale}']=float(abs(err).max())
    return row,centres


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    allocator=LockedAllocation()
    config=dict(seed=SEED,cuts=CUTS.tolist(),allocator=str(ALLOC),allocator_sha256=ALLOC_SHA,gates=GATES,
        geometry_score='max_child(diameter/sqrt(child_count))/sqrt(parent_area/n) + 0.05*max_child(log(aspect))',
        invalid='zero-budget children get infinite teacher cost; no inference repair/fallback',
        quality_tie='within 1e-12*max(1,best_score), choose lowest axis then lowest cut',
        model='shared 5->32 tanh->32 tanh->8; axis logit + seven cut logits; float64 random initialization',
        normalization='parent side lengths divided by their geometric mean; absolute scale invariance is structural',
        optimizer='Adam lr=.003 batch512 max1200 updates; training-only best checkpoint',
        geometry_only=True,placement=False,search=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    lo,hi,n=data(4096,SEED,'train');label=label_boxes(lo,hi,n,allocator)
    np.savez_compressed(OUT/'training_partitions.npz',lower=lo,upper=hi,budget=n,**label)
    hist={str(q):int(np.sum(label['cut']==k)) for k,q in enumerate(CUTS)}
    print(json.dumps(dict(event='teacher_data',cut_distribution=hist,allocator=allocator.stats())),flush=True)
    model=Splitter();initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    length=torch.from_numpy(hi-lo);budget=torch.from_numpy(n.astype(float))
    axis_target=torch.from_numpy(label['axis']);cut_target=torch.from_numpy(label['cut'])
    rng=np.random.default_rng(SEED+1);best=float('inf');best_state=None;history=[]
    for step in range(1200):
        idx=torch.from_numpy(rng.integers(0,len(n),size=512))
        a,c=model(length[idx],budget[idx]);target_axis=axis_target[idx]
        loss=F.cross_entropy(a,target_axis)+F.cross_entropy(c[torch.arange(len(idx)),target_axis],cut_target[idx])
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%100==0:
            with torch.no_grad():
                a,c=model(length,budget)
                all_loss=float(F.cross_entropy(a,axis_target)+F.cross_entropy(c[torch.arange(len(n)),axis_target],cut_target))
                acc=float(((a.argmax(1)==axis_target)&(c[torch.arange(len(n)),axis_target].argmax(1)==cut_target)).double().mean())
            if all_loss<best:best=all_loss;best_state=copy.deepcopy(model.state_dict());best_step=step+1
            history.append(dict(step=step+1,loss=all_loss,training_split_accuracy=acc))
            print(json.dumps(dict(event='training',**history[-1])),flush=True)
            if acc>.998 and all_loss<.02:break
    model.load_state_dict(best_state)
    checkpoint=OUT/'splitter_best_before_evaluation.pt'
    checkpoint_hash=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),initial_state_dict=initial,
        config=config,updates=best_step,training_loss=best,status='DIAGNOSTIC_PENDING_GATE'))
    csv_write(OUT/'splitter_training_history.csv',history)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval()
    summaries=[];rows=[]
    for i,kind in enumerate(('aspect','budget','scale','joint')):
        vl,vh,vn=data(512,SEED+10+i,kind);teacher=label_boxes(vl,vh,vn,allocator)
        np.savez_compressed(OUT/f'heldout_{kind}.npz',lower=vl,upper=vh,budget=vn,**teacher)
        stats=evaluate(model,vl,vh,vn,teacher,kind,rows);summaries.append(stats)
        print(json.dumps(dict(event='heldout_gate',**stats)),flush=True)
    single_pass=all(r['passed'] for r in summaries)
    csv_write(OUT/'region_splitter_single_split.csv',rows)
    csv_write(OUT/'region_splitter_summary.csv',summaries)
    geometry=[];recursive_pass=False
    if single_pass:
        t=(np.arange(256)+.5)/256
        ref=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(SEED+99).random((65536,2)),
                       np.stack((np.linspace(0,1,1025),np.zeros(1025)),axis=1),
                       np.stack((np.linspace(0,1,1025),np.ones(1025)),axis=1),
                       np.stack((np.zeros(1025),np.linspace(0,1,1025)),axis=1),
                       np.stack((np.ones(1025),np.linspace(0,1,1025)),axis=1)))
        sets={};recursive_pass=True
        for n in (32,64,128,256):
            for extent in ([1.,1.],[2.,.5],[.5,2.]):
                pair=[]
                for mode in ('teacher','learned'):
                    lower,upper,trace=recurse(model,allocator,n,extent,teacher_mode=mode=='teacher')
                    r,centres=recursive_metrics(lower,upper,n,extent,ref,mode)
                    r['max_depth']=max(v['depth'] for v in trace)+1;geometry.append(r);pair.append(r)
                    key=f'{mode}_{n}_{extent[0]}'
                    sets[key+'_lower']=lower;sets[key+'_upper']=upper;sets[key+'_centres']=centres
                    (OUT/f'recursion_{key}.json').write_text(json.dumps(trace))
                healthy=(pair[1]['covering_radius']<=1.1*pair[0]['covering_radius'] and
                         pair[1]['leaf_aspect_max']<=8 and pair[1]['duplicate_centres']==0 and pair[1]['large_hole_count']==0)
                pair[1]['geometry_gate']=healthy;recursive_pass &=healthy
                print(json.dumps(dict(event='recursive',**pair[1])),flush=True)
                if not healthy:break
            if not recursive_pass:break
        np.savez_compressed(OUT/'recursive_leaf_regions.npz',**sets)
        csv_write(OUT/'region_splitter_recursive_geometry.csv',geometry)
    assert digest(ALLOC)==ALLOC_SHA
    result=dict(single_split_pass=single_pass,recursive_test_run=single_pass,recursive_pass=recursive_pass,
        status='FROZEN_CANDIDATE' if single_pass and recursive_pass else 'DIAGNOSTIC_ONLY',
        checkpoint=str(checkpoint),sha256=checkpoint_hash,allocator_unchanged=True,allocator=allocator.stats(),
        elapsed_seconds=time.monotonic()-start,teacher_cut_distribution=hist,best_training_step=best_step,
        actual_training_updates=step+1,leaf_centres='TEMPORARY DIAGNOSTIC SCAFFOLDING')
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if result['status']=='FROZEN_CANDIDATE':
        (OUT/'candidate_manifest.json').write_text(json.dumps(dict(recursive_region_splitter='FROZEN_CANDIDATE',**result),indent=2))
    print(json.dumps(dict(event='finished',**result)),flush=True)


if __name__=='__main__':run()
