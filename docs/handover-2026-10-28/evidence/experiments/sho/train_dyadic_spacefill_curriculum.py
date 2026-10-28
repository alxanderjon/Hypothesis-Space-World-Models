"""Gated hierarchical dyadic curriculum on unchanged progressive inference."""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator, boundary_faces
from experiments.sho.train_anchored_discrepancy_curriculum import train_points, soft_membership
from experiments.sho.train_progressive_spacefill_smoke import sha, write_csv
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics, overlap
from experiments.sho.smoke_artifacts import save_checkpoint

N=8600
OUT=Path('outputs/sho/stage1B/dyadic_spacefill_curriculum')
PARENT=Path('outputs/sho/stage1B/progressive_scale_curriculum/stage_1_8x8_best.pt')
MONITOR=[(9600,'cell'),(9601,'cell'),(9600,'greedy')]
HELDOUT=[(100+j,'cell') for j in range(4)]+[(100,'greedy')]
PHASES=[
    (1,'halves',[(2,1),(1,2)]),
    (2,'quarters',[(4,1),(1,4)]),
    (2,'eighths',[(8,1),(1,8)]),
    (3,'coarse_2d',[(2,2),(4,2),(2,4)]),
    (4,'fine_2d_a',[(8,2),(2,8),(4,4)]),
    (4,'fine_2d_b',[(8,4),(4,8)]),
    (4,'fine_2d_c',[(8,8),(16,4),(4,16)]),
    (5,'nested',[(16,8),(8,16),(16,16)])]
NESTED=[((8,4),(16,8)),((4,8),(8,16)),((8,8),(16,16))]
RESUME_STAGE1=None


def label(f):return f'{f[0]}x{f[1]}'


def hard_counts(p,f):
    nx,ny=f
    ix=np.minimum((p[:,0]*nx).astype(int),nx-1)
    iy=np.minimum((p[:,1]*ny).astype(int),ny-1)
    return np.bincount(ix*ny+iy,minlength=nx*ny).reshape(nx,ny)


def soft_hard_bins(coordinate,k,width):
    if k==1:return torch.ones((len(coordinate),1),dtype=torch.float64)
    anchors=torch.arange(1,k+1,dtype=torch.float64)/k
    cdf=soft_membership(coordinate,anchors,width)
    soft=torch.cat((cdf[:,:1],cdf[:,1:]-cdf[:,:-1]),dim=1)
    idx=torch.floor(coordinate.detach()*k).long().clamp(0,k-1)
    hard=torch.nn.functional.one_hot(idx,num_classes=k).to(torch.float64)
    return hard+(soft-soft.detach())


def loss_for(points,active,new,width,nested):
    cache={};hists={};terms={}
    for nx,ny in active:
        for axis,k in [(0,nx),(1,ny)]:
            if (axis,k) not in cache:cache[axis,k]=soft_hard_bins(points[:,axis],k,width)
        count=cache[0,nx].T@cache[1,ny];hists[nx,ny]=count
        residual=count-N/(nx*ny)
        # Realized count error in the forward pass, normalized for cell area.
        terms[nx,ny]=(residual.square().mean()/(N/(nx*ny)))
    newest=torch.stack([terms[f] for f in new])
    prior=[terms[f] for f in active if f not in new]
    primary=newest.mean()+(.5*torch.stack(prior).mean() if prior else 0.)
    conditional=[]
    if nested:
        for parent,child in NESTED:
            px,py=parent;cx,cy=child
            children=hists[child].reshape(px,cx//px,py,cy//py).permute(0,2,1,3)
            parent_counts=children.sum((2,3),keepdim=True)
            conditional.append(((children-parent_counts/((cx//px)*(cy//py)))**2).mean()/(N/(cx*cy)))
        primary=primary+torch.stack(conditional).mean()
    return primary,terms,conditional


def imbalance(p,f):
    count=hard_counts(p,f);error=count-N/(f[0]*f[1])
    return dict(max_count=float(np.abs(error).max()),rms_count=float(np.sqrt(np.mean(error**2))),
        max_fraction=float(np.abs(error).max()/N),rms_fraction=float(np.sqrt(np.mean(error**2))/N),
        min_count=int(count.min()),max_observed_count=int(count.max()))


def family_stats(pools,families):
    result={}
    for f in families:
        v=[imbalance(p,f) for p in pools]
        result[label(f)]=dict(max_count=max(x['max_count'] for x in v),
            rms_count=float(np.sqrt(np.mean([x['rms_count']**2 for x in v]))),
            max_fraction=max(x['max_fraction'] for x in v),
            rms_fraction=float(np.sqrt(np.mean([x['rms_fraction']**2 for x in v]))))
    return result


def evaluate(model,conditions,active,reference,full=False):
    pools=[];geometry=[]
    with torch.no_grad():
        for iteration,pool in conditions:
            complete,p=model.generate(N,iteration,pool)
            assert len(p)==N and len(complete)==N+len(boundary_faces(N,2))
            assert np.array_equal(complete[np.any((complete==0)|(complete==1),axis=1)],boundary_faces(N,2))
            tree=cKDTree(p);nn=tree.query(p,k=2)[0][:,1]
            d=cKDTree(complete).query(reference)[0]
            rng=np.random.default_rng(27132027)
            a=rng.integers(0,N,131072);b=rng.integers(0,N-1,131072);b=b+(b>=a)
            pairwise=np.linalg.norm(p[a]-p[b],axis=1)
            row=dict(cover=float(d.max()),holes=int(np.sum(d>1/math.sqrt(N))),
                hole_fraction=float(np.mean(d>1/math.sqrt(N))),cv=float(nn.std()/nn.mean()),
                nn_min=float(nn.min()),nn_p01=float(np.quantile(nn,.01)),nn_p05=float(np.quantile(nn,.05)),
                pairwise_p01=float(np.quantile(pairwise,.01)),pairwise_p05=float(np.quantile(pairwise,.05)),
                cluster=float(np.mean(nn<.1/math.sqrt(N))))
            if full:row['boundary_cover']=metrics(complete,p,reference,N)['boundary_cover']
            geometry.append(row);pools.append(p)
    pairs=[(0,1),(0,2)] if len(pools)==3 else [(0,1),(1,2),(2,3),(0,4)]
    ov=[overlap(pools[i],pools[j],N) for i,j in pairs]
    result=dict(families=family_stats(pools,active),cover=max(v['cover'] for v in geometry),
        holes=max(v['holes'] for v in geometry),hole_fraction=max(v['hole_fraction'] for v in geometry),
        nn_CV=float(np.mean([v['cv'] for v in geometry])),min_nn_CV=min(v['cv'] for v in geometry),
        nn_min=min(v['nn_min'] for v in geometry),nn_p01=min(v['nn_p01'] for v in geometry),nn_p05=min(v['nn_p05'] for v in geometry),
        pairwise_p01=min(v['pairwise_p01'] for v in geometry),pairwise_p05=min(v['pairwise_p05'] for v in geometry),
        clustering=max(v['cluster'] for v in geometry),exact_overlap=max(v['exact_overlap_fraction'] for v in ov),
        near_overlap=max(v['near_overlap_fraction_005_spacing'] for v in ov),near_overlap_010=max(v['near_overlap_fraction_010_spacing'] for v in ov),
        iteration_near_overlap=max(v['near_overlap_fraction_005_spacing'] for v in ov[:-1]),
        cell_greedy_near_overlap=ov[-1]['near_overlap_fraction_005_spacing'])
    if full:result['boundary_cover']=max(v['boundary_cover'] for v in geometry)
    if all(c in active for _,c in NESTED):
        nested=[]
        for parent,child in NESTED:
            px,py=parent;cx,cy=child;errors=[]
            for p in pools:
                children=hard_counts(p,child).reshape(px,cx//px,py,cy//py).transpose(0,2,1,3)
                assert np.array_equal(children.sum((2,3)),hard_counts(p,parent))
                error=children-children.sum((2,3),keepdims=True)/((cx//px)*(cy//py))
                errors.append(error)
            nested.append(dict(parent=label(parent),child=label(child),max_child_imbalance=float(np.max(np.abs(errors))),
                rms_child_imbalance=float(np.sqrt(np.mean(np.square(errors)))),partition_identity_exact=True))
        result['nested']=nested
    return result,pools


def health(value,entry):
    return (value['cover']<=1.05*entry['cover'] and value['hole_fraction']<=entry['hole_fraction']+.0005
        and value['min_nn_CV']>=.08 and value['clustering']<=.005 and value['nn_p01']>=.95*entry['nn_p01']
        and value['exact_overlap']==0 and value['near_overlap']<=min(.03,entry['near_overlap']+.005))


def retained(value,mastered):
    return all(value['families'][f]['max_count']<=max(v['max_count']*1.10,v['max_count']+1)
        and value['families'][f]['rms_count']<=v['rms_count']*1.10+.25 for f,v in mastered.items())


def new_score(value,entry,new):
    return max(value['families'][label(f)][metric]/max(entry['families'][label(f)][metric],.5)
        for f in new for metric in ('max_count','rms_count'))


def low_halves(value,stage):
    return stage!=1 or all(value['families'][f]['max_count']<=8 for f in ('2x1','1x2'))


def baseline_models():
    from experiments.sho.fresh_spacefill_generator import FreshGenerator
    from experiments.sho import run_stage1b_fifo_vs_ranker as src
    fresh=FreshGenerator(2);fresh.load_state_dict(torch.load('outputs/sho/stage1B/fresh_spacefill_smoke/fresh_spacefill_trained_pending_evaluation.pt',weights_only=False)['state_dict'])
    previous=ProgressiveGenerator();previous.load_state_dict(torch.load('outputs/sho/stage1B/progressive_spacefill_smoke/progressive_best_before_evaluation.pt',weights_only=False)['state_dict'])
    class Sobol:
        def generate(self,n,it,pool):
            p=src.edirect._candidate_pool(86,200136,it+(1000000 if pool=='greedy' else 0))
            return p,p[~np.any((p==0)|(p==1),axis=1)]
    return {'Sobol':Sobol(),'fresh_v1':fresh,'previous_progressive':previous}


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(27132026)
    OUT.mkdir(parents=True,exist_ok=False)
    previous=json.loads(Path('outputs/sho/stage1B/anchored_discrepancy_curriculum/results.json').read_text())
    protected=dict(previous['protected_hashes'])
    protected.update({str(p):sha(p) for p in Path('outputs/sho/stage1B/anchored_discrepancy_curriculum').iterdir() if p.is_file()})
    for path in ['experiments/sho/train_anchored_discrepancy_curriculum.py','experiments/sho/progressive_spacefill_generator.py']:
        protected[path]=sha(path)
    config=dict(N=N,phases=PHASES,parent=str(PARENT),parent_sha256=sha(PARENT),optimizer='Adam',learning_rate=.0004,
        batch_contexts=2,monitor_every=8,plateau_monitors=12,
        convergence='continue while feasible actual-set max/RMS improves; no fixed short update/time cap',
        half_absolute_gate='<=8 points count imbalance in every half on both axes (<0.1% of N)',
        monitor_target=.80,heldout_target=.85,mastery_reference='first passed checkpoint for each family',
        retention='max <=max(1.10*mastery,mastery+1 point); RMS<=1.10*mastery+0.25 point',
        objective='hard-forward dyadic count squared error/expected cell count; new-family mean +0.5 prior-family mean; nested conditional balance added only in Stage5',
        gradient='smooth interval memberships + existing coordinate straight-through relaxation; no changed inference',
        input_features='all existing multiscale descriptors retained; objectives introduced only in listed phases',
        widths='0.12 / maximum active bins along either axis',
        safeguards='cover<=1.05 entry; holes fraction<=entry+.0005; NN CV>=.08; cluster fraction<=.005; NN p01>=.95 entry; exact overlap0; near<=min(.03,entry+.005)',
        monitor_conditions=MONITOR,heldout_conditions=HELDOUT,no_search=True,no_objective_labels=True,no_Sobol_supervision=True)
    if RESUME_STAGE1 is not None:
        config['resume_stage1_checkpoint']=str(RESUME_STAGE1)
        config['resume_stage1_sha256']=sha(RESUME_STAGE1)
        config['resume_rule']='same Stage1, saved weights/optimizer, original entry references; continue until full actual-set gate or plateau'
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    model=ProgressiveGenerator();model.load_state_dict(torch.load(PARENT,weights_only=False)['state_dict'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    # Verify unchanged hard generation AND realized (not expected) loss counts.
    points,hard=train_points(model,16000,'cell')
    with torch.no_grad():original=model.generate(N,16000,'cell')[1]
    assert np.array_equal(original,hard) and np.array_equal(original,points.detach().numpy())
    objective,terms,_=loss_for(points,[(2,1),(1,2)],[(2,1),(1,2)],.06,False)
    for f,value in terms.items():
        h=hard_counts(hard,f);expected=np.mean((h-N/(f[0]*f[1]))**2)/(N/(f[0]*f[1]))
        assert abs(float(value.detach())-expected)<1e-12
    objective.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
    assert math.isfinite(grad) and grad>0;optimizer.zero_grad();del points,objective,terms
    (OUT/'preflight.json').write_text(json.dumps(dict(hard_forward_bitwise_equal=True,realized_count_loss_verified=True,gradient_norm=grad),indent=2)+'\n')
    monitor_ref=np.random.default_rng(27132028).random((32768,2))
    t=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    parent=PARENT;active=[];mastered_monitor={};mastered_heldout={};history=[];rows=[];phases=[];baseline_cache={}
    for phase_id,(stage,name,new) in enumerate(PHASES,1):
        phase_start=time.monotonic();active+=new;width=.12/max(max(f) for f in active)
        persisted=torch.load(parent,weights_only=False);model.load_state_dict(persisted['state_dict'])
        if 'optimizer_state_dict' in persisted:optimizer.load_state_dict(persisted['optimizer_state_dict'])
        entry_monitor,_=evaluate(model,MONITOR,active,monitor_ref)
        entry,_=evaluate(model,HELDOUT,active,reference,True)
        best_state=copy.deepcopy(model.state_dict());best_opt=copy.deepcopy(optimizer.state_dict());best_score=1.;best_step=0;best_monitor=entry_monitor
        stale=0;step=0;reason=None
        if phase_id==1 and RESUME_STAGE1 is not None:
            resumed=torch.load(RESUME_STAGE1,weights_only=False)
            model.load_state_dict(resumed['state_dict']);optimizer.load_state_dict(resumed['optimizer_state_dict'])
            best_monitor,_=evaluate(model,MONITOR,active,monitor_ref)
            best_score=new_score(best_monitor,entry_monitor,new)
            best_state=copy.deepcopy(model.state_dict());best_opt=copy.deepcopy(optimizer.state_dict())
            step=best_step=int(resumed['best_step'])
        checked_step=-1
        print(json.dumps(dict(event='phase_start',stage=stage,phase=name,active=active,new=new,entry_monitor=entry_monitor)),flush=True)
        while True:
            step+=1;optimizer.zero_grad();batch_losses=[]
            for j in range(config['batch_contexts']):
                p,_=train_points(model,17000+phase_id*100000+step*2+j,'cell' if j==0 else 'greedy')
                loss,terms,conditional=loss_for(p,active,new,width,stage==5)
                (loss/config['batch_contexts']).backward();batch_losses.append(float(loss.detach()))
                del p,loss,terms,conditional
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm)
            optimizer.step()
            history.append(dict(stage=stage,phase=name,step=step,active_families=json.dumps(active),width=width,
                realized_dyadic_loss=float(np.mean(batch_losses)),gradient_norm=norm))
            if step%config['monitor_every']:continue
            measured,_=evaluate(model,MONITOR,active,monitor_ref)
            value=new_score(measured,entry_monitor,new)
            feasible=health(measured,entry_monitor) and retained(measured,mastered_monitor)
            if feasible and value<best_score*(1-1e-3):
                best_score=value;best_step=step;best_state=copy.deepcopy(model.state_dict());best_opt=copy.deepcopy(optimizer.state_dict());best_monitor=measured;stale=0
            else:stale+=1
            print(json.dumps(dict(event='monitor',stage=stage,phase=name,step=step,hard_score=value,best_score=best_score,
                feasible=feasible,stale=stale,new_metrics={label(f):measured['families'][label(f)] for f in new},
                seconds=time.monotonic()-phase_start)),flush=True)
            if not retained(measured,mastered_monitor):reason='mastered family materially regressed';break
            if best_score<=config['monitor_target'] and low_halves(best_monitor,stage) and checked_step!=best_step:
                # A promising small monitor is insufficient. Continue THIS
                # phase if the independent actual-set gate still fails; never
                # use those coordinates as training targets.
                current_state=copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state)
                gate_value,_=evaluate(model,HELDOUT,active,reference,True)
                model.load_state_dict(current_state);checked_step=best_step
                full_ok=(new_score(gate_value,entry,new)<=config['heldout_target'] and low_halves(gate_value,stage)
                    and retained(gate_value,mastered_heldout) and health(gate_value,entry))
                print(json.dumps(dict(event='convergence_gate',phase=name,best_step=best_step,passed=full_ok,
                    families=gate_value['families'])),flush=True)
                if full_ok:reason='monitor target reached';break
            if stale>=config['plateau_monitors']:reason='realized dyadic plateau';break
        model.load_state_dict(best_state);optimizer.load_state_dict(best_opt)
        checkpoint=OUT/f'phase_{phase_id}_{name}_best.pt'
        digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            stage=stage,phase=name,active_families=active,best_step=best_step,parent=str(parent),parent_sha256=sha(parent),
            config=config,status='PENDING_HARD_GATE',generator_source_sha256=sha('experiments/sho/progressive_spacefill_generator.py')))
        print(json.dumps(dict(event='saved',phase=name,reason=reason,checkpoint=str(checkpoint),sha256=digest)),flush=True)
        result,pools=evaluate(model,HELDOUT,active,reference,True)
        gates=dict(training_target=reason=='monitor target reached',new_families=new_score(result,entry,new)<=config['heldout_target'],
            low_halves=low_halves(result,stage),retention=retained(result,mastered_heldout),geometry=health(result,entry),boundary=result['boundary_cover']==entry['boundary_cover'])
        if stage==5:
            gates['nested']=all(v['partition_identity_exact'] and v['rms_child_imbalance']<=.85*e['rms_child_imbalance']
                for v,e in zip(result['nested'],entry['nested']))
        passed=all(gates.values())
        np.savez_compressed(OUT/f'phase_{phase_id}_{name}_heldout_interiors.npz',**{f'iteration{it}_{pool}':v for (it,pool),v in zip(HELDOUT,pools)})
        # External baselines only after training and checkpoint persistence.
        for baseline_name,baseline_model in baseline_models().items():
            if baseline_name not in baseline_cache:
                bv,bp=evaluate(baseline_model,HELDOUT,active,reference,True);baseline_cache[baseline_name]=(bv,bp)
            bv,bp=baseline_cache[baseline_name];bv['families']=family_stats(bp,active)
        item=dict(stage=stage,phase=name,passed=passed,gates=gates,reason=reason,best_step=best_step,updates=step,
            checkpoint=str(checkpoint),sha256=digest,active=active,new=new,entry=entry,result=result,
            seconds=time.monotonic()-phase_start,baselines={n:v[0] for n,v in baseline_cache.items()})
        phases.append(copy.deepcopy(item))
        for generator,v in [('entry',entry),('dyadic',result)]+[(n,v[0]) for n,v in baseline_cache.items()]:
            for f in active:
                r={key:value for key,value in v.items() if key not in ('families','nested')}
                rows.append(dict(stage=stage,phase=name,status='PASSED' if passed else 'STOPPED',N=N,generator=generator,
                    family=label(f),family_role='new' if f in new else 'retained',**v['families'][label(f)],**r,
                    mastery_reference=json.dumps(mastered_heldout.get(label(f))),reason=reason,best_step=best_step,updates=step,
                    checkpoint=str(checkpoint),sha256=digest))
        write_csv(OUT/'dyadic_curriculum_summary.csv',rows);write_csv(OUT/'training_history.csv',history)
        (OUT/f'phase_{phase_id}_{name}_gate.json').write_text(json.dumps(item,indent=2)+'\n')
        print(json.dumps(dict(event='phase_gate',stage=stage,phase=name,passed=passed,gates=gates,result=result)),flush=True)
        if not passed:
            for s,n,_ in PHASES[phase_id:]:rows.append(dict(stage=s,phase=n,N=N,status='NOT_RUN',reason=f'{name} failed'))
            break
        for f in new:
            mastered_monitor[label(f)]=best_monitor['families'][label(f)];mastered_heldout[label(f)]=result['families'][label(f)]
        assert sha(checkpoint)==digest;parent=checkpoint
    write_csv(OUT/'dyadic_curriculum_summary.csv',rows)
    unchanged=all(sha(p)==h for p,h in protected.items());assert unchanged
    final=dict(status='HIERARCHICAL_CURRICULUM_PASSED' if len(phases)==len(PHASES) and phases[-1]['passed'] else 'STOPPED_DIAGNOSTIC_ONLY',
        phases=phases,elapsed_seconds=time.monotonic()-started,protected_hashes=protected,protected_unchanged=unchanged,
        no_search=True,no_Sobol_supervision=True,no_SHO_labels_or_parameters=True,architecture_unchanged=True)
    (OUT/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    lines=['HIERARCHICAL DYADIC CURRICULUM — '+final['status'],f'N=8600 only; run time {final["elapsed_seconds"]:.1f} seconds.',
        'Existing float64 progressive architecture and inference unchanged. Warm start from last passed 8x8 occupancy checkpoint, not failed anchored weights.',
        'Loss uses actual hard-set dyadic squared count imbalance normalized by expected cell count; smooth interval/coordinate straight-through derivatives. This is a biased gradient surrogate, not differentiation through exact discrete selection.',
        'Only x/y halves trained initially. Quarters and eighths are separate gated phases. Every new 2D family retains all mastered 1D/2D losses. Conditional parent/child refinement occurs only in Stage5.',
        'Hard intervals are half-open except the final interval includes1; uniform targets are N times rectangle area. No coordinate imitation, search labels, objectives or parameters.',
        'Monitor requires20% improvement in BOTH max/RMS for EACH new family. Heldout requires15%; halves additionally require<=8 points worst imbalance per axis (<0.1% of totalN).',
        'Mastery retention compares against the first passed checkpoint per family: max increase<=max(10%,1point); RMS<=10%+.25point. Convergence continues while feasible hard metrics improve, stopping after12 stale monitors (96updates).',
        'Geometry gates preserve entry cover within5%, hole fraction within+.0005, NN CV>=.08, NN p01>=95% entry, no clustering, zero exact overlap, near overlap<=3% and increase<=.5percentage point.',
        'Discrepancy uses interior points; fixed generic codimension-one boundary support unchanged. Cover measured on524288 independent reference points; hole threshold1/sqrt(N).',
        'NN p01/p05 quantify local separation; additional p01/p05 from131072 deterministic nonself point pairs quantify sampled pairwise separation. Freshness excludes fixed boundaries; near thresholds .05/sqrt(N),.10/sqrt(N).',
        'Sobol, fresh-v1 and original progressive compared only after checkpoint persistence; baseline coordinates never used as targets.',
        'Best checkpoints saved before heldout gates, including failed diagnostics. Next phases reload only passed checkpoints and matching optimizer states. No search.','']
    for item in phases:
        lines += [f'Stage{item["stage"]} / {item["phase"]}: {"PASS" if item["passed"] else "STOP"}; {item["reason"]}; best update{item["best_step"]}/{item["updates"]}.',
            'Gates: '+json.dumps(item['gates']),'Hard family metrics: '+json.dumps(item['result']['families']),
            f'Checkpoint: {item["checkpoint"]}',f'SHA256: {item["sha256"]}','']
    (OUT/'dyadic_curriculum_report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in final.items() if k not in ('phases','protected_hashes')}),flush=True)


if __name__=='__main__':run()
