"""Soft anchored-CDF curriculum, exact unchanged hard inference.

Training-only coordinate straight-through relaxation supplies gradients through
the existing discrete placement policy. No histogram/count objective is used.
"""
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator, boundary_faces, SEED
from experiments.sho.train_progressive_spacefill_smoke import sha, write_csv
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics, overlap
from experiments.sho.smoke_artifacts import save_checkpoint

N=8600
SCALES=(4,8,16,32,64)
OUT=Path('outputs/sho/stage1B/anchored_discrepancy_curriculum')
PARENT=Path('outputs/sho/stage1B/progressive_scale_curriculum/stage_1_8x8_best.pt')
MONITOR=[(9400,'cell'),(9401,'cell'),(9400,'greedy')]
HELDOUT=[(100+j,'cell') for j in range(4)]+[(100,'greedy')]


def soft_membership(coordinate, anchors, width):
    # C1 compact smooth step. Its uniform integral equals anchor location for
    # these anchors/widths. Full-domain faces have exact membership 1.
    z=((anchors[None,:]-coordinate[:,None])/width+.5).clamp(0.,1.)
    membership=z*z*(3.-2.*z)
    return torch.where(anchors[None,:]==1.,torch.ones_like(membership),membership)


def soft_loss(points, scales, width):
    k=scales[-1]
    a=torch.arange(1,k+1,dtype=torch.float64)/k
    x=soft_membership(points[:,0],a,width)
    y=soft_membership(points[:,1],a,width)
    discrepancy=x.T@y/len(points)-a[:,None]*a[None,:]
    terms=[]
    for s in scales:
        stride=k//s
        terms.append(discrepancy[stride-1::stride,stride-1::stride].square().mean())
    # Only anchored-rectangle squared discrepancies. No count/occupancy loss.
    return terms[-1]+(torch.stack(terms[:-1]).mean() if len(terms)>1 else 0.),terms


def train_points(model, iteration, pool):
    """Forward identical to generate(); backward local soft coordinate choices.

    Hard prefix features are detached (no gradient through occupancy/KD-tree).
    Inference architecture, proposals, Gumbels and separation remain unchanged.
    """
    key=f'{SEED}:2:{N}:{iteration}:{pool}'.encode()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
    prefix=np.empty((0,2)); differentiable=[]
    while len(prefix)<N:
        b=min(128,N-len(prefix))
        candidates=rng.random((max(256,8*b),2))
        features,_,nearest=model.make_context(prefix,candidates,N,iteration,pool)
        logits=model.net(features).flatten().clamp(-8,8)
        valid=nearest>=.4/math.sqrt(N)
        logits=logits.masked_fill(torch.from_numpy(~valid),-100)
        gumbel=rng.gumbel(size=len(candidates))
        score=logits.detach().numpy()+.7*gumbel
        accepted=[]
        for j in np.argsort(-score,kind='stable'):
            if not valid[j]:continue
            if accepted and np.min(np.sum((candidates[accepted]-candidates[j])**2,axis=1))<.16/N:continue
            accepted.append(j)
            if len(accepted)==b:break
        if not accepted:raise RuntimeError('empty admissible block')
        hard=candidates[accepted]
        # Local differentiable barycentres are used ONLY for a surrogate
        # derivative. The forward values remain the exact selected coordinates.
        distance,index=cKDTree(candidates).query(hard,k=16)
        index=torch.from_numpy(index)
        local=torch.from_numpy(candidates)[index]
        bandwidth=1/math.sqrt(len(candidates))
        weights=torch.softmax(logits[index]/.7-torch.from_numpy(distance**2)/(2*bandwidth**2),dim=1)
        relaxed=(weights[:,:,None]*local).sum(1)
        straight_through=torch.from_numpy(hard)+(relaxed-relaxed.detach())
        differentiable.append(straight_through)
        prefix=np.vstack((prefix,hard))
    return torch.cat(differentiable),prefix


def hard_metrics(points, scales):
    result={}
    for k in scales:
        a=np.arange(1,k+1)/k
        # Exact inclusive rectangles [0,x] x [0,y], no shifting or tolerance.
        x=(points[:,0,None]<=a[None,:]).astype(np.float64)
        y=(points[:,1,None]<=a[None,:]).astype(np.float64)
        residual=x.T@y/len(points)-a[:,None]*a[None,:]
        result[k]=dict(max=float(np.abs(residual).max()),rms=float(np.sqrt(np.mean(residual**2))))
    return result


def evaluate(model, conditions, scales, reference, width=None, full=False):
    values=[]; complete_values=[]; geometric=[]; interiors=[]; soft=[]
    with torch.no_grad():
        for iteration,pool in conditions:
            points,interior=model.generate(N,iteration,pool)
            assert len(interior)==N
            assert np.array_equal(points[np.any((points==0)|(points==1),axis=1)],boundary_faces(N,2))
            values.append(hard_metrics(interior,scales));complete_values.append(hard_metrics(points,scales))
            interiors.append(interior)
            if width is not None:soft.append(float(soft_loss(torch.from_numpy(interior),scales,width)[0]))
            if full:
                v=metrics(points,interior,reference,N)
                geometric.append(dict(cover=v['complete_cover'],holes=v['large_hole_count'],hole_fraction=v['large_hole_fraction'],
                    cv=v['nn_CV'],cluster=v['clustered_fraction'],boundary=v['boundary_cover']))
            else:
                d=cKDTree(points).query(reference)[0]
                nn=cKDTree(interior).query(interior,k=2)[0][:,1]
                geometric.append(dict(cover=float(d.max()),holes=int(np.sum(d>1/math.sqrt(N))),
                    hole_fraction=float(np.mean(d>1/math.sqrt(N))),cv=float(nn.std()/nn.mean()),
                    cluster=float(np.mean(nn<.1/math.sqrt(N)))))
    def aggregate(v):
        return {k:dict(max=max(x[k]['max'] for x in v),rms=float(np.sqrt(np.mean([x[k]['rms']**2 for x in v])))) for k in scales}
    pairs=[(0,1),(0,len(interiors)-1)] if len(interiors)==3 else [(0,1),(1,2),(2,3),(0,4)]
    overlaps=[overlap(interiors[i],interiors[j],N) for i,j in pairs]
    result=dict(scales=aggregate(values),complete_scales=aggregate(complete_values),
        cover=max(g['cover'] for g in geometric),holes=max(g['holes'] for g in geometric),
        hole_fraction=max(g['hole_fraction'] for g in geometric),nn_CV=float(np.mean([g['cv'] for g in geometric])),
        min_nn_CV=min(g['cv'] for g in geometric),cluster=max(g['cluster'] for g in geometric),
        exact_overlap=max(v['exact_overlap_fraction'] for v in overlaps),
        near_overlap=max(v['near_overlap_fraction_005_spacing'] for v in overlaps),
        near_overlap_010=max(v['near_overlap_fraction_010_spacing'] for v in overlaps),
        cell_greedy_near_overlap=overlaps[-1]['near_overlap_fraction_005_spacing'],
        soft_mse=float(np.mean(soft)) if soft else None)
    if full:result['boundary_cover']=max(g['boundary'] for g in geometric)
    return result,interiors


def healthy(value, entry):
    return (value['cover']<=1.05*entry['cover'] and value['hole_fraction']<=entry['hole_fraction']+.0005
        and value['min_nn_CV']>=.08 and value['cluster']<=.005 and value['exact_overlap']==0
        and value['near_overlap']<=min(.03,entry['near_overlap']+.005))


def retained(value, mastered):
    return all(value['scales'][k]['max']<=1.05*v['max']+1e-15 and value['scales'][k]['rms']<=1.05*v['rms']+1e-15
               for k,v in mastered.items())


def score(value, entry, k):
    return max(value['scales'][k][metric]/entry['scales'][k][metric] for metric in ('max','rms'))


def sobol_baseline(scales):
    # Evaluation only; called AFTER the stage checkpoint has been persisted.
    from experiments.sho import run_stage1b_fifo_vs_ranker as source
    v=[];full=[]
    for it,pool in HELDOUT:
        points=source.edirect._candidate_pool(86,200136,it+(1000000 if pool=='greedy' else 0))
        interior=points[~np.any((points==0)|(points==1),axis=1)]
        assert len(interior)==N
        v.append(hard_metrics(interior,scales));full.append(hard_metrics(points,scales))
    return {k:dict(max=max(r[k]['max'] for r in v),rms=float(np.sqrt(np.mean([r[k]['rms']**2 for r in v]))),
                   complete_max=max(r[k]['max'] for r in full),complete_rms=float(np.sqrt(np.mean([r[k]['rms']**2 for r in full])))) for k in scales}


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(27122026)
    OUT.mkdir(parents=True,exist_ok=False)
    previous=json.loads(Path('outputs/sho/stage1B/progressive_scale_curriculum/scale_curriculum_results.json').read_text())
    protected=dict(previous['protected_hashes'])
    protected.update({str(p):sha(p) for p in PARENT.parent.iterdir() if p.is_file()})
    config=dict(N=N,stages=SCALES,initial_checkpoint=str(PARENT),initial_sha256=sha(PARENT),
        architecture='existing float64 34->32 tanh->1; original hard generate() unchanged',
        objective='squared soft anchored-rectangle CDF error vs x*y; current scale + mean of previous scales',
        membership='C1 compact smoothstep((anchor-coordinate)/width+0.5); exact 1 at anchor=1',
        widths=[.1,.05,.025,.0125,.00625],anchors='Cartesian {1/k,...,1}^2, nested, inclusive; no shifts',
        gradient='straight-through local 16-proposal barycentres; forward exact hard coordinates; detached prefix features',
        discrepancy_pool='8600 interiors; complete-pool hard CDF also reported; fixed faces evaluated separately',
        optimizer='Adam',learning_rate=.0005,loss_multiplier=N*N,max_updates_per_stage=100,
        monitor_every=4,plateau_monitors=8,monitor_target=.85,heldout_target=.90,previous_scale_max_ratio=1.05,
        candidate_gate='passed final reached stage; both max/RMS excess gap to Sobol reduced >=25% at every mastered scale',
        training_conditions='iteration 12000+1000*stage+step, alternating cell/greedy',monitor_conditions=MONITOR,heldout_conditions=HELDOUT,
        no_occupancy_objective=True,no_Sobol_supervision=True,no_search_labels=True)
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    model=ProgressiveGenerator();model.load_state_dict(torch.load(PARENT,weights_only=False)['state_dict'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    # Verify exact hard forward, valid coordinate gradients, and rectangle area.
    differentiable,hard=train_points(model,11500,'cell')
    with torch.no_grad():native=model.generate(N,11500,'cell')[1]
    assert np.array_equal(hard,native) and np.array_equal(differentiable.detach().numpy(),native)
    loss,_=soft_loss(differentiable,(4,),.1); loss.backward()
    grad_norm=float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
    assert math.isfinite(grad_norm) and grad_norm>0
    optimizer.zero_grad();del differentiable,loss
    probe=torch.tensor([[.24,.31]],dtype=torch.float64,requires_grad=True)
    m=soft_membership(probe[:,0],torch.tensor([.25],dtype=torch.float64),.1).sum();m.backward()
    assert probe.grad[0,0]!=0
    (OUT/'preflight.json').write_text(json.dumps(dict(hard_forward_bitwise_equal=True,finite_nonzero_weight_gradient=grad_norm,
        smooth_membership_coordinate_gradient=float(probe.grad[0,0]),architecture_parameter_count=sum(p.numel() for p in model.parameters())),indent=2)+'\n')
    monitor_ref=np.random.default_rng(27122027).random((32768,2))
    t=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    summary_rows=[];history=[];stages=[];mastered_monitor={};mastered_holdout={};initial_holdout={};parent=PARENT
    for stage,k in enumerate(SCALES,1):
        stage_start=time.monotonic();active=SCALES[:stage];width=config['widths'][stage-1]
        persisted=torch.load(parent,weights_only=False);model.load_state_dict(persisted['state_dict'])
        if 'optimizer_state_dict' in persisted:optimizer.load_state_dict(persisted['optimizer_state_dict'])
        entry_monitor,_=evaluate(model,MONITOR,active,monitor_ref,width)
        entry_holdout,_=evaluate(model,HELDOUT,active,reference,width,True)
        # Original entry metrics for final excess-gap comparison.
        original=ProgressiveGenerator();original.load_state_dict(torch.load(PARENT,weights_only=False)['state_dict'])
        if k not in initial_holdout:
            before,_=evaluate(original,HELDOUT,(k,),reference,width,True);initial_holdout[k]=before['scales'][k]
        best_state=copy.deepcopy(model.state_dict());best_opt=copy.deepcopy(optimizer.state_dict())
        best_monitor=entry_monitor;best_score=1.;best_step=0;stale=0;reason='update cap before target'
        print(json.dumps(dict(event='stage_start',stage=stage,anchors=k,width=width,entry=entry_monitor)),flush=True)
        for step in range(1,config['max_updates_per_stage']+1):
            model.train();optimizer.zero_grad()
            points,_=train_points(model,12000+1000*stage+step,'cell' if step%2 else 'greedy')
            raw,terms=soft_loss(points,active,width)
            (N*N*raw).backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            assert math.isfinite(norm) and torch.isfinite(raw)
            optimizer.step()
            history.append(dict(stage=stage,scale=k,width=width,step=step,soft_mse=float(raw.detach()),gradient_norm=norm,
                per_scale_soft_mse=json.dumps([float(v.detach()) for v in terms])))
            del points,raw,terms
            if step%config['monitor_every']:continue
            measured,_=evaluate(model,MONITOR,active,monitor_ref,width)
            value=score(measured,entry_monitor,k)
            feasible=healthy(measured,entry_monitor) and retained(measured,mastered_monitor)
            # Model selection is hard-set based; soft loss alone never passes.
            if feasible and value<best_score*(1-1e-3):
                best_score=value;best_state=copy.deepcopy(model.state_dict());best_opt=copy.deepcopy(optimizer.state_dict())
                best_step=step;best_monitor=measured;stale=0
            else:stale+=1
            print(json.dumps(dict(event='monitor',stage=stage,step=step,hard_score=value,best_score=best_score,
                soft_mse=measured['soft_mse'],feasible=feasible,stale=stale,seconds=time.monotonic()-stage_start)),flush=True)
            if not retained(measured,mastered_monitor):
                reason='mastered anchors materially regressed';break
            if best_score<=config['monitor_target']:
                reason='hard monitor target reached';break
            if stale>=config['plateau_monitors']:
                reason='hard anchored-discrepancy plateau';break
        model.load_state_dict(best_state);optimizer.load_state_dict(best_opt)
        # Persist before heldout evaluation, including failed diagnostics.
        checkpoint=OUT/f'stage_{stage}_{k}x{k}_best.pt'
        digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            stage=stage,active_scales=active,width=width,best_step=best_step,parent=str(parent),parent_sha256=sha(parent),
            config=config,status='PENDING_HARD_GATE',source_sha256=sha('experiments/sho/progressive_spacefill_generator.py')))
        print(json.dumps(dict(event='saved',stage=stage,checkpoint=str(checkpoint),sha256=digest,reason=reason)),flush=True)
        result,interiors=evaluate(model,HELDOUT,active,reference,width,True)
        sobol=sobol_baseline(active)
        gates=dict(training_target=reason=='hard monitor target reached',hard_current=score(result,entry_holdout,k)<=config['heldout_target'],
            retained_scales=retained(result,mastered_holdout),geometry_and_freshness=healthy(result,entry_holdout),
            boundary=result['boundary_cover']==entry_holdout['boundary_cover'])
        passed=all(gates.values())
        np.savez_compressed(OUT/f'stage_{stage}_heldout_interiors.npz',**{f'iteration{it}_{pool}':v for (it,pool),v in zip(HELDOUT,interiors)})
        item=dict(stage=stage,scale=k,passed=passed,reason=reason,best_step=best_step,updates=step,
            checkpoint=str(checkpoint),sha256=digest,gates=gates,result=result,entry=entry_holdout,Sobol=sobol,
            seconds=time.monotonic()-stage_start)
        stages.append(item)
        for a in active:
            summary_rows.append(dict(stage=stage,N=N,stage_anchor_grid=k,reported_anchor_grid=a,width=width,passed=passed,
                status='PASSED' if passed else 'STOPPED',reason=reason,hard_max=result['scales'][a]['max'],hard_rms=result['scales'][a]['rms'],
                entry_hard_max=entry_holdout['scales'][a]['max'],entry_hard_rms=entry_holdout['scales'][a]['rms'],
                previously_mastered_max=mastered_holdout.get(a,{}).get('max'),previously_mastered_rms=mastered_holdout.get(a,{}).get('rms'),
                Sobol_hard_max=sobol[a]['max'],Sobol_hard_rms=sobol[a]['rms'],
                complete_pool_hard_max=result['complete_scales'][a]['max'],complete_pool_hard_rms=result['complete_scales'][a]['rms'],
                Sobol_complete_pool_max=sobol[a]['complete_max'],Sobol_complete_pool_rms=sobol[a]['complete_rms'],
                covering_radius=result['cover'],entry_covering_radius=entry_holdout['cover'],large_holes=result['holes'],entry_large_holes=entry_holdout['holes'],
                nn_CV=result['nn_CV'],boundary_cover=result['boundary_cover'],exact_overlap=result['exact_overlap'],
                near_overlap_005=result['near_overlap'],near_overlap_010=result['near_overlap_010'],cell_greedy_near_overlap=result['cell_greedy_near_overlap'],
                best_step=best_step,updates=step,checkpoint=str(checkpoint),sha256=digest,seconds=item['seconds']))
        write_csv(OUT/'anchored_discrepancy_curriculum.csv',summary_rows)
        write_csv(OUT/'training_history.csv',history)
        (OUT/f'stage_{stage}_gate.json').write_text(json.dumps(item,indent=2)+'\n')
        print(json.dumps(dict(event='gate',stage=stage,passed=passed,gates=gates,result=result)),flush=True)
        if not passed:
            for j,s in enumerate(SCALES[stage:],stage+1):summary_rows.append(dict(stage=j,N=N,stage_anchor_grid=s,status='NOT_RUN',reason=f'Stage {stage} stopped'))
            break
        mastered_monitor[k]=best_monitor['scales'][k];mastered_holdout[k]=result['scales'][k]
        assert sha(checkpoint)==digest;parent=checkpoint
    write_csv(OUT/'anchored_discrepancy_curriculum.csv',summary_rows)
    unchanged=all(sha(p)==h for p,h in protected.items());assert unchanged
    last=stages[-1]
    # A failed final reached stage cannot be promoted even if an earlier passed.
    gap_gate=last['passed'] and all(last['result']['scales'][a][metric]-last['Sobol'][a][metric]
        <=.75*max(0.,initial_holdout[a][metric]-last['Sobol'][a][metric])
        for a in last['result']['scales'] for metric in ('max','rms'))
    if gap_gate:
        dest=OUT/'ANCHORED_DISCREPANCY_CURRICULUM_CANDIDATE.pt';shutil.copyfile(last['checkpoint'],dest)
        dest.with_suffix('.sha256').write_text(sha(dest)+'  '+dest.name+'\n')
    summary=dict(status='ANCHORED_DISCREPANCY_CURRICULUM_CANDIDATE' if gap_gate else 'STOPPED_DIAGNOSTIC_ONLY',
        stages=stages,elapsed_seconds=time.monotonic()-start,protected_unchanged=unchanged,protected_hashes=protected,
        architecture_unchanged=True,training_objective='soft anchored-rectangle squared discrepancy only',
        no_occupancy_loss=True,no_Sobol_supervision=True,no_search=True)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['ANCHORED-DISCREPANCY CURRICULUM — '+summary['status'],f'N=8600 only; elapsed {summary["elapsed_seconds"]:.1f} seconds.',
        'Same float64 architecture/inference. Warm start: prior passed 8x8 occupancy checkpoint. No other generator or locked component changed.',
        'Squared soft anchored discrepancy ONLY; no occupancy error or search labels. Compact smoothstep memberships integrate to uniform area at these anchors.',
        'Nested unshifted anchors {1/k,...,1}^2; current plus previous-scale losses. Widths0.1,0.05,0.025,0.0125,0.00625.',
        'Training-only straight-through local barycentres give coordinate gradients through the discrete policy. Hard forward verified bitwise equal to original inference; prefix features detached. This is a biased surrogate derivative, not an exact gradient of hard sampling.',
        'Hard evaluation uses exact inclusive [0,x]x[0,y] rectangles with no tolerance. Max and RMS reported; Sobol uses the identical anchors/contexts only for post-checkpoint evaluation.',
        'Primary discrepancy on N interior probes as in prior diagnostics; fixed boundary support preserved, complete-pool hard CDF also reported.',
        'Monitor target15% improvement in BOTH hard max/RMS; heldout target10%; previous mastered max/RMS regression<=5% against original mastery checkpoints.',
        'Coverage regression<=5%, large-hole fraction increase<=0.0005, exact overlap0, near overlap<=3% and increase<=0.5 percentage points. These preserve entry geometry, not an assertion of matching Sobol coverage.',
        'Near overlap at0.05/sqrt(N) and0.10/sqrt(N), boundaries excluded. Large holes use reference distance>1/sqrt(N). Finite-reference metrics, not exact continuous star discrepancy.',
        'Best checkpoints persisted before heldout gate. Only passed checkpoints may initialize later stages. No closed-loop search.','']
    for st in stages:
        a=st['scale'];v=st['result'];e=st['entry']
        lines.extend([f'Stage {st["stage"]}, {a}x{a}: {"PASS" if st["passed"] else "STOP"}, {st["reason"]}; best update {st["best_step"]}/{st["updates"]}.',
            f'Hard max {e["scales"][a]["max"]:.9g} -> {v["scales"][a]["max"]:.9g}; RMS {e["scales"][a]["rms"]:.9g} -> {v["scales"][a]["rms"]:.9g}.',
            f'Sobol max {st["Sobol"][a]["max"]:.9g}; RMS {st["Sobol"][a]["rms"]:.9g}.',
            f'Cover {v["cover"]:.9g}; holes {v["holes"]}; NN CV {v["nn_CV"]:.5g}; exact overlap {v["exact_overlap"]}; near {v["near_overlap"]:.5%}.',
            f'Checkpoint: {st["checkpoint"]}',f'SHA256: {st["sha256"]}',''])
    (OUT/'anchored_discrepancy_curriculum_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('stages','protected_hashes')}),flush=True)


if __name__=='__main__':run()
