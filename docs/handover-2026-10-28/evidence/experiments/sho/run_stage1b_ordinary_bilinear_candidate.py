"""Four learned bilinear weights; geometry-distribution and static fidelity smoke."""
import csv
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_spacefill_2x2 as screen
from experiments.sho.smoke_artifacts import save_checkpoint

base,cells,nearest,down=screen.base,screen.cells,screen.nearest,screen.down
edirect,selection=screen.edirect,screen.selection
OUT=Path('outputs/sho/stage1B/ordinary_cell_bilinear_candidate')
SEED=9273101
NUMERICAL_MARGIN=8*np.finfo(np.float64).eps


class BilinearScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.W=nn.Parameter(torch.empty((2,2),dtype=torch.float64).uniform_(-1,1))

    def forward(self,inputs):
        q=torch.stack((torch.ones_like(inputs[:,2]),inputs[:,2]),dim=1)
        coefficients=q@self.W
        return torch.sum(coefficients*inputs[:,:2],dim=1)


def teacher(x):
    return x[:,2]*x[:,0]-(1-x[:,2])*x[:,1]


def predict(model,x):
    with torch.no_grad():return model(torch.from_numpy(x)).numpy().ravel()


def tuple_sets(sites,losses,probes,label,state):
    distance,owner=cKDTree(sites).query(probes)
    radius=np.zeros(len(sites));np.maximum.at(radius,owner,distance)
    sizes=np.maximum(radius,.5*cKDTree(sites).query(sites,k=2)[0][:,1])
    selected=edirect._potentially_optimal_cells(sizes,losses)
    surface=edirect.GaussianLocalResponse(sites,losses)
    u=.5*edirect._normalize(sizes[selected])+.5*edirect._normalize(surface.loo_errors(selected))
    largest=int(np.argmax(sizes[selected]));result=[]
    for pos,site in enumerate(selected):
        indices=np.flatnonzero(owner==site)
        if pos==largest or len(indices)==0:continue
        x=np.column_stack((edirect._normalize(distance[indices]),edirect._normalize(surface.predict(probes[indices])),np.full(len(indices),u[pos])))
        assert x.dtype==np.float64 and np.all((x>=0)&(x<=1))
        result.append(dict(source=label,state=state,site=int(site),x=x))
    return result


def historical(payload,trajectories,indices,label):
    sets=[]
    for index in indices:
        record=payload['states'][index];n=int(record['history_count'])
        trajectory=trajectories[int(record['trajectory_id'])]
        sites=np.asarray(trajectory['evaluated_hypotheses_normalized'][:n],dtype=np.float64)
        losses=np.asarray(trajectory['evaluated_prefix_losses'][:n],dtype=np.float64)
        probes=edirect._candidate_pool(n,int(record['generator_id']),int(record['iteration']))
        sets.extend(tuple_sets(sites,losses,probes,label,index))
    return sets


def choose_states(payload,training):
    rng=np.random.default_rng(SEED+int(training))
    chosen=[]
    for lo,hi in ((10,40),(40,150),(150,400)):
        possible=[i for i,r in enumerate(payload['states']) if r['candidate_type']=='cell'
            and lo<=int(r['history_count'])<hi and int(r['trajectory_id']) not in (37,108,136)
            and (int(r['trajectory_id'])%5!=0)==training]
        chosen.extend(int(i) for i in rng.choice(possible,4,replace=False))
    return chosen


def fit(real):
    rng=np.random.default_rng(SEED)
    real=real[rng.choice(len(real),min(32768,len(real)),replace=False)]
    blocks=[rng.random((32768,3))]
    for u in (0.,1.,.01,.1,.25,.5,.75,.9,.99):
        t=np.linspace(0,1,65 if u in (0,1) else 17)
        grid=np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2)
        blocks.append(np.column_stack((grid,np.full(len(grid),u))))
    synthetic=np.concatenate(blocks)
    data=np.concatenate((real,synthetic))
    torch.manual_seed(SEED)
    model=BilinearScore()
    initial=model.W.detach().clone()
    x=torch.from_numpy(data);y=torch.from_numpy(teacher(data))
    features=torch.stack((x[:,0],x[:,1],x[:,2]*x[:,0],x[:,2]*x[:,1]),dim=1)
    hessian=2*(features.T@features)/len(x)
    history=[]
    # Exact 4x4 quadratic Hessian. All four entries are independent learned
    # parameters; no projection, snapping, prescribed coefficients or ranking loss.
    for step in range(1,9):
        model.zero_grad(set_to_none=True)
        loss=((model(x)-y)**2).mean()
        loss.backward()
        correction=torch.linalg.solve(hessian,model.W.grad.reshape(-1))
        with torch.no_grad():model.W.sub_(correction.reshape(2,2))
        error=predict(model,data)-y.numpy()
        history.append(dict(step=step,loss=float(np.mean(error**2)),max_error=float(np.max(np.abs(error))),W=model.W.detach().tolist()))
        if np.max(np.abs(error))<=4*np.finfo(np.float64).eps:break
    model.eval().requires_grad_(False)
    metadata=dict(seed=SEED,initial_W=initial.tolist(),learned_W=model.W.tolist(),updates=step,
        optimizer='full-batch quadratic Newton, four independent parameters',objective='continuous MSE only',
        real_training_tuples=len(real),synthetic_training_tuples=len(synthetic),
        data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),training_history=history)
    return model,metadata,data


def regression(model,x,source):
    y=teacher(x);p=predict(model,x);e=p-y
    return dict(record_type='regression',source=source,count=len(x),rmse=float(np.sqrt(np.mean(e*e))),
        mae=float(np.mean(abs(e))),max_error=float(np.max(abs(e))),correlation=float(np.corrcoef(y,p)[0,1]))


def synthetic_validation():
    rng=np.random.default_rng(SEED+991)
    sets=[]
    for i in range(96):
        u=[0.,1.,.001,.1,.3,.5,.7,.9,.999][i%9] if i<72 else float(rng.random())
        x=np.column_stack((rng.random((128,2)),np.full(128,u)))
        sets.append(dict(source='synthetic_unseen',state=i,site=0,x=x))
    # Exact score ties and endpoint independence, including tie-sensitive winners.
    for u in (0.,1.):
        for value in (0.,.1,.5,1.):
            x=np.column_stack((np.linspace(0,1,257),np.full(257,value),np.full(257,u))) if u==0 else np.column_stack((np.full(257,value),np.linspace(0,1,257),np.full(257,u)))
            sets.append(dict(source='synthetic_endpoint_ties',state=int(100*u+10*value),site=0,x=x))
    for i in range(128):
        u=float(rng.choice([0.,1.,.1,.5,.9]))
        x=np.tile(np.r_[rng.random(2),u],(2,1))
        axis=1 if u==0 else 0
        x[1,axis]=np.nextafter(x[0,axis],1.)
        sets.append(dict(source='synthetic_one_ulp',state=i,site=0,x=x))
    return sets


def ranking(model,old,sets):
    rows=[];reversals=[];winners=[]
    for number,item in enumerate(sets):
        x=item['x'];y=teacher(x);p=predict(model,x);old_p=predict(old,x)
        a,b,c=int(np.argmax(y)),int(np.argmax(p)),int(np.argmax(old_p))
        pair_count=strict_count=mistakes=reverse=collapsed=split_ties=meaningful=0
        n=len(x)
        for start in range(0,n,128):
            ii=np.arange(start,min(start+128,n))[:,None];jj=np.arange(n)[None,:]
            active=ii<jj
            dy=y[ii]-y[jj];dp=p[ii]-p[jj]
            wrong=active & (np.sign(dy)!=np.sign(dp))
            rev=active & (dy*dp<0)
            # Sign product could underflow on subnormal tests; signs are authoritative.
            rev=active & (dy!=0) & (dp!=0) & (np.sign(dy)!=np.sign(dp))
            pair_count+=int(active.sum());strict_count+=int(np.sum(active&(dy!=0)))
            mistakes+=int(wrong.sum());reverse+=int(rev.sum())
            collapsed+=int(np.sum(active&(dy!=0)&(dp==0)))
            split_ties+=int(np.sum(active&(dy==0)&(dp!=0)))
            meaningful+=int(np.sum(wrong&(np.abs(dy)>NUMERICAL_MARGIN)))
            for local_i,j in np.argwhere(rev):
                i=start+int(local_i);j=int(j)
                reversals.append(dict(source=item['source'],state=item['state'],site=item['site'],candidate_i=i,candidate_j=j,
                    teacher_score_i=float(y[i]),teacher_score_j=float(y[j]),predicted_score_i=float(p[i]),predicted_score_j=float(p[j]),
                    teacher_margin=float(y[i]-y[j]),predicted_margin=float(p[i]-p[j]),
                    meaningful=abs(y[i]-y[j])>NUMERICAL_MARGIN,inputs_i=json.dumps(x[i].tolist()),inputs_j=json.dumps(x[j].tolist())))
        row=dict(record_type='ranking_set',source=item['source'],state=item['state'],site=item['site'],candidates=n,
            top1_exact=a==b,top1_teacher_tie_valid=bool(y[b]==y[a]),winner_regret=float(y[a]-y[b]),
            top1_meaningful_error=bool(y[a]-y[b]>NUMERICAL_MARGIN),old_top1_exact=a==c,old_winner_regret=float(y[a]-y[c]),
            pair_count=pair_count,strict_pair_count=strict_count,pairwise_ordering_mistakes=mistakes,strict_order_reversals=reverse,
            collapsed_strict_pairs=collapsed,split_teacher_ties=split_ties,meaningful_ordering_mistakes=meaningful)
        rows.append(row)
        if a!=b or a!=c:
            winners.append(dict(**row,teacher_winner=a,bilinear_winner=b,old_winner=c))
    summaries=[]
    for source in sorted({r['source'] for r in rows}):
        part=[r for r in rows if r['source']==source]
        summary=dict(record_type='ranking_summary',source=source,set_count=len(part),
            top1_accuracy=float(np.mean([r['top1_exact'] for r in part])),old_top1_accuracy=float(np.mean([r['old_top1_exact'] for r in part])),
            top1_teacher_tie_valid_accuracy=float(np.mean([r['top1_teacher_tie_valid'] for r in part])))
        for key in ('pair_count','strict_pair_count','pairwise_ordering_mistakes','strict_order_reversals','collapsed_strict_pairs','split_teacher_ties','meaningful_ordering_mistakes','top1_meaningful_error'):
            summary[key]=sum(r[key] for r in part)
        summaries.append(summary)
    return summaries+rows,reversals,winners


def fidelity(model,old,selector,payload,trajectories,locked):
    metric,_=base._load_locked_distance()
    op=torch.load(nearest.OPERATOR,weights_only=False)
    margin=nn.Linear(2,1,bias=False,dtype=torch.float64)
    margin.load_state_dict(torch.load(op['margin_checkpoint'],weights_only=False)['state_dict'])
    threshold=nearest.threshold.NonzeroThreshold()
    threshold.load_state_dict(torch.load(op['threshold_checkpoint'],weights_only=False)['state_dict'])
    half=nn.Linear(1,1,bias=False,dtype=torch.float64)
    half.load_state_dict(torch.load(cells.OUT/'half_spacing_operator_locked.pt',weights_only=False)['state_dict'])
    models=(metric,margin,threshold,half)
    for m in models:m.eval().requires_grad_(False)
    cases=[]
    for budget,state in ((1000,17105),(8600,5660),(32768,17179)):
        record=payload['states'][state];n=int(record['history_count']);t=trajectories[int(record['trajectory_id'])]
        sites=np.asarray(t['evaluated_hypotheses_normalized'][:n],dtype=np.float64);losses=np.asarray(t['evaluated_prefix_losses'][:n],dtype=np.float64)
        for source,probes in (('sobol',edirect._candidate_pool(n,int(record['generator_id']),int(record['iteration']))),
                              ('spacefill',np.load(screen.screen.OUT/f'spacefill_probes_N{budget}.npz')['points'])):
            cases.append((source,state,record,sites,losses,probes))
    failed=np.load(screen.OUT/'closedloop_failure_state.npz')
    cases.append(('spacefill_saved_failure',-1,dict(trajectory_id=136,iteration=2),failed['sites'],failed['losses'],failed['probes']))
    rows=[]
    for source,state,record,sites,losses,probes in cases:
        print(f'Static candidate fidelity {source} state={state}',flush=True)
        reference=screen.trace_geometry(sites,losses,probes,record,state,False,models,locked)
        learned=screen.trace_geometry(sites,losses,probes,record,state,True,models,locked)
        target,_=screen.proposals(sites,losses,probes,reference['owners'],reference['distances'],reference['sizes'],reference['po'])
        for name,scorer in (('bilinear_candidate',model),('preserved_old_MLP',old)):
            proposals,records=screen.proposals(sites,losses,probes,learned['owners'],learned['distances'],learned['sizes'],learned['po'],scorer)
            features=selection.imitation._global_features(sites,losses,len(proposals),len(sites),
                160 if state==-1 else selection.NORMAL_CAP,int(record['iteration']),selection.FULL_BOUNDS,'normal_edirect')
            choice,_=selection.imitation._model_choice(selector,sites,losses,proposals,'cell',features)
            exact,total=screen.exact_counts(target,proposals)
            row=dict(source=source,state=state,scorer=name,proposals_exact=exact,proposal_count=total,
                next_selected_exact=bool(np.array_equal(target[0],proposals[choice])),choice=int(choice),
                first_proposal_teacher=json.dumps(target[0].tolist()),first_selected_learned=json.dumps(proposals[choice].tolist()))
            for stage in ('owners','distances','spacing','radius','half_spacing','sizes','membership','representatives','hull','po'):
                count,full=screen.exact_counts(reference[stage],learned[stage]);row[stage+'_exact']=count;row[stage+'_count']=full
            rows.append(row)
    return rows


def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'trained_pending_evaluation.pt').exists(),'Do not repeat a completed training run'
    hashes=json.loads((screen.OUT/'static_summary.json').read_text())['protected_hashes']
    for p,h in hashes.items():assert base._sha256(Path(p))==h,p
    payload=torch.load(base.turn.RAW_TRACE_PATH,weights_only=False);trajectories=base.ordinary_data._trajectory_map(payload)
    train_indices,valid_indices=choose_states(payload,True),choose_states(payload,False)
    training=historical(payload,trajectories,train_indices,'historical_train')
    validation=historical(payload,trajectories,valid_indices,'heldout_historical_sobol')
    train_ids={payload['states'][i]['trajectory_id'] for i in train_indices}
    valid_ids={payload['states'][i]['trajectory_id'] for i in valid_indices}
    assert not(train_ids&valid_ids)
    model,metadata,training_data=fit(np.concatenate([r['x'] for r in training]))
    metadata.update(training_states=train_indices,validation_states=valid_indices,
        training_trajectory_ids=sorted(train_ids),validation_trajectory_ids=sorted(valid_ids),
        training_source='regenerated real ordinary-cell float64 tuples, no float32 cache',
        source_trace_sha256=base._sha256(base.turn.RAW_TRACE_PATH))
    pending=dict(status='TRAINED_PENDING_EVALUATION',architecture='q=[1,u], x=[d_norm,s_norm], q^T W x; four unconstrained learned entries',
        state_dict=model.state_dict(),metadata=metadata,implementation=str(Path(__file__).resolve()),implementation_sha256=base._sha256(Path(__file__)))
    save_checkpoint(OUT/'trained_pending_evaluation.pt',pending)
    np.savez_compressed(OUT/'training_tuples.npz',inputs=training_data,labels=teacher(training_data))
    spacefill=[]
    for budget,state in ((1000,17105),(8600,5660),(32768,17179)):
        record=payload['states'][state];n=int(record['history_count']);t=trajectories[int(record['trajectory_id'])]
        probes=np.load(screen.screen.OUT/f'spacefill_probes_N{budget}.npz')['points']
        spacefill.extend(tuple_sets(np.asarray(t['evaluated_hypotheses_normalized'][:n],dtype=np.float64),
            np.asarray(t['evaluated_prefix_losses'][:n],dtype=np.float64),probes,'saved_spacefill',state))
    failed=np.load(screen.OUT/'closedloop_failure_state.npz')
    spacefill.extend(tuple_sets(failed['sites'],failed['losses'],failed['probes'],'saved_spacefill_failure',-1))
    sets=validation+spacefill+synthetic_validation()
    score_rows=[regression(model,training_data,'training')]
    for source in sorted({s['source'] for s in sets}):
        score_rows.append(regression(model,np.concatenate([s['x'] for s in sets if s['source']==source]),source))
    invariant=[]
    for u in (0.,1.):
        t=np.linspace(0,1,513);grid=np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2)
        x=np.column_stack((grid,np.full(len(grid),u)))
        p=predict(model,x).reshape(513,513)
        axis=1 if u==0 else 0
        invariant.append(dict(u=u,max_independence_violation=float(np.ptp(p,axis=axis).max()),
            max_teacher_error=float(np.max(abs(p.reshape(-1)-teacher(x))))))
    old,selector=selection._load_search_models()
    for m in (old,selector):m.eval().requires_grad_(False)
    ranking_rows,reversals,winners=ranking(model,old,sets)
    summaries=[r for r in ranking_rows if r['record_type']=='ranking_summary']
    passed=all(r['max_error']<=NUMERICAL_MARGIN for r in score_rows) and all(r['max_independence_violation']<=NUMERICAL_MARGIN for r in invariant)
    passed=passed and all(r['meaningful_ordering_mistakes']==0 and r['top1_meaningful_error']==0 for r in summaries)
    base._write_csv(OUT/'score_regression.csv',score_rows)
    base._write_csv(OUT/'endpoint_invariance.csv',invariant)
    base._write_csv(OUT/'candidate_rankings.csv',ranking_rows)
    base._write_csv(OUT/'strict_reversals.csv',reversals or [dict(record_type='summary',strict_reversals=0)])
    base._write_csv(OUT/'winner_differences.csv',winners or [dict(record_type='summary',winner_differences=0)])
    result=dict(status='CANDIDATE' if passed else 'FAILED',training=metadata,numerical_margin_threshold=NUMERICAL_MARGIN,
        score_metrics=score_rows,invariance=invariant,ranking=summaries,strict_reversal_count=len(reversals),
        no_closedloop_search=True,old_scorer_sha256=base._sha256(selection.CHECKPOINT_PATH))
    if passed:
        candidate=dict(pending,status='CANDIDATE',gate_results=result,protected_hashes=hashes)
        result['candidate_sha256']=save_checkpoint(OUT/'ordinary_bilinear_score_candidate.pt',candidate)
        # Only replay stored states. No trajectory advancement, generator update,
        # component substitution in production, or resumed closed-loop run.
        locked=down._load_downstream()
        replay=fidelity(model,old,selector,payload,trajectories,locked)
        base._write_csv(OUT/'static_fidelity_replay.csv',replay)
        result['static_candidate_fidelity_exact']=all(r['proposals_exact']==r['proposal_count'] and r['next_selected_exact'] for r in replay if r['scorer']=='bilinear_candidate')
    else:
        base._write_csv(OUT/'static_fidelity_replay.csv',[dict(status='NOT_RUN_HARD_GATE_FAILED')])
    for p,h in hashes.items():assert base._sha256(Path(p))==h,p
    result['protected_hashes_unchanged']=True
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
