"""Endpoint-supervised refinement of four unconstrained bilinear weights."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sho import run_stage1b_ordinary_bilinear_candidate as prior
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/ordinary_score_invariance_refine')
SEED=9284173


def features(x):
    return torch.stack((x[:,0],x[:,1],x[:,2]*x[:,0],x[:,2]*x[:,1]),dim=1)


def supervised_fit(model,data,endpoint,reference):
    x=torch.from_numpy(data);y=torch.from_numpy(prior.teacher(data))
    e=torch.from_numpy(endpoint);r=torch.from_numpy(reference)
    f=features(x);df=features(e)-features(r)
    hessian=2*(f.T@f)/len(x)+2*(df.T@df)/len(e)
    # Basis rows are score observations, not constraints or weight assignments.
    # Column order remains d_norm,s_norm,u. Labels come from the same teacher.
    basis=np.array([[1.,0.,0.],[0.,1.,0.],[0.,1.,1.],[1.,0.,1.]])
    history=[]
    def update_record(kind,cycle,before,gradient,rate=None,inputs=None):
        row=dict(record_type='optimizer_update',phase=kind,cycle=cycle,
            W_before=json.dumps(before.tolist()),gradient=json.dumps(gradient.tolist()),
            W_after=json.dumps(model.W.detach().tolist()),W00=float(model.W[0,0].detach()))
        if rate is not None:row['learning_rate']=float(rate)
        if inputs is not None:
            row['supervised_input']=json.dumps(inputs.tolist())
            row['teacher_score']=float(prior.teacher(inputs[None,:])[0])
        history.append(row)
    for cycle in range(1,5):
        model.zero_grad(set_to_none=True)
        loss=((model(x)-y)**2).mean()+((model(e)-model(r))**2).mean()
        loss.backward()
        before=model.W.detach().clone();gradient=model.W.grad.detach().clone()
        delta=torch.linalg.solve(hessian,gradient.reshape(-1)).reshape(2,2)
        with torch.no_grad():model.W.sub_(delta)
        update_record('mixed_score_and_endpoint_pair_MSE_Newton',cycle,before,gradient)
        for sample in basis:
            bx=torch.from_numpy(sample[None,:]);by=torch.from_numpy(prior.teacher(sample[None,:]))
            model.zero_grad(set_to_none=True)
            loss=((model(bx)-by)**2).mean()
            loss.backward()
            before=model.W.detach().clone();gradient=model.W.grad.detach().clone()
            # Standard normalized gradient step for a scalar linear regression
            # example: reciprocal of its nonzero quadratic curvature. Computed
            # from input features alone, with no reference to desired weights.
            rate=1./float(2*torch.sum(features(bx)**2))
            with torch.no_grad():model.W.add_(gradient,alpha=-rate)
            update_record('supervised_basis_normalized_gradient',cycle,before,gradient,rate,sample)
        with torch.no_grad():
            endpoint_variation=torch.max(torch.abs(model(e)-model(r))).item()
            max_error=torch.max(torch.abs(model(x)-y)).item()
        if model.W[0,0].item()==0 and endpoint_variation==0 and max_error<=prior.NUMERICAL_MARGIN:
            break
    model.eval().requires_grad_(False)
    return history,basis


def exact_score_ulp_sets():
    rng=np.random.default_rng(SEED+88)
    sets=[]
    for u in (0.,1.):
        for i in range(64):
            x=rng.uniform(.05,.95,(2,3));x[:,2]=u
            if u==0:
                x[1,1]=np.nextafter(x[0,1],1.)
            else:x[1,0]=np.nextafter(x[0,0],1.)
            y=prior.teacher(x)
            assert y[1]==np.nextafter(y[0],np.inf if u else -np.inf)
            sets.append(dict(source='exact_one_ULP_score_pairs',state=len(sets),site=0,x=x))
    attempts=0
    while len(sets)<256 and attempts<10000:
        attempts+=1
        x=np.tile(rng.uniform(.05,.95,3),(2,1))
        x[1,0]=np.nextafter(x[0,0],1.)
        y=prior.teacher(x)
        if y[1]==np.nextafter(y[0],np.inf):
            sets.append(dict(source='exact_one_ULP_score_pairs',state=len(sets),site=0,x=x))
    assert len(sets)==256
    return sets


def run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'trained_pending_evaluation.pt').exists(),'Do not rerun completed refinement'
    path=prior.OUT/'ordinary_bilinear_score_candidate.pt'
    expected=path.with_suffix('.sha256').read_text().split()[0]
    assert prior.base._sha256(path)==expected
    old_candidate=torch.load(path,weights_only=False)
    hashes=dict(old_candidate['protected_hashes'])
    hashes[str(path)]=expected
    hashes[str(prior.OUT/'training_tuples.npz')]=prior.base._sha256(prior.OUT/'training_tuples.npz')
    for p,h in hashes.items():assert prior.base._sha256(Path(p))==h,p
    torch.manual_seed(SEED)
    model=prior.BilinearScore();model.load_state_dict(old_candidate['state_dict'])
    initial=model.W.detach().clone()
    payload=torch.load(prior.base.turn.RAW_TRACE_PATH,weights_only=False)
    trajectories=prior.base.ordinary_data._trajectory_map(payload)
    validation=prior.historical(payload,trajectories,old_candidate['metadata']['validation_states'],'heldout_historical_sobol')
    spacefill=[]
    for budget,state in ((1000,17105),(8600,5660),(32768,17179)):
        record=payload['states'][state];n=int(record['history_count']);t=trajectories[int(record['trajectory_id'])]
        probes=np.load(prior.screen.screen.OUT/f'spacefill_probes_N{budget}.npz')['points']
        local=prior.tuple_sets(np.asarray(t['evaluated_hypotheses_normalized'][:n],dtype=np.float64),
            np.asarray(t['evaluated_prefix_losses'][:n],dtype=np.float64),probes,'',state)
        for i,item in enumerate(local):
            item['source']='saved_spacefill_fit_replay' if i%2==0 else 'saved_spacefill_heldout'
            spacefill.append(item)
    failure=np.load(prior.screen.OUT/'closedloop_failure_state.npz')
    spacefill.extend(prior.tuple_sets(failure['sites'],failure['losses'],failure['probes'],'saved_spacefill_failure_heldout',-1))
    endpoint=[];reference=[]
    for u in (0.,1.):
        for fixed in np.r_[np.linspace(0,1,17),np.nextafter(0.,1.),np.nextafter(1.,0.)]:
            varied=np.linspace(0,1,257)
            block=np.column_stack((varied,np.full(len(varied),fixed),np.zeros(len(varied)))) if u==0 else np.column_stack((np.full(len(varied),fixed),varied,np.ones(len(varied))))
            endpoint.append(block);reference.append(np.repeat(block[:1],len(block),axis=0))
    endpoint=np.concatenate(endpoint);reference=np.concatenate(reference)
    rng=np.random.default_rng(SEED)
    interior=rng.uniform(np.nextafter(0.,1.),1.,(16384,3))
    original=np.load(prior.OUT/'training_tuples.npz')['inputs']
    fitted_spacefill=np.concatenate([s['x'] for s in spacefill if s['source']=='saved_spacefill_fit_replay'])
    data=np.concatenate((original,fitted_spacefill,interior,endpoint))
    history,basis=supervised_fit(model,data,endpoint,reference)
    metadata=dict(initialization='reloaded prior structured candidate',initial_checkpoint_sha256=expected,
        initial_W=initial.tolist(),learned_W=model.W.tolist(),seed=SEED,optimizer_updates=len(history),
        objective='continuous score MSE + endpoint same-score difference MSE, followed by supervised basis-example MSE minibatches',
        no_weight_constraints_or_snapping=True,all_four_entries_trainable=True,
        original_training_tuple_count=len(original),spacefill_training_tuple_count=len(fitted_spacefill),
        added_synthetic_interior_count=len(interior),explicit_endpoint_count=len(endpoint),basis_examples=basis.tolist(),
        spacefill_split='alternating complete cells within each saved state; saved failed state excluded from fitting',
        original_training_recipe=old_candidate['metadata'],training_data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
        training_history=history)
    pending=dict(status='TRAINED_PENDING_EVALUATION',state_dict=model.state_dict(),metadata=metadata,
        architecture=old_candidate['architecture'],implementation=str(Path(__file__).resolve()),implementation_sha256=prior.base._sha256(Path(__file__)))
    save_checkpoint(OUT/'trained_pending_evaluation.pt',pending)
    np.savez_compressed(OUT/'refinement_training_data.npz',inputs=data,endpoint=endpoint,reference=reference,basis=basis)
    checkpoint=torch.load(OUT/'trained_pending_evaluation.pt',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    # Evaluation uses reloaded persisted weights; no subsequent training occurs.
    rows=[dict(record_type='model',learned_W=json.dumps(model.W.tolist()),initial_W=json.dumps(initial.tolist()),
        W00=float(model.W[0,0]),W00_hex=float(model.W[0,0]).hex(),W00_bits=np.float64(model.W[0,0].item()).view(np.uint64).item(),
        W00_exactly_zero=model.W[0,0].item()==0,updates=len(history))]+history
    sets=validation+spacefill+prior.synthetic_validation()+exact_score_ulp_sets()
    regression=[prior.regression(model,data,'refinement_training')]
    for source in sorted({s['source'] for s in sets}):
        regression.append(prior.regression(model,np.concatenate([s['x'] for s in sets if s['source']==source]),source))
    rows.extend(regression)
    invariance=[]
    # Fixed-value rows include non-dyadic random values and nextafter extremes;
    # each row sweeps the irrelevant input through 1025 values and random draws.
    for u in (0.,1.):
        fixed=np.r_[np.linspace(0,1,129),rng.random(128),np.nextafter(0.,1.),np.nextafter(1.,0.)]
        varied=np.r_[np.linspace(0,1,1025),rng.random(1024)]
        f,v=np.meshgrid(fixed,varied,indexing='ij')
        inputs=np.column_stack((v.ravel(),f.ravel(),np.zeros(f.size))) if u==0 else np.column_stack((f.ravel(),v.ravel(),np.ones(f.size)))
        prediction=prior.predict(model,inputs).reshape(f.shape)
        row=dict(record_type='endpoint_independence',u=u,fixed_values=len(fixed),varied_values=len(varied),
            max_output_variation=float(np.ptp(prediction,axis=1).max()),
            max_teacher_error=float(np.max(abs(prediction.ravel()-prior.teacher(inputs)))),
            learned_irrelevant_coefficient=float(model.W[0,0]) if u==0 else float(model.W[0,1]+model.W[1,1]))
        invariance.append(row)
    rows.extend(invariance)
    old,_=prior.selection._load_search_models();old.eval().requires_grad_(False)
    rankings,reversals,winners=prior.ranking(model,old,sets)
    summaries=[r for r in rankings if r['record_type']=='ranking_summary']
    # Also emit the two signed score margins for all explicitly one-ULP pairs.
    for s in sets:
        if s['source'] not in ('synthetic_one_ulp','exact_one_ULP_score_pairs'):continue
        y=prior.teacher(s['x']);p=prior.predict(model,s['x'])
        rankings.append(dict(record_type='one_ULP_pair_detail',source=s['source'],state=s['state'],
            teacher_margin=float(y[0]-y[1]),predicted_margin=float(p[0]-p[1]),
            exact_scores=bool(np.array_equal(y,p)),inputs=json.dumps(s['x'].tolist())))
    passed=(model.W[0,0].item()==0 and all(r['max_output_variation']==0 for r in invariance)
        and all(r['max_error']<=prior.NUMERICAL_MARGIN for r in regression)
        and all(r['strict_order_reversals']==0 for r in summaries if 'sobol' in r['source'] or 'spacefill' in r['source']))
    summary=dict(record_type='gate',status='CANDIDATE' if passed else 'FAILED',
        W=model.W.tolist(),W00_hex=float(model.W[0,0]).hex(),endpoint_invariance=invariance,
        rankings=summaries,no_closedloop_search=True,no_old_scorer_changes=True)
    if passed:
        candidate=dict(pending,status='CANDIDATE',gate_results=summary,protected_hashes=hashes)
        summary['candidate_sha256']=save_checkpoint(OUT/'ordinary_score_invariance_candidate.pt',candidate)
    for p,h in hashes.items():assert prior.base._sha256(Path(p))==h,p
    summary['protected_hashes_unchanged']=True
    prior.base._write_csv(OUT/'ordinary_score_invariance_refine.csv',[dict(record_type='gate',status=summary['status'],
        candidate_sha256=summary.get('candidate_sha256',''))]+rows)
    prior.base._write_csv(OUT/'ranking_stress.csv',rankings)
    prior.base._write_csv(OUT/'strict_reversals.csv',reversals or [dict(strict_reversals=0)])
    prior.base._write_csv(OUT/'winner_differences.csv',winners or [dict(winner_differences=0)])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
