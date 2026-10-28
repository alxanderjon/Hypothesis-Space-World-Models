"""Finish the same supervised zero-anchor fit; no parameter snapping."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold,load_data,evaluate,ROOT,SEED
from experiments.geometry.train_stationary_magnitude_margin import MagnitudeMargin
from experiments.geometry.train_exponent_log2_operator import sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

SOURCE=ROOT/'stationary_magnitude_threshold_smoke'
OUT=SOURCE/'zero_anchor_converged'


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    for p in (ROOT/'stationary_magnitude_margin_smoke/candidate_manifest.json',ROOT/'stationary_margin_assembly/stationary_margin_validated.json',Path('experiments/geometry/validate_stationary_margin.py'),Path('experiments/geometry/train_stationary_magnitude_margin.py')):protected[str(p)]=sha(p)
    previous=json.loads((SOURCE/'results.json').read_text()); cp=Path(previous['checkpoint']);assert sha(cp)==previous['sha256']
    checkpoint=torch.load(cp,weights_only=False);model=MagnitudeThreshold();model.load_state_dict(checkpoint['state_dict'])
    before=float(model.theta.detach());alpha=float(model.alpha().detach());history=[]
    zero_x=torch.zeros(32,dtype=torch.float64)
    for step in range(33):
        logits=model(zero_x);residual=-logits.detach().numpy()
        history.append(dict(step=step,theta=float(model.theta.detach()),alpha=float(model.alpha().detach()),max_anchor_residual=float(abs(residual).max())))
        if np.all(residual==0) or step==32:break
        jac=float(torch.autograd.grad(logits[0],model.theta)[0])
        delta=np.linalg.lstsq(np.full((len(residual),1),jac),residual,rcond=None)[0][0]
        old=float(model.theta.detach())
        with torch.no_grad():model.theta.add_(float(delta))
        if float(model.theta.detach())==old:break
    assert float(model.alpha().detach())==alpha
    config=dict(checkpoint['config'],continuation_source=str(cp),continuation_source_sha256=sha(cp),
        continuation='Same zero-logit supervised least-squares residual fit until residual is exact zero or no parameter progress; no snapping.',
        initial_continuation_theta=before,additional_anchor_updates=len(history)-1)
    path=OUT/'threshold_before_evaluation.pt'
    digest=save_checkpoint(path,dict(state_dict=model.state_dict(),config=config,status='SAVED_BEFORE_EVALUATION',
        original_training_history=checkpoint['training_history'],continuation_history=history))
    write_csv(OUT/'anchor_continuation_history.csv',history)
    print(json.dumps(dict(event='saved_before_evaluation',theta=float(model.theta.detach()),alpha=alpha,additional_updates=len(history)-1,sha256=digest)),flush=True)
    model.load_state_dict(torch.load(path,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    manifest=json.loads((ROOT/'stationary_magnitude_margin_smoke/candidate_manifest.json').read_text()); frozen_cp=Path(manifest['checkpoint'])
    assert sha(frozen_cp)==manifest['sha256']
    frozen=MagnitudeMargin();frozen.load_state_dict(torch.load(frozen_cp,weights_only=False)['state_dict']);frozen.eval().requires_grad_(False)
    margin,held,groups,inputs,counts=load_data(frozen)
    synthetic=np.unique(np.r_[np.power(10.,np.arange(-323,1,2)),np.ldexp(np.ones(1075),np.arange(-1074,1)),
        [1e-18,1e-16,1e-14,1e-12,1e-9,1e-6,1e-3,1e-1,1.]])
    rng=np.random.default_rng(SEED+1)
    positive=np.unique(np.r_[np.power(10.,np.linspace(-323,0,16385)),np.ldexp(rng.uniform(1,2,8192),rng.integers(-1074,0,8192)),np.linspace(1e-12,1e-7,4096)])
    subnormal=np.arange(1,4097,dtype=np.uint64).view(np.float64)
    sets=[('all_archived_and_dense',margin),('training_archived',margin[~held]),('heldout_archived',margin[held]),
        ('one_ulp_around_zero',np.array([np.nextafter(0.,-np.inf),-0.,0.,np.nextafter(0.,np.inf)])),
        ('first_4096_subnormals_symmetric',np.r_[-subnormal,subnormal]),('unseen_dense_symmetric',np.r_[-positive,positive]),
        ('training_synthetic_symmetric',np.r_[-synthetic,synthetic]),('signed_zero_anchors',np.array([-0.,0.]))]
    rows=[];boundary=[];failures=[]
    for name,m in sets:
        row,logit,pred=evaluate(model,m,name);rows.append(row)
        if name in ('one_ulp_around_zero','signed_zero_anchors'):
            boundary += [dict(group=name,margin=float(x),margin_hex=float(x).hex(),logit=float(l),teacher=int(np.sign(x)),decision=int(y)) for x,l,y in zip(m,logit,pred)]
        for i in np.flatnonzero(pred!=np.sign(m))[:20]:failures.append(dict(group=name,margin=float(m[i]),logit=float(logit[i]),decision=float(pred[i])))
    offset=0;scopes={}
    for g in groups:scopes.setdefault(g['scope'],[]).append(margin[offset:offset+g['cases']]);offset+=g['cases']
    for scope,chunks in scopes.items():rows.append(evaluate(model,np.concatenate(chunks),scope)[0])
    write_csv(OUT/'stationary_threshold_validation.csv',rows);write_csv(OUT/'stationary_threshold_boundary.csv',boundary)
    write_csv(OUT/'failures.csv',failures or [dict(status='none')])
    assert all(sha(Path(p))==h for p,h in {**protected,**inputs}.items())
    passed=all(r['mistakes']==r['symmetry_errors']==r['nonfinite_logits']==0 for r in rows)
    result=dict(operator='stationary_magnitude_threshold',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        learned_theta=float(model.theta),learned_alpha=float(model.alpha()),raw_alpha=float(model.raw_alpha),raw_zero_logit=float(model(torch.tensor(0.,dtype=torch.float64))),
        checkpoint=str(path),sha256=digest,module='experiments.geometry.train_stationary_magnitude_threshold',model='MagnitudeThreshold',
        implementation_sha256=sha(Path('experiments/geometry/train_stationary_magnitude_threshold.py')),
        training_continuation_sha256=sha(Path(__file__)),config=config,metrics=rows,all_prior_hashes_unchanged=True,
        protected_checkpoint_count=sum(p.endswith('.pt') for p in protected),elapsed_seconds=time.monotonic()-start,
        combined_experiment_seconds=previous['elapsed_seconds']+time.monotonic()-start)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'config.json').write_text(json.dumps(config,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2));(OUT/'input_archive_hashes.json').write_text(json.dumps(inputs,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('config','metrics')},indent=2))
    (OUT/'stationary_threshold_summary.txt').write_text(json.dumps(result,indent=2)+'\nNo derivative-sign router, root finding, recursion, or search.\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','metrics')},indent=2),flush=True)
    print(json.dumps(rows[:8],indent=2),flush=True)


if __name__=='__main__':run()
