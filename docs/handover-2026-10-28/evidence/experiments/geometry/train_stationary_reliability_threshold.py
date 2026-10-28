"""Learn only a scalar TRUST/DEFER threshold; no sign or root assembly."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'stationary_reliability_threshold_smoke'
TAU=4.735309043366555e-16
SEED=202609281149


class ReliabilityThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(.3*torch.randn((),dtype=torch.float64))
        self.raw_alpha=nn.Parameter(.3*torch.randn((),dtype=torch.float64))

    def alpha(self):return torch.nn.functional.softplus(self.raw_alpha)

    def forward(self,r):return self.alpha()*(r-self.theta)

    def decision(self,r):
        # Zero logit is DEFER; no input equality override or stored teacher tau.
        return self(r)>0


def evaluate(model,r,name):
    with torch.no_grad():
        t=torch.from_numpy(r);logit=model(t).numpy();pred=model.decision(t).numpy()
    teacher=r>TAU;bad=pred!=teacher;correct=~bad
    return dict(group=name,cases=len(r),classification_errors=int(bad.sum()),
        TRUST_errors=int(np.sum(pred&~teacher)),DEFER_errors=int(np.sum(~pred&teacher)),
        below_tau_errors=int(np.sum(bad&(r<TAU))),above_tau_errors=int(np.sum(bad&(r>TAU))),
        exact_tau_cases=int(np.sum(r==TAU)),exact_tau_errors=int(np.sum(bad&(r==TAU))),
        nonfinite_logits=int(np.sum(~np.isfinite(logit))),
        smallest_correctly_trusted=float(r[correct&pred].min()) if np.any(correct&pred) else None,
        largest_correctly_deferred=float(r[correct&~pred].max()) if np.any(correct&~pred) else None),logit,pred


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')};protected.update({str(p):sha(p) for p in ROOT.rglob('*manifest.json')})
    source_paths=[ROOT/'stationary_sign_separability_audit/sign_reliability_values.npz',ROOT/'fixed_sign_reliability_fresh_holdout/evaluated_holdout_results.npz']
    for p in source_paths:protected[str(p)]=sha(p)
    archive=np.load(source_paths[0])['gap'];fresh_file=np.load(source_paths[1]);fresh=fresh_file['r'];fresh_groups=fresh_file['group']
    assert np.isfinite(archive).all() and np.isfinite(fresh).all() and archive.min()>=0 and fresh.min()>=0
    parts=[];counts={}
    for name,values in [('archived',archive),('fresh_holdout_style',fresh)]:
        unique=np.unique(values);selected=[]
        for sign in (False,True):
            pool=unique[(unique>TAU)==sign]
            selected.extend(rng.choice(pool,size=min(4096,len(pool)),replace=False))
        parts.append(np.array(selected));counts[name]=len(selected)
    bits=np.float64(TAU).view(np.int64)
    dense=(bits+np.arange(-1024,1025,dtype=np.int64)).view(np.float64)
    # Symmetric distances around tau within the nonnegative gap domain.
    delta=TAU*2.**(-np.arange(0,54));symmetric=np.r_[TAU-delta,TAU+delta]
    broad=np.r_[0.,10.**np.linspace(-323,0,4097),rng.uniform(0,1,2048)]
    synthetic=np.unique(np.r_[dense,symmetric,broad])
    train=np.unique(np.concatenate(parts+[synthetic]));target=train>TAU
    assert all(x in train for x in [np.nextafter(TAU,0.),TAU,np.nextafter(TAU,np.inf)])
    tx=torch.from_numpy(train);labels=torch.from_numpy(target.astype(float));weights=np.empty(len(train))
    weights[target]=.5/target.sum();weights[~target]=.5/(~target).sum();tw=torch.from_numpy(weights)
    model=ReliabilityThreshold();initial=dict(theta=float(model.theta.detach()),raw_alpha=float(model.raw_alpha.detach()),alpha=float(model.alpha().detach()))
    assert initial['theta']!=TAU
    optimizer=torch.optim.Adam(model.parameters(),lr=.05);history=[]
    anchor=torch.full((32,),TAU,dtype=torch.float64)
    # Same monotone-threshold style as the earlier successful decision operators:
    # binary supervision plus zero-logit supervision at the fixed teacher boundary.
    for step in range(32):
        optimizer.zero_grad();logits=model(tx)
        bce=(tw*torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')).sum()
        anchor_loss=(model(anchor)**2).mean();loss=bce+anchor_loss
        loss.backward();optimizer.step()
        history.append(dict(phase='balanced_logistic_boundary_anchor',step=step+1,loss=float(loss.detach()),theta=float(model.theta.detach()),alpha=float(model.alpha().detach())))
    # Residual fitting updates theta from the measured supervised anchor error
    # and its autograd Jacobian. No assignment, rounding, or snapping to tau.
    for step in range(16):
        logits=model(anchor);residual=-logits.detach().numpy()
        with torch.no_grad():errors=int(torch.sum(model.decision(tx)!=(tx>TAU)))
        history.append(dict(phase='supervised_boundary_residual_fit',step=step,loss=float(np.mean(residual**2)),
            max_anchor_residual=float(abs(residual).max()),training_errors=errors,theta=float(model.theta.detach()),alpha=float(model.alpha().detach())))
        if np.all(residual==0) and errors==0:break
        jac=float(torch.autograd.grad(logits[0],model.theta)[0]);delta=np.linalg.lstsq(np.full((len(anchor),1),jac),residual,rcond=None)[0][0]
        previous=float(model.theta.detach())
        with torch.no_grad():model.theta.add_(float(delta))
        if float(model.theta.detach())==previous:break
    config=dict(seed=SEED,dtype='float64',architecture='softplus(raw_alpha)*(r-theta)',prediction='TRUST iff logit > 0; otherwise DEFER',
        teacher_tau=TAU,teacher_tau_hex=TAU.hex(),initial=initial,training_examples=len(train),source_training_counts=counts,
        synthetic_examples=len(synthetic),supervision='Balanced TRUST/DEFER labels; zero-logit supervised anchor at the fixed teacher boundary.',
        optimizer='32 Adam updates, then at most 16 measured boundary-residual least-squares updates to theta using its autograd Jacobian.',
        no_manual_snapping=True,no_input_equality_override=True,no_gap_recomputation_or_change=True,
        heldout_split='Scalar gap values in training are excluded from each held-out pool; full-pool replay additionally includes training values.',
        scope='Only threshold training. No root-location behavior, term-sign router, recursion, or global search.')
    checkpoint=OUT/'reliability_threshold_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,training_history=history,status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'training_history.csv',history);(OUT/'config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(OUT/'training_gaps.npz',r=train,teacher_trust=target)
    print(json.dumps(dict(event='saved_before_evaluation',theta=float(model.theta.detach()),alpha=float(model.alpha().detach()),sha256=digest)),flush=True)
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    archive_held=archive[~np.isin(archive,train)];fresh_held=fresh[~np.isin(fresh,train)]
    boundary=np.array([np.nextafter(TAU,0.),TAU,np.nextafter(TAU,np.inf)])
    near=(bits+np.arange(-131072,131073,dtype=np.int64)).view(np.float64)
    # Independent scalar holdout coverage; no new stationary/root experiment.
    new=np.unique(np.r_[10.**rng.uniform(-323,0,16384),rng.uniform(0,1,8192),
        (bits+rng.integers(-2**24,2**24,16384,dtype=np.int64)).view(np.float64),
        TAU*rng.uniform(0,2,8192)])
    new=new[~np.isin(new,train)]
    sets=[('training',train),('all_previous_archived_gaps',archive),('heldout_archived_gap_values',archive_held),
          ('all_fresh_generated_gaps',fresh),('heldout_fresh_gap_values',fresh_held),('new_synthetic_heldout',new),
          ('adjacent_float_boundary',boundary),('dense_boundary_262145_values',near),
          ('heldout_dense_boundary',near[~np.isin(near,train)]),
          ('ordinary_r_at_least_1e-12',np.r_[archive[archive>=1e-12],fresh[fresh>=1e-12],new[new>=1e-12]]),
          ('near_boundary_r_in_0_to_2tau',np.r_[archive[archive<=2*TAU],fresh[fresh<=2*TAU],near])]
    for group in sorted(set(fresh_groups)):sets.append(('fresh_source_'+group,fresh[fresh_groups==group]))
    metrics=[];boundary_rows=[];failures=[];largest_trust=None;largest_defer=None
    for name,r in sets:
        row,logit,pred=evaluate(model,r,name);metrics.append(row)
        if name=='adjacent_float_boundary':
            for value,l,p in zip(r,logit,pred):boundary_rows.append(dict(r=float(value),r_hex=float(value).hex(),logit=float(l),teacher='TRUST' if value>TAU else 'DEFER',prediction='TRUST' if p else 'DEFER'))
        for i in np.flatnonzero(pred!=(r>TAU))[:20]:failures.append(dict(group=name,r=float(r[i]),logit=float(logit[i]),teacher_trust=bool(r[i]>TAU),predicted_trust=bool(pred[i])))
    write_csv(OUT/'stationary_reliability_threshold_validation.csv',metrics)
    write_csv(OUT/'stationary_reliability_threshold_boundary.csv',boundary_rows)
    if failures:write_csv(OUT/'classification_failures.csv',failures)
    passed=bool(all(r['classification_errors']==r['nonfinite_logits']==0 for r in metrics))
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(operator='stationary_reliability_threshold',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        learned_theta=float(model.theta),learned_theta_hex=float(model.theta).hex(),learned_alpha=float(model.alpha()),raw_alpha=float(model.raw_alpha),
        teacher_tau=TAU,tau_unchanged=True,initial=initial,checkpoint=str(checkpoint),sha256=digest,
        module='experiments.geometry.train_stationary_reliability_threshold',model='ReliabilityThreshold',
        metrics=metrics,boundary=boundary_rows,smallest_correctly_trusted=min(r['smallest_correctly_trusted'] for r in metrics if r['smallest_correctly_trusted'] is not None),
        largest_correctly_deferred=max(r['largest_correctly_deferred'] for r in metrics if r['largest_correctly_deferred'] is not None),
        interpretation='Validation of the teacher decision r>fixed tau, not a new test of mathematical sign reliability or an assembled three-state router.',
        implementation_sha256=sha(Path(__file__)),all_protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('metrics','boundary')},indent=2))
    (OUT/'stationary_reliability_threshold_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2),flush=True)


if __name__=='__main__':run()
