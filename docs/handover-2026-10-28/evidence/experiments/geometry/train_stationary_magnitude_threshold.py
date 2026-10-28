"""Threshold-only smoke: frozen magnitude margin, explicit exact-zero TIE routing."""
import ast
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.train_stationary_magnitude_margin import Data, MagnitudeMargin, fingerprint
from experiments.geometry.validate_full_derivative import case_matrix
from experiments.geometry.validate_stationary_margin import margin_teacher
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'stationary_magnitude_threshold_smoke'
SEED=27542026


class MagnitudeThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(.3*torch.randn((),dtype=torch.float64))
        self.raw_alpha=nn.Parameter(.3*torch.randn((),dtype=torch.float64))

    def alpha(self): return torch.nn.functional.softplus(self.raw_alpha)

    def forward(self,m): return self.alpha()*(m-self.theta)

    def decision(self,m):
        # Equality is explicit deterministic bookkeeping; zero gets no binary label.
        return torch.where(m==0,torch.zeros_like(m),torch.sign(self(m)))


def load_data(frozen):
    source=Data(); source.collect(); arrays=[]; groups=[]; counts={}; reserved={}
    def add(x,b,scope):
        terms=margin_teacher(x)[:,1:]; mask=np.sign(terms[:,0])*np.sign(terms[:,1])<0
        if not mask.any(): return
        mags=abs(terms[mask]); norm=mags/mags.max(1)[:,None]
        with torch.no_grad(): margin=frozen(torch.from_numpy(norm)).numpy()
        assert np.array_equal(margin,norm[:,0]-norm[:,1])
        arrays.append(margin); key=b['group_key']
        reserved[key]=reserved.get(key,False) or b['reserved'] or int(key[:8],16)%5==0
        groups.append(dict(key=key,scope=scope,cases=len(margin),case=b['case']))
        counts[scope]=counts.get(scope,0)+len(margin)
    for b in source.batches:
        x=np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov']))
        add(x,b,b['scope'])
        if b['scope'] not in ('archived_subpiece_grids','archived_generic_subpieces'): continue
        if b['scope']=='archived_subpiece_grids':
            row=source.piece_map[ast.literal_eval(b['case'])]; lo=float(row['lower']); hi=float(row['upper'])
            roots=json.loads(row['stationary_points'])+json.loads(row['quartic_stationary_roots'])
        else:
            row=source.generic[b['case']]; lo=row['lower']; hi=row['upper']; roots=row['roots']
        grid=list(np.linspace(lo,hi,129))
        for center in sorted(set(roots)):
            if not lo<=center<=hi: continue
            width=max((hi-lo)*1e-6,128*np.spacing(center))
            grid.extend(np.linspace(max(lo,center-width),min(hi,center+width),257))
            grid.extend((np.float64(center).view(np.int64)+np.arange(-64,65,dtype=np.int64)).view(np.float64))
        grid=np.unique(grid); grid=grid[(grid>=lo)&(grid<=hi)]
        coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0])
        add(case_matrix(coeff,grid),b,'dense_root_neighborhoods')
    held=np.concatenate([np.full(g['cases'],reserved[g['key']],bool) for g in groups])
    allm=np.concatenate(arrays)
    train_keys={g['key'] for g in groups if not reserved[g['key']]}
    test_keys={g['key'] for g in groups if reserved[g['key']]}
    assert train_keys.isdisjoint(test_keys)
    return allm,held,groups,source.inputs,counts


def evaluate(model,m,name):
    with torch.no_grad():
        x=torch.from_numpy(m); logit=model(x).numpy(); p=model.decision(x).numpy()
        opposite=model.decision(-x).numpy()
    truth=np.sign(m); bad=p!=truth; nz=m!=0; near=nz&(abs(m)<=1e-8)
    correct=p==truth
    return dict(group=name,cases=len(m),mistakes=int(bad.sum()),nonzero_cases=int(nz.sum()),
        nonzero_mistakes=int(np.sum(bad&nz)),ordinary_cases=int(np.sum(abs(m)>1e-8)),ordinary_mistakes=int(np.sum(bad&(abs(m)>1e-8))),
        near_zero_cases=int(near.sum()),near_zero_mistakes=int(np.sum(bad&near)),
        exact_zero_cases=int(np.sum(~nz)),exact_zero_routing_errors=int(np.sum(bad&~nz)),
        symmetry_errors=int(np.sum(p!=-opposite)),nonfinite_logits=int(np.sum(~np.isfinite(logit))),
        nonzero_margin_zero_logit=int(np.sum(nz&(logit==0))),
        smallest_correct_positive=float(m[correct&(m>0)].min()) if np.any(correct&(m>0)) else None,
        largest_correct_negative=float(m[correct&(m<0)].max()) if np.any(correct&(m<0)) else None),logit,p


def run():
    start=time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    mp=ROOT/'stationary_magnitude_margin_smoke/candidate_manifest.json'; manifest=json.loads(mp.read_text())
    assert manifest['status']=='FROZEN_CANDIDATE'
    cp=Path(manifest['checkpoint']); assert sha(cp)==manifest['sha256']
    frozen=MagnitudeMargin(); frozen.load_state_dict(torch.load(cp,weights_only=False)['state_dict']); frozen.eval().requires_grad_(False)
    protected_extra={str(mp):sha(mp)}
    for p in [ROOT/'stationary_margin_assembly/stationary_margin_validated.json',Path('experiments/geometry/validate_stationary_margin.py'),Path('experiments/geometry/train_stationary_magnitude_margin.py')]:protected_extra[str(p)]=sha(p)
    margin,held,groups,input_hashes,counts=load_data(frozen)
    # All archived training magnitudes receive equal positive and negative weight.
    # Unique compression with exact occurrence counts preserves the full objective.
    real=abs(margin[(~held)&(margin!=0)])
    synthetic=np.unique(np.r_[np.power(10.,np.arange(-323,1,2)),np.ldexp(np.ones(1075),np.arange(-1074,1)),
        [1e-18,1e-16,1e-14,1e-12,1e-9,1e-6,1e-3,1e-1,1.]])
    mags,weights=np.unique(np.r_[real,synthetic],return_counts=True)
    mags=torch.from_numpy(mags); weights=torch.from_numpy(weights.astype(float)); weights=weights/weights.sum()
    model=MagnitudeThreshold(); initial=dict(theta=float(model.theta.detach()),alpha=float(model.alpha().detach()),raw_alpha=float(model.raw_alpha.detach()))
    assert initial['theta']!=0 and initial['alpha']>0
    optimizer=torch.optim.Adam(model.parameters(),lr=.05); history=[]
    # Binary labels only for nonzero margins; logits at zero have a separate
    # supervised zero anchor. Symmetry is enforced as a loss, never a parameter tie.
    for step in range(32):
        optimizer.zero_grad()
        pos=model(mags); neg=model(-mags)
        logistic=(weights*(torch.nn.functional.softplus(-pos)+torch.nn.functional.softplus(neg))*.5).sum()
        symmetry=(weights*((pos+neg)*.5)**2).sum()
        anchor=model(torch.zeros((),dtype=torch.float64))**2
        loss=logistic+symmetry+anchor
        loss.backward(); optimizer.step()
        history.append(dict(phase='balanced_logistic_symmetry_zero_anchor',step=step+1,loss=float(loss.detach()),
            theta=float(model.theta.detach()),alpha=float(model.alpha().detach())))
    # Supervised least-squares residual correction on the exact-zero logit anchor.
    # This updates theta from measured training residual/Jacobian, never assigns
    # or rounds it to the teacher boundary. Alpha remains at its learned value.
    zero_x=torch.zeros(32,dtype=torch.float64)
    for step in range(16):
        logits=model(zero_x); residual=-logits.detach().numpy()
        history.append(dict(phase='zero_anchor_residual_fit',step=step,loss=float(np.mean(residual*residual)),
            max_anchor_residual=float(abs(residual).max()),theta=float(model.theta.detach()),alpha=float(model.alpha().detach())))
        if np.all(residual==0): break
        jac_scalar=float(torch.autograd.grad(logits[0],model.theta)[0])
        delta=np.linalg.lstsq(np.full((len(residual),1),jac_scalar),residual,rcond=None)[0][0]
        previous=float(model.theta.detach())
        with torch.no_grad(): model.theta.add_(float(delta))
        if float(model.theta.detach())==previous: break
    config=dict(seed=SEED,dtype='float64',architecture='logit=softplus(raw_alpha)*(margin-theta)',initial=initial,
        zero_routing='Input margin exactly equal to zero bypasses binary classification and returns TIE, including signed zero.',
        training='32 Adam steps on balanced logistic + explicit symmetry + zero-logit anchor; at most 16 supervised zero-anchor residual updates',
        no_manual_threshold_snapping=True,no_binary_zero_label=True,
        all_archive_and_dense_cases=len(margin),training_archive_cases=int((~held).sum()),heldout_archive_cases=int(held.sum()),
        training_nonzero_magnitudes=len(real),training_synthetic_magnitudes=len(synthetic),source_counts=counts,
        source_split='Same parent/coefficient group reservation as frozen magnitude-margin experiment; disjoint groups.',
        frozen_margin=dict(checkpoint=str(cp),sha256=sha(cp)),no_root_finding=True,no_term_sign_router=True)
    path=OUT/'threshold_before_evaluation.pt'
    digest=save_checkpoint(path,dict(state_dict=model.state_dict(),config=config,training_history=history,status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'training_history.csv',history); (OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(event='saved_before_evaluation',theta=float(model.theta.detach()),alpha=float(model.alpha().detach()),sha256=digest)),flush=True)
    model.load_state_dict(torch.load(path,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
    boundary=np.array([np.nextafter(0.,-np.inf),-0.,0.,np.nextafter(0.,np.inf)])
    subnormal=np.arange(1,4097,dtype=np.uint64).view(np.float64)
    rng=np.random.default_rng(SEED+1)
    positive=np.unique(np.r_[np.power(10.,np.linspace(-323,0,16385)),np.ldexp(rng.uniform(1,2,8192),rng.integers(-1074,0,8192)),np.linspace(1e-12,1e-7,4096)])
    sets=[('all_archived_and_dense',margin),('training_archived',margin[~held]),('heldout_archived',margin[held]),
        ('one_ulp_around_zero',boundary),('first_4096_subnormals_symmetric',np.r_[-subnormal,subnormal]),
        ('unseen_dense_symmetric',np.r_[-positive,positive]),('training_synthetic_symmetric',np.r_[-synthetic,synthetic]),
        ('signed_zero_anchors',np.array([-0.,0.]))]
    rows=[]; boundary_rows=[]; failures=[]
    for name,m in sets:
        row,logit,pred=evaluate(model,m,name); rows.append(row)
        if name in ('one_ulp_around_zero','signed_zero_anchors'):
            boundary_rows += [dict(group=name,margin=float(x),margin_hex=float(x).hex(),logit=float(l),teacher=int(np.sign(x)),decision=int(y)) for x,l,y in zip(m,logit,pred)]
        for i in np.flatnonzero(pred!=np.sign(m))[:20]:failures.append(dict(group=name,margin=float(m[i]),logit=float(logit[i]),teacher=float(np.sign(m[i])),decision=float(pred[i])))
    offset=0; scope_arrays={}
    for g in groups:
        scope_arrays.setdefault(g['scope'],[]).append(margin[offset:offset+g['cases']]);offset+=g['cases']
    for scope,chunks in scope_arrays.items():rows.append(evaluate(model,np.concatenate(chunks),scope)[0])
    write_csv(OUT/'stationary_threshold_validation.csv',rows); write_csv(OUT/'stationary_threshold_boundary.csv',boundary_rows)
    write_csv(OUT/'failures.csv',failures or [dict(status='none')])
    unchanged=all(sha(Path(p))==h for p,h in {**protected,**protected_extra,**input_hashes}.items()); assert unchanged
    passed=all(r['mistakes']==r['symmetry_errors']==r['nonfinite_logits']==0 for r in rows)
    result=dict(operator='stationary_magnitude_threshold',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        learned_theta=float(model.theta),learned_alpha=float(model.alpha()),raw_alpha=float(model.raw_alpha),initial=initial,
        checkpoint=str(path),sha256=digest,module='experiments.geometry.train_stationary_magnitude_threshold',model='MagnitudeThreshold',
        implementation_sha256=sha(Path(__file__)),metrics=rows,config=config,
        raw_zero_logit=float(model(torch.tensor(0.,dtype=torch.float64))),
        protected_checkpoints=len(protected),all_prior_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-start)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps({**protected,**protected_extra},indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(input_hashes,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('metrics','config')},indent=2))
    (OUT/'stationary_threshold_summary.txt').write_text(json.dumps(result,indent=2)+'\nNo root finding, full derivative-sign router, recursion, or search.\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','metrics')},indent=2),flush=True)
    print(json.dumps(rows[:8],indent=2),flush=True)


if __name__=='__main__':run()
