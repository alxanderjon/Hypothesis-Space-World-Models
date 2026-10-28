"""Only two-weight continuous magnitude comparison; no threshold or root solve."""
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.validate_full_derivative import Inputs, case_matrix
from experiments.geometry.validate_stationary_margin import margin_teacher
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'stationary_magnitude_margin_smoke'
SEED=27532026
EPS=np.finfo(np.float64).eps


class MagnitudeMargin(nn.Module):
    """Independent bias-free coefficients on deterministic normalized magnitudes."""
    def __init__(self):
        super().__init__(); self.weights=nn.Parameter(.3*torch.randn(2,dtype=torch.float64))

    def forward(self,x):
        return self.weights[0]*x[...,0]+self.weights[1]*x[...,1]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Data(Inputs):
    def geometry(self,r,axis,k,c,scope,case):
        self.parent_context=r
        super().geometry(r,axis,k,c,scope,case)

    def add(self,scope,case,c,a,B,tcode,vcode,order='generic'):
        super().add(scope,case,c,a,B,tcode,vcode,order)
        b=self.batches[-1]
        if scope=='archived_generic_subpieces':
            r=self.generic[str(case)]
            key={k:r[k] for k in ('A','B','a','beta','t','v','lower','upper')}
            held='heldout' in str(case) or 'holdout' in str(case) or 'selection' in str(case)
        else:
            if scope=='archived_subpiece_grids': r=self.parents[case[0]]; held=case[0]>=1280
            else:
                r=self.parent_context
                held=(int(str(case).split(':')[0])>=1280 if scope=='archived_interval_grids' else
                      any(tag in str(case) for tag in ('heldout','holdout','selection')))
            key={k:r[k] for k in ('lower','upper','n','d')}
        b['group_key']=fingerprint(key); b['reserved']=held


def predict(model,x):
    with torch.no_grad(): return model(torch.from_numpy(x)).numpy()


def metrics(x,p):
    y=x[:,0]-x[:,1]; error=p-y; signs=np.sign(p)!=np.sign(y)
    return dict(cases=len(x),rmse=float(np.sqrt(np.mean(error*error))),mae=float(np.mean(abs(error))),
        max_absolute_error=float(abs(error).max()),exact_matches=int(np.sum(p==y)),nonfinite=int(np.sum(~np.isfinite(p))),
        sign_disagreements=int(signs.sum()),opposite_signs=int(np.sum(p*y<0)),
        teacher_zero_learned_nonzero=int(np.sum((y==0)&(p!=0))),learned_zero_teacher_nonzero=int(np.sum((p==0)&(y!=0))),
        sign_disagreements_away_from_numerical_ties=int(np.sum(signs&(abs(y)>32*EPS))),
        smallest_nonzero_teacher_margin=float(abs(y[y!=0]).min()) if np.any(y!=0) else None)


def run():
    start=time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    assembly=ROOT/'stationary_margin_assembly/stationary_margin_validated.json'
    manifest=json.loads(assembly.read_text()); assert manifest['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE'
    protected_other={str(assembly):sha(assembly),str(Path('experiments/geometry/validate_stationary_margin.py')):sha(Path('experiments/geometry/validate_stationary_margin.py'))}
    assert protected_other[str(Path('experiments/geometry/validate_stationary_margin.py'))]==manifest['implementation_sha256']
    source=Data(); source.collect()
    xs=[]; groups=[]; info=[]; reserved={}; offset=0; source_counts={}; global_min=np.inf; global_max=0.
    def magnitudes(m,key,held,scope,case):
        nonlocal offset,global_min,global_max
        assert np.all(np.isfinite(m)) and np.all(m>0)
        scale=m.max(1); x=m/scale[:,None]
        if not len(x): return
        global_min=min(global_min,float(m.min())); global_max=max(global_max,float(m.max()))
        group=len(groups); groups.append(key); reserved[key]=reserved.get(key,False) or held or int(key[:8],16)%5==0
        xs.append(x); info.append(dict(start=offset,end=offset+len(x),group=group,scope=scope,case=str(case)))
        offset+=len(x); source_counts[scope]=source_counts.get(scope,0)+len(x)
    def add(data,key,held,scope,case):
        terms=margin_teacher(data)[:,1:]
        mask=np.sign(terms[:,0])*np.sign(terms[:,1])<0
        if mask.any(): magnitudes(abs(terms[mask]),key,held,scope,case)
    for b in source.batches:
        x=np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov']))
        add(x,b['group_key'],b['reserved'],b['scope'],b['case'])
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
        add(case_matrix(coeff,grid),b['group_key'],b['reserved'],'dense_subpiece_and_root_neighborhoods',b['case'])
    del source.batches
    real_min,real_max=global_min,global_max
    # Disjoint synthetic scales; all normalization/absolute values are deterministic.
    rng=np.random.default_rng(SEED)
    for held in (False,True):
        exponents=np.arange(-900,901); exponents=exponents[(exponents%5==0)==held]
        for kind in ('broad','near_equal','equal','one_ulp'):
            base=np.ldexp(rng.uniform(1,2,4096),rng.choice(exponents,4096))
            if kind=='broad': other=base*np.exp(rng.uniform(-35,0,4096))
            elif kind=='near_equal': other=base*(1-np.power(10.,rng.uniform(-17,-3,4096)))
            elif kind=='equal': other=base.copy()
            else: other=np.nextafter(base,0.)
            pair=np.column_stack((base,other)); pair[::2]=pair[::2,::-1]
            key=fingerprint(dict(synthetic=kind,heldout=held))
            magnitudes(pair,key,held,'synthetic_'+kind,('heldout' if held else 'train'))
            reserved[key]=held  # Entire synthetic exponent bands withheld.
    x=np.concatenate(xs); del xs
    split=np.empty(len(x),bool); tags=np.empty(len(x),np.int16)
    scopes=sorted(source_counts); scope_ids={s:i for i,s in enumerate(scopes)}
    for row in info:
        sl=slice(row['start'],row['end']); split[sl]=reserved[groups[row['group']]]; tags[sl]=scope_ids[row['scope']]
    train_keys={groups[r['group']] for r in info if not reserved[groups[r['group']]]}
    test_keys={groups[r['group']] for r in info if reserved[groups[r['group']]]}
    assert train_keys.isdisjoint(test_keys)
    base=x[~split]
    anchors=np.unique(np.r_[1.,base[np.linspace(0,len(base)-1,min(len(base),2048),dtype=int)].ravel()])
    eq=np.column_stack((anchors,anchors))
    # Swapped targets supply equal + / - weight. Extra identity rows explicitly
    # supervise m(x)+m(swap(x))=0 without tying or constraining either coefficient.
    sym=base.sum(1); symmetry=np.column_stack((sym,sym))*.5
    design=np.concatenate((base,base[:,::-1],eq,symmetry))
    targets=np.r_[base[:,0]-base[:,1],base[:,1]-base[:,0],np.zeros(len(eq)+len(symmetry))]
    model=MagnitudeMargin(); initial=model.weights.detach().numpy().copy()
    assert initial[0]!=1 and initial[1]!=-1
    history=[]; best=(np.inf,initial.copy())
    # Full-batch supervised least-squares residual updates from random weights.
    # No coefficient targets, zeroing, snapping, or analytic formula assignment.
    for update in range(9):
        pred=predict(model,design); residual=targets-pred; loss=float(np.mean(residual*residual))
        w=model.weights.detach().numpy().copy()
        history.append(dict(update=update,w1=w[0],w2=w[1],rmse=float(np.sqrt(loss)),max_error=float(abs(residual).max())))
        if loss<best[0]: best=(loss,w.copy())
        if loss==0 or update==8: break
        delta=np.linalg.lstsq(design,residual,rcond=None)[0]; proposal=w+delta
        if np.array_equal(w,proposal): break
        with torch.no_grad(): model.weights.copy_(torch.from_numpy(proposal))
    with torch.no_grad(): model.weights.copy_(torch.from_numpy(best[1]))
    config=dict(seed=SEED,dtype='float64',bias=False,model='w1*x1+w2*x2',initial_weights=initial.tolist(),
        teacher='x1-x2 after deterministic abs/max normalization',optimizer='Full-batch supervised least-squares residual corrections; maximum 8 updates',
        source_counts=source_counts,original_pairs=len(x),train_original_pairs=len(base),heldout_original_pairs=int(split.sum()),
        training_rows=len(design),swap_rows=len(base),symmetry_rows=len(symmetry),equality_rows=len(eq),
        train_source_groups=len(train_keys),heldout_source_groups=len(test_keys),source_groups_disjoint=True,
        split_rule='Archive heldout/selection geometry or coefficient groups reserved; additional SHA256 group modulo 5 holdout. All members of a duplicate group share reservation. Synthetic exponent%5==0 bands held out.',
        real_magnitude_min=real_min,real_magnitude_max=real_max,synthetic_and_real_min=global_min,synthetic_and_real_max=global_max,
        numerical_tie_definition='0<abs(x1-x2)<=32*float64_eps; diagnostic only, no threshold trained',absolute_precision_gate=32*EPS,
        no_coefficient_snapping=True,no_threshold_training=True,no_root_solving=True)
    checkpoint=OUT/'stationary_magnitude_margin_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),config=config,status='SAVED_BEFORE_EVALUATION',training_history=history))
    write_csv(OUT/'training_history.csv',history); (OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(event='saved_before_evaluation',weights=best[1].tolist(),initial=initial.tolist(),sha256=digest,
        training_rows=len(design),heldout_pairs=int(split.sum()))),flush=True)
    del design,targets,symmetry,base
    model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
    rows=[]; properties=[]
    for role,mask in [('all',np.ones(len(x),bool)),('train',~split),('heldout',split)]:
        xx=x[mask]; p=predict(model,xx); y=xx[:,0]-xx[:,1]
        rows.append(dict(role=role,group='all',**metrics(xx,p)))
        categories={'ordinary':abs(y)>1e-8,'near_tie':(abs(y)<=1e-8)&(y!=0),'numerical_tie':(abs(y)<=32*EPS)&(y!=0),'exact_teacher_zero':y==0}
        for name,subset in categories.items():
            if subset.any(): rows.append(dict(role=role,group=name,**metrics(xx[subset],p[subset])))
        swapped=predict(model,np.ascontiguousarray(xx[:,::-1])); equal=predict(model,np.column_stack((xx[:,0],xx[:,0])))
        properties.append(dict(role=role,antisymmetry_max_error=float(abs(p+swapped).max()),equality_anchor_max_error=float(abs(equal).max())))
    for scope,sid in scope_ids.items():
        mask=split&(tags==sid)
        if mask.any(): rows.append(dict(role='heldout',group=scope,**metrics(x[mask],predict(model,x[mask]))))
    exact_examples=np.array([[1.,1.],[1.,np.nextafter(1.,0.)],[np.nextafter(1.,0.),1.],[.5,.5]])
    write_csv(OUT/'boundary_examples.csv',[dict(x1=a,x2=b,teacher=a-b,learned=p) for (a,b),p in zip(exact_examples,predict(model,exact_examples))])
    write_csv(OUT/'stationary_magnitude_margin_validation.csv',rows); write_csv(OUT/'symmetry_equality.csv',properties)
    p=predict(model,x); y=x[:,0]-x[:,1]; bad=np.flatnonzero(np.sign(p)!=np.sign(y))
    write_csv(OUT/'sign_disagreements.csv',[dict(index=int(i),heldout=bool(split[i]),x1=x[i,0],x2=x[i,1],teacher=y[i],learned=p[i],numerical_tie=abs(y[i])<=32*EPS) for i in bad[:1000]] or [dict(status='none')])
    unchanged=all(sha(Path(p))==h for p,h in {**protected,**protected_other,**source.inputs}.items()); assert unchanged
    overall=rows[0]; passed=bool(overall['max_absolute_error']<=32*EPS and overall['nonfinite']==0
        and overall['sign_disagreements_away_from_numerical_ties']==0
        and max(r['antisymmetry_max_error'] for r in properties)<=32*EPS
        and max(r['equality_anchor_max_error'] for r in properties)<=32*EPS)
    result=dict(operator='stationary_magnitude_margin',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        learned_weights=model.weights.tolist(),weight_sum=float(model.weights.sum()),bias=False,checkpoint=str(checkpoint),sha256=digest,
        module='experiments.geometry.train_stationary_magnitude_margin',model='MagnitudeMargin',implementation_sha256=sha(Path(__file__)),
        metrics=rows,properties=properties,training_history=history,protected_checkpoints=len(protected),
        all_prior_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-start,no_threshold_trained=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps({**protected,**protected_other},indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(source.inputs,indent=2))
    if passed: (OUT/'candidate_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('metrics','properties','training_history')},indent=2))
    (OUT/'stationary_magnitude_margin_summary.txt').write_text(json.dumps(result,indent=2)+'\nOnly independent comparison weights trained. No threshold, root finding, recursion, or search.\n')
    print(json.dumps(dict(status=result['status'],weights=result['learned_weights'],weight_sum=result['weight_sum'],overall=overall,
        properties=properties,updates=len(history)-1,sha256=digest,seconds=result['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__': run()
