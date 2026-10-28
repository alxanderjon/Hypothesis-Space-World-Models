"""Fresh holdout only, immutable tau, no iterative root or bracket search."""
import json
import time
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
import numpy as np
import torch
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_full_derivative import case_matrix
from experiments.sho.train_recursive_region_splitter_poc import LockedAllocation

OUT=ref.ROOT/'fixed_sign_reliability_fresh_holdout'
TAU=4.735309043366555e-16
SEED=202609280731


def generate():
    rng=np.random.default_rng(SEED);mp.mp.dps=90
    arrays=[];labels=[];ids=[];roots=[];subpieces=[]
    def add(x,label,case):
        x=np.asarray(x,dtype=np.float64).reshape(-1,7)
        keep=np.sign(x[:,0])*np.where(x[:,5]==0,1,-1)*np.sign(x[:,3])*np.where(x[:,6]==0,1,-1)<0
        x=x[keep]
        arrays.append(x);labels.extend([label]*len(x));ids.extend([case]*len(x))
    # New broad ordinary combinations; no coefficients copied from archives.
    n=32768;A=10.**rng.uniform(-5,5,n);a=10.**rng.uniform(-5,5,n);B=a*a*10.**rng.uniform(-6,6,n)
    ot=rng.integers(0,2,n);ov=rng.integers(0,2,n);beta=-np.where(ot==ov,1,-1)*A*a*10.**rng.uniform(-4,1,n)
    c=rng.uniform(.00001,.99999,n);x=np.column_stack((A,B,a,beta,c,ot,ov))
    add(x,'ordinary','new_independent_coefficients')
    # Same-orientation roots from a closed algebraic quadratic expression. No
    # bracketing, iteration, findroot, or numerical optimization is performed.
    for i in range(768):
        A=10.**rng.uniform(-5,5);a=10.**rng.uniform(-5,5);B=a*a*10.**rng.uniform(-6,6);o=i%2
        planted=float(rng.uniform(.01,.99));tt=mp.mpf(planted)
        beta=-float(mp.mpf(A)*mp.mpf(a)**2*tt**2/mp.sqrt(mp.mpf(B)+mp.mpf(a)**2*tt**2))
        k=abs(mp.mpf(beta))/(mp.mpf(A)*mp.mpf(a));b=mp.mpf(B)/mp.mpf(a)**2
        tstar=mp.sqrt((k*k+mp.sqrt(k**4+4*k*k*b))/2);star=tstar if o==0 else 1-tstar
        if 0<star<1:roots.append(dict(case=f'algebraic_same_{i}',coeff=(A,B,a,beta,o,o),root=mp.nstr(star,85),method='closed algebraic expression'))
    # Exact dyadic construction covers all four orientation combinations with
    # B>0. B=a^2(h^2-t^2), beta=-tprime/vprime*A*a*t*v/h.
    # Every quantity is exactly representable here; exact rational verification
    # checks that the planted coordinate really is a mathematical stationary point.
    for i in range(1024):
        ot=(i//2)%2;ov=i%2;c=float(rng.integers(41,4055))/4096
        A=2.**int(rng.integers(-16,17));a=2.**int(rng.integers(-16,17));h=2.**int(rng.integers(0,4))
        t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
        B=a*a*(h*h-t*t);beta=-(1 if ot==ov else -1)*A*a*t*v/h
        af,bf,aa,be,cc=map(F.from_float,(A,B,a,beta,c));tf=cc if ot==0 else 1-cc;vf=cc if ov==0 else 1-cc
        assert (af*aa*aa*tf*vf)**2==be*be*(bf+aa*aa*tf*tf)
        roots.append(dict(case=f'exact_planted_{i}',coeff=(A,B,a,beta,ot,ov),root=mp.nstr(mp.mpf(c),85),method='exact dyadic stationary construction'))
    # B=0 crossed orientations also have a direct linear root expression.
    for i in range(256):
        ot=i%2;ov=1-ot;A=10.**rng.uniform(-5,5);a=10.**rng.uniform(-5,5)
        beta=float(A*a*rng.uniform(.01,.99));vstar=mp.mpf(beta)/(mp.mpf(A)*mp.mpf(a));star=vstar if ov==0 else 1-vstar
        if 0<star<1:roots.append(dict(case=f'algebraic_crossed_{i}',coeff=(A,0.,a,beta,ot,ov),root=mp.nstr(star,85),method='closed linear expression'))
    for i,r in enumerate(roots):
        center=float(r['root']);width=min(center,1-center)*10.**rng.uniform(-7,-3)
        grid=np.linspace(center-width,center+width,129);grid=grid[(grid>0)&(grid<1)]
        add(case_matrix(r['coeff'],grid),'near_root',r['case'])
        near=(np.float64(center).view(np.int64)+np.arange(-64,65,dtype=np.int64)).view(np.float64)
        near=near[(near>0)&(near<1)];add(case_matrix(r['coeff'],near),'one_ULP_neighborhood',r['case'])
        if i%4==0:
            coef=list(r['coeff']);beta0=coef[3]
            # New coefficient combinations around the cancellation surface;
            # none is used to solve for a new root.
            for j in (-4,-2,-1,1,2,4):
                b=beta0
                for _ in range(abs(j)):b=float(np.nextafter(b,np.inf if j>0 else -np.inf))
                coef[3]=b;cuts=(np.float64(center).view(np.int64)+np.arange(-2,3,dtype=np.int64)).view(np.float64)
                add(case_matrix(coef,cuts),'severe_cancellation',r['case']+f':beta_offset_{j}')
    # Fresh boxes and fixed-allocation/active-branch subsegments. Conservative
    # interior segments of constant-label grid runs suffice; no maximal interval
    # discovery and no stationary-point solver is called.
    allocator=LockedAllocation();audit=ref.Audit.__new__(ref.Audit)
    for pi in range(160):
        d=2+pi%4;n=int(rng.integers(2,21)) if pi%5 else int(rng.choice([29,73,128,251]))
        sides=10.**(rng.uniform(-5,5)+rng.uniform(-1.5,1.5,d));axis=int(rng.integers(d))
        parent=dict(lower=[0.]*d,upper=sides.tolist(),d=d,n=n);audit.setup(parent,axis)
        grid=np.linspace(.001,.999,257);ell=audit.geometry(grid)
        counts=allocator.allocate(ell/sides,np.full(len(grid),n))[0]
        tags=[]
        for c,cc in zip(grid,counts):
            k=int(cc[0])
            if min(cc)<1:tags.append(None);continue
            vals=audit.formula([c],k);iq=int(vals[3][0]);ia=int(vals[4][0]);sg=int(vals[5][0,ia]);tags.append((k,iq,ia,sg))
        start=0
        while start<len(grid):
            end=start+1
            while end<len(grid) and tags[end]==tags[start]:end+=1
            tag=tags[start]
            if tag is not None and end-start>=5:
                k,iq,ia,sg=tag;lo=grid[start+1];hi=grid[end-2];samples=np.linspace(lo,hi,17)
                cc=allocator.allocate(audit.geometry(samples)/sides,np.full(len(samples),n))[0]
                vals=audit.formula(samples,k)
                assert np.all(cc==[k,n-k]) and np.all(vals[3]==iq) and np.all(vals[4]==ia) and np.all(vals[5][:,ia]==sg)
                nj=k if iq==0 else n-k;A=1/(audit.h*nj**(1/d));beta=.05*sg
                coeff=(A,audit.B,audit.a,beta,iq,ia);key=f'box_{pi}_segment_{start}'
                if (1 if iq==0 else -1)*np.sign(beta)*(1 if ia==0 else -1)<0:
                    subpieces.append(dict(case=key,parent=parent,axis=axis,counts=[k,n-k],lower=float(lo),upper=float(hi),coeff=coeff))
                    add(case_matrix(coeff,samples),'ordinary_fixed_allocation_subpiece',key)
            start=end
    data=np.concatenate(arrays);labels=np.array(labels);ids=np.array(ids)
    order=rng.permutation(len(data))  # fixed mixing avoids source-order bias if stopped
    return data[order],labels[order],ids[order],roots,subpieces


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ref.ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ref.ROOT.rglob('*manifest.json')})
    manifest=json.loads((ref.OUT/'candidate_manifest.json').read_text());cp=Path(manifest['checkpoint']);assert ref.sha(cp)==manifest['sha256']
    sqrt=ref.ScalarPower();sqrt.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);sqrt.eval().requires_grad_(False);_,router=ref.routers(sqrt)
    x,groups,cases,roots,subpieces=generate()
    (OUT/'fresh_root_constructions.json').write_text(json.dumps(roots,indent=2));(OUT/'fresh_fixed_allocation_subpieces.json').write_text(json.dumps(subpieces,indent=2))
    # The previous pool is read only to certify disjointness, never for labels,
    # parameter selection, ranges, or adaptation of this fixed new sample plan.
    prior_path=ref.ROOT/'stationary_sign_separability_audit/sign_reliability_values.npz'
    prior=np.load(prior_path)['inputs'];protected[str(prior_path)]=ref.sha(prior_path)
    dtype=np.dtype((np.void,7*8));oldkeys=np.ascontiguousarray(prior).view(dtype).ravel();newkeys=np.ascontiguousarray(x).view(dtype).ravel()
    overlap=np.isin(newkeys,oldkeys);overlap_count=int(overlap.sum());del prior,oldkeys
    if overlap_count:
        x=x[~overlap];groups=groups[~overlap];cases=cases[~overlap];newkeys=newkeys[~overlap]
    unique_generated=len(np.unique(newkeys));n=len(x)
    np.savez_compressed(OUT/'fresh_holdout_inputs.npz',inputs=x,group=groups,case=cases)
    config=dict(seed=SEED,tau=TAU,tau_hex=TAU.hex(),generated_cases=n,unique_generated_inputs=unique_generated,
        previous_audit_overlap_removed=overlap_count,new_fixed_allocation_subpieces=len(subpieces),fresh_stationary_constructions=len(roots),
        no_iterative_root_search=True,root_generation='Closed algebraic/linear formulas and exactly planted dyadic roots; no optimization/root iteration.',
        order='Fixed seeded shuffle; stop after first failing evaluation batch, with all failures in that batch saved.',batch_size=16384)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(event='fresh_pool_ready',**config)),flush=True)
    allterms=[];allgap=[];allpred=[];allteacher=[];evaluated=0;fallbacks=0
    with torch.no_grad():
        for begin in range(0,n,16384):
            end=min(begin+16384,n);xx=x[begin:end];terms=router.terms(torch.from_numpy(xx)).numpy()
            pred=router.route_terms(torch.from_numpy(terms)).numpy().astype(np.int8)
            mag=abs(terms);gap=abs(mag[:,0]-mag[:,1])/mag.max(1)
            teacher,fallback=ref.mathematical_sign(xx);fallbacks+=fallback
            bad=(pred!=teacher)|(teacher==0);trusted=gap>TAU
            allterms.append(terms);allgap.append(gap);allpred.append(pred);allteacher.append(teacher);evaluated=end
            if np.any(bad&trusted) or not np.isfinite(gap).all():break
    terms=np.concatenate(allterms);gap=np.concatenate(allgap);pred=np.concatenate(allpred);teacher=np.concatenate(allteacher)
    bad=(pred!=teacher)|(teacher==0);trusted=gap>TAU
    def metric(ids,group):
        b=bad[ids];tr=trusted[ids];g=gap[ids]
        return dict(group=group,cases=len(ids),trusted=int(tr.sum()),deferred=int((~tr).sum()),trusted_failures=int(np.sum(b&tr)),
            reliable_deferred=int(np.sum(~b&~tr)),ambiguous_total=int(b.sum()),reliable_total=int((~b).sum()),
            maximum_ambiguous_r=float(g[b].max()) if b.any() else None,minimum_reliable_r=float(g[~b].min()) if (~b).any() else None,
            exact_teacher_zeros=int(np.sum(teacher[ids]==0)),false_ZERO=int(np.sum((teacher[ids]!=0)&(pred[ids]==0))))
    metrics=[metric(np.arange(evaluated),'all_evaluated')]
    for group in sorted(set(groups)):metrics.append(metric(np.flatnonzero(groups[:evaluated]==group),group))
    ref.write_csv(OUT/'fixed_tau_holdout_validation.csv',metrics)
    failures=[]
    for i in np.flatnonzero(bad&trusted):
        row=x[i];A,B,a,beta,c,ot,ov=row;t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
        af,bf,aa,be,cc=map(F.from_float,(A,B,a,beta,c));tf=cc if ot==0 else 1-cc;vf=cc if ov==0 else 1-cc
        X=(af*aa*aa*tf*vf)**2;Y=be*be*(bf+aa*aa*tf*tf);diff=X-Y
        tx=torch.from_numpy(row[None,:]);m=router.components
        with torch.no_grad():
            tt=m['orientation'](torch.tensor([[c,ot]],dtype=torch.float64));vv=m['orientation'](torch.tensor([[c,ov]],dtype=torch.float64))
            q=m['quadratic'](torch.stack((tx[:,1],tx[:,2],tt),1));rr=m['sqrt'](q)
        failures.append(dict(index=int(i),group=str(groups[i]),case=str(cases[i]),A=A,B=B,a=a,beta=beta,c=c,c_hex=float(c).hex(),ot=int(ot),ov=int(ov),
            learned_t=float(tt[0]),learned_v=float(vv[0]),learned_q=float(q[0]),learned_sqrt=float(rr[0]),direct_float64_sqrt=float(np.sqrt(float(q[0]))),
            T1=terms[i,0],T2=terms[i,1],T1_hex=float(terms[i,0]).hex(),T2_hex=float(terms[i,1]).hex(),
            r=gap[i],r_hex=float(gap[i]).hex(),tau=TAU,learned_sign=int(pred[i]),mathematical_sign=int(teacher[i]),
            exact_squared_difference_numerator=str(diff.numerator),exact_squared_difference_denominator=str(diff.denominator),
            exact_normalized_squared_difference=float(diff/max(X,Y))))
    if failures:ref.write_csv(OUT/'trusted_sign_failures.csv',failures)
    np.savez_compressed(OUT/'evaluated_holdout_results.npz',inputs=x[:evaluated],group=groups[:evaluated],case=cases[:evaluated],terms=terms,r=gap,learned_sign=pred,teacher_sign=teacher)
    passed=bool(not failures and evaluated==n and np.isfinite(gap).all())
    result=dict(status='fixed reliability screen validated on fresh holdout' if passed else 'FAILED_STOP',passed=passed,**config,
        evaluated_cases=evaluated,unique_evaluated_inputs=len(np.unique(newkeys[:evaluated])),unevaluated_cases=n-evaluated,
        metrics=metrics,tau_unchanged=True,exact_rational_fallbacks=fallbacks,nonfinite_gap_cases=int(np.sum(~np.isfinite(gap))),
        no_training=True,no_checkpoint_changes=True,no_root_search=True,implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started)
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    result['protected_hashes_unchanged']=True
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'fixed_tau_holdout_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
