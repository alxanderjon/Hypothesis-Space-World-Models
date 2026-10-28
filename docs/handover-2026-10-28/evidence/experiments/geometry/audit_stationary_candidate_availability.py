"""Read-only availability/source isolation, no learned locator or recursion."""
import csv
import json
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from scipy.optimize import brentq
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_stationary_three_state_router import load_router, DEFER
from experiments.geometry.validate_terminal_candidate_full import original_parents
from experiments.geometry.train_parent_split_selector import safe_json
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.train_subpiece_stationary_solver import roots_high_precision, derivative, score

ROOT=ref.ROOT;OUT=ROOT/'stationary_candidate_availability_audit';SEED=28362026


def scan_roots(fun,l,r):
    # Verbatim availability convention from the current continuous verifier.
    ll=np.nextafter(l,r);rr=np.nextafter(r,l)
    if ll>=rr:return []
    xx=np.linspace(ll,rr,129);yy=np.asarray(fun(xx));roots=[]
    for i in np.flatnonzero(yy[:-1]*yy[1:]<0):
        roots.append(float(brentq(lambda c:float(np.atleast_1d(fun(c))[0]),xx[i],xx[i+1],xtol=5e-324,rtol=1e-15)))
    return sorted(set(roots))


def rec(coeff):
    return dict(zip(('A','B','a','beta','t','v'),coeff[:6]))


@lru_cache(maxsize=None)
def math_roots(coeff):
    return roots_high_precision(rec(coeff))


def mpder(coeff,c):
    A,B,a,beta,ot,ov=coeff[:6];A,B,a,beta,c=map(mp.mpf,(A,B,a,beta,c))
    t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
    return A*a*a*t*(1 if ot==0 else -1)/mp.sqrt(B+a*a*t*t)+beta*(1 if ov==0 else -1)/v


def ordered_steps(x,y):
    return abs(int(np.float64(x).view(np.uint64))-int(np.float64(y).view(np.uint64)))


def run():
    start=time.monotonic();torch.set_num_threads(1);mp.mp.dps=80;OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('manifest','locked','validated'))})
    router,refs=load_router();geometry=Audit();parents=original_parents();pieces=[]
    def geo_piece(group,case,parent,axis,k,l,r,stored=None):
        geometry.setup(parent,axis);a=geometry;f=a.formula(np.array([(l+r)/2]),k)
        iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
        A=1/(a.h*np.power(float(k if iq==0 else parent['n']-k),1/parent['d']))
        C=0. if parent['d']==1 else .05*np.log(a.a/a.omin) if power==1 else .05*np.log(a.omax/a.a) if power==-1 else .05*np.log(a.omax/a.omin)
        pieces.append(dict(group=group,case=case,parent=parent,axis=axis,k=k,l=float(l),r=float(r),
            coeff=(A,a.B,a.a,.05*power,iq,ia),C=float(C),stored_roots=stored))
    archive_path=ROOT/'fixed_allocation_interval_audit/fixed_interval_score_structure.csv'
    archived=list(csv.DictReader(archive_path.open()))
    for p in archived:
        pi,axis,k,j=[int(p[s]) for s in ('parent','axis','interval','piece')]
        geo_piece('archived_geometry',f'{pi}:{axis}:{k}:{j}',parents[pi],axis,k,float(p['lower']),float(p['upper']),json.loads(p['stationary_points']))
    existing=json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text())
    ep=defaultdict(dict)
    for p in csv.DictReader((ROOT/'stationary_terminal_candidate_training/fresh_candidate_sets.csv').open()):
        if p['kind'] in ('left','right'):ep[p['case']][p['kind']]=p
    for case,ends in ep.items():
        p=ends['left'];pi=int(p['parent']);axis=int(p['axis']);k=int(p['k'])
        geo_piece('existing_fresh_geometry',case,existing[str(pi)],axis,k,float(p['teacher_cut']),float(ends['right']['teacher_cut']))
    # Reuse the exact saved failing box plus new independent geometry, not recursion.
    failed=json.loads((ROOT/'learned_continuous_recursive_verification/first_failure.json').read_text())['selector_context']['parent']
    rng=np.random.default_rng(SEED);fresh=[failed]
    for d in (2,3,4):
        for n in (3,4,8,16,32):
            scale=10**rng.uniform(-2,2);s=scale*np.exp(rng.uniform(-1,1,d));lo=scale*rng.uniform(-.2,.2,d)
            fresh.append(dict(d=d,n=n,lower=lo.tolist(),upper=(lo+s).tolist()))
    for pi,parent in enumerate(fresh):
        for axis in range(parent['d']):
            geometry.setup(parent,axis);a=geometry;n=parent['n']
            edge=max(1e-15,4*np.spacing(abs(a.lo[axis])+abs(a.hi[axis]))/a.a)
            xx=np.linspace(edge,1-edge,max(2049,32*n+1));cc=a.counts(xx)[:,0]
            lo=np.array([xx[np.flatnonzero(cc<k)[-1]] for k in range(1,n+1)])
            hi=np.array([xx[np.flatnonzero(cc>=k)[0]] for k in range(1,n+1)])
            for _ in range(60):
                mid=(lo+hi)/2;mask=a.counts(mid)[:,0]<np.arange(1,n+1)
                lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
            for k in range(1,n):
                br=a.breaks(hi[k-1],lo[k],k)
                for j,(l,r) in enumerate(zip(br[:-1],br[1:])):
                    geo_piece('saved_failure_geometry' if pi==0 else 'new_fresh_geometry',f'{pi}:{axis}:{k}:{j}',parent,axis,k,l,r)
    (OUT/'fresh_parents.json').write_text(safe_json(fresh))
    # Explicit coefficient constructions around both interval endpoints. Float64
    # coefficients are fixed first; exact mathematical roots locate the bounds.
    seeds=[]
    for i in range(32):
        A=10**rng.uniform(-4,4);a=10**rng.uniform(-4,4);B=a*a*10**rng.uniform(-2,2)
        ot=i%2;ov=(i//2)%2;c=float(rng.uniform(.15,.85));t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
        beta=-A*a*a*t*(1 if ot==0 else -1)*v/((1 if ov==0 else -1)*np.sqrt(B+a*a*t*t))
        coeff=(A,B,a,beta,ot,ov);rr=math_roots(coeff)
        if rr:seeds.append((coeff,min(rr,key=lambda x:abs(float(x)-c))))
    seeds += [((1.,0.,1.,-.5,o,o),mp.mpf('.5')) for o in (0,1)]
    for i,(coeff,root) in enumerate(seeds):
        c=float(root);bits=int(np.float64(c).view(np.uint64))
        for side in ('left','right'):
            for offset in (-1,0,1,2,4,8,16,32,64):
                near=np.uint64(bits+(-offset if side=='left' else offset)).view(np.float64)
                l,r=(near,min(.999,c+.04)) if side=='left' else (max(.001,c-.04),near)
                pieces.append(dict(group='explicit_endpoint_neighborhoods',case=f'{i}:{side}:{offset}',parent=None,axis=0,k=0,l=float(l),r=float(r),coeff=coeff,C=0.,stored_roots=None))
    # Include generic legacy intervals separately, including all two-root cases.
    folder=ROOT/'split_subpiece_stationary_solver_smoke'
    for path in sorted(folder.glob('stage_*_train.json'))+sorted(folder.glob('stage_*_selection.json'))+sorted(folder.glob('stage_*_heldout.json'))+[folder/'interval_location_holdouts.json']:
        for i,p in enumerate(json.loads(path.read_text())):
            pieces.append(dict(group='legacy_generic_diagnostics',case=f'{path.stem}:{i}',parent=None,axis=0,k=0,l=p['lower'],r=p['upper'],
                coeff=(p['A'],p['B'],p['a'],p['beta'],p['t'],p['v']),C=0.,stored_roots=p['roots']))
    print(safe_json(dict(stage='data_ready',groups=dict(Counter(p['group'] for p in pieces)),seconds=time.monotonic()-start)),flush=True)
    coeff=np.array([p['coeff'] for p in pieces]);L=np.array([p['l'] for p in pieces]);R=np.array([p['r'] for p in pieces])
    def matrix(c):return np.column_stack((coeff[:,:4],c,coeff[:,4:6]))
    def endpoint(c):
        tx=torch.from_numpy(matrix(c))
        with torch.no_grad():
            state,gap,_,terms=router(tx,details=True)
            # Replace only the sqrt output, without changing modules/weights.
            m=router.term_assembly.components;A,B,a,beta,c,ot,ov=tx.unbind(1)
            t=m['orientation'](torch.stack((c,ot),1));v=m['orientation'](torch.stack((c,ov),1))
            vp=m['orientation'].weights[2]+m['orientation'].weights[3]*ov
            q=m['quadratic'](torch.stack((B,a,t),1));rr=torch.from_numpy(np.sqrt(q.numpy()))
            term2=m['multiply'](torch.stack((m['multiply'](torch.stack((beta,vp),1)),rr),1))
            directterms=torch.stack((terms[:,0],term2),1);ds=router.route_terms(directterms)
            raw=router.term_assembly.route_terms(terms)
        return state.numpy(),gap.numpy(),terms.numpy(),ds.numpy(),raw.numpy()
    sl,gl,tl,dl,rawL=endpoint(L);sr,gr,tr,dr,rawR=endpoint(R)
    available=(sl!=DEFER)&(sr!=DEFER)&(sl!=sr)
    sqrt_avail=(dl!=DEFER)&(dr!=DEFER)&(dl!=dr)
    rows=[];mismatches=[];allroots=[];legacy_stored_changes=0
    for i,p in enumerate(pieces):
        l,r=p['l'],p['r'];cr=rec(p['coeff'])
        if p['parent'] is not None:
            geometry.setup(p['parent'],p['axis']);k=p['k']
            fun=lambda c:geometry.formula(c,k)[1]
            sf=lambda c:geometry.formula(c,k)[0]
        else:
            fun=lambda c:derivative(cr,np.asarray(c));sf=lambda c:score(cr,np.asarray(c))+p['C']
        roots=scan_roots(fun,l,r)
        # Legacy generic reference defines availability from high-precision roots;
        # keep its native labels while separately logging the current scan rule.
        teacher_roots=p['stored_roots'] if p['group']=='legacy_generic_diagnostics' else roots
        yes=bool(teacher_roots);learned=bool(available[i]);mismatch=yes!=learned
        if p['stored_roots'] is not None and p['group']=='archived_geometry':legacy_stored_changes+=int(bool(p['stored_roots'])!=yes)
        row=dict(group=p['group'],case=p['case'],teacher_available=yes,learned_available=learned,match=not mismatch,
                 teacher_root_count=len(teacher_roots),current_scan_available=bool(roots),lower=l,upper=r,
                 learned_left=int(sl[i]),learned_right=int(sr[i]),left_gap=gl[i],right_gap=gr[i],sqrt_only_available=bool(sqrt_avail[i]))
        rows.append(row)
        if mismatch:
            hp=math_roots(tuple(p['coeff']));inside=[c for c in hp if mp.mpf(l)<c<mp.mpf(r)]
            hpder=[mpder(p['coeff'],z) for z in (l,r)]
            fd=np.array([derivative(cr,l),derivative(cr,r)]);dynamic=np.asarray(fun(np.array([l,r])))
            mathbr=mp.sign(hpder[0])*mp.sign(hpder[1])<0
            floatbr=fd[0]*fd[1]<0;rawbr=rawL[i]*rawR[i]<0
            fixedscan=scan_roots(lambda c:derivative(cr,np.asarray(c)),l,r)
            if yes and not learned:
                if sqrt_avail[i]:cause='learned_sqrt_precision'
                elif rawbr:cause='reliability_DEFER_screening'
                elif floatbr:cause='learned_arithmetic_precision'
                elif not inside:cause='float64_teacher_active_branch_or_rounding_artifact'
                elif mathbr:cause='float64_endpoint_sign_rounding'
                else:cause='endpoint_bracket_rule_multiple_or_tangent_roots'
                first='sqrt_only' if sqrt_avail[i] else 'direct_float64_fixed_endpoint_signs' if floatbr else 'high_precision_fixed_condition' if inside else 'not_restored_by_fixed_subpiece_precision'
            else:
                cause='teacher_scan_exact_zero_or_trimmed_endpoint_omission' if inside else 'learned_added_bracket_without_fixed_mathematical_root'
                first='not_a_missed_teacher_candidate'
            if roots and not fixedscan and p['parent'] is not None:
                source_detail='Dynamic teacher reselects max branches; fixed-branch float64 scan does not reproduce its candidate.'
            elif inside and not roots:source_detail='Mathematical interior root exists but strict-product trimmed teacher scan omits it.'
            else:source_detail='See endpoint signs, scan availability and high-precision root lists.'
            targets=teacher_roots or [float(c) for c in inside] or [(l+r)/2]
            dists=[min(abs(c-l),abs(r-c)) for c in targets]
            ulps=[min(ordered_steps(c,l),ordered_steps(c,r)) for c in targets]
            near=min(range(len(targets)),key=lambda j:dists[j]);c=targets[near]
            vals=np.asarray(sf(np.array([l,r,c])));bestend=min(vals[:2]);tol=1e-12*max(1.,min(vals))
            arithmetic=dict(A=p['coeff'][0],B=p['coeff'][1],a=p['coeff'][2],beta=p['coeff'][3],ot=p['coeff'][4],ov=p['coeff'][5],C=p['C'])
            z=dict(row,**arithmetic,classification=cause,source_detail=source_detail,first_restoring_stage=first,
                teacher_candidate_roots=json.dumps(teacher_roots),mathematical_interior_roots=json.dumps([mp.nstr(c,70) for c in inside]),
                mathematical_roots_full=json.dumps([mp.nstr(c,70) for c in hp]),mathematical_interior_exists=bool(inside),
                teacher_candidate=float(c),candidate_distance_to_endpoint=dists[near],candidate_distance_ULPs=ulps[near],
                all_candidate_distances=json.dumps(dists),all_candidate_ULP_distances=json.dumps(ulps),
                mathematical_nearest_endpoint_distance=min((float(min(c-mp.mpf(l),mp.mpf(r)-c)) for c in inside),default=None),
                direct_teacher_endpoint_derivatives=json.dumps(dynamic.tolist()),fixed_float64_endpoint_derivatives=json.dumps(fd.tolist()),
                fixed_high_precision_endpoint_derivatives=json.dumps([mp.nstr(v,60) for v in hpder]),
                left_T1=tl[i,0],left_T2=tl[i,1],right_T1=tr[i,0],right_T2=tr[i,1],
                sqrt_only_left=int(dl[i]),sqrt_only_right=int(dr[i]),raw_magnitude_left=int(rawL[i]),raw_magnitude_right=int(rawR[i]),
                raw_without_reliability_available=bool(rawbr),direct_float64_fixed_endpoint_available=bool(floatbr),
                high_precision_fixed_endpoint_available=bool(mathbr),direct_float64_fixed_scan_available=bool(fixedscan),
                direct_float64_dynamic_endpoint_available=bool(dynamic[0]*dynamic[1]<0),
                teacher_candidate_score=float(vals[2]),left_endpoint_score=float(vals[0]),right_endpoint_score=float(vals[1]),
                candidate_exactly_equals_an_endpoint_score=bool(vals[2]==vals[0] or vals[2]==vals[1]),
                candidate_equivalent_to_best_endpoint=bool(abs(vals[2]-bestend)<=tol),candidate_minus_best_endpoint=float(vals[2]-bestend),
                parent=json.dumps(p['parent']))
            # One-ULP and two-ULP neighborhoods around both endpoints, saved for
            # every mismatch without changing the tested interval.
            neighbors=[]
            for endpoint_value in (l,r):
                b=int(np.float64(endpoint_value).view(np.uint64))
                for delta in (-2,-1,0,1,2):neighbors.append(np.uint64(b+delta).view(np.float64))
            x=np.column_stack((np.tile(p['coeff'][:4],(10,1)),neighbors,np.tile(p['coeff'][4:6],(10,1))))
            with torch.no_grad():ns,ng,_,_=router(torch.from_numpy(x),details=True)
            z['endpoint_neighborhoods']=json.dumps([dict(c=float(c),learned_state=int(s),gap=float(g),teacher_derivative=float(np.atleast_1d(fun(c))[0]),fixed_math_sign=int(mp.sign(mpder(p['coeff'],float(c))))) for c,s,g in zip(neighbors,ns.numpy(),ng.numpy())])
            mismatches.append(z)
        if (i+1)%2000==0:print(safe_json(dict(evaluated=i+1,total=len(pieces),mismatches=len(mismatches),seconds=time.monotonic()-start)),flush=True)
    ref.write_csv(OUT/'stationary_availability_all.csv',rows)
    ref.write_csv(OUT/'stationary_availability_mismatches.csv',mismatches)
    def summary(rr):
        return dict(total=len(rr),matches=sum(r['match'] for r in rr),teacher_yes_learned_no=sum(r['teacher_available'] and not r['learned_available'] for r in rr),
                    teacher_no_learned_yes=sum(not r['teacher_available'] and r['learned_available'] for r in rr))
    groups={g:summary([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
    missed=[r for r in mismatches if r['teacher_available'] and not r['learned_available']]
    dist={}
    for group in ['all','active_geometric','explicit_endpoint_neighborhoods','legacy_generic_diagnostics']:
        rr=missed if group=='all' else [r for r in missed if (r['group'].endswith('geometry') if group=='active_geometric' else r['group']==group)]
        ds=[r['candidate_distance_ULPs'] for r in rr]
        dist[group]=dict(missed_subpieces=len(rr),one_ULP=sum(d==1 for d in ds),ULP_2_to_4=sum(2<=d<=4 for d in ds),
            ULP_5_to_16=sum(5<=d<=16 for d in ds),ULP_above_16=sum(d>16 for d in ds),zero_ULP=sum(d==0 for d in ds),
            maximum_candidate_distance_ULPs=max(ds,default=0),maximum_physical_candidate_distance=max((r['candidate_distance_to_endpoint'] for r in rr),default=0),
            classifications=dict(Counter(r['classification'] for r in rr)),
            exact_endpoint_score_ties=sum(r['candidate_exactly_equals_an_endpoint_score'] for r in rr),
            equivalent_best_endpoint_scores=sum(r['candidate_equivalent_to_best_endpoint'] for r in rr))
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    result=dict(status='AUDIT_ONLY_COMPLETE',overall=summary(rows),groups=groups,missed_distance_distribution=dist,
        all_mismatch_classifications=dict(Counter(r['classification'] for r in mismatches)),archived_scan_vs_saved_availability_changes=legacy_stored_changes,
        elapsed_seconds=time.monotonic()-start,protected_files=len(protected),all_protected_hashes_unchanged=True,
        high_precision='80 decimal digits, exact float64 coefficients; mathematical roots are NOT invented for float64 branch-switch artifacts',
        distance_definition='Nearest saved teacher candidate to an endpoint per mismatched subpiece; all root distances are also retained in CSV. Mathematical distances separately reported when roots exist.',
        source_isolation='Square-root-only substitution retains frozen routing. Fixed float64 and high-precision endpoint signs retain endpoint rule. Dynamic teacher scan additionally logged to separate arithmetic from rule changes.',
        no_training=True,no_recursion=True,no_global_search=True,implementation_sha256=ref.sha(Path(__file__)))
    (OUT/'results.json').write_text(safe_json(result));(OUT/'protected_hashes.json').write_text(safe_json(protected))
    print(safe_json(result),flush=True)


if __name__=='__main__':run()
