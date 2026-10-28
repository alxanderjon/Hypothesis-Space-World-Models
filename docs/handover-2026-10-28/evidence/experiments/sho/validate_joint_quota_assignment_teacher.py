"""New generic deterministic integer-assignment teacher; count-only validation.

No learned model, point placement, or changes to existing operators. scipy/HiGHS
is only a feasibility/optimization engine: every output is integer checked and
canonicalized. Incidence columns can represent finite candidates or region slots.
"""
import gzip, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, vstack

ROOT=Path('outputs/sho/stage1B'); OUT=ROOT/'joint_quota_assignment_teacher'
CALLS=0

def solve(c,A,lo,hi,lb,ub):
    global CALLS
    CALLS+=1
    result=milp(np.asarray(c,float),integrality=np.ones(len(c)),
                bounds=Bounds(lb,ub),constraints=LinearConstraint(csr_matrix(A),lo,hi),
                options={'mip_rel_gap':0.0,'presolve':True})
    if result.status==2:return None
    if not result.success:raise RuntimeError(f'No certified optimum: {result.message}')
    z=np.rint(result.x).astype(np.int64)
    # Inputs to this solver are integral, so this is an exact integer validation.
    Az=np.asarray(A@z).ravel()
    if not (np.all(z>=lb) and np.all(z<=ub) and np.all(Az>=lo) and np.all(Az<=hi)):
        raise RuntimeError('MILP result failed exact integer verification')
    return z

def canonical(A,lo,hi,cap):
    """Lexicographically maximize counts at ascending canonical candidate IDs."""
    cap=np.asarray(cap,dtype=np.int64);lb=np.zeros(len(cap),dtype=np.int64);ub=cap.copy()
    solution=solve(np.zeros(len(cap)),A,lo,hi,lb,ub)
    if solution is None:return None
    for j in range(len(cap)):
        if lb[j]==ub[j]:continue
        if solution[j]!=ub[j]:
            cost=np.zeros(len(cap),dtype=np.int64);cost[j]=-1
            solution=solve(cost,A,lo,hi,lb,ub)
            assert solution is not None
        lb[j]=ub[j]=solution[j]
    return solution

def teacher(incidence,capacity,current,lower,upper,remaining,frozen=None):
    I=np.asarray(incidence,dtype=np.int64);cap=np.asarray(capacity,dtype=np.int64)
    current=np.asarray(current,dtype=np.int64);lower=np.asarray(lower,dtype=np.int64);upper=np.asarray(upper,dtype=np.int64)
    assert set(np.unique(I)).issubset({0,1}) and np.all(cap>=0) and remaining>=0
    assert np.all(current>=0) and np.all(lower<=upper)
    frozen=np.zeros(len(current),bool) if frozen is None else np.asarray(frozen,bool)
    # Already satisfied coarse rows cannot be incremented. This is a hard constraint.
    assert np.all((current[frozen]>=lower[frozen]) & (current[frozen]<=upper[frozen]))
    lo=lower-current;hi=upper-current;lo[frozen]=hi[frozen]=0
    if cap.sum()<remaining:
        return None,{'kind':'capacity_shortage','remaining':int(remaining),'total_capacity':int(cap.sum())}
    if np.any(current>upper):
        rows=np.flatnonzero(current>upper)
        return None,{'kind':'immutable_prefix_exceeds_upper','rows':rows.tolist(),'current':current[rows].tolist(),'upper':upper[rows].tolist()}
    if remaining==int(cap.sum()):
        counts=current+I@cap
        if np.all((counts>=lower)&(counts<=upper)) and np.all((I@cap)[frozen]==0):return cap, None
        return None,{'kind':'budget_forces_every_candidate','forced_counts':counts.tolist(),'violating_rows':np.flatnonzero((counts<lower)|(counts>upper)|(frozen & ((I@cap)!=0))).tolist(), 'frozen_rows_violated':np.flatnonzero(frozen & ((I@cap)!=0)).tolist(), 'unique_budget_assignment':cap.tolist()}
    A=np.vstack((np.ones(len(cap),dtype=np.int64),I));L=np.r_[remaining,lo];H=np.r_[remaining,hi]
    z=canonical(A,L,H,cap)
    if z is None:return None,{'kind':'integer_infeasible','solver':'HiGHS status 2; see capacity-cut diagnostic if supplied'}
    return z,None

def lexicographic_relaxation(I,cap,current,target,remaining,tiers,frozen=None):
    """Small infeasible cases: exact lex max-error then SSE per historical tier.

    Native integral targets here; rational historical targets use floor/ceil
    feasibility above. One-hot count levels make squared error an exact integer
    objective. Large forced-pool cases have a unique solution, requiring no IP.
    """
    I=np.asarray(I,dtype=np.int64);cap=np.asarray(cap,dtype=np.int64);current=np.asarray(current,dtype=np.int64);target=np.asarray(target,dtype=np.int64)
    if cap.sum()<remaining:return None,{'hard_infeasible':'budget exceeds capacity'}
    if cap.sum()==remaining:return cap,{'unique_solution':True}
    m,n=I.shape; levels=np.arange(remaining+1);size=n+m*len(levels)
    A=[];lo=[];hi=[]
    row=np.zeros(size,dtype=np.int64);row[:n]=1;A.append(row);lo.append(remaining);hi.append(remaining)
    for i in range(m):
        row=np.zeros(size,dtype=np.int64);row[n+i*len(levels):n+(i+1)*len(levels)]=1;A.append(row);lo.append(1);hi.append(1)
        row=np.zeros(size,dtype=np.int64);row[:n]=I[i];row[n+i*len(levels):n+(i+1)*len(levels)]=-levels;A.append(row);lo.append(0);hi.append(0)
    lb=np.zeros(size,dtype=np.int64);ub=np.r_[cap,np.ones(m*len(levels),dtype=np.int64)]
    if frozen is not None:
        for i in np.flatnonzero(frozen):ub[n+i*len(levels)+1:n+(i+1)*len(levels)]=0
    err=np.abs(current[:,None]+levels-target[:,None]);objective=[]
    for tier in sorted(set(tiers)):
        rows=np.flatnonzero(np.asarray(tiers)==tier); found=None
        # Minimum max error via integer bound enumeration (small diagnostic only).
        for bound in range(int(err[rows].max())+1):
            test=ub.copy()
            for i in rows:test[n+i*len(levels):n+(i+1)*len(levels)][err[i]>bound]=0
            z=solve(np.zeros(size),np.array(A),np.array(lo),np.array(hi),lb,test)
            if z is not None:ub=test;found=bound;break
        if found is None:return None,{'hard_infeasible':'frozen rows incompatible with remaining budget/capacity'}
        cost=np.zeros(size,dtype=np.int64)
        for i in rows:cost[n+i*len(levels):n+(i+1)*len(levels)]=err[i]**2
        z=solve(cost,np.array(A),np.array(lo),np.array(hi),lb,ub);opt=int(cost@z)
        A.append(cost);lo.append(opt);hi.append(opt);objective.append({'tier':int(tier),'max_error':found,'sum_squared_error':opt})
    z=canonical(np.array(A),np.array(lo),np.array(hi),ub)
    return z[:n],{'tier_optima':objective}

def incidence(points,families):
    arrays=[];meta=[];levels=[]
    for tier,f in families:
        f=np.array(f,dtype=np.int64);idx=np.minimum(np.floor(points*f).astype(np.int64),f-1)
        flat=np.ravel_multi_index(idx.T,tuple(f));k=int(np.prod(f))
        block=(np.arange(k)[:,None]==flat).astype(np.int64);arrays.append(block)
        for cell in range(k):meta.append({'family':f.tolist(),'cell':cell,'tier':tier,'axis_levels':np.log2(f).astype(int).tolist()});levels.append(tier)
    return np.vstack(arrays),meta,np.array(levels)

def aggregate(I):
    # Canonical incidence-bit tuple order, independent of coordinates and column input order.
    patterns,inv,cap=np.unique(I.T,axis=0,return_inverse=True,return_counts=True)
    return patterns.T,cap,inv

def bounds(meta,N):
    sizes=np.array([np.prod(v['family']) for v in meta],dtype=np.int64)
    return N//sizes,(N+sizes-1)//sizes

def metrics(counts,meta,N):
    out={}
    for f in dict.fromkeys(tuple(v['family']) for v in meta):
        ids=[i for i,v in enumerate(meta) if tuple(v['family'])==f]
        error=counts[ids]-N/len(ids)
        out[str(f)]={'max_count':float(abs(error).max()),'rms_count':float(np.sqrt(np.mean(error**2)))}
    return out

def trace(name,I,cap,current,target,remaining,z,meta):
    # Lossless event sourcing: full input once, then every unit assignment/update.
    # Input state for event t is initial plus every preceding sparse event.
    path=OUT/(name+'.jsonl.gz')
    with gzip.open(path,'wt') as f:
        f.write(json.dumps({'type':'initial','incidence':I.tolist(),'capacities':cap.tolist(),'counts':current.tolist(),'integer_targets':target.tolist(),'remaining_budget':remaining,'constraints':meta,'tie_rule':'lexicographically maximal assignment-count vector in ascending incidence-bit candidate-ID order; emit earliest ID with positive planned count'})+'\n')
        counts=current.copy();capacities=cap.copy();step=0
        for j,n in enumerate(z):
            affected=np.flatnonzero(I[:,j])
            for _ in range(int(n)):
                counts[affected]+=1;capacities[j]-=1;step+=1
                f.write(json.dumps({'type':'assignment','input_state_ref':step-1,'output_state_id':step,'selected_assignment':j,'updated_constraint_ids':affected.tolist(),'updated_counts':counts[affected].tolist(),'updated_deficits':(target-counts)[affected].tolist(),'candidate_remaining_capacity':int(capacities[j]),'remaining_budget':remaining-step,'tie_break':'earliest positive witness count'})+'\n')
                assert np.all(counts<=target)
        assert step==remaining and np.array_equal(counts,target) and np.all(capacities>=0)
    return str(path)

def validate_trace(path):
    with gzip.open(path,'rt') as f:
        initial=json.loads(next(f));I=np.array(initial['incidence']);cap=np.array(initial['capacities']);count=np.array(initial['counts']);target=np.array(initial['integer_targets']);remaining=initial['remaining_budget'];step=0
        for line in f:
            e=json.loads(line);j=e['selected_assignment'];assert e['input_state_ref']==step;step+=1
            assert cap[j]>0;cap[j]-=1;count+=I[:,j];remaining-=1
            ids=np.flatnonzero(I[:,j]);assert ids.tolist()==e['updated_constraint_ids']
            assert count[ids].tolist()==e['updated_counts'] and (target-count)[ids].tolist()==e['updated_deficits']
            assert e['remaining_budget']==remaining and cap[j]==e['candidate_remaining_capacity'] and np.all(count<=target)
        assert remaining==0 and np.array_equal(count,target)
    return step

def run():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.pt')}
    from experiments.sho.train_dyadic_spacefill_curriculum import PHASES
    source=ROOT/'dyadic_spacefill_curriculum/converged_continuation/phase_1_halves_heldout_interiors.npz'
    pools=dict(np.load(source)); records=[];traces=[];failures=[];states=[]
    # Historical completed pools and immutable prefixes with precisely their saved remaining points.
    # Preserve the exact historical phase order for coarse-to-fine priorities.
    active=[(phase,tuple(f)) for phase,(_,_,fs) in enumerate(PHASES) for f in fs]
    for name,p in pools.items():
        N=len(p);I,meta,tiers=incidence(p,active);lo,hi=bounds(meta,N)
        for prefix in (0,128,N//2,N):
            current=I[:,:prefix].sum(1);J,cap,_=aggregate(I[:,prefix:]) if prefix<N else (np.empty((len(meta),0),int),np.zeros(0,int),None)
            z,cert=teacher(J,cap,current,lo,hi,N-prefix)
            # Budget equals capacity: unique lex optimum is all remaining candidates.
            forced=current+J@cap
            rec={'id':name+f'_fixed_prefix{prefix}','kind':'saved_capacity','prefix':prefix,'dimension':p.shape[1],'N':N,'rows':len(meta),'atoms':len(cap),'feasible':z is not None,'certificate':cert,'best_lex_counts':forced.tolist(),'counts':metrics(forced,meta,N),'budget_error':0,'unique_best_lex_solution':True}
            records.append(rec)
            if cert:failures.append(rec)
    # Historical candidates pooled into a declared NEW finite reservoir, not claimed saved online trace.
    reservoir=np.concatenate(list(pools.values())); assert len(np.unique(reservoir,axis=0))==len(reservoir)
    warm_pools=dict(np.load(ROOT/'progressive_scale_curriculum/stage_1_heldout_interiors.npz'))
    previous=None
    for phase,(_,name,fs) in enumerate(PHASES):
        fam=[(i,tuple(f)) for i,(_,_,ff) in enumerate(PHASES[:phase+1]) for f in ff]
        I,meta,tiers=incidence(reservoir,fam);J,cap,inv=aggregate(I);N=8600;lo,hi=bounds(meta,N);cur=np.zeros(len(meta),dtype=np.int64)
        if previous is not None:
            # Keep the exact already-selected coarse integer targets during refinement.
            lo[:len(previous)]=hi[:len(previous)]=previous
        t=time.monotonic();z,cert=teacher(J,cap,cur,lo,hi,N)
        if z is None:
            failures.append({'id':'historical_reservoir_'+name,'certificate':cert});raise RuntimeError('Historical reservoir refinement infeasible; stop')
        count=J@z;retention=0 if previous is None else int(np.count_nonzero(count[:len(previous)]!=previous));previous=count.copy()
        baseline={k:metrics(incidence(p,fam)[0].sum(1),meta,N) for k,p in pools.items()}
        row={'id':'historical_reservoir_'+name,'kind':'historical_reconstructed_reservoir','dimension':2,'phase':name,'N':N,'candidate_capacity':int(cap.sum()),'atoms':len(cap),'rows':len(meta),'feasible':True,'seconds':time.monotonic()-t,'counts':metrics(count,meta,N),'baseline_by_pool':baseline,'budget_error':int(z.sum()-N),'coarse_retention_errors':retention}
        warm={k:metrics(incidence(p,fam)[0].sum(1),meta,N) for k,p in warm_pools.items()}
        ratios=[]
        for f in fs:
            key=str(tuple(f)); mm=row['counts'][key]
            baseline_max=max(v[key]['max_count'] for v in warm.values())
            baseline_rms=float(np.sqrt(np.mean([v[key]['rms_count']**2 for v in warm.values()])))
            ratios.extend([mm['max_count']/max(baseline_max,.5),mm['rms_count']/max(baseline_rms,.5)])
        row['historical_new_family_ratio_vs_saved_warm_entry']=max(ratios)
        row['historical_count_improvement_gate']=max(ratios)<=.85
        row['halves_absolute_gate']=all(row['counts'][str(tuple(f))]['max_count']<=8 for f in PHASES[0][2])
        records.append(row)
        states.append({'id':row['id'],'incidence':J.tolist(),'capacity':cap.tolist(),'current':cur.tolist(),'lower':lo.tolist(),'upper':hi.tolist(),'integer_targets':count.tolist(),'assignment_counts':z.tolist(),'metadata':meta})
        # Save traces for every coarse/fine phase (same reservoir, distinct teacher tasks).
        path=trace(row['id'],J,cap,cur,count,N,z,meta);traces.append(path)
        print(json.dumps({k:v for k,v in row.items() if k not in ['baseline_by_pool','counts']}),flush=True)
    # Fresh generic rectangular family structures, no branch on dimension identity.
    rng=np.random.default_rng(29093026)
    for d in (1,2,3):
        for trial in range(3):
            N=int([64,128,256][trial]);candidate_count=8*N
            points=rng.random((candidate_count,d)); families=[]
            for level in (1,2,3):
                for axis in range(d):
                    f=[1]*d;f[axis]=2**level;families.append((level-1,tuple(f)))
            for axes in itertools.combinations(range(d),2):
                f=[1]*d
                for axis in axes:f[axis]=2
                families.append((3,tuple(f)))
            families.extend([(4,(2,)*d),(5,(4,)*d)])
            families=list({f:(tier,f) for tier,f in reversed(families)}.values()); families.sort(key=lambda v:(v[0],v[1]));I,meta,_=incidence(points,families);J,cap,_=aggregate(I);lo,hi=bounds(meta,N);cur=np.zeros(len(meta),dtype=np.int64)
            t=time.monotonic();z,cert=teacher(J,cap,cur,lo,hi,N);sec=time.monotonic()-t
            if z is None:
                failures.append({'id':f'fresh_d{d}_{trial}','certificate':cert});raise RuntimeError('Fresh quota feasibility failure; stop')
            target=J@z
            path=trace(f'fresh_d{d}_{trial}',J,cap,cur,target,N,z,meta);traces.append(path)
            # Resume with actual partial witness: includes saturated constraints; forbid any increments there.
            consumed=np.minimum(z,np.maximum(1,z//2));prefix=J@consumed;remaining=int(N-consumed.sum());frozen=prefix==target
            zz,cert2=teacher(J,cap-consumed,prefix,target,target,remaining,frozen)
            assert zz is not None and np.all((J@zz)[frozen]==0)
            path=trace(f'fresh_resume_d{d}_{trial}',J,cap-consumed,prefix,target,remaining,zz,meta);traces.append(path)
            # Repeat exact input: canonical plan must not depend on MILP's first feasible witness.
            repeat,_=teacher(J,cap,cur,lo,hi,N);assert np.array_equal(repeat,z)
            records.append({'id':f'fresh_d{d}_{trial}','kind':'fresh','dimension':d,'levels':[1,2,3],'N':N,'candidate_capacity':candidate_count,'atoms':len(cap),'rows':len(meta),'feasible':True,'seconds':sec,'counts':metrics(target,meta,N),'budget_error':int(z.sum()-N),'coarse_retention_errors':int(np.count_nonzero((J@zz)[frozen])),'saturated_rows_checked':int(frozen.sum()),'resume_feasible':True,'repeat_exact':True})
            states.append({'id':f'fresh_d{d}_{trial}','incidence':J.tolist(),'capacity':cap.tolist(),'current':cur.tolist(),'lower':lo.tolist(),'upper':hi.tolist(),'integer_targets':target.tolist(),'assignment_counts':z.tolist(),'metadata':meta})
            print(f'fresh d={d} N={N} atoms={len(cap)} seconds={sec:.3f}',flush=True)
    # Analytically certified incompatible overlapping families, with exact lex fallback.
    I=np.array([[1,1,0],[0,0,1],[1,0,1],[0,1,0]],dtype=np.int64);cap=np.array([4,4,4]);cur=np.zeros(4,dtype=np.int64);target=np.array([1,3,1,3]);N=4
    z,cert=teacher(I,cap,cur,target,target,N);assert z is None
    best,details=lexicographic_relaxation(I,cap,cur,target,N,[0,0,1,1]);assert best is not None
    assert np.array_equal((I@best)[:2],target[:2])
    failures.append({'id':'incompatible_cross_family_control','certificate':{'kind':'capacity_cut','rows':[1,3],'requested_sum':6,'maximum_possible_sum':4,'reason':'each assignment hits at most one of these two rows and budget is four'},'best_assignment':best.tolist(),'best_counts':(I@best).tolist(),'lexicographic_result':details,'budget_error':int(best.sum()-N)})
    # Coarse lock makes a capacity conflict hard-infeasible; no silent relaxation.
    frozen=np.array([True,False,False,False]);prefix=np.array([1,0,1,0]);short_cap=np.array([2,0,0]);zz,cert=teacher(I,short_cap,prefix,target,target,2,frozen)
    assert zz is None
    failures.append({'id':'frozen_coarse_capacity_control','certificate':{'kind':'saturated_row_cut','row':0,'allowed_increment':0,'every_available_assignment_increments_row':True,'remaining_budget':2},'hard_infeasible':True})
    shortage,cert=teacher(I,np.zeros(3,dtype=np.int64),cur,target,target,4);assert shortage is None
    failures.append({'id':'capacity_shortage_control','certificate':cert,'hard_infeasible':True})
    replayed=sum(validate_trace(p) for p in traces)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    assert unchanged
    # Gate applies to supported candidate assignment problems; impossible states must reject correctly.
    passing=[r for r in records if r['kind']!='saved_capacity']
    gate=all(r['feasible'] and r['budget_error']==0 and r['coarse_retention_errors']==0 and r.get('historical_count_improvement_gate',True) and r.get('halves_absolute_gate',True) for r in passing)
    report={'status':'VALIDATED_DETERMINISTIC_TEACHER' if gate else 'FAILED','scope':'generic finite-incidence, count-only assignment; jointly floor/ceil-balanced supported pools and explicit infeasibility handling; not full geometric N1 acceptance','records':records,'infeasibility_diagnostics':failures,'trace_files':traces,'replayed_unit_assignments':replayed,'milp_calls':CALLS,'runtime_seconds':time.monotonic()-start,'protected_checkpoint_count':len(protected),'protected_checkpoints_unchanged':unchanged,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (OUT/'results.json').write_text(json.dumps(report,indent=2));(OUT/'input_states.json').write_text(json.dumps(states));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    manifest={'joint_quota_assignment_teacher':report['status'],'scope':report['scope'],'implementation':'experiments/sho/validate_joint_quota_assignment_teacher.py','implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),'tie_break':'lexicographically maximize counts by ascending incidence-bit tuple; emit smallest remaining ID','training':False}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','infeasibility_diagnostics','trace_files']},indent=2),flush=True)

if __name__=='__main__':run()
