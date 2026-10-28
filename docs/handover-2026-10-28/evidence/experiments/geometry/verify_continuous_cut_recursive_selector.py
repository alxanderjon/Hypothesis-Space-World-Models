"""Fresh continuous teacher versus hybrid partitioning; no training/menu search."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_stationary_three_state_router import load_router,DEFER
from experiments.geometry.train_mantissa_log import MonotoneMantissaLog
from experiments.geometry.train_exponent_log2_operator import ExponentLog2
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.train_finite_split_score_smoke import upstream,predict
from experiments.sho.train_split_validity_smoke import route
from experiments.sho.train_recursive_region_splitter_poc import children

ROOT=ref.ROOT;OUT=ROOT/'continuous_cut_recursive_verification/parent_impact_gate';SEED=28262026


class VerificationFailure(RuntimeError):pass


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if 'manifest' in p.name or 'validated' in p.name})
    sources={}
    for folder,name in [('fixed_allocation_interval_minimizer_audit','fixed_allocation_interval_minimizer_validated.json'),
                        ('stationary_float64_minimizer_full_audit','stationary_float64_minimizer_validated.json'),
                        ('defer_aware_stationary_locator_audit','defer_aware_stationary_locator_validated.json')]:
        p=ROOT/folder/name;m=json.loads(p.read_text());assert m['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE'
        assert ref.sha(p)==json.loads((p.parent/'results.json').read_text())['assembly_sha256'];sources[str(p)]=ref.sha(p)
    audit=Audit();router,router_refs=load_router();base,arithmetic_refs=ref.load_terms();models=base.components
    for name,folder,cls in [('sqrt','sqrt_operator_failure_refine',ref.ScalarPower),('mantissa','mantissa_log_operator_support_refine',MonotoneMantissaLog),('exponent','exponent_log2_operator_smoke',ExponentLog2)]:
        m=json.loads((ROOT/folder/'candidate_manifest.json').read_text());p=Path(m['checkpoint']);assert ref.sha(p)==m['sha256']
        net=cls();net.load_state_dict(torch.load(p,weights_only=False)['state_dict']);models[name]=net.eval().requires_grad_(False);arithmetic_refs[name]=m
    stats=dict(selector_calls=0,axes=0,allocation_intervals=0,subpieces=0,teacher_stationary_candidates=0,hybrid_stationary_candidates=0,
        router_queries=0,DEFER_queries=0,locator_iterations=0,enumerated_stationary_floats=0,endpoint_defers=0,
        stationary_bracket_changes=0,allocation_boundary_candidate_reassignments=0,scalar_comparator_errors=0)
    phase1=[];recursive=[];partition_rows=[];trees={};failure=None;last_context={};trace=[];bracket_diagnostics=[]
    def deadline():
        if time.monotonic()-started>240:raise VerificationFailure('Quick-screen execution budget reached; partial results only.')
    def signs(coeff,c):
        coeff=np.asarray(coeff,float);c=np.asarray(c,float)
        x=np.column_stack((coeff[:,:4],c,coeff[:,4:6]))
        with torch.no_grad():out=router(torch.from_numpy(x)).numpy()
        stats['router_queries']+=len(c);stats['DEFER_queries']+=int(np.sum(out==DEFER));return out
    def generic_learned(coeff,c):
        A,B,a,beta,ot,ov,C=coeff;c=torch.from_numpy(np.asarray(c,float));n=len(c)
        def full(x):return torch.full((n,),float(x),dtype=torch.float64)
        def mul(x,y):return models['multiply'](torch.stack((x,y),1))
        def add(x,y):return models['addition'](torch.stack((x,y),1))
        with torch.no_grad():
            t=models['orientation'](torch.stack((c,full(ot)),1));v=models['orientation'](torch.stack((c,full(ov)),1))
            q=models['quadratic'](torch.stack((full(B),full(a),t),1));first=mul(full(A),models['sqrt'](q))
            m,e=np.frexp(v.numpy());m=2*m;e=e-1
            lv=add(models['mantissa'](torch.from_numpy(m)),models['exponent'](torch.from_numpy(e.astype(float))))
            return add(add(first,mul(full(beta),lv)),full(C)).numpy()
    def locate(pieces,sl,sr):
        if not pieces:return []
        coeff=np.array([p['coeff'] for p in pieces]);L=np.array([p['l'] for p in pieces]);R=np.array([p['r'] for p in pieces]);active=np.ones(len(L),bool)
        while active.any():
            deadline();ids=np.flatnonzero(active);q=(L[ids]+R[ids])/2;done=(np.nextafter(L[ids],R[ids])==R[ids])|(q==L[ids])|(q==R[ids])
            active[ids[done]]=False;ids=ids[~done];q=q[~done]
            if not len(ids):break
            stats['locator_iterations']+=len(ids);s=signs(coeff[ids],q)
            def update(ii,c,ss):
                left=(ss!=DEFER)&(ss==sl[ii]);right=(ss!=DEFER)&(ss==sr[ii]);assert np.all(left|right|(ss==DEFER))
                L[ii[left]]=c[left];R[ii[right]]=c[right]
            trust=s!=DEFER;update(ids[trust],q[trust],s[trust]);di=ids[~trust];dq=q[~trust]
            if len(di):
                oldL=L[di].copy();oldR=R[di].copy();qL=(oldL+dq)/2;qR=(dq+oldR)/2
                ls=signs(coeff[di],qL);rs=signs(coeff[di],qR);update(di,qL,ls);update(di,qR,rs)
                assert np.all(L[di]>=oldL) and np.all(R[di]<=oldR)
                active[di[(L[di]==oldL)&(R[di]==oldR)]]=False
            assert np.all(L<R) and np.all(sl!=sr) and np.all(sl!=DEFER) and np.all(sr!=DEFER)
        out=[]
        for i,p in enumerate(pieces):
            lo=int(L[i].view(np.uint64));hi=int(R[i].view(np.uint64));best=np.inf;chosen=None
            for begin in range(lo,hi+1,16384):
                deadline();c=np.arange(begin,min(begin+16384,hi+1),dtype=np.uint64).view(np.float64);scores=generic_learned(p['coeff'],c);j=int(np.argmin(scores))
                if scores[j]<best:best=float(scores[j]);chosen=float(c[j])
                stats['enumerated_stationary_floats']+=len(c)
            out.append(chosen)
        return out
    def canonical(candidates,mode):
        field=mode+'_score';ordered=sorted(candidates,key=lambda r:(r['axis'],r['cut'],r['identity']))
        values=np.array([r[field] for r in ordered]);assert np.isfinite(values).all()
        if mode=='teacher':best=float(values.min());eligible=values<=best+1e-12*max(1,best)
        else:
            best=float(values[0])
            for v in values[1:]:
                if v==best:continue
                replace=bool(audit.execution.logits(np.array([best]),np.array([v]))[0]>0)
                stats['scalar_comparator_errors']+=int(replace!=(v<best))
                if replace:best=float(v)
            bound=best+1e-12*max(1,best);same=values==bound;eligible=same.copy();idx=~same
            eligible[idx]=audit.execution.logits(values[idx],np.full(idx.sum(),bound))<=0
            stats['scalar_comparator_errors']+=int(np.sum(eligible!=(values<=bound)))
        return dict(ordered[int(np.flatnonzero(eligible)[0])]),best
    def select(record,label):
        nonlocal last_context
        deadline();stats['selector_calls']+=1;last_context=dict(label=label,parent=record)
        all_candidates={'teacher':[],'hybrid':[]};p=[]
        for axis in range(record['d']):
            audit.setup(record,axis);n=record['n'];stats['axes']+=1
            edge=max(1e-15,4*np.spacing(abs(audit.lo[axis])+abs(audit.hi[axis]))/audit.a)
            scan=np.linspace(edge,1-edge,max(2049,32*n+1));counts=audit.counts(scan)[:,0]
            if np.any(np.diff(counts)<0) or counts[0]!=0 or counts[-1]!=n:raise VerificationFailure('Allocator interval-discovery scan is not monotone/complete.')
            lower=np.array([scan[np.flatnonzero(counts<k)[-1]] for k in range(1,n+1)]);upper=np.array([scan[np.flatnonzero(counts>=k)[0]] for k in range(1,n+1)])
            for _ in range(60):
                mid=(lower+upper)/2;left=audit.counts(mid)[:,0]<np.arange(1,n+1);lower=np.where(left,mid,lower);upper=np.where(left,upper,mid)
            if not np.array_equal(audit.counts(lower)[:,0],np.arange(n)) or not np.array_equal(audit.counts(upper)[:,0],np.arange(1,n+1)):
                raise VerificationFailure('Fresh allocation-interval endpoint ownership fails replay.')
            for k in range(1,n):
                stats['allocation_intervals']+=1;l=upper[k-1];r=lower[k];br=audit.breaks(l,r,k)
                for j,(a,b) in enumerate(zip(br[:-1],br[1:])):
                    identity=f'{axis}:{k}:{j}';stats['subpieces']+=1
                    for mode in all_candidates:
                        all_candidates[mode].extend(dict(axis=axis,k=k,cut=float(c),identity=identity+':'+kind,kind=kind) for c,kind in [(a,'left'),(b,'right')])
                    l0=np.nextafter(a,b);r0=np.nextafter(b,a)
                    if l0>=r0:continue
                    f=audit.formula(np.array([(a+b)/2]),k);iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
                    A=1/(audit.h*np.power(float(k if iq==0 else n-k),1/audit.d));beta=.05*power
                    C=.05*np.log(audit.a/audit.omin) if power==1 else .05*np.log(audit.omax/audit.a) if power==-1 else .05*np.log(audit.omax/audit.omin)
                    coeff=[A,audit.B,audit.a,beta,iq,ia,C]
                    # Existing deterministic continuous reference: analytic derivative
                    # scan and safeguarded root solve inside each active subpiece.
                    xx=np.linspace(l0,r0,129);der=audit.formula(xx,k)[1];roots=[]
                    for h in np.flatnonzero(der[:-1]*der[1:]<0):
                        roots.append(float(brentq(lambda c:float(audit.formula(c,k)[1][0]),xx[h],xx[h+1],xtol=5e-324,rtol=1e-15)))
                    for ri,c in enumerate(sorted(set(roots))):all_candidates['teacher'].append(dict(axis=axis,k=k,cut=c,identity=identity+':stationary'+str(ri),kind='stationary'))
                    stats['teacher_stationary_candidates']+=len(set(roots));p.append(dict(axis=axis,k=k,identity=identity,l=float(a),r=float(b),coeff=coeff,teacher_roots=roots))
        if p:
            coeff=np.array([r['coeff'] for r in p]);sl=signs(coeff,np.array([r['l'] for r in p]));sr=signs(coeff,np.array([r['r'] for r in p]))
            deferred=(sl==DEFER)|(sr==DEFER);stats['endpoint_defers']+=int(deferred.sum())
            bracket=(sl!=DEFER)&(sr!=DEFER)&(sl!=sr);teacher_has=np.array([bool(r['teacher_roots']) for r in p])
            bad=bracket!=teacher_has;stats['stationary_bracket_changes']+=int(bad.sum())
            # No teacher signs or roots substitute for hybrid control flow.
            if bad.any():
                # The continuous reference's dynamic max can flip active child a
                # float before an approximate switch boundary. Keep ALL reference
                # candidates; do not impose an extra candidate-list identity gate.
                # The requested gate is the resulting parent/recursive decision.
                bracket_diagnostics.extend(dict(label=label,**p[i],left_state=int(sl[i]),right_state=int(sr[i]),hybrid_bracket=bool(bracket[i])) for i in np.flatnonzero(bad))
            selected=np.flatnonzero(bracket);cuts=locate([p[i] for i in selected],sl[selected],sr[selected])
            for i,c in zip(selected,cuts):
                pp=p[i];all_candidates['hybrid'].append(dict(axis=pp['axis'],k=pp['k'],cut=c,identity=pp['identity']+':stationary0',kind='stationary'))
            stats['hybrid_stationary_candidates']+=len(cuts)
        # Physical child geometry and fresh allocator outputs, including boundary ownership.
        for mode,cs in all_candidates.items():
            parent=np.array(record['upper'])-record['lower'];lower=np.array(record['lower']);upper=np.array(record['upper'])
            boxes=[children(lower,upper,r['axis'],r['cut']) for r in cs];cl=np.array([b[0] for b in boxes]);ch=np.array([b[1] for b in boxes]);ell=ch-cl
            budgets=np.full(len(cs),record['n']);cc,_,_=audit.allocator.allocate(ell/parent,budgets)
            _,_,_,valid=route(audit.execution,audit.gate['margin'],audit.gate['threshold'],cc)
            if not np.all(valid) or not np.all(cc.sum(1)==record['n']):raise VerificationFailure('Constructed minimum candidate invalid or nonconserving.')
            stats['allocation_boundary_candidate_reassignments']+=int(np.sum(cc[:,0]!=np.array([r['k'] for r in cs])))
            # Direct teacher evaluated on actual child geometry and actual counts.
            h=np.exp((np.log(np.prod(parent))-np.log(record['n']))/record['d']);q=np.sqrt(np.sum(ell*ell,axis=2))/np.power(cc.astype(float),1/record['d'])
            ts=np.max(q,axis=1)/h+.05*np.max(np.log(np.max(ell,axis=2)/np.min(ell,axis=2)),axis=1)
            learned=predict(audit.finite,upstream(audit.models,audit.execution,np.tile(parent,(len(cs),1)),budgets,ell,cc))
            for i,candidate in enumerate(cs):candidate.update(teacher_score=float(ts[i]),hybrid_score=float(learned[i]),counts=cc[i].tolist(),child_lower=cl[i].tolist(),child_upper=ch[i].tolist())
        t,tbest=canonical(all_candidates['teacher'],'teacher');l,lbest=canonical(all_candidates['hybrid'],'hybrid')
        last_context.update(teacher=t,hybrid=l,teacher_best=tbest,candidates=all_candidates)
        equivalent=l['teacher_score']<=tbest+1e-12*max(1,tbest)
        check=dict(label=label,d=record['d'],n=record['n'],teacher_axis=t['axis'],hybrid_axis=l['axis'],teacher_cut=t['cut'],hybrid_cut=l['cut'],
            teacher_counts=json.dumps(t['counts']),hybrid_counts=json.dumps(l['counts']),teacher_identity=t['identity'],hybrid_identity=l['identity'],
            teacher_score=t['teacher_score'],hybrid_cut_teacher_score=l['teacher_score'],hybrid_score=l['hybrid_score'],
            exact_axis_cut=t['axis']==l['axis'] and t['cut']==l['cut'],allocation_match=t['counts']==l['counts'],identity_changed=t['identity']!=l['identity'],
            outside_equivalence=not equivalent,teacher_equivalent_coordinate_difference=equivalent and t['cut']!=l['cut'],
            max_child_geometry_difference=float(np.max(abs(np.array(t['child_lower'])-l['child_lower']))+np.max(abs(np.array(t['child_upper'])-l['child_upper']))),
            teacher_stationary_candidates=sum(c['kind']=='stationary' for c in all_candidates['teacher']),candidate_count_teacher=len(all_candidates['teacher']),candidate_count_hybrid=len(all_candidates['hybrid']))
        trace.append(check)
        if not equivalent:raise VerificationFailure('Fresh canonical split changes outside teacher equivalence.')
        return t,l,check
    rng=np.random.default_rng(SEED);fresh=[]
    for d in (2,3,4):
        for n in (3,7,16,32):
            scale=10**rng.uniform(-1,1);sides=scale*np.exp(rng.uniform(-.8,.8,d));lo=scale*rng.uniform(-.3,.3,d)
            fresh.append(dict(d=d,n=n,lower=lo.tolist(),upper=(lo+sides).tolist()))
    # Freshness is geometry, not merely a new seed.
    archived=[]
    for name in ('train_regions.json','heldout_regions.json'):archived+=json.loads((ROOT/'learned_split_proposer_smoke'/name).read_text())
    archive_keys={(r['d'],r['n'],tuple(r['lower']),tuple(r['upper'])) for r in archived}
    assert all((r['d'],r['n'],tuple(r['lower']),tuple(r['upper'])) not in archive_keys for r in fresh)
    (OUT/'fresh_parents.json').write_text(json.dumps(fresh,indent=2))
    config=dict(seed=SEED,phase1_parents=12,dimensions=[2,3,4],phase1_budgets=[3,7,16,32],recursive_budgets=[8,16],
        geometry_tolerance='256*float64_epsilon per coordinate normalized by root box side lengths; same axis/count/path required.',
        teacher='Existing analytic active-subpiece derivative scan + scipy brentq, not old fixed cut menu.',
        canonical='ascending axis/cut/identity, first score <= best + 1e-12*max(1,best)',
        scaffold='existing actual-allocator interval discovery, active-piece breakpoints, midpoint/DEFER localization, exhaustive final-float enumeration')
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    phase='phase1';completed_partition_count=0
    try:
        for i,r in enumerate(fresh):
            t,l,row=select(r,f'fresh_{i}');phase1.append(row)
            print(json.dumps(dict(phase=phase,completed=len(phase1),exact=row['exact_axis_cut'],equivalent=not row['outside_equivalence'],seconds=time.monotonic()-started)),flush=True)
        phase='recursive'
        for d in (2,3,4):
            for n in (8,16):
                scale=10**rng.uniform(-.5,.5);sides=scale*np.exp(rng.uniform(-.5,.5,d));lo=scale*rng.uniform(-.1,.1,d)
                root=dict(d=d,n=n,lower=lo.tolist(),upper=(lo+sides).tolist());name=f'd{d}_n{n}'
                stack=[('r',root,root,0)];leaves={'teacher':{},'hybrid':{}};nodes=[]
                while stack:
                    path,tr,lr,depth=stack.pop();deadline()
                    if tr['n']!=lr['n']:raise VerificationFailure('Recursive path budget mismatch.')
                    if tr['n']==1:
                        leaves['teacher'][path]=dict(tr,depth=depth);leaves['hybrid'][path]=dict(lr,depth=depth);continue
                    tt,tl,_=select(tr,name+':'+path+':teacher_parent')
                    if tr==lr:ll=tl
                    else:_,ll,_=select(lr,name+':'+path+':hybrid_parent')
                    same_axis=tt['axis']==ll['axis'];same_counts=tt['counts']==ll['counts']
                    err=max(float(np.max(abs(np.array(tt[k])-ll[k])/sides)) for k in ('child_lower','child_upper'))
                    equivalent=ll['teacher_score']<=tt['teacher_score']+1e-12*max(1,tt['teacher_score'])
                    row=dict(case=name,path=path,depth=depth,d=d,budget=tr['n'],teacher_axis=tt['axis'],hybrid_axis=ll['axis'],teacher_cut=tt['cut'],hybrid_cut=ll['cut'],
                        exact_decision=same_axis and same_counts and tt['cut']==ll['cut'],axis_match=same_axis,allocation_match=same_counts,
                        normalized_geometry_difference=err,outside_equivalence=not equivalent,counts=json.dumps(ll['counts']))
                    recursive.append(row);nodes.append(dict(path=path,teacher=tt,hybrid=ll))
                    if not same_axis or not same_counts or err>256*np.finfo(float).eps or not equivalent:
                        last_context=dict(case=name,path=path,teacher_parent=tr,hybrid_parent=lr,teacher=tt,hybrid=ll,comparison=row)
                        raise VerificationFailure('Meaningful recursive decision or downstream geometry divergence.')
                    for child in (1,0):
                        def cr(node):return dict(d=d,n=int(node['counts'][child]),lower=node['child_lower'][child],upper=node['child_upper'][child])
                        a,b=cr(tt),cr(ll)
                        if not (0<a['n']<tr['n'] and 0<b['n']<lr['n']) or sum(tt['counts'])!=tr['n'] or sum(ll['counts'])!=lr['n']:raise VerificationFailure('Recursive budgets not conserved.')
                        stack.append((path+str(child),a,b,depth+1))
                assert len(leaves['teacher'])==len(leaves['hybrid'])==n and set(leaves['teacher'])==set(leaves['hybrid'])
                leaf_error=max(float(np.max(abs(np.array(leaves['teacher'][p][k])-leaves['hybrid'][p][k])/sides)) for p in leaves['teacher'] for k in ('lower','upper'))
                depths=[r['depth'] for r in leaves['hybrid'].values()]
                partition_rows.append(dict(case=name,d=d,n=n,leaf_counts_exact=True,budgets_conserved=True,leaf_geometry_bit_exact=leaves['teacher']==leaves['hybrid'],
                    maximum_normalized_leaf_error=leaf_error,partition_matches=leaf_error<=256*np.finfo(float).eps,depth_min=min(depths),depth_median=float(np.median(depths)),depth_max=max(depths)))
                trees[name]=dict(root=root,nodes=nodes,leaves=leaves);completed_partition_count+=1
                print(json.dumps(dict(phase=phase,partition=name,splits=len(nodes),leaf_error=leaf_error,seconds=time.monotonic()-started)),flush=True)
    except (VerificationFailure,AssertionError) as e:
        failure=dict(phase=phase,reason=str(e) or 'Invariant assertion failed',context=last_context)
        (OUT/'first_failure.json').write_text(json.dumps(failure,indent=2))
    ref.write_csv(OUT/'parent_fidelity.csv',trace) if trace else None
    ref.write_csv(OUT/'recursive_fidelity.csv',recursive) if recursive else None
    ref.write_csv(OUT/'partition_geometry.csv',partition_rows) if partition_rows else None
    (OUT/'completed_recursive_trees.json').write_text(json.dumps(trees))
    (OUT/'stationary_candidate_list_differences.json').write_text(json.dumps(bracket_diagnostics,indent=2))
    passed=failure is None and len(phase1)==12 and completed_partition_count==6
    summary=dict(operator='continuous_cut_recursive_selector',status='VALIDATED_UNDER_TEACHER_EQUIVALENCE' if passed else 'STOPPED_NOT_VALIDATED',passed=passed,
        phase1_completed=len(phase1),phase1_exact_axis_cut=sum(r['exact_axis_cut'] for r in phase1),phase1_equivalent_coordinate_differences=sum(r['teacher_equivalent_coordinate_difference'] for r in phase1),
        phase1_identity_changes=sum(r['identity_changed'] for r in phase1),phase1_outside_equivalence=sum(r['outside_equivalence'] for r in phase1),
        recursive_partitions_completed=completed_partition_count,recursive_split_decisions=len(recursive),recursive_exact_decisions=sum(r['exact_decision'] for r in recursive),
        recursive_outside_equivalence=sum(r['outside_equivalence'] for r in recursive),final_partitions_matching=sum(r['partition_matches'] for r in partition_rows),
        statistics=stats,failure_reason=failure['reason'] if failure else None,failure_phase=failure['phase'] if failure else None,
        sources=sources,router_references=router_refs,implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,
        no_training=True,no_checkpoint_changes=True,no_global_search=True)
    summary=json.loads(json.dumps(summary,default=lambda value:value.item() if isinstance(value,np.generic) else str(value)))
    assert all(ref.sha(Path(p))==h for p,h in protected.items());summary['all_protected_hashes_unchanged']=True
    if passed:
        dest=OUT/'continuous_cut_recursive_selector_validated.json';dest.write_text(json.dumps(summary,indent=2));digest=ref.sha(dest)
        dest.with_suffix('.sha256').write_text(digest+'  '+dest.name+'\n');summary['assembly_sha256']=digest
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'continuous_cut_recursive_summary.txt').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
