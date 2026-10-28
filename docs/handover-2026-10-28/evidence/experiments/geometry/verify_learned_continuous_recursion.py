"""Fresh end-to-end learned continuous selector versus analytic recursion.

Read-only models. Teacher roots and choices never enter learned selection.
"""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq

from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_terminal_candidate_full import FrozenLocator
from experiments.geometry.train_stationary_terminal_candidate import TerminalCandidate
from experiments.geometry.train_subpiece_minimum_selector import SubpieceSelector
from experiments.geometry.train_interval_minimum_reducer import IntervalMinimumReducer
from experiments.geometry.train_parent_split_selector import ParentSplitSelector, load_locked, safe_json
from experiments.geometry.validate_stationary_three_state_router import DEFER
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.train_finite_split_score_smoke import upstream, predict
from experiments.sho.train_split_validity_smoke import route
from experiments.sho.train_recursive_region_splitter_poc import children

ROOT=ref.ROOT
OUT=ROOT/'learned_continuous_recursive_verification'
SEED=28352026
GEOMETRY_TOL=256*np.finfo(float).eps  # Established recursive geometry convention.


class VerificationFailure(RuntimeError): pass


class Selector:
    def __init__(self):
        self.audit=Audit(); self.locator=FrozenLocator(); self.refs={}
        meta=json.loads((ROOT/'stationary_terminal_candidate_training/candidate_manifest.json').read_text())
        assert meta['status']=='FROZEN_CANDIDATE' and ref.sha(Path(meta['checkpoint']))==meta['checkpoint_sha256']
        self.terminal=TerminalCandidate()
        self.terminal.load_state_dict(torch.load(meta['checkpoint'],weights_only=False)['state_dict'])
        self.terminal.eval().requires_grad_(False)
        self.refs['terminal']=dict(checkpoint=meta['checkpoint'],sha256=meta['checkpoint_sha256'])
        self.refs['locator']=self.locator.refs
        sm,sw=load_locked(ROOT/'subpiece_minimum_selector_training','subpiece_minimum_selector_locked.json')
        im,iw=load_locked(ROOT/'interval_minimum_reducer_training','interval_minimum_reducer_locked.json')
        pm,pw=load_locked(ROOT/'parent_split_selector_training','parent_split_selector_locked.json')
        self.sub=SubpieceSelector(self.audit.execution);self.sub.route.load_state_dict(sw)
        self.interval=IntervalMinimumReducer(self.sub);self.interval.head.load_state_dict(iw)
        self.parent=ParentSplitSelector(self.sub);self.parent.head.load_state_dict(pw)
        for name,model,m in [('subpiece',self.sub,sm),('interval',self.interval,im),('parent',self.parent,pm)]:
            model.eval().requires_grad_(False);self.refs[name]=dict(checkpoint=m['checkpoint'],sha256=m['checkpoint_sha256'])
        self.stats=dict(calls=0,intervals=0,subpieces=0,teacher_stationary=0,learned_stationary=0,
                        router_queries=0,locator_iterations=0,endpoint_defers=0,stationary_availability_differences=0,
                        interval_reductions=0,parent_reductions=0)
        self.context={};self.availability=[]

    def score(self,record,candidates):
        """Frozen whole split-score chain, evaluated on actual physical children."""
        a=self.audit;side=np.array(record['upper'])-record['lower'];n=record['n'];d=record['d']
        boxes=[children(np.array(record['lower']),np.array(record['upper']),e['axis'],e['cut']) for e in candidates]
        lo=np.array([b[0] for b in boxes]);hi=np.array([b[1] for b in boxes]);ell=hi-lo
        budgets=np.full(len(candidates),n);counts,_,_=a.allocator.allocate(ell/side,budgets)
        _,_,_,valid=route(a.execution,a.gate['margin'],a.gate['threshold'],counts)
        if not np.all(valid) or not np.all(counts.sum(1)==n):
            raise VerificationFailure('Invalid candidate or integer budget conservation failure.')
        h=np.exp((np.log(np.prod(side))-np.log(n))/d)
        q=np.sqrt(np.sum(ell*ell,axis=2))/np.power(counts.astype(float),1/d)
        teacher=np.max(q,axis=1)/h+.05*np.max(np.log(np.max(ell,axis=2)/np.min(ell,axis=2)),axis=1)
        learned=predict(a.finite,upstream(a.models,a.execution,np.tile(side,(len(candidates),1)),budgets,ell,counts))
        for i,e in enumerate(candidates):
            e.update(teacher_score=float(teacher[i]),score=float(learned[i]),counts=counts[i].tolist(),child_lower=lo[i].tolist(),child_upper=hi[i].tolist())

    def select(self,record,label):
        self.stats['calls']+=1;self.context=dict(label=label,parent=record)
        a=self.audit;pieces=[];teacher=[];learned=[]
        # Unchanged interval and active-subpiece construction scaffold.
        for axis in range(record['d']):
            a.setup(record,axis);n=record['n']
            edge=max(1e-15,4*np.spacing(abs(a.lo[axis])+abs(a.hi[axis]))/a.a)
            scan=np.linspace(edge,1-edge,max(2049,32*n+1));counts=a.counts(scan)[:,0]
            if np.any(np.diff(counts)<0) or counts[0]!=0 or counts[-1]!=n:
                raise VerificationFailure('Interval discovery is not complete/monotone.')
            lower=np.array([scan[np.flatnonzero(counts<k)[-1]] for k in range(1,n+1)])
            upper=np.array([scan[np.flatnonzero(counts>=k)[0]] for k in range(1,n+1)])
            for _ in range(60):
                mid=(lower+upper)/2;mask=a.counts(mid)[:,0]<np.arange(1,n+1)
                lower=np.where(mask,mid,lower);upper=np.where(mask,upper,mid)
            if not np.array_equal(a.counts(lower)[:,0],np.arange(n)) or not np.array_equal(a.counts(upper)[:,0],np.arange(1,n+1)):
                raise VerificationFailure('Allocation interval endpoint ownership changed.')
            for k in range(1,n):
                self.stats['intervals']+=1;br=a.breaks(upper[k-1],lower[k],k)
                for j,(l,r) in enumerate(zip(br[:-1],br[1:])):
                    identity=f'{axis}:{k}:{j}'
                    f=a.formula(np.array([(l+r)/2]),k);iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
                    A=1/(a.h*np.power(float(k if iq==0 else n-k),1/a.d));beta=.05*power
                    C=.05*np.log(a.a/a.omin) if power==1 else .05*np.log(a.omax/a.a) if power==-1 else .05*np.log(a.omax/a.omin)
                    coeff=[A,a.B,a.a,beta,iq,ia,C]
                    base=dict(axis=axis,k=k,piece=identity)
                    for kind,c in [('left',l),('right',r)]:
                        e=dict(base,identity=identity+':'+kind,kind=kind,cut=float(c))
                        teacher.append(e.copy());learned.append(e.copy())
                    # Existing analytic continuous teacher, for evaluation only.
                    ll=np.nextafter(l,r);rr=np.nextafter(r,l);roots=[]
                    if ll<rr:
                        xx=np.linspace(ll,rr,129);der=a.formula(xx,k)[1]
                        for h in np.flatnonzero(der[:-1]*der[1:]<0):
                            roots.append(float(brentq(lambda c:float(a.formula(c,k)[1][0]),xx[h],xx[h+1],xtol=5e-324,rtol=1e-15)))
                    roots=sorted(set(roots))
                    for ri,c in enumerate(roots):
                        teacher.append(dict(base,identity=identity+':stationary'+str(ri),kind='stationary',cut=c))
                    pieces.append(dict(**base,l=float(l),r=float(r),coeff=coeff,teacher_roots=roots))
        self.stats['subpieces']+=len(pieces)
        coeff=np.array([p['coeff'] for p in pieces]);L=np.array([p['l'] for p in pieces]);R=np.array([p['r'] for p in pieces])
        def signs(c):
            with torch.no_grad():
                return self.locator.sign(torch.from_numpy(np.column_stack((coeff[:,:4],c,coeff[:,4:6])))).numpy()
        sl=signs(L);sr=signs(R)
        bracket=(sl!=DEFER)&(sr!=DEFER)&(sl!=sr)&(L<R)
        self.stats['endpoint_defers']+=int(np.sum((sl==DEFER)|(sr==DEFER)))
        # Only learned signs decide whether to call the learned locator.
        ids=np.flatnonzero(bracket)
        if len(ids):
            brackets,cuts=self.locator.stationary_candidates(coeff[ids],L[ids],R[ids],self.terminal)
            self.stats['router_queries']+=int(brackets['queries'].sum())
            self.stats['locator_iterations']+=int(brackets['iterations'].sum())
            for idx,c in zip(ids,cuts):
                p=pieces[idx]
                learned.append(dict(axis=p['axis'],k=p['k'],piece=p['piece'],identity=p['piece']+':stationary0',kind='stationary',cut=float(c)))
        thas=np.array([bool(p['teacher_roots']) for p in pieces]);bad=thas!=bracket
        self.stats['stationary_availability_differences']+=int(bad.sum())
        self.availability.extend(dict(label=label,**pieces[i],learned_bracket=bool(bracket[i]),left_sign=int(sl[i]),right_sign=int(sr[i])) for i in np.flatnonzero(bad))
        self.stats['teacher_stationary']+=sum(len(p['teacher_roots']) for p in pieces)
        self.stats['learned_stationary']+=len(ids)
        self.score(record,teacher);self.score(record,learned)
        order=lambda e:(e['axis'],e['cut'],e['identity'])
        # Reference canonical choice uses full teacher list, without any learned filtering.
        best=min(e['teacher_score'] for e in teacher)
        t=next(e for e in sorted(teacher,key=order) if e['teacher_score']<=best+1e-12*max(1.,best))
        pools=defaultdict(list)
        for e in learned:pools[e['piece']].append(e)
        slots=[];ss=[];valid=[]
        for pool in pools.values():
            by={e['kind']:e for e in pool}
            row=[by['left'],by['right'],by.get('stationary')]
            slots.append(row);ss.append([e['score'] if e else 0 for e in row]);valid.append([e is not None for e in row])
        ss=np.array(ss);valid=np.array(valid)
        subidx=self.sub.select(self.sub.normalize(ss,valid),valid)
        ipools=defaultdict(list)
        for row,idx in zip(slots,subidx):
            e=row[idx]
            if e is None:raise VerificationFailure('Learned subpiece selector chose invalid candidate.')
            ipools[(e['axis'],e['k'])].append(e)
        ilists=[sorted(ipools[k],key=order) for k in sorted(ipools)]
        iidx,_,_=self.interval.reduce([np.array([e['score'] for e in pool]) for pool in ilists])
        winners=sorted([pool[idx] for pool,idx in zip(ilists,iidx)],key=order)
        pidx,_,_=self.parent.reduce([np.array([e['score'] for e in winners])])
        result=winners[pidx[0]]
        self.stats['interval_reductions']+=len(ilists);self.stats['parent_reductions']+=1
        self.context.update(teacher=t,learned=result,teacher_best=best,teacher_candidates=teacher,learned_candidates=learned,
                            pieces=pieces,interval_winners=winners)
        return t,result,best


def compare(t,l,best,label,root_sides):
    err=max(float(np.max(abs(np.array(t[k])-l[k])/root_sides)) for k in ('child_lower','child_upper'))
    return dict(label=label,teacher_axis=t['axis'],learned_axis=l['axis'],axis_match=t['axis']==l['axis'],
        teacher_cut=t['cut'],learned_cut=l['cut'],exact_cut=t['cut']==l['cut'],
        teacher_counts=json.dumps(t['counts']),learned_counts=json.dumps(l['counts']),allocation_match=t['counts']==l['counts'],
        teacher_identity=t['identity'],learned_identity=l['identity'],identity_match=t['identity']==l['identity'],
        teacher_score=t['teacher_score'],learned_score=l['score'],learned_teacher_score=l['teacher_score'],
        absolute_teacher_score_difference=abs(t['teacher_score']-l['teacher_score']),
        outside_equivalence=l['teacher_score']>best+1e-12*max(1.,best),
        normalized_child_geometry_difference=err)


def enforce(row):
    if not all(row[k] for k in ('axis_match','allocation_match','identity_match')) or row['outside_equivalence'] or row['normalized_child_geometry_difference']>GEOMETRY_TOL:
        raise VerificationFailure('Canonical split or recursive geometry gate failed.')


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    sel=Selector();rng=np.random.default_rng(SEED)
    def fresh(d,n):
        scale=10**rng.uniform(-.6,.6);sides=scale*np.exp(rng.uniform(-.8,.8,d));lo=scale*rng.uniform(-.2,.2,d)
        return dict(d=d,n=n,lower=lo.tolist(),upper=(lo+sides).tolist())
    phase1=[fresh(d,n) for d in (2,3,4) for n in (8,16,32)]
    recursive=[fresh(d,n) for d in (2,3,4) for n in (8,16,32)]
    old=[]
    for name in ('train_regions.json','heldout_regions.json'):
        old+=json.loads((ROOT/'learned_split_proposer_smoke'/name).read_text())
    old+=list(json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text()).values())
    key=lambda p:(p['d'],p['n'],tuple(p['lower']),tuple(p['upper']))
    oldkeys={key(p) for p in old}
    assert all(key(p) not in oldkeys for p in phase1+recursive)
    (OUT/'fresh_roots.json').write_text(safe_json(dict(seed=SEED,phase1=phase1,recursive=recursive)))
    (OUT/'config.json').write_text(safe_json(dict(geometry_tolerance=GEOMETRY_TOL,teacher='Existing deterministic analytic continuous reference (active-subpiece derivative scan and brentq)',
        learned='FrozenLocator + learned terminal + locked subpiece, interval, parent selectors; no enumeration or teacher choice substitution',
        phases=[9,9],dimensions=[2,3,4],budgets=[8,16,32])))
    quick=[];nodes=[];partitions=[];trees={};failure=None;phase='fresh_screen';context={}
    try:
        for i,record in enumerate(phase1):
            context=dict(phase=phase,index=i,parent=record)
            t,l,b=sel.select(record,f'fresh_{i}');side=np.array(record['upper'])-record['lower']
            row=compare(t,l,b,f'fresh_{i}',side);row.update(d=record['d'],budget=record['n']);quick.append(row)
            enforce(row)
            print(safe_json(dict(phase=phase,completed=len(quick),seconds=time.monotonic()-started)),flush=True)
        phase='recursion'
        for ri,root in enumerate(recursive):
            name=f'fresh_d{root["d"]}_n{root["n"]}';side=np.array(root['upper'])-root['lower']
            stack=[('r',root,root,0)];leaves={'teacher':{},'learned':{}};tree=[]
            while stack:
                path,tr,lr,depth=stack.pop();context=dict(phase=phase,case=name,path=path,teacher_parent=tr,learned_parent=lr)
                if tr['n']!=lr['n']:raise VerificationFailure('Tree-path budgets differ.')
                if tr['n']==1:
                    leaves['teacher'][path]=dict(tr,depth=depth);leaves['learned'][path]=dict(lr,depth=depth);continue
                t,l,b=sel.select(tr,name+':'+path)
                teacher_context=sel.context
                if tr!=lr:
                    _,l,_=sel.select(lr,name+':'+path+':learned_geometry')
                    # Evaluate the learned choice on reference geometry only for quality audit.
                    reevaluated=dict(l);sel.score(tr,[reevaluated])
                    if reevaluated['teacher_score']>b+1e-12*max(1.,b):
                        context.update(reference_geometry_score=reevaluated['teacher_score'],best=b)
                        raise VerificationFailure('Learned recursive cut outside teacher equivalence on reference geometry.')
                row=compare(t,l,b,name+':'+path,side);row.update(path=path,depth=depth,d=root['d'],budget=tr['n'])
                nodes.append(row);context.update(teacher_context=teacher_context,learned_context=sel.context,comparison=row)
                enforce(row)
                if sum(t['counts'])!=tr['n'] or sum(l['counts'])!=lr['n'] or any(n<1 for n in t['counts']+l['counts']):
                    raise VerificationFailure('Recursive budget conservation failure.')
                tree.append(dict(path=path,teacher=t,learned=l))
                for child in (1,0):
                    def cr(e):return dict(d=root['d'],n=e['counts'][child],lower=e['child_lower'][child],upper=e['child_upper'][child])
                    stack.append((path+str(child),cr(t),cr(l),depth+1))
            paths=set(leaves['teacher']);samepaths=paths==set(leaves['learned'])
            if not samepaths or len(paths)!=root['n']:raise VerificationFailure('Terminal leaf count/path mismatch.')
            err=max(float(np.max(abs(np.array(leaves['teacher'][p][k])-leaves['learned'][p][k])/side)) for p in paths for k in ('lower','upper'))
            depths={arm:sorted(e['depth'] for e in leaves[arm].values()) for arm in leaves}
            entry=dict(case=name,d=root['d'],N=root['n'],split_decisions=len(tree),tree_paths_match=samepaths,
                teacher_leaves=len(paths),learned_leaves=len(leaves['learned']),budget_conservation=True,
                depth_distribution_match=depths['teacher']==depths['learned'],depth_distribution=json.dumps(depths),
                leaf_geometry_bit_exact=leaves['teacher']==leaves['learned'],max_normalized_leaf_difference=err,
                partition_match=err<=GEOMETRY_TOL)
            partitions.append(entry);trees[name]=dict(root=root,nodes=tree,leaves=leaves)
            if not entry['partition_match'] or not entry['depth_distribution_match']:raise VerificationFailure('Final partition differs beyond established geometry convention.')
            print(safe_json(dict(phase=phase,partition=name,splits=len(tree),leaf_error=err,seconds=time.monotonic()-started)),flush=True)
    except (VerificationFailure,AssertionError) as ex:
        failure=dict(reason=str(ex) or 'Invariant assertion failed',phase=phase,context=context,selector_context=sel.context)
        (OUT/'first_failure.json').write_text(safe_json(failure))
    for filename,rows in [('fresh_parent_fidelity.csv',quick),('recursive_split_fidelity.csv',nodes),('partition_fidelity.csv',partitions)]:
        if rows:ref.write_csv(OUT/filename,rows)
    (OUT/'recursive_trees.json').write_text(safe_json(trees))
    (OUT/'stationary_availability_diagnostics.json').write_text(safe_json(sel.availability))
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    def metrics(rows):
        return dict(split_decisions=len(rows),exact_axis_matches=sum(r['axis_match'] for r in rows),
            exact_allocation_matches=sum(r['allocation_match'] for r in rows),exact_identity_matches=sum(r['identity_match'] for r in rows),
            exact_cut_matches=sum(r['exact_cut'] for r in rows),equivalent_coordinate_differences=sum(not r['exact_cut'] and not r['outside_equivalence'] for r in rows),
            outside_equivalence=sum(r['outside_equivalence'] for r in rows),
            max_normalized_geometry_difference=max((r['normalized_child_geometry_difference'] for r in rows),default=0),
            max_teacher_score_difference=max((r['absolute_teacher_score_difference'] for r in rows),default=0))
    passed=failure is None and len(quick)==9 and len(partitions)==9
    summary=dict(operator='learned_continuous_recursive_selector',status='VALIDATED_UNDER_TEACHER_EQUIVALENCE' if passed else 'STOPPED_NOT_VALIDATED',passed=passed,
        phase1=metrics(quick),recursive=metrics(nodes),completed_partitions=len(partitions),
        final_partition_matches=sum(r['partition_match'] for r in partitions),
        leaf_count_matches=sum(r['teacher_leaves']==r['learned_leaves'] for r in partitions),
        depth_distribution_matches=sum(r['depth_distribution_match'] for r in partitions),
        max_final_leaf_geometry_difference=max((r['max_normalized_leaf_difference'] for r in partitions),default=0),
        first_failure_reason=failure['reason'] if failure else None,failure_phase=phase if failure else None,
        stats=sel.stats,refs=sel.refs,elapsed_seconds=time.monotonic()-started,protected_files=len(protected),all_protected_hashes_unchanged=True,
        implementation_sha256=ref.sha(Path(__file__)),no_training=True,no_global_search=True)
    if passed:
        dest=OUT/'learned_continuous_recursive_selector_validated.json';dest.write_text(safe_json(summary));h=ref.sha(dest)
        dest.with_suffix('.sha256').write_text(h+'  '+dest.name+'\n');summary['assembly_sha256']=h
        (OUT/'remaining_deterministic_scaffold.txt').write_text('''Remaining deterministic operations (audit only; nothing learned here):
1. Recursive stack traversal, node/path bookkeeping, n==1 termination, child enqueue.
2. Axis enumeration; valid allocation-count enumeration; actual-allocator interval
   discovery by grid scan and bisection, including float64 endpoint ownership.
3. Active-subpiece discovery: geometry branch comparisons, norm/aspect switching
   equations and deterministic brentq breakpoint calculations (Audit.breaks).
4. Construction of local analytic coefficients and stationary-bracket eligibility
   from trusted opposite-sign endpoint states; endpoint candidate construction.
5. Coordinate-bank/state orchestration around the learned locator routers;
   nextafter/equality/movement predicates used as learned stopping inputs.
6. Canonical candidate sorting (axis, cut, identity), validity masks, role packing,
   shared-endpoint identity bookkeeping, and grouping by subpiece/interval/parent.
7. Frozen comparison execution loops and tolerance arithmetic; learned reducers
   route the decisions but loop scheduling/global-min first pass is scaffold.
8. Affine cut-to-child-box conversion; deterministic integer conservation after
   learned proportional allocation; validity control and budget bookkeeping.
9. Deterministic terminal-feature normalization and fraction-to-bracket conversion.
The analytic teacher root solve and teacher canonical selector run only for
reference evaluation. Float64 stationary-candidate enumeration is absent.
''')
    (OUT/'results.json').write_text(safe_json(summary));(OUT/'recursive_verification_summary.txt').write_text(safe_json(summary))
    (OUT/'protected_hashes.json').write_text(safe_json(protected))
    print(safe_json(summary),flush=True)


if __name__=='__main__':run()
