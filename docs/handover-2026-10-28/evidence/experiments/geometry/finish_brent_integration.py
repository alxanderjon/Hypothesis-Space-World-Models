"""Geometry regeneration and strictly gated parent/recursive verification."""
import inspect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.geometry.finish_learned_brent import ROOT, OUT, dump
from experiments.geometry import audit_exact_branch_brent as authority
from experiments.geometry.train_branch_switch_location import load_locked_models
from experiments.geometry.validate_stationary_three_state_router import load_router
from experiments.geometry.validate_terminal_candidate_full import original_parents
from experiments.geometry import validate_branch_boundary_baseline as baseline
from experiments.geometry.verify_learned_continuous_recursion import compare
from experiments.geometry.validate_locked_brent_linear import read,write,bits
from experiments.geometry.train_division_operator import sha


def geometry_key(p):
    return (p['d'],p['n'],tuple(p['lower']),tuple(p['upper']))


class GeometryRequests:
    """No cached cuts or teacher availability labels enter generation."""
    def __init__(self):
        self.audit=authority.Audit()
        self.transition,self.availability,self.refs=load_locked_models()
        self.router,self.router_refs=load_router()
        self.cache={}

    def perturbed_fixture(self,case):
        """Restore declared synthetic geometry; never read a target cut/label."""
        if not case.get('label','').startswith('previous_fresh_perturbations:'):
            return None
        name=case['label'].split(':',1)[1]
        records=read(ROOT/'branch_availability_false_positive_audit/fresh_boundary_scan_states.csv')
        hits=[r for r in records if r['case']==name and bits(float(r['pair_left']))==bits(case['left']) and bits(float(r['pair_right']))==bits(case['right'])]
        assert len(hits)==1
        provenance=hits[0]
        source=read(ROOT/'branch_switch_candidate_availability_training/availability_predictions.csv')[int(provenance['source_state'])]
        side=int(source['boundary_side']);piece=source['case'].split(':',1)[1]
        base=[r for r in self.axis(case['parent'],case['axis']) if r['piece']==piece and r['side']==side]
        assert len(base)==1
        base=base[0];l=base['subpiece_left'];u=base['subpiece_right']
        assert bits(l)==bits(float(source['lower'])) and bits(u)==bits(float(source['upper']))
        b=l if side==0 else u;shift=int(provenance['boundary_shift_ULPs'])
        shifted=np.uint64(int(bits(b))+shift).view(np.float64)
        if side==0:l=float(shifted)
        else:u=float(shifted)
        scan=np.linspace(np.nextafter(l,u),np.nextafter(u,l),129)
        c=scan[[0,1] if side==0 else [127,128]]
        a=self.audit;a.setup(case['parent'],case['axis']);k=case['count']
        f=a.formula(c,k);iq=f[3];ia=f[4];power=f[5][np.arange(2),ia]
        A=1/(a.h*np.power(np.where(iq==0,k,a.n-k).astype(float),1/a.d))
        x=np.column_stack((A,np.full(2,a.B),np.full(2,a.a),.05*power,c,iq,ia))
        changed=int(iq[0]!=iq[1] or ia[0]!=ia[1] or power[0]!=power[1])
        with torch.no_grad():
            terms=self.router.term_assembly.terms(torch.from_numpy(x)).numpy()
            m=(terms.sum(1)/abs(terms).sum(1)).reshape(1,2)
            transition=int(self.transition(torch.from_numpy(m)).argmax(1)[0])
            available=bool(self.availability(torch.tensor([[transition,changed,int(l<c[0]<c[1]<u),1]],dtype=torch.float64)).argmax(1)[0])
        return dict(base,left=float(c[0]),right=float(c[1]),subpiece_left=l,subpiece_right=u,
                    available=available,margin_left=float(m[0,0]),margin_right=float(m[0,1]),
                    declared_boundary_shift_ULPs=shift,source_state=int(provenance['source_state']))

    def axis(self,p,axis):
        key=(geometry_key(p),axis)
        if key in self.cache:return self.cache[key]
        a=self.audit;a.setup(p,axis);n=p['n']
        edge=max(1e-15,4*np.spacing(abs(a.lo[axis])+abs(a.hi[axis]))/a.a)
        scan=np.linspace(edge,1-edge,max(2049,32*n+1));counts=a.counts(scan)[:,0]
        low=np.array([scan[np.flatnonzero(counts<k)[-1]] for k in range(1,n+1)])
        high=np.array([scan[np.flatnonzero(counts>=k)[0]] for k in range(1,n+1)])
        for _ in range(60):
            mid=(low+high)/2;mask=a.counts(mid)[:,0]<np.arange(1,n+1)
            low=np.where(mask,mid,low);high=np.where(mask,high,mid)
        rows=[];inputs=[];flags=[]
        for k in range(1,n):
            br=a.breaks(high[k-1],low[k],k)
            for j,(l,r) in enumerate(zip(br[:-1],br[1:])):
                xx=np.linspace(np.nextafter(l,r),np.nextafter(r,l),129)
                for side,idx in ((0,[0,1]),(1,[127,128])):
                    c=xx[idx];f=a.formula(c,k);iq=f[3];ia=f[4];power=f[5][np.arange(2),ia]
                    A=1/(a.h*np.power(np.where(iq==0,k,n-k).astype(float),1/a.d))
                    inputs.extend(np.column_stack((A,np.full(2,a.B),np.full(2,a.a),.05*power,c,iq,ia)))
                    changed=bool(iq[0]!=iq[1] or ia[0]!=ia[1] or power[0]!=power[1])
                    flags.append([int(changed),int(l<c[0]<c[1]<r),1])
                    rows.append(dict(parent=p,axis=axis,count=k,left=float(c[0]),right=float(c[1]),piece=f'{axis}:{k}:{j}',side=side,
                                     subpiece_left=float(l),subpiece_right=float(r),signature_change=changed))
        with torch.no_grad():
            terms=self.router.term_assembly.terms(torch.from_numpy(np.array(inputs))).numpy()
            scale=abs(terms).sum(1);m=np.divide(terms.sum(1),scale,out=np.zeros(len(terms)),where=scale>0).reshape(-1,2)
            transition=self.transition(torch.from_numpy(m)).argmax(1).numpy()
            yes=self.availability(torch.from_numpy(np.column_stack((transition,flags)).astype(float))).argmax(1).numpy()
        for i,r in enumerate(rows):r.update(available=bool(yes[i]),margin_left=float(m[i,0]),margin_right=float(m[i,1]))
        self.cache[key]=rows
        return rows


def integrate(run, resume_regeneration=False):
    engine=run.engine
    generator=GeometryRequests()
    learned,code=engine.build(4)
    # End-to-end initial-zero and adjacent-float terminal controls. No roots or
    # later state transitions are fabricated for these immediate-return cases.
    terminal_controls=[]
    for fp in (0.,-0.,1.,-1.):
        for fc in (0.,-0.,1.,-1.):
            if fp!=0 and fc!=0:continue
            lo=.25;hi=.75
            def callback(c,fp=fp,fc=fc):return (fp if c==lo else fc),{}
            tr,tt,tn,why=authority.traced_brent(callback,lo,hi,-1)
            lr,lt,ln,lwhy=learned(callback,lo,hi,-1)
            run.gate(bits(tr)==bits(lr) and tt==lt and tn==ln and why==lwhy,
                     dict(reason='Initial-zero full solver control failed',fpre=fp,fcur=fc))
            terminal_controls.append(dict(kind='initial_zero',left_function_hex=fp.hex(),right_function_hex=fc.hex(),exact=True))
    for lo in (.125,.25,.5,.875):
        hi=float(np.nextafter(lo,np.inf))
        for sign in (1.,-1.):
            def callback(c,sign=sign):return (-sign if c==lo else sign),{}
            tr,tt,tn,why=authority.traced_brent(callback,lo,hi,-1)
            lr,lt,ln,lwhy=learned(callback,lo,hi,-1)
            run.gate(bits(tr)==bits(lr) and tt==lt and tn==ln and why==lwhy,
                     dict(reason='Adjacent-float full solver control failed',left=lo,right=hi))
            terminal_controls.append(dict(kind='adjacent_float',left_function_hex=(-sign).hex(),right_function_hex=sign.hex(),exact=True))
    write(OUT/'terminal_solver_controls.csv',terminal_controls)
    rng=np.random.default_rng(928202642)
    newparents=[];newcases=[]
    # New independent geometries, after all heads have been fitted and frozen.
    for d in (2,3,4):
        scale=10**rng.uniform(-1,1);lo=scale*rng.uniform(-.2,.2,d);side=scale*np.exp(rng.uniform(-.7,.7,d))
        p=dict(d=d,n=8,lower=lo.tolist(),upper=(lo+side).tolist());newparents.append(p)
        for axis in range(d):
            for r in generator.axis(p,axis):
                if not r['available']:continue
                cb=run.callback(r);root,_,_,_=authority.traced_brent(cb,r['left'],r['right'],len(run.cases)+len(newcases))
                newcases.append(dict(r,teacher_cut=root,label=f'new_independent_d{d}:'+r['piece']))
    dump(OUT/'new_independent_parents.json',newparents);dump(OUT/'new_independent_cases.json',newcases)
    first_new=len(run.cases)
    run.cases.extend(newcases)
    run.splits.update({i:'new_independent' for i in range(first_new,len(run.cases))})
    if not resume_regeneration:
        run.stage='E_fresh'
        run.replay(4,'E_complete_with_new_fresh')
    run.stage='F'
    rows=[]
    for i,case in enumerate(run.cases):
        requests=generator.axis(case['parent'],case['axis'])
        matches=[r for r in requests if r['count']==case['count'] and bits(r['left'])==bits(case['left']) and bits(r['right'])==bits(case['right'])]
        if not matches:
            fixture=generator.perturbed_fixture(case)
            if fixture is not None:
                matches=[fixture] if bits(fixture['left'])==bits(case['left']) and bits(fixture['right'])==bits(case['right']) else []
        if len(matches)!=1 or not matches[0]['available']:
            write(OUT/'branch_candidate_regeneration.csv',rows)
        run.gate(len(matches)==1 and matches[0]['available'],dict(reason='Geometry-regenerated branch candidate not available',case_id=i,case=case,matching_requests=matches))
        request=matches[0]
        got,tr,n,why=learned(run.callback(request),request['left'],request['right'],i)
        exact=got.hex()==float(case['teacher_cut']).hex()
        rows.append(dict(case_id=i,split=run.splits.get(i,'additional_fresh' if i<875 else 'new_independent'),
                         label=case.get('label',''),piece=request['piece'],teacher_cut=case['teacher_cut'],learned_cut=got,
                         teacher_hex=float(case['teacher_cut']).hex(),learned_hex=got.hex(),exact_coordinate=exact,queries=n,
                         declared_boundary_shift_ULPs=request.get('declared_boundary_shift_ULPs',0)))
        if not exact:
            write(OUT/'branch_candidate_regeneration.csv',rows)
            run.gate(False,dict(reason='Regenerated branch candidate coordinate mismatch',case_id=i,request=request,teacher=case['teacher_cut'],learned=got,trace=tr))
        if (i+1)%100==0:
            write(OUT/'branch_candidate_regeneration.csv',rows)
            print(json.dumps(dict(stage='F_regeneration',cases=i+1)),flush=True)
    write(OUT/'branch_candidate_regeneration.csv',rows)
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv')
    reports=[]
    for group in range(21):
        ids=sorted({int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==group})
        reports.append(dict(group=group,cases=len(ids),exact=sum(rows[i]['exact_coordinate'] for i in ids),case_ids=ids))
    dump(OUT/'conflict_groups.json',reports)
    dump(OUT/'recursive_fixture_candidate.json',[rows[i] for i,c in enumerate(run.cases) if 'recursive_d4_n16:r10' in c.get('label','')])
    run.lock('branch_switch_candidate_generator','AVAILABILITY_ONLY',dict(cases=len(rows),exact_coordinates=len(rows),missing=0,conflict_groups=21,new_independent=len(newcases)),
             dict(pipeline='Geometry -> allocation intervals -> active subpieces -> two interior endpoint scan pairs -> frozen transition/availability -> complete learned Brent.',
                  remaining_scaffold='Geometry interval/branch construction, including deterministic branch-boundary discovery. No cached teacher candidate is used for generation.',
                  geometry_implementation=str(Path(__file__)),geometry_implementation_sha256=sha(Path(__file__)),
                  availability_dependencies=generator.refs,sign_dependencies=generator.router_refs))

    # Replace exactly the old projected-boundary baseline assignment. Existing
    # smooth path, score chain, selectors and reference remain byte-for-byte code.
    cls=inspect.getsource(baseline.BoundarySelector)
    cls=cls.replace('class BoundarySelector(Selector):','class CompletedSelector(Selector):',1)
    old='cut=float(np.clip(boundary,c[0],c[1]))'
    assert cls.count(old)==1
    cls=cls.replace(old,'cut=self.locate_branch(record,p,c)')
    env=dict(baseline.__dict__)
    exec(compile(cls,'completed_branch_candidate_selector','exec'),env)
    C=env['CompletedSelector']
    def locate_branch(self,record,p,c):
        request=dict(parent=record,axis=p['axis'],count=p['k'],left=float(c[0]),right=float(c[1]))
        cut,_,_,_=learned(run.callback(request),float(c[0]),float(c[1]),-1)
        return cut
    C.locate_branch=locate_branch
    (OUT/'integrated_selector.py').write_text(cls)
    sel=C()
    run.stage='G'
    # Former failures first, then every archived/fresh parent previously available.
    first=json.loads((ROOT/'learned_continuous_recursive_verification/first_failure.json').read_text())['context']['parent']
    failure=json.loads((ROOT/'branch_boundary_candidate_baseline_validation/first_failure.json').read_text())
    parent_cases=[('former_d2_n8',first)]
    for k in ('teacher_parent','baseline_parent'):
        if k in failure['context']:parent_cases.append(('former_d4_recursive_'+k,failure['context'][k]))
    original=original_parents()
    archive_ids=sorted({int(r['parent']) for r in read(ROOT/'fixed_allocation_interval_audit/fixed_interval_score_structure.csv')})
    parent_cases.extend((f'archived_{i}',original[i]) for i in archive_ids)
    planned=json.loads((ROOT/'branch_boundary_candidate_baseline_validation/planned_cases.json').read_text())
    parent_cases.extend((label,p) for label,p in planned['parents'])
    parent_cases.extend((f'saved_candidate_parent_{i}',c['parent']) for i,c in enumerate(run.cases))
    parent_cases.extend((f'new_independent_{i}',p) for i,p in enumerate(newparents))
    seen=set();unique=[]
    for label,p in parent_cases:
        if geometry_key(p) not in seen:unique.append((label,p));seen.add(geometry_key(p))
    dump(OUT/'planned_parent_validation.json',unique)
    results=[]
    def validate_parent(label,p):
        t,l,best=sel.select(p,label)
        row=compare(t,l,best,label,np.array(p['upper'])-p['lower'])
        row.update(d=p['d'],n=p['n'])
        inv=sel.inventory[-1]
        bad=not all(row[k] for k in ('axis_match','allocation_match','identity_match','exact_cut')) or row['outside_equivalence'] or inv['missing_count'] or inv['extra_count']
        if bad:
            tm={e['identity']:e for e in sel.context['teacher_candidates']};lm={e['identity']:e for e in sel.context['learned_candidates']}
            differences=[dict(identity=key,teacher=tm.get(key),learned=lm.get(key)) for key in sorted(set(tm)|set(lm))
                         if key not in tm or key not in lm or bits(tm[key]['cut'])!=bits(lm[key]['cut'])]
            selected_generation=tm.get(t['identity']) is not None and (t['identity'] not in lm or bits(tm[t['identity']]['cut'])!=bits(lm[t['identity']]['cut']))
            selected=lm.get(t['identity'])
            origin=('missing candidate' if selected is None else
                    'regenerated branch-switch candidate' if selected.get('branch_boundary_baseline') else
                    'unchanged smooth stationary candidate' if selected.get('kind')=='stationary' else 'endpoint') if selected_generation else 'locked selection/reduction'
            results.append(row);write(OUT/'parent_validation.csv',results)
            run.gate(False,dict(reason='Exact parent split gate failed',label=label,parent=p,comparison=row,
                first_differing_operation='candidate generation' if selected_generation else 'selection/reduction',
                differing_component=origin,
                candidate_coordinate_differences=differences,teacher_selection=t,learned_selection=l,selector_context=sel.context))
        return row,t,l
    for i,(label,p) in enumerate(unique):
        row,_,_=validate_parent(label,p);results.append(row)
        write(OUT/'parent_validation.csv',results)
        if (i+1)%10==0:print(json.dumps(dict(stage='G_parents',completed=i+1,total=len(unique))),flush=True)
    run.lock('learned_continuous_parent_selector','PARTIALLY_VALIDATED',dict(parents=len(results),axis_errors=0,allocation_errors=0,identity_errors=0,coordinate_errors=0,score_errors=0),
             'Exact parent comparison; unmodified smooth stationary machinery plus regenerated branch candidates and locked reducers.')
    run.stage='H'
    roots=list(planned['recursive'])+[(f'new_independent_recursive_{i}',p) for i,p in enumerate(newparents)]
    node_rows=[];partition_rows=[]
    for label,root in roots:
        stack=[('r',root,0)];leaves=[]
        while stack:
            path,p,depth=stack.pop()
            if p['n']==1:leaves.append(dict(path=path,parent=p,depth=depth));continue
            row,t,l=validate_parent(label+':'+path,p);row.update(path=path,depth=depth)
            run.gate(sum(t['counts'])==p['n'] and sum(l['counts'])==p['n'] and min(l['counts'])>0,
                     dict(reason='Budget conservation failed',parent=p,teacher=t,learned=l))
            run.gate(t['child_lower']==l['child_lower'] and t['child_upper']==l['child_upper'],
                     dict(reason='Exact child geometry mismatch',parent=p,teacher=t,learned=l))
            node_rows.append(row);write(OUT/'recursive_nodes.csv',node_rows)
            for j in (1,0):
                stack.append((path+str(j),dict(d=p['d'],n=l['counts'][j],lower=l['child_lower'][j],upper=l['child_upper'][j]),depth+1))
        run.gate(len(leaves)==root['n'],dict(reason='Leaf count mismatch',root=root))
        partition_rows.append(dict(label=label,leaves=len(leaves),exact_geometry=True,exact_topology=True,budget_conserved=True))
        write(OUT/'recursive_partitions.csv',partition_rows)
        dump(OUT/(label+'_leaves.json'),leaves)
    run.lock('learned_continuous_recursive_selector','NOT_VALIDATED',dict(partitions=len(partition_rows),split_decisions=len(node_rows),errors=0),
             'Exact coordinates, axis, allocation, identities, child geometry and paths at every split; exact leaves and budgets.')
