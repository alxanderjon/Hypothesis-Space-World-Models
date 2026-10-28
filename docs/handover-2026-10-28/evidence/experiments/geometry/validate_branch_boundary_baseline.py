"""End-to-end projected-boundary candidate baseline, read-only learned models."""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from experiments.geometry.verify_learned_continuous_recursion import Selector,VerificationFailure,compare,GEOMETRY_TOL
from experiments.geometry.train_branch_switch_location import load_locked_models,json_write
from experiments.geometry.validate_terminal_candidate_full import original_parents
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv

OUT=ROOT/'branch_boundary_candidate_baseline_validation'
SEED=28442026


class BoundarySelector(Selector):
    def __init__(self):
        super().__init__();self.transition,self.combiner,self.branch_refs=load_locked_models()
        self.branch_rows=[];self.inventory=[];self.combiner_states=0

    def select(self,record,label):
        # Executes the unchanged smooth stationary path and analytic teacher.
        # The old learned choice is discarded; only its generated candidate list
        # and unchanged analytic reference are retained.
        teacher,_,best=super().select(record,label)
        context=self.context;a=self.audit;learned=context['learned_candidates'];pieces=context['pieces']
        requests=[];inputs=[];discrete=[]
        for p in pieces:
            a.setup(record,p['axis']);l,r=p['l'],p['r'];scan=np.linspace(np.nextafter(l,r),np.nextafter(r,l),129)
            for side in (0,1):
                ii=[0,1] if side==0 else [127,128];c=scan[ii];f=a.formula(c,p['k']);iq=f[3];ia=f[4];power=f[5][np.arange(2),ia]
                A=1/(a.h*np.power(np.where(iq==0,p['k'],a.n-p['k']).astype(float),1/a.d))
                inputs.extend(np.column_stack((A,np.full(2,a.B),np.full(2,a.a),.05*power,c,iq,ia)))
                signature=int(iq[0]!=iq[1] or ia[0]!=ia[1] or power[0]!=power[1]);valid=int(l<c[0]<c[1]<r)
                discrete.append([signature,valid,1]);requests.append((p,side,c))
        with torch.no_grad():
            terms=self.locator.sign.term_assembly.terms(torch.from_numpy(np.array(inputs))).numpy()
            den=abs(terms).sum(1);m=np.divide(terms.sum(1),den,out=np.zeros(len(terms)),where=den>0).reshape(-1,2)
            transition=self.transition(torch.from_numpy(m)).argmax(1).numpy()
            availability=self.combiner(torch.from_numpy(np.column_stack((transition,discrete)).astype(float))).argmax(1).numpy()
        self.combiner_states+=len(requests)
        generated=defaultdict(list)
        for i in np.flatnonzero(availability):
            p,side,c=requests[i];boundary=p['l'] if side==0 else p['r'];cut=float(np.clip(boundary,c[0],c[1]))
            e=dict(axis=p['axis'],k=p['k'],piece=p['piece'],identity=p['piece']+':stationary0',kind='stationary',cut=cut,branch_boundary_baseline=True)
            generated[p['piece']].append(e)
            self.branch_rows.append(dict(label=label,dimension=record['d'],budget=record['n'],piece=p['piece'],side=side,
                boundary=boundary,scan_left=float(c[0]),scan_right=float(c[1]),projected_cut=cut,
                margin_left=m[i,0],margin_right=m[i,1],outside_pair=not c[0]<=cut<=c[1]))
        # Preserve all smooth candidates. Never silently replace or discard one
        # if a new class collides with the current single-stationary slot.
        bypiece=defaultdict(list)
        for e in learned:
            if e['kind']=='stationary':bypiece[e['piece']].append(e)
        for key,entries in generated.items():
            if bypiece[key] or len(entries)>1:
                context['candidate_class_collision']=dict(piece=key,smooth=bypiece[key],branch=entries)
                raise VerificationFailure('Multiple stationary candidates collide with the locked three-slot subpiece interface.')
            learned.extend(entries)
        self.score(record,learned)
        tm={e['identity']:e for e in context['teacher_candidates']};lm={e['identity']:e for e in learned}
        missing=sorted(set(tm)-set(lm));extra=sorted(set(lm)-set(tm))
        self.inventory.append(dict(label=label,teacher_candidates=len(tm),baseline_candidates=len(lm),missing=json.dumps(missing),extra=json.dumps(extra),
            missing_count=len(missing),extra_count=len(extra),branch_candidates=sum(len(v) for v in generated.values())))
        for row in self.branch_rows:
            if row['label']!=label:continue
            identity=row['piece']+':stationary0';t=tm.get(identity);l=lm[identity]
            row.update(teacher_identity_exists=t is not None,teacher_cut=t['cut'] if t else None,
                exact_coordinate=t['cut']==l['cut'] if t else False,
                coordinate_difference=abs(t['cut']-l['cut']) if t else None,
                coordinate_ULPs=abs(int(np.float64(t['cut']).view(np.uint64))-int(np.float64(l['cut']).view(np.uint64))) if t else None,
                teacher_score_difference=abs(t['teacher_score']-l['teacher_score']) if t else None,
                outside_equivalence=l['teacher_score']>t['teacher_score']+1e-12*max(1.,t['teacher_score']) if t else True,
                allocation_changed=t['counts']!=l['counts'] if t else True)
        slots=[];pools=defaultdict(list)
        for e in learned:pools[e['piece']].append(e)
        for pool in pools.values():
            by={e['kind']:e for e in pool};slots.append([by['left'],by['right'],by.get('stationary')])
        ss=np.array([[e['score'] if e else 0 for e in row] for row in slots]);valid=np.array([[e is not None for e in row] for row in slots])
        indices=self.sub.select(self.sub.normalize(ss,valid),valid);ip=defaultdict(list)
        for row,j in zip(slots,indices):
            e=row[j];ip[(e['axis'],e['k'])].append(e)
        order=lambda e:(e['axis'],e['cut'],e['identity'])
        ilists=[sorted(ip[k],key=order) for k in sorted(ip)]
        chosen,_,_=self.interval.reduce([np.array([e['score'] for e in pool]) for pool in ilists])
        winners=sorted([pool[j] for pool,j in zip(ilists,chosen)],key=order)
        chosen,_,_=self.parent.reduce([np.array([e['score'] for e in winners])]);result=winners[chosen[0]]
        context.update(learned=result,learned_candidates=learned,interval_winners=winners,candidate_inventory=self.inventory[-1])
        return teacher,result,best


def enforce(row):
    if not all(row[k] for k in ('axis_match','allocation_match','identity_match')) or row['outside_equivalence'] or row['normalized_child_geometry_difference']>GEOMETRY_TOL:
        raise VerificationFailure('Canonical candidate/axis/allocation/score or geometry gate failed.')


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('manifest','locked','validated'))})
    json_write(OUT/'protected_hashes.json',protected)
    sel=BoundarySelector();rng=np.random.default_rng(SEED)
    def fresh(d,n):
        scale=10**rng.uniform(-1,1);lo=scale*rng.uniform(-.3,.3,d);sides=scale*np.exp(rng.uniform(-1,1,d))
        return dict(d=d,n=n,lower=lo.tolist(),upper=(lo+sides).tolist())
    fixture=json.loads((ROOT/'learned_continuous_recursive_verification/first_failure.json').read_text())['context']['parent']
    oldfresh=list(json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text()).values())
    originals=original_parents();archive_ids=sorted({int(r['parent']) for r in csv.DictReader((ROOT/'fixed_allocation_interval_audit/fixed_interval_score_structure.csv').open())})
    parent_cases=[('previous_d2_n8',fixture)]+[(f'new_d{d}_n{n}',fresh(d,n)) for d in (2,3,4) for n in (8,16,32)]
    parent_cases += [(f'existing_fresh_{i}',r) for i,r in enumerate(oldfresh)]
    parent_cases += [(f'archived_{i}',originals[i]) for i in archive_ids]
    recursive=[('previous_d2_n8_recursive',fixture)]+[(f'recursive_d{d}_n{n}',fresh(d,n)) for d in (2,3,4) for n in (8,16)]
    json_write(OUT/'planned_cases.json',dict(seed=SEED,parents=parent_cases,recursive=recursive,geometry_tolerance=GEOMETRY_TOL,
        coordinate_convention='Raw coordinate differences reported separately; canonical change means axis/allocation/candidate identity or score outside existing equivalence.'))
    comparisons=[];nodes=[];partitions=[];trees={};failure=None;context={};phase='parent'
    try:
        for i,(label,record) in enumerate(parent_cases):
            context=dict(label=label,parent=record);t,l,best=sel.select(record,label)
            row=compare(t,l,best,label,np.array(record['upper'])-record['lower']);row.update(d=record['d'],budget=record['n'])
            comparisons.append(row);enforce(row)
            if sel.inventory[-1]['missing_count'] or sel.inventory[-1]['extra_count']:
                raise VerificationFailure('Candidate inventory differs from analytic teacher.')
            if i%10==0:print(json.dumps(dict(phase='parents',completed=i+1,total=len(parent_cases),seconds=time.monotonic()-start)),flush=True)
        phase='recursive'
        for label,root in recursive:
            sides=np.array(root['upper'])-root['lower'];stack=[('r',root,root,0)];leaves={'teacher':{},'baseline':{}};tree=[]
            while stack:
                path,tr,lr,depth=stack.pop();context=dict(label=label,path=path,teacher_parent=tr,baseline_parent=lr)
                if tr['n']!=lr['n']:raise VerificationFailure('Recursive child budget mismatch.')
                if tr['n']==1:
                    leaves['teacher'][path]=dict(tr,depth=depth);leaves['baseline'][path]=dict(lr,depth=depth);continue
                t,l,best=sel.select(tr,label+':'+path);teacher_context=sel.context
                if tr!=lr:
                    _,l,_=sel.select(lr,label+':'+path+':baseline_geometry')
                    checked=dict(l);sel.score(tr,[checked])
                    if checked['teacher_score']>best+1e-12*max(1.,best):raise VerificationFailure('Baseline cut outside equivalence on teacher geometry.')
                row=compare(t,l,best,label+':'+path,sides);row.update(path=path,depth=depth,d=root['d'],budget=tr['n']);nodes.append(row)
                context['reference_selector_context']=teacher_context;enforce(row)
                if sum(l['counts'])!=lr['n'] or sum(t['counts'])!=tr['n'] or min(l['counts']+t['counts'])<1:raise VerificationFailure('Budget conservation violation.')
                tree.append(dict(path=path,teacher=t,baseline=l))
                for j in (1,0):
                    def child(e):return dict(d=root['d'],n=e['counts'][j],lower=e['child_lower'][j],upper=e['child_upper'][j])
                    stack.append((path+str(j),child(t),child(l),depth+1))
            tp=set(leaves['teacher']);bp=set(leaves['baseline']);depth_t=sorted(e['depth'] for e in leaves['teacher'].values());depth_b=sorted(e['depth'] for e in leaves['baseline'].values())
            if tp!=bp or len(tp)!=root['n']:raise VerificationFailure('Leaf paths/count mismatch.')
            err=max(float(np.max(abs(np.array(leaves['teacher'][p][key])-leaves['baseline'][p][key])/sides)) for p in tp for key in ('lower','upper'))
            p=dict(label=label,d=root['d'],N=root['n'],leaf_count=len(tp),leaf_paths_match=tp==bp,budget_conservation=True,
                depth_distribution_match=depth_t==depth_b,max_normalized_leaf_difference=err,leaf_geometry_bit_exact=leaves['teacher']==leaves['baseline'])
            partitions.append(p);trees[label]=dict(root=root,nodes=tree,leaves=leaves)
            if err>GEOMETRY_TOL or depth_t!=depth_b:raise VerificationFailure('Leaf geometry or depth mismatch.')
            print(json.dumps(dict(phase='recursive',completed=label,seconds=time.monotonic()-start)),flush=True)
    except (VerificationFailure,AssertionError) as ex:
        failure=dict(reason=str(ex) or 'assertion failed',phase=phase,context=context,selector_context=sel.context)
        json_write(OUT/'first_failure.json',failure)
    def write(name,rows):
        if rows:write_csv(OUT/name,rows)
    write('parent_fidelity.csv',comparisons);write('recursive_fidelity.csv',nodes);write('partition_fidelity.csv',partitions)
    write('branch_candidate_coordinates.csv',sel.branch_rows);write('candidate_inventory.csv',sel.inventory);json_write(OUT/'completed_trees.json',trees)
    assert all(sha(Path(p))==h for p,h in protected.items())
    def summarize(rows):
        return dict(decisions=len(rows),axis_changes=sum(not r['axis_match'] for r in rows),allocation_changes=sum(not r['allocation_match'] for r in rows),
            candidate_identity_changes=sum(not r['identity_match'] for r in rows),raw_cut_coordinate_changes=sum(not r['exact_cut'] for r in rows),
            outside_equivalence=sum(r['outside_equivalence'] for r in rows),max_teacher_score_difference=max((r['absolute_teacher_score_difference'] for r in rows),default=0.),
            max_normalized_geometry_difference=max((r['normalized_child_geometry_difference'] for r in rows),default=0.))
    passed=failure is None and len(partitions)==len(recursive) and len(comparisons)==len(parent_cases)
    result=dict(operator='branch_boundary_candidate_baseline',status='VALIDATED_UNDER_TEACHER_EQUIVALENCE' if passed else 'STOPPED_NOT_VALIDATED',passed=passed,
        planned_parent_cases=len(parent_cases),planned_recursive_partitions=len(recursive),parents=summarize(comparisons),recursive=summarize(nodes),
        completed_partitions=len(partitions),branch_candidates=len(sel.branch_rows),branch_coordinate_differences=sum(not r.get('exact_coordinate',False) for r in sel.branch_rows),
        max_branch_coordinate_difference=max((r.get('coordinate_difference') or 0 for r in sel.branch_rows),default=0),
        branch_score_equivalence_failures=sum(r.get('outside_equivalence',False) for r in sel.branch_rows),
        candidate_inventory_misses=sum(r['missing_count'] for r in sel.inventory),candidate_inventory_extras=sum(r['extra_count'] for r in sel.inventory),
        first_failure=failure['reason'] if failure else None,failure_phase=phase if failure else None,
        protected_files=len(protected),all_protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start,
        refs=sel.refs,branch_refs=sel.branch_refs,no_training=True,no_global_search=True,implementation_sha256=sha(Path(__file__)))
    json_write(OUT/'results.json',result)
    if passed:
        path=OUT/'branch_boundary_candidate_baseline_validated.json';json_write(path,result);path.with_suffix('.sha256').write_text(sha(path)+'  '+path.name+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
