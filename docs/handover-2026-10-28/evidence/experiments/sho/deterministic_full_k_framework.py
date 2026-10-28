"""Dimension-generic canonical DIRECT from the repository's pure-DIRECT lineage.

The search receives only an objective, box, budget and the existing loss gate.
Reference parameters are consumed exclusively by post-search validation.
"""
from __future__ import annotations
import json, math, time, hashlib, csv, types, inspect, ast
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from experiments.sho import run_full_k_transfer as prior
from experiments.sho import run_stage1b_n3_gradient_polishing as polish

OUT=Path('outputs/sho/stage1B/deterministic_full_k_framework')

@dataclass
class Rectangle:
    rid:int
    center:np.ndarray
    exponents:np.ndarray
    value:float
    @property
    def sides(self): return np.power(3.,-self.exponents)
    @property
    def diameter(self): return float(np.linalg.norm(self.sides)*.5)


def potentially_optimal(rectangles):
    # Canonical size-class minima and lower-right hull: same pure-DIRECT rule,
    # including repository tolerances, with vector-derived half diagonals.
    ordered=sorted(rectangles.values(),key=lambda r:(r.diameter,r.value,r.rid))
    groups=[]
    for r in ordered:
        if not groups or abs(r.diameter-groups[-1].diameter)>1e-14: groups.append(r)
        else:
            old=groups[-1];tol=5e-14*max(1.,abs(old.value),abs(r.value))
            if r.value<old.value-tol or (abs(r.value-old.value)<=tol and r.rid<old.rid):groups[-1]=r
    hull=[]
    for r in groups:
        while len(hull)>=2:
            a,b=hull[-2:]
            cross=(b.diameter-a.diameter)*(r.value-a.value)-(b.value-a.value)*(r.diameter-a.diameter)
            if cross < -5e-14*max(1.,abs(a.value),abs(b.value),abs(r.value)):hull.pop()
            else:break
        hull.append(r)
    selected=[]
    for i,r in enumerate(hull):
        lo=-math.inf if i==0 else (r.value-hull[i-1].value)/(r.diameter-hull[i-1].diameter)
        hi=math.inf if i==len(hull)-1 else (hull[i+1].value-r.value)/(hull[i+1].diameter-r.diameter)
        if hi>0 and max(lo,0.)<=hi+5e-14:selected.append(r.rid)
    return sorted(selected,key=lambda rid:(-rectangles[rid].diameter,rectangles[rid].value,rid))


def search(objective,lower,upper,cap=5000,gate=1e-4):
    lower=np.asarray(lower,float);upper=np.asarray(upper,float);d=len(lower)
    assert lower.shape==upper.shape and np.all(upper>lower)
    points=[];values=[];cache={};events=[];active={};next_id=1;iteration=0;skipped=0
    def evaluate(u):
        key=tuple(u.tolist())
        if key in cache:return cache[key]
        value=float(objective(lower+(upper-lower)*u))
        if not math.isfinite(value):raise ValueError('nonfinite objective')
        cache[key]=value;points.append(u.copy());values.append(value);return value
    root=np.full(d,.5);active[0]=Rectangle(0,root,np.zeros(d,int),evaluate(root))
    while len(values)<cap and min(values)>gate:
        selection=potentially_optimal(active);progress=False
        for rid in selection:
            parent=active[rid];axes=np.flatnonzero(parent.exponents==min(parent.exponents))
            proposals=[]
            for axis in axes:
                for sign in (-1,1):
                    point=parent.center.copy();point[axis]+=sign*parent.sides[axis]/3;proposals.append(point)
            needed=sum(tuple(p.tolist()) not in cache for p in proposals)
            if len(values)+needed>cap:skipped+=1;continue
            losses=[evaluate(p) for p in proposals]
            axis_order=sorted(range(len(axes)),key=lambda i:(min(losses[2*i:2*i+2]),int(axes[i])))
            del active[rid];e=parent.exponents.copy();children=[]
            for i in axis_order:
                axis=int(axes[i]);e[axis]+=1
                for j in (2*i,2*i+1):
                    r=Rectangle(next_id,proposals[j],e.copy(),losses[j]);active[next_id]=r;children.append(r);next_id+=1
            central=Rectangle(next_id,parent.center.copy(),e.copy(),parent.value)
            active[next_id]=central;children.append(central);next_id+=1
            vol=sum(float(np.prod(r.sides)) for r in children)
            assert np.isclose(vol,np.prod(parent.sides),rtol=1e-14,atol=0.)
            for r in children:
                assert np.all(r.center-r.sides/2>=parent.center-parent.sides/2-1e-14)
                assert np.all(r.center+r.sides/2<=parent.center+parent.sides/2+1e-14)
            events.append(dict(iteration=iteration,parent=rid,parent_center=parent.center.tolist(),parent_exponents=parent.exponents.tolist(),parent_value=parent.value,
                selected_ids=selection,axes=axes.tolist(),axis_order=[int(axes[i]) for i in axis_order],probe_values=losses,
                children=[dict(id=r.rid,center=r.center.tolist(),exponents=r.exponents.tolist(),value=r.value) for r in children],evaluations=len(values),best=min(values)))
            progress=True
            if min(values)<=gate:break
        iteration+=1
        if not progress:break
    index=int(np.argmin(values));theta=lower+(upper-lower)*points[index]
    # Generalize the EXISTING nominal spacing, width/sqrt(n), to width/n**(1/d).
    # No reference matrix, gradients or new localization criterion is used.
    radius=(upper-lower)/len(values)**(1/d)
    box_lower=np.maximum(lower,theta-radius);box_upper=np.minimum(upper,theta+radius)
    return dict(theta=theta,loss=values[index],evaluations=len(values),iterations=iteration,rectangles=active,
        events=events,points=np.stack(points),values=np.array(values),lower=box_lower,upper=box_upper,
        skipped_incomplete_splits=skipped,success=values[index]<=gate)


def deterministic_finish(objective,torch_objective,start):
    # Repository multidimensional Powell policy, existing 1500-evaluation control.
    best=[start['theta'].copy(),start['loss']]
    def counted(x):
        f=objective(x)
        if f<best[1]:best[:]=[np.array(x).copy(),f]
        return f
    result=minimize(counted,start['theta'],method='Powell',bounds=list(zip(start['lower'],start['upper'])),
        options=dict(xtol=1e-12,ftol=1e-15,maxiter=200,maxfev=1500,disp=False))
    # Execute a PRIVATE copy of the unchanged safeguarded Newton teacher.  Only
    # objective callback and its already-required runtime parameter dimension vary.
    namespace=dict(polish.__dict__)
    namespace['_scalar_loss']=lambda theta,position:torch_objective(theta)
    def derivatives(theta,position):
        t=theta.detach().clone().requires_grad_(True);f=torch_objective(t)
        g=torch.autograd.grad(f,t,create_graph=True)[0]
        H=torch.stack([torch.autograd.grad(g[i],t,retain_graph=True)[0] for i in range(len(t))])
        return f.detach(),g.detach(),(.5*(H+H.T)).detach()
    namespace['_gradient_hessian']=derivatives
    source=inspect.getsource(polish._newton_direction)
    assert 'torch.eye(2, dtype=DTYPE)' in source
    source=source.replace('torch.eye(2, dtype=DTYPE)','torch.eye(len(gradient), dtype=DTYPE)')
    exec(compile(source,polish.__file__,'exec'),namespace)
    run=types.FunctionType(polish._polish.__code__,namespace)
    endpoint=polish.N2Endpoint(0,best[0],best[1],'deterministic_Powell','full_K',False)
    finish=run(endpoint,torch.empty(0))
    return dict(n2_theta=best[0].tolist(),n2_loss=best[1],n2_evaluations=int(result.nfev),n2_message=str(result.message),
        n2_gate=bool(best[1]<1e-20),final_theta=finish.theta_history[-1].tolist(),final_loss=finish.loss_history[-1],
        n3_accepted_steps=finish.accepted_steps,n3_evaluations=finish.actual_loss_evaluations,n3_stop=finish.stop_reason)


def hashes():
    # Every checkpoint file in the workspace, not only selected manifests.
    result={}
    for ext in ('*.pt','*.pth','*.ckpt'):
        for p in Path('outputs').rglob(ext):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1<<20),b''):h.update(b)
            result[str(p)]=h.hexdigest()
    return result


def validate_lineage():
    from experiments.sho import run_stage1b_pure_direct_unique_scaling as old
    rng=np.random.default_rng(20261030);matches=0
    for _ in range(100):
        rects={};state=old.DirectState(0,{}, {},0,[],0,0,0,set())
        for rid in range(50):
            e=rng.integers(0,6,size=2);center=rng.random(2);value=float(rng.random())
            # Our normalized half-diameter and original physical half-diameter
            # differ by a common positive factor; use physical widths for audit.
            r=Rectangle(rid,center,e,value);rects[rid]=r
            q=old.DirectRectangle(rid,float(center[0]),float(center[1]),int(e[0]),int(e[1]),value);old._add_rectangle(state,q)
        got=potentially_optimal(rects);expected=old._potentially_optimal_ids(state)
        assert got==expected,(got,expected);matches+=1
    return dict(two_dimensional_po_replay_states=matches,errors=0)


def main():
    OUT.mkdir(exist_ok=True);started=time.time();protected=hashes()
    (OUT/'checkpoint_hashes_before.json').write_text(json.dumps(protected,indent=2))
    lineage=validate_lineage();rows=[];runs=[]
    for seed,count,label in ((prior.DATA_SEED,4,'original'),(prior.DATA_SEED+1,8,'fresh')):
        positions,_,coeff=prior.active_n1.im._make_exact_labeled_split(count,seed)
        for i in range(count):
            case=f'{label}_{i}';p=positions[i].numpy();pt=positions[i]
            objective=lambda x:float(prior.objective_numpy(np.asarray(x)[None,:],p)[0])
            t0=time.time();r=search(objective,prior.LOWER,prior.UPPER)
            (OUT/f'{case}_trace.json').write_text(json.dumps(r['events']))
            np.savez_compressed(OUT/f'{case}_samples.npz',normalized_points=r['points'],values=r['values'])
            # Everything below is evaluation-only; reference never enters search/finish.
            teacher=np.array([0.,1.,-1.,float(coeff[i])]);u=(teacher-prior.LOWER)/prior.WIDTH
            containing=[x for x in r['rectangles'].values() if np.all(abs(u-x.center)<=x.sides/2+1e-14)]
            assert containing
            cell=min(containing,key=lambda x:np.prod(x.sides))
            refined=[e for e in r['events'] if np.all(abs(u-np.array(e['parent_center']))<=3.**(-np.array(e['parent_exponents']))/2+1e-14)]
            row=dict(case=case,c_teacher=float(coeff[i]),n1_success=r['success'],n1_loss=r['loss'],n1_theta=r['theta'].tolist(),
                n1_evaluations=r['evaluations'],iterations=r['iterations'],rectangle_count=len(r['rectangles']),
                split_count=len(r['events']),teacher_containing_exponents=cell.exponents.tolist(),teacher_containing_sides=cell.sides.tolist(),
                teacher_region_refinements=len(refined),teacher_inside_n2_box=bool(np.all(teacher>=r['lower']) and np.all(teacher<=r['upper'])),
                local_lower=r['lower'].tolist(),local_upper=r['upper'].tolist(),
                per_axis_probe_counts=[sum(2*(j in e['axes']) for e in r['events']) for j in range(4)],
                duplicate_evaluations=len(r['points'])-len(np.unique(r['points'],axis=0)),
                volume_error=abs(sum(np.prod(x.sides) for x in r['rectangles'].values())-1),
                teacher_k=teacher.reshape(2,2).tolist())
            if r['success']:
                row.update(deterministic_finish(objective,lambda x:prior.objective_torch_native(x,pt),r))
                theta=np.array(row['final_theta']);diff=theta-teacher
                truth=prior.rollout_matrix(teacher.reshape(2,2),p[126:128],4096)
                pred=prior.rollout_matrix(theta.reshape(2,2),p[126:128],4096);err=pred-truth
                row.update(k_error=diff.reshape(2,2).tolist(),matrix_norm=float(np.linalg.norm(diff)),matrix_max_error=float(max(abs(diff))),
                    rollout_max_error=float(np.max(abs(err))),rollout_mse=float(np.mean(err*err)),recovered_k=theta.reshape(2,2).tolist())
            row['seconds']=time.time()-t0;rows.append(row)
            (OUT/'results_partial.json').write_text(json.dumps(rows,indent=2))
            print(case,'N1',r['evaluations'],r['loss'],'box',row['teacher_inside_n2_box'],'final',row.get('final_loss'),'Kerr',row.get('matrix_max_error'),flush=True)
    after=hashes();unchanged=protected==after
    (OUT/'checkpoint_hashes_after.json').write_text(json.dumps(after,indent=2))
    passed=all(r['n1_success'] and r['teacher_inside_n2_box'] and r.get('final_loss',math.inf)<1e-20 for r in rows) and unchanged
    report=dict(status='VALIDATED_DETERMINISTIC_TEACHER' if passed else 'FAILED_DETERMINISTIC_FRAMEWORK',lineage=lineage,cases=rows,
        n1_successes=sum(r['n1_success'] for r in rows),n2_gate_successes=sum(r.get('n2_gate',False) for r in rows),
        final_successes=sum(r.get('final_loss',math.inf)<1e-20 for r in rows),checkpoint_count=len(protected),checkpoints_unchanged=unchanged,
        algorithm='Canonical unbiased DIRECT: lower-right hull, vector ternary exponents, all longest-axis ties, paired center sampling, value-ranked sequential partition',
        gate=1e-4,cap=5000,local_region='clip(best +/- domain_width / evaluations**(1/d), global box)',
        downstream='existing deterministic bounded Powell settings (1500 calls), then safeguarded Newton even on a Powell endpoint above 1e-20, as original SHO precision-polishing lineage',
        seconds=time.time()-started)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    if passed:
        (OUT/'deterministic_full_k_teacher.json').write_text(json.dumps(dict(status='VALIDATED_DETERMINISTIC_TEACHER',implementation=__file__,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),report='report.json',validated_dimension=4,validation_only=True,no_learned_replacement=True),indent=2))
    print(report['status'],report['n1_successes'],report['final_successes'],flush=True)

if __name__=='__main__':main()
