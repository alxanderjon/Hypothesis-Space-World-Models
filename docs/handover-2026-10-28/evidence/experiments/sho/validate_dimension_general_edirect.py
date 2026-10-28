"""Structural controls and stop-on-failure full-K deterministic E-DIRECT replay."""
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import dimension_general_edirect as ed
from experiments.sho.deterministic_full_k_framework import deterministic_finish, hashes, prior

OUT = Path('outputs/sho/stage1B/dimension_general_edirect')


def structural():
    rows = []; rng = np.random.default_rng(20261101)
    for d in (2, 3, 4, 8):
        v = ed.directions(d)
        assert np.array_equal(v, ed.directions(d))
        assert len(np.unique(v, axis=0))==len(v)
        assert np.max(abs(np.linalg.norm(v, axis=1)-1))<1e-14
        lo = -np.arange(1,d+1, dtype=float); hi = np.arange(1,d+1, dtype=float)
        b, ids = ed.boundary(d, 2000, 9101, lo, hi)
        assert np.array_equal(b, ed.boundary(d, 2000, 9101, lo, hi)[0])
        assert np.isfinite(b).all() and np.all(b>=lo) and np.all(b<=hi)
        for point, (axis, side) in zip(b, ids):
            assert point[axis]==(hi[axis] if side else lo[axis])
        p = ed.pool(d, 20, 13, 7)
        assert np.array_equal(p, ed.pool(d, 20, 13, 7))
        assert len(p)==len(np.unique(p,axis=0)) and np.isfinite(p).all()
        assert np.all((p>=0)&(p<=1))
        assert np.array_equal(p, np.unique(p,axis=0))
        # Historical nearest-neighbor/coverage quantities, reported without a
        # newly invented optimization gate. This only validates sampling structure.
        query = rng.normal(size=(4096,d)); query /= np.linalg.norm(query,axis=1)[:,None]
        sphere_dist = cKDTree(v).query(query)[0]
        row = dict(d=d,directions=len(v),raw_boundary=len(b),unique_boundary=len(np.unique(b,axis=0)),
                   pool_count=len(p),radius_max_error=float(np.max(abs(np.linalg.norm(v,axis=1)-1))),
                   directional_coverage=float(max(sphere_dist)),direction_mean_norm=float(np.linalg.norm(v.mean(axis=0))),
                   face_counts=[int(np.sum(np.all(ids==[a,s],axis=1))) for a in range(d) for s in (0,1)],
                   deterministic=True,merged_duplicates=0)
        if d==2:
            points=rng.random((30,2)); losses=rng.random(30)
            assert np.array_equal(p, ed.teacher._candidate_pool(20,13,7))
            assert np.array_equal(ed.local_greedy(points,losses),ed.teacher._local_greedy_candidates(points,losses))
            api=ed.composition(2)
            assert np.array_equal(api['_cell_proposals'](points,losses,13,7),ed.teacher._cell_proposals(points,losses,13,7))
            assert np.array_equal(api['_greedy_proposal'](points,losses,13,7),ed.teacher._greedy_proposal(points,losses,13,7))
            row['historical_pool_local_cell_greedy_bit_exact']=True
        rows.append(row)
    return rows


def cases():
    rng=np.random.default_rng(20261102)
    positions,_,coeff=prior.active_n1.im._make_exact_labeled_split(32,prior.DATA_SEED)
    result=[]
    for i in range(32):
        p=positions[i].numpy()
        states=np.column_stack((p[:127],p[1:128]))
        result.append((f'companion_{i}', 'companion',np.array([[0.,1.],[-1.,float(coeff[i])]]),states))
    # Stable non-companion state transitions use the same mean-squared one-step
    # two-component residual. Generic K cannot be generated as overlapping scalar
    # delay pairs: observe both components, rather than secretly fixing row one.
    for category,count in (('interior',64),('single_face',64),('multi_face',32),('edge',32),('corner',32)):
        for i in range(count):
            for attempt in range(100000):
                k=rng.uniform(-1.7,1.7,(2,2))
                if category=='single_face':
                    k.flat[i%4]=(-1 if (i//4)%2 else 1)*(1.99-rng.uniform(.001,.049))
                elif category in ('multi_face','edge','corner'):
                    # Near a stable nilpotent corner, then free perturbations.
                    a=1.99-rng.uniform(.001,.02)
                    k=np.array([[a,a],[-a,-a]])+rng.uniform(-.0004,.0004,(2,2))
                    if i%2:k=-k
                    if i%4>=2:k=k.T
                    if category=='multi_face':
                        k[0,0]*=.9; k[1,1]*=.9
                    if category=='edge':k.flat[i%4]*=.95
                if np.max(abs(k))>=1.99 or np.max(abs(np.linalg.eigvals(k)))>=1:continue
                z=rng.normal(size=2); states=[z.copy()]
                for _ in range(127):z=k@z;states.append(z.copy())
                states=np.asarray(states)
                if np.isfinite(states).all() and np.linalg.matrix_rank(states[:-1])==2:break
            else:raise RuntimeError('could not construct stable identifiable '+category)
            result.append((f'{category}_{i}',category,k,states))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True); started=time.time()
    before=hashes();(OUT/'checkpoint_hashes_before.json').write_text(json.dumps(before,indent=2))
    controls=structural();(OUT/'structural.json').write_text(json.dumps(controls,indent=2))
    data=cases()
    np.savez_compressed(OUT/'validation_cases.npz',names=np.array([x[0] for x in data]),
                        categories=np.array([x[1] for x in data]),teachers=np.stack([x[2] for x in data]),
                        **{f'prefix_{i}':x[3] for i,x in enumerate(data)})
    rows=[];stop=None
    print('structural PASS; planned',len(data),'cases',flush=True)
    for index,(name,category,k,states) in enumerate(data):
        x=states[:-1].copy(); y=states[1:].copy(); xt=torch.from_numpy(x);yt=torch.from_numpy(y)
        def objective(theta):
            pred=np.einsum('ij,nj->ni',np.asarray(theta).reshape(2,2),x)
            return float(np.mean((pred-y)**2))
        def torch_objective(theta):return torch.mean((xt@theta.reshape(2,2).T-yt)**2)
        t0=time.time();r=ed.search(objective,prior.LOWER,prior.UPPER,index)
        np.savez_compressed(OUT/f'{name}_samples.npz',points=r['points'],losses=r['values'])
        (OUT/f'{name}_trace.json').write_text(json.dumps(r['events']))
        truth=k.ravel();contained=bool(np.all(truth>=r['lower']) and np.all(truth<=r['upper']))
        row=dict(case=name,category=category,teacher=k.tolist(),prefix_rank=int(np.linalg.matrix_rank(x)),
                 prefix_condition=float(np.linalg.cond(x)),teacher_baseline=objective(truth),n1_success=r['success'],
                 n1_evaluations=r['evaluations'],n1_loss=r['loss'],n1_theta=r['theta'].tolist(),rounds=r['iterations'],
                 greedy_evaluations=sum(e['source']=='greedy' for e in r['events']),
                 boundary_evaluations=sum(e['boundary'] for e in r['events']),teacher_contained=contained,
                 local_lower=r['lower'].tolist(),local_upper=r['upper'].tolist())
        if r['success']:
            row.update(deterministic_finish(objective,torch_objective,r))
            final=np.array(row['final_theta']).reshape(2,2);diff=final-k
            true=states[-1].copy();pred=true.copy();errors=[]
            for _ in range(4096):true=k@true;pred=final@pred;errors.append(pred-true)
            errors=np.asarray(errors)
            row.update(k_errors=diff.tolist(),matrix_error=float(np.linalg.norm(diff)),
                       rollout_max_error=float(np.max(abs(errors))),rollout_mse=float(np.mean(errors**2)))
        row['seconds']=time.time()-t0;rows.append(row)
        (OUT/'results_partial.json').write_text(json.dumps(rows,indent=2))
        print(name,'N1',r['evaluations'],r['loss'],'greedy',row['greedy_evaluations'],'contained',contained,
              'final',row.get('final_loss'),flush=True)
        if not r['success'] or not contained or row.get('final_loss',float('inf'))>=1e-20:
            stop=('N1_BUDGET' if not r['success'] else 'HANDOFF_REGION_EXCLUDES_TEACHER' if not contained else 'DOWNSTREAM_RECOVERY')
            break
    after=hashes();assert before==after
    (OUT/'checkpoint_hashes_after.json').write_text(json.dumps(after,indent=2))
    report=dict(status='STOPPED_'+stop if stop else 'VALIDATED_DETERMINISTIC_TEACHER',
                planned_cases=len(data),completed_cases=len(rows),cases=rows,structural=controls,
                checkpoint_count=len(before),checkpoints_unchanged=True,seconds=time.time()-started,
                first_failure=stop,unchanged_downstream='existing deterministic_finish; no N2/N3 policy change',
                sobol='installed production SciPy oracle, recovered bit-exact by locked Sobol assembly')
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    if not stop:
        (OUT/'dimension_general_edirect_teacher.json').write_text(json.dumps(dict(
            status=report['status'],implementation=ed.__file__,sha256=hashlib.sha256(Path(ed.__file__).read_bytes()).hexdigest()),indent=2))
    print(report['status'],flush=True)


if __name__=='__main__':main()
