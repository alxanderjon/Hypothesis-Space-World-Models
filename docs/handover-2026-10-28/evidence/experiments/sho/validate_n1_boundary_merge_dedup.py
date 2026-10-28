"""Validate only production N1 boundary support and np.unique row canonicalization."""
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

import experiments.sho.run_stage1b_edirect_vs_direct as production
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol, schedule

BASE=Path('outputs/sho/stage1B')
OUT=BASE/'n1_boundary_merge_dedup'
SOURCE=BASE/'probe_generation_audit/probe_generation_counts.csv'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def capture(sample_count,generator,iteration):
    saved={};code=production._candidate_pool.__code__;previous=sys.getprofile()
    def observe(frame,event,arg):
        if event=='return' and frame.f_code is code:
            for key in ('requested','exponent','seed','interior','edge_count','edge_coordinate','boundaries'):
                value=frame.f_locals[key];saved[key]=value.copy() if isinstance(value,np.ndarray) else value
    try:
        sys.setprofile(observe);final=production._candidate_pool(sample_count,generator,iteration)
    finally:sys.setprofile(previous)
    return final,saved


def boundaries(n):
    k=min(257,max(9,int(math.ceil(math.sqrt(n)))))
    t=np.linspace(0.,1.,k,dtype=np.float64);z=np.zeros_like(t);o=np.ones_like(t)
    return np.concatenate((np.column_stack((t,z)),np.column_stack((t,o)),
                           np.column_stack((z,t)),np.column_stack((o,t)))),k,t


def independent_unique(raw):
    # Finite production coordinates: stable index is only a deterministic tiebreak.
    keys=tuple([np.arange(len(raw))]+[raw[:,j] for j in range(raw.shape[1]-1,-1,-1)])
    order=np.lexsort(keys);ordered=raw[order]
    starts=np.r_[0,np.flatnonzero(np.any(ordered[1:]!=ordered[:-1],axis=1))+1]
    return ordered[starts],order[starts]


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if (OUT/'n1_boundary_merge_dedup_locked.json').exists():raise RuntimeError('Refuse to overwrite lock')
    protected=[]
    for folder in ('sobol_scramble','shared_one_bit_xor','sobol_float_scaling','sobol_batching_truncation'):
        protected += [p for p in (BASE/folder).rglob('*') if p.is_file()]
    protected += [Path(production.__file__)]
    before={str(p):sha(p) for p in protected};dump('protected_before.json',before)
    with SOURCE.open() as f:
        rows=[r for r in csv.DictReader(f) if r.get('requested_sobol') and r.get('seed')]
    assert len(rows)==320

    historical=[];total_final=total_boundary=total_raw=total_duplicates=0
    for index,r in enumerate(rows):
        sample_count=int(r['evaluated_sites']);generator=int(r['generator_id']);iteration=int(r['iteration'])
        final,state=capture(sample_count,generator,iteration)
        n=int(r['requested_sobol']);edge,k,t=boundaries(n)
        assert state['requested']==n and state['edge_count']==k
        assert np.array_equal(edge.view('u8'),state['boundaries'].view('u8'))
        assert np.array_equal(t.view('u8'),state['edge_coordinate'].view('u8'))
        raw=np.concatenate((state['interior'],edge));expected,first=independent_unique(raw)
        assert np.array_equal(expected.view('u8'),final.view('u8'))
        assert len(final)==int(r['final_probe_count'])
        # np.unique's documented index is the first original occurrence.
        teacher_unique,teacher_first=np.unique(raw,axis=0,return_index=True)
        assert np.array_equal(teacher_unique.view('u8'),final.view('u8'))
        assert np.array_equal(teacher_first,first)
        duplicates=len(raw)-len(final);assert duplicates==int(r['duplicates_removed'])
        total_final+=len(final);total_boundary+=len(edge);total_raw+=len(raw);total_duplicates+=duplicates
        historical.append(dict(index=index,split=r['split'],sample_count=sample_count,generator_id=generator,
          iteration=iteration,requested=n,edge_count=k,raw_boundary=len(edge),distinct_boundary=4*k-4,
          raw_total=len(raw),duplicates_removed=duplicates,final_count=len(final),
          boundary_errors=0,count_errors=0,coordinate_errors=0,order_errors=0,dedup_errors=0,
          final_sha256=hashlib.sha256(final.tobytes()).hexdigest()))
    dump('historical_replay.json',historical)

    # All attainable production edge resolutions and budget transition states.
    resolution=[]
    for k in range(9,183):
        t=np.linspace(0.,1.,k,dtype=np.float64)
        alt=np.arange(k,dtype=np.float64)*(1.0/(k-1));alt[-1]=1.
        assert np.array_equal(t.view('u8'),alt.view('u8'))
        edge,_,_=boundaries((k-1)**2+1 if k>9 else 1)
        unique=np.unique(edge,axis=0)
        assert len(edge)==4*k and len(unique)==4*k-4
        resolution.append(dict(edge_count=k,raw=4*k,unique=4*k-4,corner_duplicates=4,errors=0))
    dump('resolution_validation.json',resolution)

    # Collision and exact-order controls. Production inputs are finite and nonnegative.
    t=np.array([0.,.25,.5,.75,1.],np.float64);z=np.zeros(5);o=np.ones(5)
    edge=np.concatenate((np.column_stack((t,z)),np.column_stack((t,o)),np.column_stack((z,t)),np.column_stack((o,t))))
    near=np.nextafter(.5,1.)
    interior=np.array([[0.,0.],[.5,0.],[.5,0.],[near,0.],[1.,1.],[.25,.75],[.25,.75]],np.float64)
    raw=np.concatenate((interior,edge));teacher,indices=np.unique(raw,axis=0,return_index=True);got,first=independent_unique(raw)
    assert np.array_equal(got.view('u8'),teacher.view('u8')) and np.array_equal(indices,first)
    collision=dict(raw_rows=len(raw),unique_rows=len(teacher),duplicates_removed=len(raw)-len(teacher),
      first_occurrence_indices=indices.tolist(),near_boundary_distinct=bool(any(np.array_equal(row,[near,0.]) for row in teacher)),
      exact_half_boundary_present=bool(any(np.array_equal(row,[.5,0.]) for row in teacher)),errors=0)
    dump('collision_controls.json',collision)

    # Generic row canonicalizer supports dimensions above two; boundary generator does not.
    generic=[]
    rng=np.random.default_rng(9312026)
    for d in (1,3,8,64):
        a=rng.integers(0,17,size=(1000,d)).astype(np.float64);a=np.concatenate((a,a[:100],a[::37]))
        teacher=np.unique(a,axis=0);got,_=independent_unique(a)
        assert np.array_equal(got.view('u8'),teacher.view('u8'))
        generic.append(dict(dimension=d,raw=len(a),unique=len(teacher),errors=0))
    dump('generic_merge_controls.json',generic)

    # Full locked path: archived train/held-out budget range plus fresh seeds.
    def pick(split,nmin=False,nmax=False):
        candidates=[r for r in rows if r['split']==split]
        return min(candidates,key=lambda r:int(r['requested_sobol'])) if nmin else max(candidates,key=lambda r:int(r['requested_sobol'])) if nmax else candidates[len(candidates)//2]
    selected=[pick('training',nmin=True),pick('training'),pick('training',nmax=True),
              pick('held_out',nmin=True),pick('held_out'),pick('held_out',nmax=True)]
    learned=LearnedSobol();composed=[]
    for r in selected:
        sample_count=int(r['evaluated_sites']);generator=int(r['generator_id']);iteration=int(r['iteration'])
        n=int(r['requested_sobol']);seed=int(r['seed']);e,count=schedule(n)
        block,state,_,teacher,_=learned.full_block(2,e,seed);interior=block[:n];edge,k,_=boundaries(n)
        got=np.unique(np.concatenate((interior,edge)),axis=0);expected=production._candidate_pool(sample_count,generator,iteration)
        assert np.array_equal(got.view('u8'),expected.view('u8')) and len(got)==int(r['final_probe_count'])
        # Boundary/merge consumes no engine state or RNG. Captured teacher remains immediately post-scramble.
        assert np.array_equal(state,(teacher.random_base2(e)[-1]/teacher._scale).astype(teacher.dtype_i))
        composed.append(dict(kind='archived_'+r['split'],seed=seed,requested=n,generated=count,
          edge_count=k,final_count=len(got),coordinates=len(got)*2,errors=0))
    for n,seed in ((1,950001),(2,950002),(3,950003),(255,950004),(256,950005),(257,950006),(32768,950007)):
        e,count=schedule(n);block,state,_,teacher,_=learned.full_block(2,e,seed);edge,k,_=boundaries(n)
        got=np.unique(np.concatenate((block[:n],edge)),axis=0)
        reference_engine=qmc.Sobol(2,scramble=True,seed=seed);interior=reference_engine.random_base2(e)[:n]
        expected=np.unique(np.concatenate((interior,edge)),axis=0)
        assert np.array_equal(got.view('u8'),expected.view('u8')) and np.array_equal(state,reference_engine._quasi)
        composed.append(dict(kind='fresh',seed=seed,requested=n,generated=count,edge_count=k,
          final_count=len(got),coordinates=len(got)*2,errors=0))
    dump('complete_path_replay.json',composed)

    after={str(p):sha(p) for p in protected};changes=[p for p,h in before.items() if after[p]!=h];assert not changes
    totals=dict(historical_calls=len(historical),historical_boundary_rows=total_boundary,historical_raw_rows=total_raw,
      historical_final_rows=total_final,historical_duplicates_removed=total_duplicates,
      historical_final_coordinates=total_final*2,resolutions=len(resolution),collision_controls=1,
      generic_controls=len(generic),complete_path_calls=len(composed),complete_path_coordinates=sum(r['coordinates'] for r in composed),errors=0)
    result=dict(status='LOCKED_ASSEMBLY',name='n1_boundary_merge_dedup',new_parameters=0,
      production_dimension=2,higher_dimension_boundary_support=False,
      boundary_rule='k=min(257,max(9,int(ceil(sqrt(N))))); t=linspace(0,1,k,float64); concatenate bottom,top,left,right',
      merge_rule='concatenate interior first, boundaries second; np.unique(axis=0) exact finite-row equality; lexicographically sorted rows',
      duplicate_retention='return_index audit identifies first original occurrence; production returns values only, so identical-bit duplicates have no retained identity',
      totals=totals,protected_files=len(protected),protected_changes=changes,seconds=time.time()-started,
      next_operation='Voronoi candidate-to-site assignment in downstream N1; objective/search not run')
    dump('results.json',result)
    dump('n1_boundary_merge_dedup_locked.json',dict(status='LOCKED_ASSEMBLY',name='n1_boundary_merge_dedup',
      result_sha256=sha(OUT/'results.json'),implementation_sha256=sha(Path(__file__)),new_checkpoint=None,
      batching_manifest=dict(path=str(BASE/'sobol_batching_truncation/sobol_batching_truncation_locked.json'),
                             sha256=sha(BASE/'sobol_batching_truncation/sobol_batching_truncation_locked.json'))))
    print('PASS',totals,flush=True)


if __name__=='__main__':main()
