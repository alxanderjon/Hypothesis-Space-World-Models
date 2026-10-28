"""Validate only production power-of-two Sobol generation and prefix truncation."""
import copy
import csv
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from experiments.sho.validate_sobol_scramble import CountParity, ScrambleAssembly, Capture
from experiments.sho.validate_shared_one_bit_xor import XorAssembly, direction_column, update
from experiments.sho.validate_sobol_float_scaling import ScalingAssembly, scan_parameters

BASE=Path('outputs/sho/stage1B')
OUT=BASE/'sobol_batching_truncation'
HIST=BASE/'probe_generation_audit/probe_generation_counts.csv'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def state_hash(rng): return hashlib.sha256(pickle.dumps(rng.bit_generator.state,protocol=4)).hexdigest()


def schedule(n):
    # This is the production expression, retained exactly rather than replaced by bit_length.
    exponent=int(math.ceil(math.log2(max(2,n))))
    return exponent,2**exponent


class LearnedSobol:
    """Previously locked scramble -> recurrence -> scale; batching lives above it."""
    def __init__(self):
        self.arithmetic=XorAssembly()
        self.scaling=ScalingAssembly(self.arithmetic.mul)
        self.w,self.b,_=scan_parameters()
        parity=CountParity()
        checkpoint=BASE/'sobol_scramble/sobol_sum_parity.pt'
        manifest=json.loads((BASE/'sobol_scramble/sobol_sum_parity_locked.json').read_text())
        assert sha(checkpoint)==manifest['sha256']
        parity.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict'])
        parity.eval().requires_grad_(False)
        self.scramble=ScrambleAssembly(self.arithmetic,parity)

    def full_block(self,d,exponent,seed,bits=30):
        teacher=Capture(d,scramble=True,bits=bits,seed=seed)
        directions,_=self.scramble.transform(teacher.before_v,teacher.raw_matrix)
        state=self.scramble.shift(teacher.raw_shift).astype(np.uint64)
        assert np.array_equal(directions,teacher._sv) and np.array_equal(state,teacher._shift)
        count=2**exponent; points=np.empty((count,d),np.float64)
        points[0]=self.scaling(state,teacher._scale)
        snapshots={1:state.copy()}
        for index in range(count-1):
            column=direction_column(index,bits,self.w,self.b)
            state=update(self.arithmetic,state,directions[:,column].astype(np.uint64),bits)
            points[index+1]=self.scaling(state,teacher._scale)
            generated=index+2
            if generated&(generated-1)==0:snapshots[generated]=state.copy()
        return points,state,snapshots,teacher,directions


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if (OUT/'sobol_batching_truncation_locked.json').exists():raise RuntimeError('Refuse to overwrite lock')
    protected=[]
    for folder in ('sobol_digital_primitives','shared_one_bit_xor','sobol_float_scaling','sobol_scramble'):
        protected += [p for p in (BASE/folder).rglob('*') if p.is_file()]
    before={str(p):sha(p) for p in protected};dump('protected_before.json',before)

    # Finite schedule audit around every power-of-two boundary through cap.
    boundary=[];values={1,2,32768}
    for k in range(1,16):values.update((max(1,(1<<k)-1),1<<k,min(32768,(1<<k)+1)))
    for n in sorted(values):
        e,count=schedule(n)
        expected=max(1,(max(2,n)-1).bit_length())
        assert e==expected and count>=n and (count//2<n or n==1)
        boundary.append(dict(requested=n,exponent=e,generated=count,returned=n,discarded=count-n))
    dump('power_boundary_schedule.json',boundary)

    # All archived production states: exact Python rule, two APIs, state, order, determinism.
    with HIST.open() as f:
        historical=[r for r in csv.DictReader(f) if r.get('requested_sobol') and r.get('seed')]
    assert len(historical)==320
    archived=[];returned_coordinates=generated_coordinates=0
    for index,r in enumerate(historical):
        n=int(r['requested_sobol']);seed=int(r['seed']);e,count=schedule(n)
        assert e==int(r['sobol_power']) and count==int(r['generated_sobol_block'])
        a=qmc.Sobol(2,scramble=True,seed=seed);rng_after_init=state_hash(a.rng)
        full=a.random_base2(e);prefix=full[:n];rng_after_draw=state_hash(a.rng)
        assert rng_after_init==rng_after_draw and a.num_generated==count
        expected_state=(full[-1]/a._scale).astype(a.dtype_i)
        assert np.array_equal(a._quasi,expected_state)
        b=qmc.Sobol(2,scramble=True,seed=seed);alternate=b.random(n=count)[:n]
        c=qmc.Sobol(2,scramble=True,seed=seed);repeat=c.random_base2(e)[:n]
        assert np.array_equal(prefix.view('u8'),alternate.view('u8'))
        assert np.array_equal(prefix.view('u8'),repeat.view('u8'))
        assert b.num_generated==c.num_generated==count and np.array_equal(a._quasi,b._quasi) and np.array_equal(a._quasi,c._quasi)
        returned_coordinates+=n*2;generated_coordinates+=count*2
        archived.append(dict(index=index,split=r['split'],seed=seed,requested=n,exponent=e,
             generated=count,returned=n,discarded=count-n,num_generated=a.num_generated,
             coordinate_errors=0,order_errors=0,state_errors=0,determinism_errors=0,rng_draw_changes=0,
             prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),final_state_sha256=hashlib.sha256(a._quasi.tobytes()).hexdigest()))
    dump('archived_replay.json',archived)

    learned=LearnedSobol()
    # One production-size locked-path block proves every requested prefix and full-block state.
    seed=10654473;points,final_state,snapshots,teacher,_=learned.full_block(2,15,seed)
    reference_engine=qmc.Sobol(2,scramble=True,seed=seed);reference=reference_engine.random_base2(15)
    assert np.array_equal(points.view('u8'),reference.view('u8')) and np.array_equal(final_state,reference_engine._quasi)
    requested=sorted(set([int(r['requested_sobol']) for r in historical]+[x['requested'] for x in boundary]))
    composed=[]
    for n in requested:
        e,count=schedule(n);t=qmc.Sobol(2,scramble=True,seed=seed);block=t.random_base2(e)
        got=points[:n];assert np.array_equal(got.view('u8'),block[:n].view('u8'))
        assert np.array_equal(snapshots[count],t._quasi)
        composed.append(dict(requested=n,generated=count,coordinates=n*2,coordinate_errors=0,
                             order_errors=0,state_errors=0))
    dump('learned_prefix_replay.json',composed)

    # Higher-dimensional controls around boundaries; complete locked path each time.
    controls=[]
    for d,n,seed in [(1,1,40001),(3,2,40002),(4,3,40003),(8,31,40004),(16,32,40005),
                     (32,33,40006),(128,255,40007),(1024,17,40008),(21201,3,40009)]:
        e,count=schedule(n);got,state,_,_,_=learned.full_block(d,e,seed)
        t=qmc.Sobol(d,scramble=True,seed=seed);ref=t.random_base2(e)
        assert np.array_equal(got[:n].view('u8'),ref[:n].view('u8')) and np.array_equal(state,t._quasi)
        assert len(got[:n])==n and t.num_generated==count
        controls.append(dict(dimension=d,seed=seed,requested=n,generated=count,coordinates=n*d,
             generated_coordinates=count*d,coordinate_errors=0,order_errors=0,state_errors=0,count_errors=0))
    dump('higher_dimension_controls.json',controls)

    after={str(p):sha(p) for p in protected};changes=[p for p,h in before.items() if after[p]!=h];assert not changes
    totals=dict(historical_states=len(archived),historical_unique_requested=len(set(r['requested'] for r in archived)),
      historical_returned_coordinates=returned_coordinates,historical_generated_coordinates=generated_coordinates,
      learned_composed_prefix_cases=len(composed),learned_composed_prefix_coordinates=sum(r['coordinates'] for r in composed),
      learned_full_block_coordinates=len(points)*2,higher_dimension_coordinates=sum(r['coordinates'] for r in controls),
      higher_dimension_generated_coordinates=sum(r['generated_coordinates'] for r in controls),errors=0)
    result=dict(status='LOCKED_ASSEMBLY',name='sobol_batching_truncation',new_parameters=0,
      production_rule='exponent=int(ceil(log2(max(2,N)))); full=random_base2(exponent); returned=full[:N]',
      state_semantics='engine advances by 2**exponent; slicing does not alter engine state; fresh engine starts at index zero',
      seed_semantics='batch size depends only on N; seed selects scramble at engine construction and no RNG is consumed by drawing',
      boundary=boundary,totals=totals,protected_files=len(protected),protected_changes=changes,
      seconds=time.time()-started,next_operation='boundary support generation (dedup/order remains after that)')
    dump('results.json',result)
    dump('sobol_batching_truncation_locked.json',dict(status='LOCKED_ASSEMBLY',name='sobol_batching_truncation',
      result_sha256=sha(OUT/'results.json'),implementation_sha256=sha(Path(__file__)),new_checkpoint=None,
      component_manifests={name:dict(path=str(path),sha256=sha(path)) for name,path in {
       'scramble':BASE/'sobol_scramble/sobol_scramble_locked.json','xor':BASE/'shared_one_bit_xor/shared_one_bit_xor_locked.json',
       'scaling':BASE/'sobol_float_scaling/sobol_float_scaling_locked.json'}.items()}))
    print('PASS',totals,flush=True)


if __name__=='__main__':main()
