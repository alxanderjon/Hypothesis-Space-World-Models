"""Saved production seeds; reuse new parity checkpoint read-only, no training."""
import csv
import json
import numpy as np
import torch
from scipy.stats import _sobol
from experiments.sho.validate_sobol_scramble import (
    OUT,BASE,CountParity,ScrambleAssembly,Capture,dump,sha,ScalingAssembly,
    XorAssembly,scan_parameters,direction_column,update)


def main():
    torch.set_num_threads(1)
    source=BASE/'probe_generation_audit/probe_generation_counts.csv'
    with source.open() as f:
        states=[r for r in csv.DictReader(f) if r.get('seed') and r.get('sobol_power')]
    assert states
    parity=CountParity();parity.load_state_dict(torch.load(OUT/'sobol_sum_parity.pt',weights_only=False)['state_dict'])
    parity.eval().requires_grad_(False)
    arithmetic=XorAssembly();assembly=ScrambleAssembly(arithmetic,parity)
    scaling=ScalingAssembly(arithmetic.mul);w,b,_=scan_parameters()
    rows=[]
    for r in states:
        seed=int(r['seed']);t=Capture(2,scramble=True,seed=seed)
        sv,_=assembly.transform(t.before_v,t.raw_matrix);shift=assembly.shift(t.raw_shift)
        assert np.array_equal(sv,t._sv) and np.array_equal(shift,t._shift)
        rows.append(dict(seed=seed,split=r['split'],iteration=r['iteration'],
                         generated_points=int(r['generated_sobol_block']),direction_words=60,shift_words=2,errors=0))
    replays=[]
    for split in sorted(set(r['split'] for r in rows)):
        chosen=max((r for r in rows if r['split']==split),key=lambda r:r['generated_points'])
        seed=chosen['seed'];n=chosen['generated_points'];t=Capture(2,scramble=True,seed=seed)
        sv,_=assembly.transform(t.before_v,t.raw_matrix);shift=assembly.shift(t.raw_shift)
        reference=t.random_base2(int(np.log2(n)));state=shift.astype(np.uint64);oracle=shift.copy()
        assert np.array_equal(scaling(state,t._scale).view('u8'),reference[0].view('u8'))
        for i in range(n-1):
            state=update(arithmetic,state,sv[:,direction_column(i,30,w,b)].astype(np.uint64),30)
            temp=np.empty((1,2));_sobol._draw(1,i,2,t._scale,t._sv,oracle,temp)
            assert np.array_equal(state,oracle)
            assert np.array_equal(scaling(state,t._scale).view('u8'),reference[i+1].view('u8'))
        replays.append(dict(seed=seed,split=split,points=n,dimension=2,coordinates=n*2,
                            digital_updates=(n-1)*2,digital_errors=0,coordinate_errors=0))
        print(replays[-1],flush=True)
    dump('saved_production_validation.json',dict(source=str(source),source_sha256=sha(source),
         saved_states=len(rows),unique_seeds=len(set(r['seed'] for r in rows)),
         initialization=rows,replays=replays,errors=0))
    print('PASS saved',len(rows),flush=True)


if __name__=='__main__':main()
