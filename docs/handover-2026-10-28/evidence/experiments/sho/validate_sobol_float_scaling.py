"""Validate only SciPy Sobol unsigned-word -> float64 coordinate scaling."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc, _sobol
import torch

from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.sho.validate_shared_one_bit_xor import XorAssembly, direction_column, update

ROOT=Path('outputs/sho/stage1B')
XOR_ROOT=ROOT/'shared_one_bit_xor'
SCAN_ROOT=ROOT/'sobol_digital_primitives'
OUT=ROOT/'sobol_float_scaling'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def load_multiply():
    meta=json.loads((ROOT/'multiplication_operator_smoke/candidate_manifest.json').read_text())
    path=Path(meta['checkpoint']); assert sha(path)==meta['sha256']
    model=QuadraticArithmetic(); model.load_state_dict(torch.load(path,weights_only=False)['state_dict'])
    model.eval().requires_grad_(False); return model,meta


class ScalingAssembly:
    """Production order: uint word -> float64 conversion -> multiply by stored scale."""
    def __init__(self,multiply): self.multiply=multiply
    def __call__(self,words,scale):
        # Deterministic dtype conversion matches Cython's unsigned-to-double conversion.
        values=np.asarray(words).astype(np.float64)
        shape=values.shape
        xy=np.column_stack((values.ravel(),np.full(values.size,np.float64(scale))))
        with torch.no_grad(): out=self.multiply(torch.from_numpy(xy)).numpy()
        return out.reshape(shape)


def scan_parameters():
    meta=json.loads((SCAN_ROOT/'sobol_scan_bit_locked.json').read_text())
    assert sha(Path(meta['checkpoint']))==meta['sha256']
    state=meta['learned']; return state['net.0.weight'][0][0],state['net.0.bias'][0],meta


def compiled_oracle(words,bits):
    words=np.asarray(words); dtype=np.uint32 if bits<=32 else np.uint64
    chunks=[]
    for start in range(0,len(words),20000):
        w=words[start:start+20000].astype(dtype)
        d=len(w); sv=np.zeros((d,bits),dtype=dtype); sv[:,0]=w
        quasi=np.zeros(d,dtype=dtype); sample=np.empty((1,d),np.float64)
        _sobol._draw(1,0,d,1.0/2**bits,sv,quasi,sample)
        assert np.array_equal(quasi,w)
        chunks.append(sample[0])
    return np.concatenate(chunks) if chunks else np.empty(0,np.float64)


def boundary_words(bits,rng):
    top=(1<<bits)-1
    vals={0,1,2,3,top,max(0,top-1),max(0,top-2)}
    for j in range(bits):
        p=1<<j
        vals.update((p,max(0,p-1),min(top,p+1)))
    # Conversion boundaries, where present, plus broad fresh state words.
    for p in (1<<24,1<<30,1<<31,1<<52,1<<53,1<<63):
        if p<=top: vals.update((p-1,p,min(top,p+1)))
    dtype=np.uint32 if bits<=32 else np.uint64
    random=rng.integers(0,top,50000,dtype=dtype,endpoint=True)
    return np.concatenate((np.array(sorted(vals),dtype=dtype),random))


def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    multiply,mm=load_multiply(); scale_op=ScalingAssembly(multiply)
    xor_manifest=json.loads((XOR_ROOT/'shared_one_bit_xor_locked.json').read_text())
    assert xor_manifest['status']=='LOCKED_ASSEMBLY'
    assert sha(XOR_ROOT/'results.json')==xor_manifest['result_sha256']
    w,b,scan_meta=scan_parameters(); xor=XorAssembly()
    protected={
      'multiply':(Path(mm['checkpoint']),mm['sha256']),
      'scan':(Path(scan_meta['checkpoint']),scan_meta['sha256']),
      'xor_results':(XOR_ROOT/'results.json',xor_manifest['result_sha256'])}
    before={k:sha(v[0]) for k,v in protected.items()}

    rng=np.random.default_rng(9302026); boundaries=[]
    for bits in (30,32,53,64):
        words=boundary_words(bits,rng); scale=np.float64(1.0/2**bits)
        teacher=compiled_oracle(words,bits); got=scale_op(words,scale)
        same=got.view('u8')==teacher.view('u8')
        signed_zero=int(np.count_nonzero((got==0)&(teacher==0)&~same))
        nonfinite=int(np.count_nonzero(np.isfinite(got)!=np.isfinite(teacher)))
        row=dict(bits=bits,scale=float(scale),scale_bits=int(np.asarray(scale).view('u8')),
                 cases=len(words),bit_exact=int(same.sum()),errors=int((~same).sum()),
                 signed_zero_mismatches=signed_zero,nonfinite_mismatches=nonfinite)
        boundaries.append(row); assert row['errors']==signed_zero==nonfinite==0
    write('integer_boundary_validation.json',boundaries)

    cases=[(1,32768,'saved'),(2,32768,'saved'),(3,32768,'saved'),
           (4,8192,'held_out'),(8,4096,'held_out'),(16,2048,'fresh'),
           (32,1024,'fresh'),(64,512,'fresh'),(128,256,'higher_dimension'),
           (1024,64,'higher_dimension'),(21201,8,'higher_dimension')]
    replay=[]
    for d,n,split in cases:
        engine=qmc.Sobol(d,scramble=False,bits=30); teacher=engine.random_base2(int(np.log2(n)))
        state=np.zeros(d,np.uint64); digital_oracle=np.zeros(d,engine.dtype_i)
        state_errors=0; coordinate_errors=0; signed_zero=0; nonfinite=0
        # First unscrambled state is integer zero. Scale through the exact same assembly.
        got=scale_op(state,engine._scale); ref=teacher[0]
        same=got.view('u8')==ref.view('u8'); coordinate_errors+=int((~same).sum())
        signed_zero+=int(np.count_nonzero((got==0)&(ref==0)&~same))
        for i in range(n-1):
            column=direction_column(i,30,w,b)
            state=update(xor,state,engine._sv[:,column].astype(np.uint64),30)
            oracle_sample=np.empty((1,d)); _sobol._draw(1,i,d,engine._scale,engine._sv,digital_oracle,oracle_sample)
            state_errors+=int(np.count_nonzero(state!=digital_oracle))
            got=scale_op(state,engine._scale); ref=teacher[i+1]
            same=got.view('u8')==ref.view('u8')
            coordinate_errors+=int((~same).sum())
            signed_zero+=int(np.count_nonzero((got==0)&(ref==0)&~same))
            nonfinite+=int(np.count_nonzero(np.isfinite(got)!=np.isfinite(ref)))
        assert state_errors==coordinate_errors==signed_zero==nonfinite==0
        row=dict(dimension=d,points=n,split=split,digital_words=(n-1)*d,
                 coordinates=n*d,digital_state_errors=state_errors,coordinate_errors=coordinate_errors,
                 signed_zero_mismatches=signed_zero,nonfinite_mismatches=nonfinite)
        replay.append(row);write('replay_progress.json',replay);print(row,flush=True)

    resumed=[]
    for d in (1,7,31,257,4096):
        engine=qmc.Sobol(d,scramble=False,bits=30); counters=np.concatenate((
          np.array([(1<<j)-1 for j in range(30)],np.uint64),rng.integers(0,2**29,256,dtype=np.uint64)))
        errors=zero=nonfinite=state_errors=0
        for counter in counters:
            n=int(counter); gray=n^(n>>1); state=np.zeros(d,np.uint64)
            for j in range(30):
                if (gray>>j)&1: state^=engine._sv[:,j].astype(np.uint64)
            state=update(xor,state,engine._sv[:,direction_column(n,30,w,b)].astype(np.uint64),30)
            oracle_state=np.zeros(d,engine.dtype_i)
            # Supply current state as quasi: _draw mutates it once and returns scaled result.
            gray0=n^(n>>1); oracle_state.fill(0)
            for j in range(30):
                if (gray0>>j)&1: oracle_state^=engine._sv[:,j]
            sample=np.empty((1,d));_sobol._draw(1,n,d,engine._scale,engine._sv,oracle_state,sample)
            state_errors+=int(np.count_nonzero(state!=oracle_state))
            got=scale_op(state,engine._scale);same=got.view('u8')==sample[0].view('u8')
            errors+=int((~same).sum());zero+=int(np.count_nonzero((got==0)&(sample[0]==0)&~same))
            nonfinite+=int(np.count_nonzero(np.isfinite(got)!=np.isfinite(sample[0])))
        assert state_errors==errors==zero==nonfinite==0
        resumed.append(dict(dimension=d,states=len(counters),coordinates=len(counters)*d,
                            digital_state_errors=state_errors,coordinate_errors=errors,
                            signed_zero_mismatches=zero,nonfinite_mismatches=nonfinite))

    after={k:sha(v[0]) for k,v in protected.items()};changes=[k for k in before if before[k]!=after[k]]
    assert not changes
    totals=dict(boundary_coordinates=sum(x['cases'] for x in boundaries),
                sequence_coordinates=sum(x['coordinates'] for x in replay),
                resumed_coordinates=sum(x['coordinates'] for x in resumed),
                total_coordinates=sum(x['cases'] for x in boundaries)+sum(x['coordinates'] for x in replay)+sum(x['coordinates'] for x in resumed),
                bit_exact_matches=sum(x['cases'] for x in boundaries)+sum(x['coordinates'] for x in replay)+sum(x['coordinates'] for x in resumed),
                errors=0,signed_zero_mismatches=0,nonfinite_mismatches=0)
    result=dict(status='LOCKED_ASSEMBLY',name='sobol_float_scaling',scipy=scipy.__version__,
      production_formula='sample[i,j] = float64(qtmp) * scale; scale = float64(1.0 / 2**bits)',
      evaluation_order=['construct Python scale as 1.0 / 2**bits','convert unsigned digital word to float64','locked multiplication(converted_word, scale)','store float64 coordinate'],
      deterministic_scaffolding=['scale construction','unsigned-to-float64 conversion','shape/index handling','float64 output storage'],
      learned_or_locked_arithmetic='existing frozen multiplication_operator',multiplication=mm,
      boundaries=boundaries,replays=replay,resumed=resumed,totals=totals,
      protected_changes=changes,seconds=time.time()-start,
      stop='Complete unscrambled recurrence through float coordinate scaling exact. Scrambling untouched.')
    write('results.json',result)
    manifest=dict(status='LOCKED_ASSEMBLY',name='sobol_float_scaling',checkpoint=None,
      components=dict(multiplication=dict(path=mm['checkpoint'],sha256=mm['sha256']),
                      digital_recurrence=dict(manifest=str(XOR_ROOT/'shared_one_bit_xor_locked.json'),sha256=sha(XOR_ROOT/'shared_one_bit_xor_locked.json'))),
      result_sha256=sha(OUT/'results.json'),implementation_sha256=sha(Path(__file__)),
      next_operation='Sobol scrambling/digital shift (explicitly excluded)')
    write('sobol_float_scaling_locked.json',manifest)
    print('PASS',totals,flush=True)


if __name__=='__main__':main()
