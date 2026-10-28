"""Production LMS scramble + shift only. Existing Sobol operators are read-only.

Retains sum-then-modulo order of SciPy _cscramble; no XOR reduction substitute.
"""
import copy
import hashlib
import inspect
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import qmc, _sobol
from scipy._lib._util import rng_integers

from experiments.sho.validate_shared_one_bit_xor import XorAssembly, direction_column, update
from experiments.sho.validate_sobol_float_scaling import ScalingAssembly, scan_parameters

BASE=Path('outputs/sho/stage1B')
OUT=BASE/'sobol_scramble'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def digest(value): return hashlib.sha256(pickle.dumps(value,protocol=4)).hexdigest()


class Capture(qmc.Sobol):
    """Read-only capture using cloned RNG. Superclass executes actual production."""
    def _scramble(self):
        self.before_rng=copy.deepcopy(self.rng)
        self.before_v=self._sv.copy()
        probe=copy.deepcopy(self.rng)
        self.raw_shift=rng_integers(probe,2,size=(self.d,self.bits),dtype=self.dtype_i)
        self.raw_matrix=rng_integers(probe,2,size=(self.d,self.bits,self.bits),dtype=self.dtype_i)
        self.expected_after_rng=digest(probe)
        super()._scramble()
        assert digest(self.rng)==self.expected_after_rng


class CountParity(torch.nn.Module):
    """Bias-free categorical logit, same operation for any dimension/row/column.

    Input is integer sum n, encoded as its one-hot category. It never receives n%2.
    Gathering its trained logit is exactly a bias-free one-hot linear evaluation.
    """
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.randn(33,dtype=torch.float64)*.2)

    def forward(self,n): return self.logits[n]

    def predict(self,n):
        n=np.asarray(n)
        if not np.all((n>=0)&(n<=32)&(n==np.floor(n))):
            raise ValueError('Parity domain is integral counts 0..32')
        return (self.logits.detach().numpy()[n.astype(np.int64)]>0).astype(np.float64)


def parity_screen():
    torch.manual_seed(202609301)
    m=CountParity(); initial=m.logits.detach().tolist()
    n=torch.arange(33); y=(n%2).double()
    opt=torch.optim.Adam(m.parameters(),lr=.08)
    for epoch in range(300):
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(m(n),y)
        loss.backward();opt.step()
        if bool((m(n).detach()*(2*y-1)>2).all()):break
    m.eval().requires_grad_(False)
    errors=int(np.count_nonzero(m.predict(np.arange(33))!=np.arange(33)%2))
    report=dict(parameters=33,architecture='bias-free categorical linear head, 33 count categories',
                initial=initial,learned=m.logits.tolist(),steps=epoch+1,truth_cases=33,errors=errors)
    dump('parity_screen.json',report)
    assert errors==0,'STOP: parity screen failed'
    torch.save(dict(state_dict=m.state_dict(),report=report),OUT/'sobol_sum_parity.pt')
    dump('sobol_sum_parity_locked.json',dict(status='LOCKED',domain='integer sums 0..32',
         checkpoint=str(OUT/'sobol_sum_parity.pt'),sha256=sha(OUT/'sobol_sum_parity.pt'),**report))
    return m


class ScrambleAssembly:
    def __init__(self,arithmetic,parity): self.a=arithmetic; self.parity=parity
    def mul(self,x,y): return self.a.call(self.a.mul,x,y)
    def add(self,x,y): return self.a.call(self.a.add,x,y)

    def shift(self,bits):
        d,width=bits.shape
        if width>32: raise ValueError('Validated scope bits<=32')
        result=np.zeros(d,np.float64)
        for k in range(width):
            result=self.add(result,self.mul(bits[:,k],float(1<<k)))
        return result.astype(bits.dtype)

    def transform(self,v,raw):
        d,width=v.shape
        if width>32: raise ValueError('Validated scope bits<=32')
        matrix=np.tril(raw.copy())
        idx=np.arange(width);matrix[:,idx,idx]=1
        transformed=np.zeros_like(v); parity_hist=np.zeros(33,np.int64)
        # Chunking only across independent dimensions. k summation order unchanged.
        for start in range(0,d,128):
            end=min(start+128,d);vv=v[start:end].astype(np.uint64); mat=matrix[start:end]
            # Native audit expansion for intermediate comparison, never fed to assembly.
            packed=np.zeros((end-start,width),np.uint64)
            packed_hat=np.zeros((end-start,width),np.float64)
            for k in range(width):
                packed += mat[:,:,width-1-k].astype(np.uint64)*np.uint64(1<<k)
                packed_hat=self.add(packed_hat,self.mul(mat[:,:,width-1-k],float(1<<k)))
            assert np.array_equal(packed_hat.astype(np.uint64),packed)
            total=np.zeros((end-start,width,width),np.float64)
            target=np.zeros_like(total,dtype=np.uint64)
            for k in range(width):
                left=((packed_hat.astype(np.uint64)>>np.uint64(k))&1)[:,:,None]
                right=((vv>>np.uint64(k))&1)[:,None,:]
                product=self.mul(left,right)
                assert np.array_equal(product,left*right)
                total=self.add(total,product)
                target+=left*right
                assert np.array_equal(total,target)
            remainder=self.parity.predict(total)
            assert np.array_equal(remainder,target%2)
            parity_hist+=np.bincount(total.astype(np.int64).ravel(),minlength=33)
            result=np.zeros((end-start,width),np.float64)
            for p in range(width-1,-1,-1):
                result=self.add(result,self.mul(remainder[:,p,:],float(1<<(width-1-p))))
            transformed[start:end]=result.astype(v.dtype)
        return transformed,parity_hist


def run():
    started=time.time();OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(1)
    if (OUT/'sobol_scramble_locked.json').exists():raise RuntimeError('Refuse to overwrite locked assembly')
    protected=[]
    for folder in ['sobol_digital_primitives','shared_one_bit_xor','sobol_float_scaling',
                   'multiplication_operator_smoke','addition_operator_smoke','subtraction_operator_training']:
        protected += [p for p in (BASE/folder).rglob('*') if p.is_file()]
    protected += [Path(inspect.getfile(qmc.Sobol)),Path(_sobol.__file__),
                  Path(inspect.getfile(XorAssembly)),Path(inspect.getfile(ScalingAssembly))]
    before={str(p):sha(p) for p in protected};dump('protected_before.json',before)
    (OUT/'installed_scramble_source.txt').write_text(inspect.getsource(qmc.Sobol._scramble)+
             '\n'+inspect.getsource(rng_integers))
    dump('authority.json',dict(scipy=scipy.__version__,binary=str(_sobol.__file__),
         binary_hash=sha(_sobol.__file__),word_scope=[30,32],
         upstream='https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_sobol.pyx'))
    arithmetic=XorAssembly(); scaling=ScalingAssembly(arithmetic.mul);w,b,scanmeta=scan_parameters()
    # First operations: shift generation and row packing use already frozen arithmetic.
    rng=np.random.default_rng(770103)
    bitpairs=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    assert np.array_equal(arithmetic.call(arithmetic.mul,bitpairs[:,0],bitpairs[:,1]),[0,0,0,1])
    shift_checks=[]
    dummy=ScrambleAssembly(arithmetic,None)
    for width in (30,32):
        raw=rng.integers(0,2,size=(10000,width),dtype=np.uint32)
        got=dummy.shift(raw);expected=np.dot(raw,2**np.arange(width,dtype=np.uint32))
        assert np.array_equal(got,expected)
        shift_checks.append(dict(bits=width,cases=len(raw),errors=0))
    dump('shift_packing_validation.json',shift_checks)
    parity=parity_screen();assembly=ScrambleAssembly(arithmetic,parity)
    print('PASS shift packing and parity primitive',flush=True)
    # Captured original production seeds (integer-seed path uses actual SciPy RNG).
    # Existing replay dimensions/lengths retained, split roles are whole sequences.
    cases=[(1,32768,30,0,'saved'),(2,32768,30,1729,'saved'),(3,32768,30,9282026,'saved'),
           (4,8192,30,100003,'held_out'),(8,4096,30,700001,'held_out'),
           (16,2048,30,9302026,'fresh'),(32,1024,30,9302027,'fresh'),
           (64,512,30,9302028,'fresh'),(128,256,30,9302029,'higher_dimension'),
           (1024,64,30,9302030,'higher_dimension'),(21201,8,30,9302031,'higher_dimension'),
           (7,1024,32,9302032,'fresh_width')]
    rows=[]
    for ci,(d,n,width,seed,split) in enumerate(cases):
        teacher=Capture(d,scramble=True,bits=width,seed=seed)
        shift=assembly.shift(teacher.raw_shift)
        assert np.array_equal(shift,teacher._shift)
        sv,hist=assembly.transform(teacher.before_v,teacher.raw_matrix)
        assert np.array_equal(sv,teacher._sv),'STOP: scrambled direction disagreement'
        assert np.array_equal(shift,teacher._quasi)
        np.savez_compressed(OUT/f'state_{ci:02d}.npz',raw_shift=teacher.raw_shift,
             raw_matrix=teacher.raw_matrix,original_directions=teacher.before_v,
             scrambled_directions=sv,shift=shift)
        # Copy actual post-scramble engine for an independent compiled per-step oracle.
        oracle=shift.copy();reference=teacher.random_base2(int(np.log2(n)))
        state=shift.astype(np.uint64); coord=scaling(state,teacher._scale)
        assert np.array_equal(coord.view('u8'),reference[0].view('u8'))
        for i in range(n-1):
            col=direction_column(i,width,w,b)
            state=update(arithmetic,state,sv[:,col].astype(np.uint64),width)
            sample=np.empty((1,d));_sobol._draw(1,i,d,teacher._scale,teacher._sv,oracle,sample)
            assert np.array_equal(state,oracle),'STOP: digital state mismatch'
            coord=scaling(state,teacher._scale)
            assert np.array_equal(coord.view('u8'),reference[i+1].view('u8')),'STOP: coordinate mismatch'
        row=dict(seed=seed,split=split,dimension=d,bits=width,points=n,
                 shift_words=d,direction_words=d*width,parity_calls=d*width*width,
                 parity_count_histogram=hist.tolist(),digital_updates=(n-1)*d,coordinates=n*d,
                 shift_errors=0,direction_errors=0,digital_errors=0,coordinate_errors=0,
                 signed_zero_errors=0,nonfinite_errors=0,rng_before=digest(teacher.before_rng),
                 rng_after=teacher.expected_after_rng,rng_class=type(teacher.rng).__name__)
        rows.append(row);dump('progress.json',rows);print(row|{'parity_count_histogram':'saved'},flush=True)
    # Fresh caller-supplied Generator and RandomState semantics, initialization only.
    rng_controls=[]
    for kind in ('Generator','RandomState'):
        obj=np.random.default_rng(381920) if kind=='Generator' else np.random.RandomState(381920)
        t=Capture(19,scramble=True,seed=obj)
        sv,_=assembly.transform(t.before_v,t.raw_matrix); shift=assembly.shift(t.raw_shift)
        assert np.array_equal(sv,t._sv) and np.array_equal(shift,t._shift)
        rng_controls.append(dict(input_rng=kind,dimension=19,direction_words=570,shift_words=19,errors=0))
    changes=[p for p,h in before.items() if sha(p)!=h];assert not changes
    totals={k:sum(r[k] for r in rows) for k in ['shift_words','direction_words','parity_calls','digital_updates','coordinates']}
    result=dict(status='LOCKED_ASSEMBLY',name='sobol_scramble',new_parameters=33,new_model='sobol_sum_parity',
         parity_domain='integer sums 0..32; one-hot count input, no teacher parity input',rows=rows,
         rng_controls=rng_controls,totals=totals,errors=0,protected_files=len(before),protected_changes=changes,
         seconds=time.time()-started,scope='SciPy production bits=30 plus bits=32 extension; no 64-bit packing claim',
         next_operation='existing power-of-two generation / first-N truncation, then boundary support and dedup')
    dump('results.json',result)
    dump('sobol_scramble_locked.json',dict(status='LOCKED_ASSEMBLY',name='sobol_scramble',
         result_sha256=sha(OUT/'results.json'),implementation_sha256=sha(Path(__file__)),
         parity_checkpoint=str(OUT/'sobol_sum_parity.pt'),parity_sha256=sha(OUT/'sobol_sum_parity.pt'),
         component_hashes=before))
    print('PASS ALL',totals,flush=True)


if __name__=='__main__':run()
