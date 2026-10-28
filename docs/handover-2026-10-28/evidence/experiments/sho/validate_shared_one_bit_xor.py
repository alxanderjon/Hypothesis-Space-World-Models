"""Validate x+y-2xy from locked arithmetic inside unscrambled Sobol recurrence."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc, _sobol

from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_subtraction_operator import Subtraction

ROOT = Path('outputs/sho/stage1B')
SOURCE = ROOT/'sobol_digital_primitives'
OUT = ROOT/'shared_one_bit_xor'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')


def load(cls, manifest_path):
    meta=json.loads(manifest_path.read_text()); checkpoint=Path(meta['checkpoint'])
    assert sha(checkpoint)==meta['sha256']
    model=cls(); model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    return model,meta


class XorAssembly:
    """Exact audited order: p=mul(x,y), two_p=mul(2,p), s=add(x,y), sub(s,two_p)."""
    def __init__(self):
        self.mul,self.mm=load(QuadraticArithmetic,ROOT/'multiplication_operator_smoke/candidate_manifest.json')
        self.add,self.am=load(Addition,ROOT/'addition_operator_smoke/candidate_manifest.json')
        self.sub,self.sm=load(Subtraction,ROOT/'subtraction_operator_training/subtraction_operator_locked.json')

    @staticmethod
    def call(model,x,y):
        x,y=np.broadcast_arrays(np.asarray(x,np.float64),np.asarray(y,np.float64)); shape=x.shape
        xy=np.column_stack((x.ravel(),y.ravel()))
        with torch.no_grad(): result=model(torch.from_numpy(xy)).numpy()
        return result.reshape(shape)

    def __call__(self,x,y):
        p=self.call(self.mul,x,y)
        two_p=self.call(self.mul,np.full_like(p,2.),p)
        summed=self.call(self.add,x,y)
        return self.call(self.sub,summed,two_p)


def scan_params():
    meta=json.loads((SOURCE/'sobol_scan_bit_locked.json').read_text())
    assert sha(Path(meta['checkpoint']))==meta['sha256']
    learned=meta['learned']; return learned['net.0.weight'][0][0],learned['net.0.bias'][0],meta


def direction_column(counter,bits,w,b):
    for j in range(bits):
        if not (w*((counter>>j)&1)+b>0): return j
    raise ValueError('counter exhausted')


def update(assembly,state,direction,bits):
    state=np.asarray(state,np.uint64); direction=np.asarray(direction,np.uint64)
    shifts=np.arange(bits,dtype=np.uint64)
    a=((state[:,None]>>shifts)&np.uint64(1)).astype(np.float64)
    b=((direction[:,None]>>shifts)&np.uint64(1)).astype(np.float64)
    z=assembly(a,b)
    if not np.all((z==0)|(z==1)): raise AssertionError('non-bit assembly output')
    return np.bitwise_or.reduce(z.astype(np.uint64)<<shifts,axis=1)


def main():
    start=time.time(); OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    assembly=XorAssembly(); w,b,scan_meta=scan_params()
    checkpoints={
      'multiply':dict(path=assembly.mm['checkpoint'],sha256=assembly.mm['sha256'],status=assembly.mm['status']),
      'addition':dict(path=assembly.am['checkpoint'],sha256=assembly.am['sha256'],status=assembly.am['status']),
      'subtraction':dict(path=assembly.sm['checkpoint'],sha256=assembly.sm['sha256'],status=assembly.sm['status']),
      'scan':dict(path=scan_meta['checkpoint'],sha256=scan_meta['sha256'],status=scan_meta['status'])}
    before={k:sha(Path(v['path'])) for k,v in checkpoints.items()}
    x=np.array([0.,0.,1.,1.]); y=np.array([0.,1.,0.,1.]); teacher=np.array([0.,1.,1.,0.])
    p=assembly.call(assembly.mul,x,y); two=assembly.call(assembly.mul,np.full(4,2.),p)
    summed=assembly.call(assembly.add,x,y); got=assembly.call(assembly.sub,summed,two)
    table=[]
    for i in range(4):
        table.append(dict(x=int(x[i]),y=int(y[i]),product=float(p[i]),twice_product=float(two[i]),
                          sum=float(summed[i]),output=float(got[i]),teacher=int(teacher[i]),
                          output_bits=int(got[i:i+1].view('u8')[0]),exact=bool(got[i]==teacher[i])))
    assert np.array_equal(got.view('u8'),teacher.view('u8'))
    write('truth_table.json',table)

    cases=[(1,32768,'saved'),(2,32768,'saved'),(3,32768,'saved'),
           (4,8192,'held_out'),(8,4096,'held_out'),(16,2048,'fresh'),
           (32,1024,'fresh'),(64,512,'fresh'),(128,256,'higher_dimension'),
           (1024,64,'higher_dimension'),(21201,8,'higher_dimension')]
    rows=[]
    for d,n,split in cases:
        eng=qmc.Sobol(d,scramble=False,bits=30); teacher_coords=eng.random_base2(int(np.log2(n)))
        learned=np.zeros(d,np.uint64); oracle=np.zeros(d,eng.dtype_i)
        digital_errors=0; coordinate_errors=0
        # Point zero is initialized state. Stop before coordinate scaling: comparison below is diagnostic
        # confirmation only and never feeds state.
        for i in range(n-1):
            col=direction_column(i,30,w,b)
            learned=update(assembly,learned,eng._sv[:,col].astype(np.uint64),30)
            sample=np.empty((1,d)); _sobol._draw(1,i,d,eng._scale,eng._sv,oracle,sample)
            digital_errors += int(np.count_nonzero(learned!=oracle))
            # No learned scaling: compare integer state to teacher coordinate's exact inverse scale.
            expected_integer=(teacher_coords[i+1]/eng._scale).astype(np.uint64)
            coordinate_errors += int(np.count_nonzero(learned!=expected_integer))
        assert digital_errors==coordinate_errors==0
        row=dict(dimension=d,points=n,split=split,updates=n-1,digital_words=(n-1)*d,
                 bit_decisions=(n-1)*d*30,digital_state_errors=digital_errors,
                 integer_vs_saved_coordinate_errors=coordinate_errors)
        rows.append(row); write('replay_progress.json',rows); print(row,flush=True)

    rng=np.random.default_rng(9292026); resumed=[]
    for d in (1,7,31,257,4096):
        eng=qmc.Sobol(d,scramble=False,bits=30); errors=0
        counters=np.concatenate((np.array([(1<<j)-1 for j in range(30)],np.uint64),
                                 rng.integers(0,2**29,256,dtype=np.uint64)))
        for counter in counters:
            n=int(counter); gray=n^(n>>1); state=np.zeros(d,np.uint64)
            for j in range(30):
                if (gray>>j)&1: state ^= eng._sv[:,j].astype(np.uint64)
            learned=update(assembly,state,eng._sv[:,direction_column(n,30,w,b)].astype(np.uint64),30)
            oracle=state.astype(eng.dtype_i); sample=np.empty((1,d))
            _sobol._draw(1,n,d,eng._scale,eng._sv,oracle,sample)
            errors += int(np.count_nonzero(learned!=oracle))
        assert errors==0
        resumed.append(dict(dimension=d,states=len(counters),digital_words=len(counters)*d,errors=errors))

    after={k:sha(Path(v['path'])) for k,v in checkpoints.items()}; changes=[k for k in before if before[k]!=after[k]]
    assert not changes
    result=dict(status='LOCKED_ASSEMBLY',name='shared_one_bit_xor',formula='subtract(add(x,y), multiply(2, multiply(x,y)))',
                evaluation_order=['p=multiply(x,y)','two_p=multiply(2,p)','sum=addition(x,y)','out=subtraction(sum,two_p)'],
                truth_table=table,truth_table_errors=0,replays=rows,resumed=resumed,
                totals=dict(digital_words=sum(r['digital_words'] for r in rows),
                            bit_decisions=sum(r['bit_decisions'] for r in rows),
                            digital_state_errors=0,resumed_digital_words=sum(r['digital_words'] for r in resumed)),
                checkpoints=checkpoints,protected_checkpoint_changes=changes,seconds=time.time()-start,
                scope='Unscrambled integer digital recurrence only. Float scaling and scrambling were not executed as learned stages.')
    write('results.json',result)
    manifest=dict(status='LOCKED_ASSEMBLY',name='shared_one_bit_xor',components=checkpoints,
                  result_sha256=sha(OUT/'results.json'),implementation_sha256=sha(Path(__file__)),
                  next_operation='float coordinate scaling (explicitly excluded from this run)')
    write('shared_one_bit_xor_locked.json',manifest)
    print('PASS',result['totals'],flush=True)


if __name__=='__main__': main()
