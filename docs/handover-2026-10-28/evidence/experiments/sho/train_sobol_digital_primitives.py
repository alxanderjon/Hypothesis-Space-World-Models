"""Isolated production-Sobol digital peelback. No scramble or N1 execution.

Authority: SciPy 1.13.1 stats/_sobol.pyx low_0_bit/draw and installed _draw.
Representational scaffolding: bit serialization, counters, indexing, packing.
Learned operations: scan-bit routing and shared one-bit XOR.
"""
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc, _sobol, _qmc
import torch
from torch import nn

OUT = Path('outputs/sho/stage1B/sobol_digital_primitives')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')


class BitHead(nn.Module):
    def __init__(self, xor=False):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
                    if xor else nn.Sequential(nn.Linear(1, 1)))
        self.double()

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(name, x, y, xor=False):
    # One fixed small architecture per operation. No restart/capacity search.
    torch.manual_seed(20260930 if xor else 20260929)
    m = BitHead(xor)
    initial = {k: v.tolist() for k, v in m.state_dict().items()}
    opt = torch.optim.Adam(m.parameters(), lr=.03)
    xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    for epoch in range(3000):
        opt.zero_grad()
        logits = m(xt)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss.backward(); opt.step()
        with torch.no_grad():
            signed = m(xt) * (2*yt-1)
            if bool((signed > 2).all()):
                break
    m.eval().requires_grad_(False)
    errors = int(((m(xt) > 0).numpy() != np.array(y)).sum())
    report = dict(parameters=sum(p.numel() for p in m.parameters()), epochs=epoch+1,
                  initial=initial, learned={k: v.tolist() for k,v in m.state_dict().items()},
                  truth_table_errors=errors, logits=m(xt).tolist())
    save(name+'_screen.json', report)
    if errors:
        raise RuntimeError(name+' failed; stop without enlarging')
    return m, report


class Digital:
    def __init__(self, scan, xor=None):
        self.scan = scan
        self.xor = xor

    def column(self, counters, bits):
        counters = np.asarray(counters, dtype=np.uint64).reshape(-1)
        found = np.zeros(len(counters), bool)
        result = np.full(len(counters), -1, np.int64)
        w = self.scan.net[0].weight.numpy()[0, 0]
        b = self.scan.net[0].bias.numpy()[0]
        for j in range(bits):
            bit = ((counters >> np.uint64(j)) & np.uint64(1)).astype(np.float64)
            # Learned 1 = CONTINUE, learned 0 = SELECT. No bit-based fallback.
            select = ~(w*bit+b > 0)
            take = ~found & select
            result[take] = j; found |= take
        if not found.all():
            raise ValueError('No valid direction column: exhausted sequence')
        return result

    def update(self, state, direction, bits):
        a, b = np.broadcast_arrays(np.asarray(state, np.uint64), np.asarray(direction, np.uint64))
        result = np.zeros(a.shape, np.uint64)
        w1 = self.xor.net[0].weight.numpy(); b1 = self.xor.net[0].bias.numpy()
        w2 = self.xor.net[2].weight.numpy()[0]; b2 = self.xor.net[2].bias.numpy()[0]
        for j in range(bits):
            u = ((a >> np.uint64(j)) & np.uint64(1)).astype(np.float64)
            v = ((b >> np.uint64(j)) & np.uint64(1)).astype(np.float64)
            h0 = np.tanh(w1[0,0]*u+w1[0,1]*v+b1[0])
            h1 = np.tanh(w1[1,0]*u+w1[1,1]*v+b1[1])
            bit = (w2[0]*h0+w2[1]*h1+b2 > 0).astype(np.uint64)
            result |= bit << np.uint64(j)
        return result


def lock(name, model, report):
    path = OUT / (name+'.pt')
    torch.save(dict(state_dict=model.state_dict(), report=report), path)
    save(name+'_locked.json', dict(status='LOCKED', checkpoint=str(path), sha256=sha(path), **report))


def main():
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    assert scipy.__version__ in ('1.13.0', '1.13.1'), 'Reaudit actual source on version change'
    if (OUT/'protected_before.json').exists():
        protected=json.loads((OUT/'protected_before.json').read_text())
    else:
        protected = {str(p):sha(p) for p in Path('outputs').rglob('*')
                     if p.is_file() and OUT not in p.parents and p.suffix in ('.pt','.pth','.npz','.json')}
        save('protected_before.json', protected)
    save('authority.json', dict(scipy=scipy.__version__, binary=str(_sobol.__file__),
         binary_sha256=sha(Path(_sobol.__file__)),
         upstream='https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_sobol.pyx',
         production='experiments/sho/run_stage1b_enhanced_edirect_boundary_rescue.py:_candidate_pool',
         init_scaffolding='SciPy _initialize_v is retained, including its polynomial recurrence; not learned here',
         scaling_scaffolding='uint-to-float conversion then multiplication by 2**(-bits) retained'))
    (OUT/'installed_qmc_source.txt').write_text(inspect.getsource(_qmc.Sobol))
    scan, sr = fit('sobol_scan_bit', [[0],[1]], [0,1])
    digital = Digital(scan)
    rng = np.random.default_rng(4072026)
    scan_rows=[]
    for bits in (30,32,53,64):
        values = rng.integers(0, np.iinfo(np.uint64).max, 12000, dtype=np.uint64)
        values &= np.uint64((1<<bits)-1)
        values = values[values != np.uint64((1<<bits)-1)]
        values = np.concatenate((values, np.array([(1<<j)-1 for j in range(bits)],np.uint64)))
        expected=np.array([((int(n)+1)&-(int(n)+1)).bit_length()-1 for n in values])
        got=digital.column(values,bits)
        # Independent installed Cython oracle: one-hot direction columns reveal selected bit.
        dtype=np.uint32 if bits<=32 else np.uint64
        sv=np.array([[1<<j for j in range(bits)]],dtype=dtype)
        oracle_count=0
        for n,j in zip(values,got):
            # Installed Cython low_0_bit hangs when the zero index is >=32.
            # This optional extension is outside production's bits=30 domain.
            if j>=32:
                continue
            state=np.zeros(1,dtype=dtype); sample=np.empty((1,1))
            _sobol._draw(1,int(n),1,2.**-bits,sv,state,sample)
            assert int(state[0]) == 1<<int(j)
            oracle_count+=1
        errors=int(np.count_nonzero(got!=expected)); assert errors==0
        scan_rows.append(dict(bits=bits,cases=len(values),errors=errors,installed_oracle_cases=oracle_count,
                              unavailable_installed_oracle_cases=len(values)-oracle_count))
    sr['fresh_counter_and_installed_draw_validation']=scan_rows
    lock('sobol_scan_bit',scan,sr)
    print('scan locked',scan_rows,flush=True)
    xor,xr=fit('sobol_bit_xor',[[0,0],[0,1],[1,0],[1,1]],[0,1,1,0],True)
    digital.xor=xor
    xor_rows=[]
    for bits in (30,32,53,64):
        a=rng.integers(0,np.iinfo(np.uint64).max,24000,dtype=np.uint64)&np.uint64((1<<bits)-1)
        b=rng.integers(0,np.iinfo(np.uint64).max,24000,dtype=np.uint64)&np.uint64((1<<bits)-1)
        got=digital.update(a,b,bits)
        errors=int(np.count_nonzero(got != (a^b))); assert errors==0
        xor_rows.append(dict(bits=bits,words=len(a),bit_decisions=len(a)*bits,errors=errors))
    xr['fresh_word_validation']=xor_rows
    lock('sobol_bit_xor',xor,xr)
    print('xor locked',xor_rows,flush=True)
    # Full recurrent digital execution, not independent teacher-fed updates.
    replay=[]
    cases=[(1,32768,30,'held_out'),(2,32768,30,'held_out'),(3,32768,30,'held_out'),
           (4,8192,30,'fresh'),(8,4096,30,'fresh'),(16,2048,30,'higher_dimension'),
           (32,1024,30,'higher_dimension'),(64,512,30,'higher_dimension'),
           (128,256,30,'higher_dimension'),(1024,64,30,'higher_dimension'),
           (21201,8,30,'higher_dimension'),(7,1024,32,'fresh_width'),
           (17,512,53,'fresh_width'),(33,256,64,'fresh_width')]
    for d,n,bits,split in cases:
        eng=qmc.Sobol(d,scramble=False,bits=bits)
        reference=eng.random_base2(int(np.log2(n)))
        q=np.zeros(d,np.uint64); coords=np.zeros((n,d)); states=[]
        cols=digital.column(np.arange(n-1,dtype=np.uint64),bits)
        teacher_state=np.zeros(d,dtype=eng.dtype_i)
        state_errors=0
        for i,j in enumerate(cols):
            q=digital.update(q,eng._sv[:,j],bits)
            oracle=np.empty((1,d))
            _sobol._draw(1,i,d,eng._scale,eng._sv,teacher_state,oracle)
            state_errors+=int(np.count_nonzero(q!=teacher_state))
            coords[i+1]=q.astype(np.float64)*eng._scale
        errors=int(np.count_nonzero(coords.view(np.uint64)!=reference.view(np.uint64)))
        assert state_errors==errors==0,(d,n,bits,state_errors,errors)
        row=dict(dimension=d,points=n,bits=bits,split=split,digital_words=(n-1)*d,
                 digital_errors=state_errors,coordinates=n*d,coordinate_errors=errors)
        replay.append(row); save('replay_progress.json',replay); print(row,flush=True)
    # Fresh resumed states at every counter-bit boundary, and arbitrary valid counters.
    resumed=[]
    for bits in (30,32,53,64):
        d=19; eng=qmc.Sobol(d,scramble=False,bits=bits)
        indices=[(1<<j)-1 for j in range(min(bits,32))] + [int(v) for v in rng.integers(0,2**29,200)]
        for n in indices:
            # Genuine sequence state from Gray-code/direction reference, teacher only.
            gray=n^(n>>1); state=np.zeros(d,dtype=eng.dtype_i)
            for j in range(bits):
                if (gray>>j)&1: state ^= eng._sv[:,j]
            col=digital.column([n],bits)[0]
            got=digital.update(state,eng._sv[:,col],bits)
            oracle=state.copy(); sample=np.empty((1,d))
            _sobol._draw(1,n,d,eng._scale,eng._sv,oracle,sample)
            assert np.array_equal(got,oracle)
            assert np.array_equal((got.astype(np.float64)*eng._scale).view(np.uint64),sample[0].view(np.uint64))
        resumed.append(dict(bits=bits,states=len(indices),digital_words=len(indices)*d,errors=0))
    changed=[p for p,h in protected.items() if sha(Path(p))!=h]
    assert not changed,changed
    result=dict(status='VALIDATED_CANDIDATE',name='sobol_unscrambled_digital_sequence',
                scan=sr,xor=xr,replays=replay,resumed=resumed,
                protected_files=len(protected),protected_changes=changed,seconds=time.time()-started,
                stop='Unscrambled sequence exact on tested domain. Scrambling untouched.',
                next_operation='SciPy Sobol._scramble: RNG bit matrix for digital shift, then LMS direction-number scramble',
                scope='Learned scan decision and XOR only. Bit access/packing, loop/index state, direction initialization, cast/scale remain deterministic.')
    save('results.json',result)
    save('sobol_unscrambled_digital_sequence_candidate.json',dict(status='VALIDATED_CANDIDATE',
         components={n:sha(OUT/(n+'.pt')) for n in ('sobol_scan_bit','sobol_bit_xor')},
         result_sha256=sha(OUT/'results.json'),script_sha256=sha(Path(__file__)),scope=result['scope']))
    print('PASS',result['seconds'],flush=True)


if __name__=='__main__':
    main()
