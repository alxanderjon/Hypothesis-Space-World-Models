"""Read-only dependency/domain audit. No correction router or output assembly."""
import csv
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import torch

from experiments.geometry import train_division_operator as src
from experiments.geometry.compensated_division_residual import CompensatedDivisionResidual
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'division_remaining_primitives_audit'


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    protected={str(p):src.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):src.sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('locked','manifest','validated'))})
    for name in ('division_rounding_diagnosis','compensated_division_residual_validation'):
        protected.update({str(p):src.sha(p) for p in (ROOT/name).iterdir() if p.is_file()})
    assembly=CompensatedDivisionResidual()
    th,thmeta=src.load(MagnitudeThreshold,'stationary_magnitude_threshold_smoke/zero_anchor_converged')
    with np.load(ROOT/'compensated_division_residual_validation/compensated_residual_outputs.npz') as data:
        z={k:data[k] for k in data.files}
    x,y,q,hi,lo=[z[k] for k in ('x','y','q','hi','lo')];n_saved=len(x)
    # Saved true-halfway controls; no solve, training, or generated search state.
    with (ROOT/'division_rounding_diagnosis/explicit_halfway_controls.csv').open() as f:controls=list(csv.DictReader(f))
    cx=np.array([float(r['x']) for r in controls]);cy=np.array([float(r['y']) for r in controls]);cq=np.array([float(r['model']) for r in controls])
    ce=assembly.construct(cx,cy,cq)
    x=np.r_[x,cx];y=np.r_[y,cy];q=np.r_[q,cq];hi=np.r_[hi,ce.hi];lo=np.r_[lo,ce.lo]
    rc=assembly._call(assembly.add,hi,lo)
    with torch.no_grad():learned_direction=th.decision(torch.from_numpy(rc)).numpy().astype(int)
    direction=np.sign(rc).astype(int);Q=abs(q);mx,ex=np.frexp(abs(x));my,ey=np.frexp(abs(y));e=ex-ey
    active=np.flatnonzero(direction!=0)
    neighbor=np.full(len(x),np.nan)
    neighbor[active]=np.nextafter(Q[active],np.where(direction[active]>0,np.inf,-np.inf))
    used=active[np.isfinite(neighbor[active])]
    d=np.full(len(x),np.nan);g=d.copy();h=d.copy();margin=d.copy()
    d[used]=abs(assembly._call(assembly.subtract,neighbor[used],Q[used]))
    g[used]=np.ldexp(d[used],-e[used])
    intermediate=assembly._call(assembly.multiply,g[used],my[used])
    h[used]=assembly._call(assembly.multiply,intermediate,.5)
    margin[used]=assembly._call(assembly.subtract,abs(rc[used]),h[used])
    learned_comparison=np.full(len(x),9,dtype=int)
    with torch.no_grad():learned_comparison[used]=th.decision(torch.from_numpy(margin[used])).numpy().astype(int)
    # Characterize inputs/labels only. Do NOT implement move/keep/parity routing.
    parity=(Q.view('u8')&np.uint64(1)).astype(int)
    exact_direction=np.zeros(len(x),dtype=int);exact_comparison=np.full(len(x),9,dtype=int)
    threshold_exact=np.ones(len(x),dtype=bool);residual_scalar_exact=threshold_exact.copy()
    count_state={};oracle_rows=[]
    used_set=set(used.tolist())
    for i in range(len(x)):
        r=F(float(hi[i]))+F(float(lo[i]));exact_direction[i]=(r>0)-(r<0)
        residual_scalar_exact[i]=F(float(rc[i]))==r
        if i in used_set:
            # Exact normalized equivalent of the actual rational diagnostic's
            # half=abs(F(neighbor)-F(Q))*abs(y)/2, retaining its meaning.
            exact_h=abs(F(float(neighbor[i]))-F(float(Q[i])))*F(float(my[i]))*F(2)**(-int(e[i]))/2
            threshold_exact[i]=F(float(h[i]))==exact_h
            diff=abs(r)-exact_h;exact_comparison[i]=(diff>0)-(diff<0)
            label=('LT','EQ','GT')[exact_comparison[i]+1]
            state=(label,int(parity[i]));count_state[state]=count_state.get(state,0)+1
            if (not residual_scalar_exact[i] or not threshold_exact[i]) and len(oracle_rows)<60:
                oracle_rows.append(dict(index=i,hi_hex=float(hi[i]).hex(),lo_hex=float(lo[i]).hex(),rounded_hex=float(rc[i]).hex(),
                    threshold_hex=float(h[i]).hex(),comparison_exact=int(exact_comparison[i]),comparison_learned=int(learned_comparison[i]),
                    scalar_residual_exact=bool(residual_scalar_exact[i]),threshold_exact=bool(threshold_exact[i])))
    subsets={'required':np.arange(119210),'Brent':np.arange(45582),
             'fresh_and_extreme_residual_validation':np.arange(119210,n_saved),
             'saved_halfway_controls':np.arange(n_saved,len(x)),'all_audited':np.arange(len(x))}
    metrics=[];domains=[]
    for name,ii in subsets.items():
        jj=ii[np.isin(ii,used)]
        metrics.append(dict(group=name,cases=len(ii),zero_residual=int((direction[ii]==0).sum()),
            positive_residual=int((direction[ii]>0).sum()),negative_residual=int((direction[ii]<0).sum()),
            active_finite_neighbor_cases=len(jj),nonfinite_neighbor_guard=int(((direction[ii]!=0)&~np.isfinite(neighbor[ii])).sum()),
            learned_residual_sign_errors=int((learned_direction[ii]!=exact_direction[ii]).sum()),
            exact_threshold_value_errors=int((~threshold_exact[jj]).sum()),
            scalar_residual_loses_exact_bits=int((~residual_scalar_exact[ii]).sum()),
            learned_threeway_comparison_errors=int((learned_comparison[jj]!=exact_comparison[jj]).sum()),
            LT=int((exact_comparison[jj]<0).sum()),EQ=int((exact_comparison[jj]==0).sum()),GT=int((exact_comparison[jj]>0).sum()),
            odd_Q=int(parity[ii].sum()),even_Q=int((parity[ii]==0).sum())))
        for quantity,arr,j in [('Q',Q,ii),('mx',mx,ii),('my',my,ii),('e',e,ii),('hi',hi,ii),('lo',lo,ii),
                               ('rounded_residual',rc,ii),('neighbor',neighbor,jj),('directional_output_gap',d,jj),
                               ('normalized_gap',g,jj),('half_threshold',h,jj),('comparison_margin',margin,jj)]:
            v=arr[j];nz=abs(v[v!=0])
            domains.append(dict(group=name,quantity=quantity,cases=len(v),minimum=float(v.min()),maximum=float(v.max()),
                minimum_nonzero_absolute=float(nz.min()) if len(nz) else None,
                zeros=int((v==0).sum()),subnormals=int(((abs(v)<np.finfo(float).tiny)&(v!=0)).sum()),
                nonfinite=int((~np.isfinite(v)).sum())))
    write('existing_decision_reuse.csv',metrics);write('primitive_domains.csv',domains)
    write('rounding_state_coverage.csv',[dict(comparison=k[0],odd=k[1],cases=v) for k,v in sorted(count_state.items())])
    if oracle_rows:write('expansion_vs_scalar_comparison_cases.csv',oracle_rows)
    unchanged=all(src.sha(Path(p))==v for p,v in protected.items());assert unchanged
    result=dict(status='ISOLATION_AUDIT_ONLY',metrics=metrics,
        reused_threshold={k:thmeta.get(k) for k in ('status','checkpoint','sha256','learned_theta','learned_alpha')},
        first_missing_learnable_decision='rounding move eligibility: GT OR (EQ AND odd)',
        later_new_learnable_primitives=[],caveat='Scoped to successful diagnostic and tested domain. Arbitrary expansion-aware comparison beyond it is not promoted by this audit.',
        protected_files=len(protected),protected_hashes_unchanged=unchanged,training=False,
        new_checkpoint=False,rounding_router_implemented=False,final_quotient_assembly=False,Brent=False,recursion=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
