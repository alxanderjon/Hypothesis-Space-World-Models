"""One frozen full-derivative audit; no training, minimization or root solving."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.validate_sqrt_derivative_branch import SqrtDerivativeBranch,DerivativeInputs,COMPONENTS as SQ_COMPONENTS
from experiments.geometry.validate_log_derivative_branch import LogDerivativeBranch

OUT=ROOT/'full_derivative_assembly'
EPS=np.finfo(float).eps
BLOCK=8192


class FullDerivative(nn.Module):
    """Inputs A,B,a,beta,c,o_t,o_v; only validated branches and frozen addition."""
    def __init__(self,sqrt_branch,log_branch,addition):
        super().__init__(); self.sqrt_branch=sqrt_branch; self.log_branch=log_branch; self.addition=addition

    def forward(self,x,details=False):
        ds=self.sqrt_branch(x[:,[0,1,2,4,5]])
        dl=self.log_branch(x[:,[3,4,6]])
        value=self.addition(torch.stack((ds,dl),1))
        return torch.stack((value,ds,dl),1) if details else value


class Inputs(DerivativeInputs):
    def collect(self):
        self.pieces=self.read(ROOT/'fixed_allocation_interval_audit/fixed_interval_score_structure.csv')
        self.piece_map={tuple(int(r[k]) for k in ('parent','axis','interval','piece')):r for r in self.pieces}
        super().collect()

    def geometry(self,r,axis,k,c,scope,case):
        self.log_context=r,axis
        super().geometry(r,axis,k,c,scope,case)

    def add(self,scope,case,c,a,B,tcode,vcode,order='generic'):
        super().add(scope,case,c,a,B,tcode,vcode,order)
        b=self.batches[-1]; ov=np.broadcast_to(np.asarray(vcode,float),b['c'].shape).copy()
        if scope=='archived_generic_subpieces': beta=np.full(len(c),self.generic[str(case)]['beta'])
        elif scope=='archived_subpiece_grids': beta=np.full(len(c),.05*int(self.piece_map[tuple(case)]['aspect_log_power']))
        else:
            r,axis=self.log_context; sides=np.asarray(r['upper'])-r['lower']; other=np.delete(sides,axis)
            side=a*b['v']
            beta=.05*np.where(side>other.max(),1.,np.where(side<other.min(),-1.,0.)) if len(other) else np.zeros(len(c))
        b['beta']=beta; b['ov']=ov


def teacher(x):
    A,B,a,beta,c,ot,ov=x.T
    t=np.where(ot==0,c,1-c); v=np.where(ov==0,c,1-c)
    tp=np.where(ot==0,1.,-1.); vp=np.where(ov==0,1.,-1.)
    ds=(((A*(a*a))*t)*tp)/np.sqrt(B+(a*a)*(t*t))
    dl=(beta*vp)/v
    return np.column_stack((ds+dl,ds,dl))


def case_matrix(coeff,c):
    c=np.asarray(c,dtype=float)
    A,B,a,beta,ot,ov=coeff
    return np.column_stack((np.full(len(c),A),np.full(len(c),B),np.full(len(c),a),np.full(len(c),beta),c,
                            np.full(len(c),ot),np.full(len(c),ov)))


def classification(pair):
    a,b=pair
    if not np.isfinite(a) or not np.isfinite(b): return 'nonfinite'
    if a==0 or b==0: return 'endpoint_zero'
    if a>0 and b>0: return 'positive_both'
    if a<0 and b<0: return 'negative_both'
    return 'sign_change'


def stats(truth,pred):
    y=truth[:,0]; p=pred[:,0]; scale=abs(truth[:,1])+abs(truth[:,2]); error=p-y
    ratio=np.divide(abs(y),scale,out=np.zeros_like(y),where=scale>0)
    meaningful=(abs(y)>=np.finfo(float).tiny)&(ratio>=1e-8)
    nonzero=y!=0; sign=np.sign(y)!=np.sign(p); normalizer=scale>0
    relative=error[meaningful]/y[meaningful]
    normalized=error[normalizer]/scale[normalizer]
    largest=float(abs(error).max())
    strict=(np.sign(y)*np.sign(p))<0
    return dict(cases=len(y),rmse=float(largest*np.sqrt(np.mean((error/largest)**2))) if largest else 0.,
        mae=float(np.mean(abs(error))),max_absolute_error=largest,
        meaningful_relative_cases=int(meaningful.sum()),max_relative_error_meaningful=float(abs(relative).max()) if len(relative) else None,
        max_relative_error_all_nonzero=float(abs(error[nonzero]/y[nonzero]).max()) if nonzero.any() else None,
        max_branch_normalized_error=float(abs(normalized).max()) if len(normalized) else 0.,
        nonfinite_outputs=int(np.sum(~np.isfinite(pred))),sign_mismatches=int(sign.sum()),strict_opposite_signs=int(strict.sum()),
        teacher_zero_learned_nonzero=int(np.sum((y==0)&(p!=0))),learned_zero_teacher_nonzero=int(np.sum((p==0)&(y!=0))),
        smallest_abs_teacher_at_disagreement=float(abs(y[sign]).min()) if sign.any() else None,
        smallest_nonzero_abs_teacher_at_disagreement=float(abs(y[sign&nonzero]).min()) if np.any(sign&nonzero) else None,
        sign_mismatches_outside_branch_roundoff=int(np.sum(sign&(abs(y)>32*EPS*scale))),
        exact_matches=int(np.sum(y==p)))


def run():
    start=time.monotonic(); torch.set_num_threads(1); OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    refs={}; branches={}; assembly_sources={}
    for name,folder,file,cls in [('sqrt','sqrt_derivative_branch_assembly','sqrt_derivative_branch_validated.json',SqrtDerivativeBranch),
                               ('log','log_derivative_branch_assembly','log_derivative_branch_validated.json',LogDerivativeBranch)]:
        path=ROOT/folder/file; m=json.loads(path.read_text()); result=json.loads((path.parent/'results.json').read_text())
        assert m['status']=='VALIDATED_ASSEMBLY' and sha(path)==result['assembly_sha256']
        code=Path('experiments/geometry')/('validate_sqrt_derivative_branch.py' if name=='sqrt' else 'validate_log_derivative_branch.py')
        assert sha(code)==m['implementation_sha256']
        assembly_sources[str(path)]=sha(path); assembly_sources[str(code)]=sha(code)
        models={}
        for key,ref in m['components'].items():
            cp=Path(ref['checkpoint']); assert sha(cp)==ref['sha256']
            model=SQ_COMPONENTS[key][1](); model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
            models[key]=model
        branches[name]=cls(models).eval().requires_grad_(False)
        refs[name]=dict(manifest=str(path),sha256=sha(path))
    ap=ROOT/'addition_operator_smoke/candidate_manifest.json'; am=json.loads(ap.read_text()); ac=Path(am['checkpoint'])
    assert sha(ac)==am['sha256']
    adder=Addition(); adder.load_state_dict(torch.load(ac,weights_only=False)['state_dict']); adder.eval().requires_grad_(False)
    refs['addition']=dict(checkpoint=str(ac),sha256=sha(ac))
    model=FullDerivative(branches['sqrt'],branches['log'],adder).eval().requires_grad_(False)
    source=Inputs(); source.collect()
    arrays=[]; groups=[]; offset=0
    def add(x,kind,scope,case,**meta):
        nonlocal offset
        groups.append(dict(start=offset,end=offset+len(x),kind=kind,scope=scope,case=str(case),**meta))
        arrays.append(x); offset+=len(x)
    for b in source.batches:
        x=np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov']))
        add(x,'archive',b['scope'],b['case'])
    archived_count=offset
    subpieces=[]
    for b in source.batches:
        if b['scope']=='archived_subpiece_grids':
            key=tuple(int(v.strip()) for v in b['case'].strip('()').split(',')); row=source.piece_map[key]
            roots=json.loads(row['stationary_points'])+json.loads(row['quartic_stationary_roots'])
            subpieces.append(dict(scope='geometric',case=b['case'],coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),
                                  lower=float(row['lower']),upper=float(row['upper']),roots=roots))
    for key,r in source.generic.items():
        subpieces.append(dict(scope='generic_coefficients',case=key,coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),
                              lower=r['lower'],upper=r['upper'],roots=r['roots']))
    local_count=0; crossing_subpieces=0
    for sp in subpieces:
        lo,hi=sp['lower'],sp['upper']; coeff=sp['coeff']
        add(case_matrix(coeff,[lo,hi]),'endpoints',sp['scope'],sp['case'],lower=lo,upper=hi)
        grid=np.unique(np.r_[np.linspace(lo,hi,129),[r for r in sp['roots'] if lo<=r<=hi]])
        x=case_matrix(coeff,grid); yt=teacher(x)[:,0]
        add(x,'dense_subpiece',sp['scope'],sp['case'])
        brackets=np.flatnonzero(np.sign(yt[:-1])*np.sign(yt[1:])<0)
        zero_locations=grid[yt==0]
        # Fixed local sampling only. Saved roots are inputs, never recomputed.
        centers=np.unique(np.r_[[r for r in sp['roots'] if lo<=r<=hi],zero_locations])
        local=[]
        for i in brackets: local.extend(np.linspace(grid[i],grid[i+1],257))
        for center in centers:
            width=max((hi-lo)*1e-6,128*np.spacing(center))
            local.extend(np.linspace(max(lo,center-width),min(hi,center+width),257))
            bits=np.float64(center).view(np.int64)+np.arange(-64,65,dtype=np.int64)
            floats=bits.view(np.float64); local.extend(floats[(floats>=lo)&(floats<=hi)])
        if local:
            local=np.unique(local); local=local[(local>=lo)&(local<=hi)]
            add(case_matrix(coeff,local),'local_crossing',sp['scope'],sp['case'],saved_roots=json.dumps(sp['roots']),
                coarse_teacher_sign_brackets=len(brackets),coarse_teacher_zeros=len(zero_locations))
            local_count+=len(local); crossing_subpieces+=1
    rng=np.random.default_rng(27492026); synth=[]
    for _ in range(256):
        A=10**rng.uniform(-8,8); B=10**rng.uniform(-20,12); a=10**rng.uniform(-8,6)
        c=rng.uniform(.001,.999); ot,ov=rng.integers(0,2,2)
        base=case_matrix((A,B,a,0.,ot,ov),[c]); ds=teacher(base)[0,1]
        v=c if ov==0 else 1-c; vp=1. if ov==0 else -1.
        cancellation_beta=-(ds*v)/vp
        for delta in (0.,-EPS,EPS,-8*EPS,8*EPS,-1e-12,1e-12,-1e-8,1e-8):
            synth.append([A,B,a,cancellation_beta*(1+delta),c,ot,ov])
    add(np.array(synth),'synthetic_cancellation','synthetic','coefficient_balanced_pairs')
    data=np.concatenate(arrays); del arrays
    print(json.dumps(dict(event='one_audit_pool_ready',cases=len(data),archived=archived_count,subpieces=len(subpieces),
                          local_crossing_subpieces=crossing_subpieces,local_points=local_count)),flush=True)
    truth=teacher(data); assert np.all(np.isfinite(truth))
    output=np.empty_like(truth)
    # All requested diagnostics are evaluated in this single batch audit.
    with torch.no_grad():
        for start_i in range(0,len(data),BLOCK):
            end=min(start_i+BLOCK,len(data)); output[start_i:end]=model(torch.from_numpy(data[start_i:end]),details=True).numpy()
    rows=[dict(group='all',**stats(truth,output))]
    for kind in sorted(set(g['kind'] for g in groups)):
        ids=np.concatenate([np.arange(g['start'],g['end']) for g in groups if g['kind']==kind])
        rows.append(dict(group=kind,**stats(truth[ids],output[ids])))
    scale=abs(truth[:,1])+abs(truth[:,2]); ratio=np.divide(abs(truth[:,0]),scale,out=np.ones(len(data)),where=scale>0)
    opposite=np.sign(truth[:,1])*np.sign(truth[:,2])<0
    for threshold in (1e-4,1e-8,1e-12,32*EPS):
        mask=opposite&(ratio<=threshold)
        if mask.any(): rows.append(dict(group=f'near_cancellation_ratio_le_{threshold:.17g}',**stats(truth[mask],output[mask])))
    endpoints=[]; crossing_rows=[]; endpoint_failures=[]
    for g in groups:
        sl=slice(g['start'],g['end']); t=truth[sl,0]; p=output[sl,0]
        if g['kind']=='endpoints':
            tc,pc=classification(t),classification(p)
            tm=tc in ('endpoint_zero','sign_change'); pm=pc in ('endpoint_zero','sign_change')
            row=dict(scope=g['scope'],case=g['case'],lower=g['lower'],upper=g['upper'],teacher_class=tc,learned_class=pc,
                teacher_left=float(t[0]),teacher_right=float(t[1]),learned_left=float(p[0]),learned_right=float(p[1]),
                classification_changed=tc!=pc,missed_strict_sign_bracket=tc=='sign_change' and pc!='sign_change',
                added_strict_sign_bracket=pc=='sign_change' and tc!='sign_change',
                missed_endpoint_or_bracket=tm and not pm,added_endpoint_or_bracket=pm and not tm)
            endpoints.append(row)
            if tc!=pc: endpoint_failures.append(row)
        elif g['kind']=='local_crossing':
            ts,ps=np.sign(t),np.sign(p)
            transitions_t=np.flatnonzero(ts[1:]!=ts[:-1]); transitions_p=np.flatnonzero(ps[1:]!=ps[:-1])
            crossing_rows.append(dict(scope=g['scope'],case=g['case'],points=len(t),sign_disagreements=int(np.sum(ts!=ps)),
                strict_opposite_signs=int(np.sum(ts*ps<0)),teacher_zero_learned_nonzero=int(np.sum((t==0)&(p!=0))),
                learned_zero_teacher_nonzero=int(np.sum((p==0)&(t!=0))),
                transition_indices_differ=not np.array_equal(transitions_t,transitions_p),
                teacher_transition_indices=json.dumps(transitions_t.tolist()),learned_transition_indices=json.dumps(transitions_p.tolist())))
    sign_mask=np.sign(truth[:,0])!=np.sign(output[:,0]); failures=[]; ends=np.array([g['end'] for g in groups])
    for i in np.flatnonzero(sign_mask):
        g=groups[int(np.searchsorted(ends,i,side='right'))]; x=data[i]; t,p=truth[i],output[i]
        failures.append(dict(kind=g['kind'],scope=g['scope'],case=g['case'],A=float(x[0]),B=float(x[1]),a=float(x[2]),beta=float(x[3]),
            c=float(x[4]),c_hex=float(x[4]).hex(),orientation_t=int(x[5]),orientation_v=int(x[6]),
            teacher_sqrt=float(t[1]),teacher_log=float(t[2]),learned_sqrt=float(p[1]),learned_log=float(p[2]),
            teacher_derivative=float(t[0]),learned_derivative=float(p[0]),absolute_error=float(abs(p[0]-t[0])),
            cancellation_ratio=float(ratio[i]),branch_normalized_error=float(abs(p[0]-t[0])/scale[i]) if scale[i]>0 else 0.,
            strict_opposite_sign=bool(np.sign(t[0])*np.sign(p[0])<0)))
    write_csv(OUT/'full_derivative_validation.csv',rows)
    write_csv(OUT/'subpiece_endpoint_classifications.csv',endpoints)
    write_csv(OUT/'root_existence_changes.csv',endpoint_failures or [dict(status='no_changes')])
    write_csv(OUT/'dense_zero_crossing_audit.csv',crossing_rows)
    write_csv(OUT/'full_derivative_sign_failures.csv',failures or [dict(status='no_failures')])
    root=dict(subpieces=len(endpoints),classification_changes=len(endpoint_failures),
        missed_strict_sign_brackets=sum(r['missed_strict_sign_bracket'] for r in endpoints),
        added_strict_sign_brackets=sum(r['added_strict_sign_bracket'] for r in endpoints),
        missed_endpoint_or_bracket=sum(r['missed_endpoint_or_bracket'] for r in endpoints),
        added_endpoint_or_bracket=sum(r['added_endpoint_or_bracket'] for r in endpoints))
    crossing=dict(subpieces=len(crossing_rows),subpieces_with_sign_discrepancies=sum(r['sign_disagreements']>0 for r in crossing_rows),
        subpieces_with_transition_order_changes=sum(r['transition_indices_differ'] for r in crossing_rows),
        sign_disagreements=sum(r['sign_disagreements'] for r in crossing_rows))
    addition_error=output[:,0]-(output[:,1]+output[:,2])
    unchanged=all(sha(Path(p))==h for p,h in protected.items()); assert unchanged
    assert all(sha(Path(p))==h for p,h in {**source.inputs,**assembly_sources}.items())
    # User's explicit stop rule applies even when a sign disagreement is within
    # branch-level float64 rounding. Do not waive a zero/root classification change.
    passed=bool(rows[0]['sign_mismatches']==0 and root['classification_changes']==0 and rows[0]['nonfinite_outputs']==0
                and rows[0]['max_branch_normalized_error']<=32*EPS)
    result=dict(operator='full_derivative_operator',status='VALIDATED_ASSEMBLY' if passed else 'DIAGNOSTIC_ONLY_STOP',gate_passed=passed,
        metrics=rows,root_existence=root,dense_crossings=crossing,addition_max_error_on_same_branch_outputs=float(abs(addition_error).max()),
        first_sign_disagreement=failures[0] if failures else None,first_endpoint_classification_change=endpoint_failures[0] if endpoint_failures else None,
        precision_conventions=dict(meaningful_relative='|teacher sum|/(|teacher sqrt|+|teacher log|)>=1e-8 and normal nonzero sum',
            cancellation='Opposite branch signs; report four explicit cancellation-ratio thresholds',
            sign_gate='Any raw sign/zero disagreement fails; branch-normalized roundoff reported separately, never used to waive it',
            root_existence='Endpoint sign classifications only, not a proof that same-sign endpoints exclude multiple interior roots'),
        components=refs,protected_checkpoints=len(protected),all_prior_checkpoint_hashes_unchanged=unchanged,
        no_training=True,no_root_solving=True,no_interval_minimization=True,elapsed_seconds=time.monotonic()-start)
    if passed:
        manifest=dict(operator='full_derivative_operator',status='VALIDATED_ASSEMBLY',components=refs,
            module='experiments.geometry.validate_full_derivative',model='FullDerivative',implementation_sha256=sha(Path(__file__)),
            inputs=['A','B','a','beta','c','orientation_t','orientation_v'],validation=str(OUT/'results.json'))
        path=OUT/'full_derivative_validated.json'; path.write_text(json.dumps(manifest,indent=2))
        digest=sha(path); path.with_suffix('.sha256').write_text(f'{digest}  {path.name}\n'); result['assembly_sha256']=digest
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps({**source.inputs,**assembly_sources},indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    lines=[f"full_derivative_operator = {result['status']}",f'Global: {json.dumps(rows[0])}',
        f'Endpoint classifications: {json.dumps(root)}',f'Dense local grids: {json.dumps(crossing)}',
        f'Learned addition error on identical branch operands: {result["addition_max_error_on_same_branch_outputs"]:.17g}',
        'All requested diagnostics evaluated as one static audit. No root solver, minimization, training, recursion or search.',
        'Dense local grids use saved stationary locations and fixed subdivisions of teacher sign-change grid cells. They do not refine or solve for a root.',
        'Any sign/zero or endpoint-classification disagreement blocks promotion, even if its amplitude is at branch-level float64 rounding.',
        f'All {len(protected)} prior checkpoint hashes unchanged. Execution: {result["elapsed_seconds"]:.3f} seconds.']
    if failures: lines.append('First sign disagreement: '+json.dumps(failures[0]))
    if endpoint_failures: lines.append('First endpoint change: '+json.dumps(endpoint_failures[0]))
    (OUT/'full_derivative_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],global_metrics=rows[0],root_existence=root,dense_crossings=crossing,
        first_sign_disagreement=result['first_sign_disagreement'],first_endpoint_change=result['first_endpoint_classification_change'],
        addition_error=result['addition_max_error_on_same_branch_outputs'],seconds=result['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':
    run()
