"""Frozen stationary margin: value, bracket and canonical minimum validation only."""
import ast
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

from experiments.geometry.validate_full_derivative import Inputs, case_matrix, teacher, stats, classification
from experiments.geometry.validate_sqrt_derivative_branch import SqrtDerivativeBranch, COMPONENTS
from experiments.geometry.validate_log_derivative_branch import LogDerivativeBranch
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.train_subpiece_stationary_solver import score as generic_score

OUT = ROOT / 'stationary_margin_assembly'


class StationaryMargin(torch.nn.Module):
    """Raw inputs A,B,a,beta,c,o_t,o_v; only the specified frozen primitives."""
    def __init__(self, components):
        super().__init__(); self.components = torch.nn.ModuleDict(components)

    def forward(self, x, details=False):
        A,B,a,beta,c,ot,ov = x.unbind(1); m = self.components
        t = m['orientation'](torch.stack((c,ot),1))
        v = m['orientation'](torch.stack((c,ov),1))
        w = m['orientation'].weights
        tp = w[2]+w[3]*ot; vp = w[2]+w[3]*ov
        q = m['quadratic'](torch.stack((B,a,t),1)); r = m['sqrt'](q)
        def mul(x,y): return m['multiply'](torch.stack((x,y),1))
        a2 = mul(a,a); z1 = mul(A,a2); z2 = mul(z1,t); z3 = mul(z2,tp)
        term1 = mul(z3,v); z4 = mul(beta,vp); term2 = mul(z4,r)
        value = m['addition'](torch.stack((term1,term2),1))
        return torch.stack((value,term1,term2),1) if details else value


def load_model():
    models={}; refs={}
    for name in ('orientation','multiply','quadratic','sqrt','addition'):
        file, cls = COMPONENTS[name] if name!='addition' else ('addition_operator_smoke/candidate_manifest.json',Addition)
        path=ROOT/file; manifest=json.loads(path.read_text()); cp=Path(manifest['checkpoint'])
        assert sha(cp)==manifest['sha256']
        model=cls(); model.load_state_dict(torch.load(cp,weights_only=False)['state_dict'])
        models[name]=model.eval().requires_grad_(False)
        refs[name]=dict(manifest=str(path),checkpoint=str(cp),sha256=sha(cp))
    return StationaryMargin(models).eval().requires_grad_(False), refs


def margin_teacher(x):
    A,B,a,beta,c,ot,ov=x.T
    t=np.where(ot==0,c,1-c); v=np.where(ov==0,c,1-c)
    tp=np.where(ot==0,1.,-1.); vp=np.where(ov==0,1.,-1.)
    r=np.sqrt(B+(a*a)*(t*t))
    term1=((((A*(a*a))*t)*tp)*v); term2=(beta*vp)*r
    return np.column_stack((term1+term2,term1,term2))



def audit_values(source,pieces,model):
    arrays=[]; groups=[]; offset=0
    def add(x,kind,case):
        nonlocal offset
        arrays.append(x); groups.append(dict(start=offset,end=offset+len(x),kind=kind,case=case)); offset+=len(x)
    for b in source.batches:
        x=np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov']))
        add(x,'archive',f"{b['scope']}:{b['case']}")
    for sp in pieces:
        coeff=sp['coeff']; lo,hi=sp['lower'],sp['upper']; name=sp['scope']+':'+sp['case']
        add(case_matrix(coeff,[lo,hi]),'endpoints',name)
        grid=np.unique(np.r_[np.linspace(lo,hi,129),[r for r in sp['roots'] if lo<=r<=hi]])
        x=case_matrix(coeff,grid); td=teacher(x)[:,0]
        add(x,'dense_subpiece',name)
        brackets=np.flatnonzero(np.sign(td[:-1])*np.sign(td[1:])<0)
        centers=np.unique(np.r_[[r for r in sp['roots'] if lo<=r<=hi],grid[td==0]])
        local=[]
        for j in brackets: local.extend(np.linspace(grid[j],grid[j+1],257))
        for center in centers:
            width=max((hi-lo)*1e-6,128*np.spacing(center))
            local.extend(np.linspace(max(lo,center-width),min(hi,center+width),257))
            bits=np.float64(center).view(np.int64)+np.arange(-64,65,dtype=np.int64)
            local.extend(bits.view(np.float64))
        if local:
            local=np.unique(local); local=local[(local>=lo)&(local<=hi)]
            add(case_matrix(coeff,local),'local_crossing',name)
    rng=np.random.default_rng(27492026); synthetic=[]; eps=np.finfo(float).eps
    for _ in range(256):
        A=10**rng.uniform(-8,8); B=10**rng.uniform(-20,12); a=10**rng.uniform(-8,6)
        c=rng.uniform(.001,.999); ot,ov=rng.integers(0,2,2)
        ds=teacher(case_matrix((A,B,a,0.,ot,ov),[c]))[0,1]
        v=c if ov==0 else 1-c; vp=1. if ov==0 else -1.; beta=-(ds*v)/vp
        for delta in (0.,-eps,eps,-8*eps,8*eps,-1e-12,1e-12,-1e-8,1e-8):
            synthetic.append([A,B,a,beta*(1+delta),c,ot,ov])
    add(np.array(synthetic),'synthetic_cancellation','balanced_terms')
    data=np.concatenate(arrays); del arrays
    truth=margin_teacher(data); derivative=teacher(data)
    assert np.all(np.isfinite(truth)) and np.all(np.isfinite(derivative))
    c=data[:,4]; t=np.where(data[:,5]==0,c,1-c); v=np.where(data[:,6]==0,c,1-c)
    r=np.sqrt(data[:,1]+(data[:,2]*data[:,2])*(t*t))
    assert np.all(v>0) and np.all(r>0)
    output=np.empty_like(truth)
    with torch.no_grad():
        for start in range(0,len(data),8192):
            end=min(start+8192,len(data)); output[start:end]=model(torch.from_numpy(data[start:end]),details=True).numpy()
    def metric(y,p):
        m=stats(y,p); m['max_error_relative_to_term_magnitudes']=m.pop('max_branch_normalized_error')
        m['sign_mismatches_outside_term_roundoff']=m.pop('sign_mismatches_outside_branch_roundoff')
        return m
    rows=[dict(group='all',**metric(truth,output))]
    for kind in sorted(set(g['kind'] for g in groups)):
        ids=np.concatenate([np.arange(g['start'],g['end']) for g in groups if g['kind']==kind])
        rows.append(dict(group=kind,**metric(truth[ids],output[ids])))
    scale=abs(truth[:,1])+abs(truth[:,2]); ratio=np.divide(abs(truth[:,0]),scale,out=np.ones(len(data)),where=scale>0)
    opposite=np.sign(truth[:,1])*np.sign(truth[:,2])<0
    for limit in (1e-4,1e-8,1e-12,32*eps):
        mask=opposite&(ratio<=limit)
        if mask.any(): rows.append(dict(group=f'near_cancellation_ratio_le_{limit:.17g}',**metric(truth[mask],output[mask])))
    endpoints=[]
    for g in groups:
        if g['kind']!='endpoints': continue
        sl=slice(g['start'],g['end']); td=derivative[sl,0]; mt=truth[sl,0]; ml=output[sl,0]
        tc,pc=classification(td),classification(ml)
        endpoints.append(dict(case=g['case'],lower=data[g['start'],4],upper=data[g['end']-1,4],
            teacher_derivative_left=td[0],teacher_derivative_right=td[1],teacher_margin_left=mt[0],teacher_margin_right=mt[1],
            learned_margin_left=ml[0],learned_margin_right=ml[1],teacher_class=tc,learned_class=pc,
            classification_changed=tc!=pc,missed_bracket=tc=='sign_change' and pc!='sign_change',
            added_bracket=tc!='sign_change' and pc=='sign_change'))
    dscale=abs(derivative[:,1])+abs(derivative[:,2]); away=abs(derivative[:,0])>32*eps*dscale
    derivative_sign=np.sign(derivative[:,0]); margin_sign=np.sign(output[:,0])
    td_vs_mt=np.sign(derivative[:,0])!=np.sign(truth[:,0]); sign=np.sign(truth[:,0])!=margin_sign
    failing=np.flatnonzero(sign); ends=np.array([g['end'] for g in groups]); details=[]
    errnorm=np.divide(abs(output[:,0]-truth[:,0]),scale,out=np.zeros(len(data)),where=scale>0)
    worst=failing[np.argsort(errnorm[failing])[-20:][::-1]]
    for i in worst:
        g=groups[int(np.searchsorted(ends,i,side='right'))]
        details.append(dict(kind=g['kind'],case=g['case'],A=data[i,0],B=data[i,1],a=data[i,2],beta=data[i,3],c=data[i,4],
            c_hex=float(data[i,4]).hex(),orientation_t=int(data[i,5]),orientation_v=int(data[i,6]),
            teacher_margin=truth[i,0],learned_margin=output[i,0],teacher_derivative=derivative[i,0],
            teacher_term1=truth[i,1],teacher_term2=truth[i,2],learned_term1=output[i,1],learned_term2=output[i,2],
            cancellation_ratio=ratio[i],term_normalized_error=errnorm[i]))
    w=model.components['orientation'].weights.detach().numpy()
    signs=w[2]+w[3]*np.array([0.,1.]); assert np.array_equal(signs,np.array([1.,-1.]))
    result=dict(metrics=rows,endpoint_subpieces=len(endpoints),endpoint_classification_changes=sum(x['classification_changed'] for x in endpoints),
        missed_brackets=sum(x['missed_bracket'] for x in endpoints),added_brackets=sum(x['added_bracket'] for x in endpoints),
        exact_orientation_derivatives=signs.tolist(),valid_v_min=float(v.min()),valid_sqrt_min=float(r.min()),
        learned_margin_vs_teacher_derivative_sign_disagreements=int(np.sum(derivative_sign!=margin_sign)),
        learned_margin_vs_teacher_derivative_sign_disagreements_away_from_cancellation=int(np.sum(away&(derivative_sign!=margin_sign))),
        away_from_cancellation_cases=int(away.sum()),away_definition='abs(S_prime_teacher)>32*eps*(abs(D_sqrt_teacher)+abs(D_log_teacher))',
        teacher_margin_vs_teacher_derivative_sign_disagreements=int(td_vs_mt.sum()),
        teacher_margin_vs_teacher_derivative_disagreements_away_from_cancellation=int(np.sum(away&td_vs_mt)),
        learned_addition_error_on_identical_terms=float(abs(output[:,0]-(output[:,1]+output[:,2])).max()))
    write_csv(OUT/'stationary_margin_values.csv',rows)
    write_csv(OUT/'stationary_margin_endpoints.csv',endpoints)
    write_csv(OUT/'stationary_margin_sign_worst20.csv',details or [dict(status='none')])
    write_csv(OUT/'stationary_margin_bracket_changes.csv',[x for x in endpoints if x['classification_changed']] or [dict(status='none')])
    print(json.dumps(dict(event='values_complete',cases=len(data),endpoint_changes=result['endpoint_classification_changes'],
        sign_disagreements=rows[0]['sign_mismatches'],away_sign_disagreements=result['learned_margin_vs_teacher_derivative_sign_disagreements_away_from_cancellation'])),flush=True)
    return result


def canonical(candidates):
    # Existing canonical order: axis ascending, cut ascending, first within tolerance.
    ordered = sorted(candidates, key=lambda x: (x['axis'], x['c'], x['identity']))
    best = min(x['score'] for x in ordered)
    return next(x for x in ordered if x['score'] <= best + 1e-12*max(1., best))


def run():
    started = time.monotonic(); torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=False)
    protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
    model, refs = load_model()
    source = Inputs(); source.collect()
    pieces = []
    for b in source.batches:
        if b['scope'] != 'archived_subpiece_grids': continue
        key = ast.literal_eval(b['case']); r = source.piece_map[key]
        pieces.append(dict(scope='geometric', case=b['case'], key=key,
            coeff=(b['A'][0], b['B'], b['a'], b['beta'][0], b['o'][0], b['ov'][0]),
            lower=float(r['lower']), upper=float(r['upper']),
            roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots'])))
    for key, r in source.generic.items():
        pieces.append(dict(scope='generic_coefficients', case=key, record=r,
            coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),
            lower=r['lower'], upper=r['upper'], roots=r['roots']))
    value_result = audit_values(source, pieces, model)
    del source.batches
    endpoint_rows = list(csv.DictReader((ROOT/'full_derivative_assembly/subpiece_endpoint_classifications.csv').open()))
    expected = {(r['scope'],r['case']) for r in endpoint_rows if r['teacher_class']=='sign_change'}
    print(json.dumps(dict(event='start',total_subpieces=len(pieces),bracketed_subpieces=len(expected))), flush=True)
    rows=[]; neighborhoods=[]; pools={}; failures=[]; completed=0
    teacher_calls=learned_calls=0
    # Setup/formula are read-only analytic teacher arithmetic; no Audit constructor,
    # allocator calls, new interval discovery, or minimization is executed here.
    geometry = Audit.__new__(Audit)

    with torch.no_grad():
        for sp in pieces:
            coeff=sp['coeff']; lo,hi=sp['lower'],sp['upper']
            if sp['scope']=='geometric':
                parent,axis,k,piece=sp['key']; geometry.setup(source.parents[parent],axis)
                def score(c): return np.asarray(geometry.formula(c,k)[0])
                group=(parent,axis,k)
            else:
                axis=0; group=None
                def score(c): return np.atleast_1d(generic_score(sp['record'],np.asarray(c)))
            def td(c):
                nonlocal teacher_calls
                teacher_calls+=1
                return float(teacher(case_matrix(coeff,[c]))[0,0])
            def ld(c):
                nonlocal learned_calls
                learned_calls+=1
                return float(model(torch.from_numpy(case_matrix(coeff,[c])))[0])
            endpoints=teacher(case_matrix(coeff,[lo,hi]))[:,0]
            bracket=bool(np.sign(endpoints[0])*np.sign(endpoints[1])<0)
            assert bracket == ((sp['scope'],sp['case']) in expected)
            candidates={arm:[] for arm in ('teacher','learned')}
            for j,c in enumerate((lo,hi)):
                for arm in candidates:
                    candidates[arm].append(dict(c=c,score=float(score([c])[0]),axis=axis,identity=f"{sp['case']}:endpoint{j}",kind=f'endpoint{j}'))
            if not bracket:
                # Unchanged archived candidates retain context for interval/parent
                # canonical selection. Only bracketed subpieces are root-solved.
                for j,c in enumerate(sorted(set(sp['roots']))):
                    if not lo<=c<=hi: continue
                    for arm in candidates:
                        candidates[arm].append(dict(c=c,score=float(score([c])[0]),axis=axis,identity=f"{sp['case']}:saved_root{j}",kind='saved_root'))
            else:
                if time.monotonic()-started>480:
                    raise TimeoutError('Audit exceeded eight minutes; no further computation authorized in this script.')
                roots={}; converged={}; calls={}; errors={}
                for arm,fn in (('teacher',td),('learned',ld)):
                    try:
                        root,result=brentq(fn,lo,hi,xtol=5e-324,rtol=1e-15,maxiter=100,full_output=True,disp=False)
                        roots[arm]=float(root); converged[arm]=bool(result.converged); calls[arm]=result.function_calls
                    except (ValueError,RuntimeError) as exc:
                        roots[arm]=None; converged[arm]=False; calls[arm]=0; errors[arm]=str(exc)
                row=dict(scope=sp['scope'],case=sp['case'],lower=lo,upper=hi,
                    A=coeff[0],B=coeff[1],a=coeff[2],beta=coeff[3],orientation_t=int(coeff[4]),orientation_v=int(coeff[5]),
                    teacher_found=converged['teacher'],learned_found=converged['learned'],
                    root_found_mismatch=converged['teacher']!=converged['learned'],
                    teacher_solver_calls=calls['teacher'],learned_solver_calls=calls['learned'],solver_errors=json.dumps(errors))
                if all(converged.values()):
                    ct,cl=roots['teacher'],roots['learned']; st,sl=map(float,score([ct,cl])); tol=1e-12*max(1.,st)
                    row.update(c_teacher=ct,c_learned=cl,root_displacement=abs(cl-ct),
                        root_displacement_ulps=abs(cl-ct)/abs(np.spacing(ct)),
                        teacher_root_score=st,learned_root_teacher_score=sl,score_difference=abs(sl-st),
                        teacher_equivalence_tolerance=tol,score_difference_outside_equivalence=abs(sl-st)>tol)
                    for arm in candidates:
                        candidates[arm].append(dict(c=roots[arm],score=st if arm=='teacher' else sl,axis=axis,
                            identity=f"{sp['case']}:stationary",kind='stationary'))
                    selected={arm:canonical(cs) for arm,cs in candidates.items()}
                    bt,bl=selected['teacher'],selected['learned']; best=min(x['score'] for x in candidates['teacher'])
                    row.update(teacher_minimum_kind=bt['kind'],learned_minimum_kind=bl['kind'],
                        subpiece_minimum_identity_changed=bt['kind']!=bl['kind'],
                        canonical_cut_changed_raw=bt['c']!=bl['c'],
                        minimum_change_outside_equivalence=abs(bt['score']-bl['score'])>1e-12*max(1.,best))
                    # All actual adjacent floats in the union of +/-32 ULP neighborhoods.
                    around=[]
                    for center in (ct,cl):
                        bits=np.float64(center).view(np.int64)+np.arange(-32,33,dtype=np.int64)
                        around.extend(bits.view(np.float64))
                    grid=np.unique(around); grid=grid[(grid>=lo)&(grid<=hi)]
                    x=case_matrix(coeff,grid); yt=teacher(x)[:,0]; yl=model(torch.from_numpy(x)).numpy()
                    def switches(y):
                        signs=np.sign(y); nz=signs[signs!=0]
                        return int(np.sum(nz[1:]!=nz[:-1]))
                    row.update(local_teacher_sign_switches=switches(yt),local_learned_sign_switches=switches(yl),
                        local_teacher_zero_count=int(np.sum(yt==0)),local_learned_zero_count=int(np.sum(yl==0)),
                        local_sign_disagreements=int(np.sum(np.sign(yt)!=np.sign(yl))),
                        local_teacher_multiple_sign_changes=switches(yt)>1,local_learned_multiple_sign_changes=switches(yl)>1)
                    neighborhoods.append(dict(scope=sp['scope'],case=sp['case'],c=grid.tolist(),teacher=yt.tolist(),learned=yl.tolist()))
                rows.append(row); completed+=1
                if row.get('root_found_mismatch') or not all(converged.values()) or row.get('minimum_change_outside_equivalence') or row.get('score_difference_outside_equivalence'):
                    failures.append(row)
                if completed%500==0:
                    write_csv(OUT/'stationary_point_impact.csv',rows)
                    print(json.dumps(dict(event='progress',completed=completed,seconds=time.monotonic()-started,failures=len(failures))),flush=True)
            if group is not None:
                dest=pools.setdefault(group,{arm:[] for arm in candidates})
                for arm in candidates: dest[arm].extend(candidates[arm])
            if failures: break

    comparisons=[]; parent_pools={}
    for group,candidates in pools.items():
        parent,axis,k=group
        for arm in candidates: parent_pools.setdefault(parent,{a:[] for a in candidates})[arm].extend(candidates[arm])
        comparisons.append(compare_pool('allocation_interval',str(group),candidates))
    for parent,candidates in parent_pools.items(): comparisons.append(compare_pool('archived_parent',str(parent),candidates))
    valid=[r for r in rows if 'root_displacement' in r]
    disp=np.array([r['root_displacement'] for r in valid])
    summary=dict(operator='stationary_margin_operator',status='EVALUATION_ONLY',tested_subpieces=len(rows),expected_subpieces=len(expected),
        complete=len(rows)==len(expected),by_scope={s:sum(r['scope']==s for r in rows) for s in ('geometric','generic_coefficients')},
        root_found_mismatches=sum(r['root_found_mismatch'] for r in rows),
        teacher_not_found=sum(not r['teacher_found'] for r in rows),learned_not_found=sum(not r['learned_found'] for r in rows),
        root_displacement_max=float(disp.max()),root_displacement_median=float(np.median(disp)),root_displacement_p99=float(np.quantile(disp,.99)),
        max_root_displacement_ulps=max(r['root_displacement_ulps'] for r in valid),
        max_score_difference=max(r['score_difference'] for r in valid),
        stationary_scores_outside_equivalence=sum(r['score_difference_outside_equivalence'] for r in valid),
        subpiece_minimum_identity_changes=sum(r['subpiece_minimum_identity_changed'] for r in valid),
        subpiece_minimum_changes_outside_equivalence=sum(r['minimum_change_outside_equivalence'] for r in valid),
        subpiece_raw_canonical_coordinate_changes=sum(r['canonical_cut_changed_raw'] for r in valid),
        teacher_multiple_local_sign_changes=sum(r['local_teacher_multiple_sign_changes'] for r in valid),
        learned_multiple_local_sign_changes=sum(r['local_learned_multiple_sign_changes'] for r in valid),
        teacher_function_calls=teacher_calls,learned_function_calls=learned_calls,
        canonical_pool_comparisons={scope:dict(pools=sum(r['scope']==scope for r in comparisons),
            identity_changes=sum(r['identity_changed'] for r in comparisons if r['scope']==scope),
            raw_coordinate_changes=sum(r['raw_cut_changed'] for r in comparisons if r['scope']==scope),
            outside_equivalence=sum(r['outside_equivalence'] for r in comparisons if r['scope']==scope))
            for scope in ('allocation_interval','archived_parent')},
        solver=dict(name='scipy.optimize.brentq',xtol=5e-324,rtol=1e-15,maxiter=100),
        canonical_semantics='Existing score tolerance then axis/cut ordering. Root identity is its subpiece/stationary slot; actual cut coordinate differences are reported separately, not rounded or snapped.',
        geometric_score='Original archived Audit.formula full max-over-children score; fixed integer counts. Generic score uses original train_subpiece_stationary_solver.score.',
        global_pool_scope='All archived geometric subpieces, retaining saved stationary candidates on unbracketed pieces; no new partitions or allocation discovery.',
        value_audit=value_result,components=refs,protected_checkpoints=len(protected),elapsed_seconds=time.monotonic()-started)
    summary['all_checkpoint_hashes_unchanged']=all(sha(Path(p))==h for p,h in protected.items())
    assert summary['all_checkpoint_hashes_unchanged']
    meaningful=failures or any(r['outside_equivalence'] or r['identity_changed'] for r in comparisons) or summary['subpiece_minimum_identity_changes']
    summary['decision']='STOP_MEANINGFUL_CHANGE' if meaningful else 'NO_MEANINGFUL_MINIMUM_CHANGE_ON_AUDITED_POOL'
    passed = bool(summary['complete'] and not meaningful and summary['root_found_mismatches']==0
        and summary['teacher_not_found']==0 and summary['learned_not_found']==0
        and value_result['endpoint_classification_changes']==0 and value_result['metrics'][0]['nonfinite_outputs']==0)
    summary['gate_passed']=passed
    summary['status']='VALIDATED_UNDER_TEACHER_EQUIVALENCE' if passed else 'DIAGNOSTIC_ONLY_STOP'
    assert all(sha(Path(p))==h for p,h in source.inputs.items())
    (OUT/'input_archive_hashes.json').write_text(json.dumps(source.inputs,indent=2))
    if passed:
        manifest=dict(operator='stationary_margin_operator',status=summary['status'],components=refs,
            module='experiments.geometry.validate_stationary_margin',model='StationaryMargin',
            inputs=['A','B','a','beta','c','orientation_t','orientation_v'],
            implementation_sha256=sha(Path(__file__)),validation=str(OUT/'results.json'),
            scope='Audited positive-v, positive-sqrt domain; teacher-equivalent root/minimum recovery. Pointwise cancellation sign mismatches are reported, not repaired.',
            no_training=True)
        path=OUT/'stationary_margin_validated.json'; path.write_text(json.dumps(manifest,indent=2))
        digest=sha(path); path.with_suffix('.sha256').write_text(f'{digest}  {path.name}\n')
        summary['assembly_sha256']=digest
    write_csv(OUT/'stationary_point_impact.csv',rows)
    write_csv(OUT/'canonical_minimum_comparison.csv',comparisons)
    worst=sorted(valid,key=lambda r:(r['score_difference'],r['root_displacement']),reverse=True)[:20]
    near_map={(r['scope'],r['case']):r for r in neighborhoods}
    worst=[dict(r,derivative_neighborhood=json.dumps(near_map[(r['scope'],r['case'])])) for r in worst]
    write_csv(OUT/'stationary_point_worst20.csv',worst)
    biggest=sorted(valid,key=lambda r:r['root_displacement'],reverse=True)[:20]
    write_csv(OUT/'root_displacement_worst20.csv',biggest)
    ids={(r['scope'],r['case']) for r in worst+biggest}
    (OUT/'worst_root_neighborhoods.json').write_text(json.dumps([r for r in neighborhoods if (r['scope'],r['case']) in ids],indent=2))
    write_csv(OUT/'multiple_local_sign_changes.csv',[r for r in valid if r['local_teacher_multiple_sign_changes'] or r['local_learned_multiple_sign_changes']] or [dict(status='none')])
    if failures: write_csv(OUT/'failures.csv',failures)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'stationary_margin_summary.txt').write_text(report(summary))
    print(json.dumps(summary,indent=2),flush=True)


def compare_pool(scope,case,candidates):
    t,l=canonical(candidates['teacher']),canonical(candidates['learned'])
    best=min(x['score'] for x in candidates['teacher']); tol=1e-12*max(1.,best)
    return dict(scope=scope,case=case,teacher_axis=t['axis'],learned_axis=l['axis'],teacher_cut=t['c'],learned_cut=l['c'],
        teacher_identity=t['identity'],learned_identity=l['identity'],identity_changed=t['identity']!=l['identity'],
        raw_cut_changed=t['c']!=l['c'] or t['axis']!=l['axis'],teacher_score=t['score'],learned_cut_teacher_score=l['score'],
        score_difference=abs(l['score']-t['score']),outside_equivalence=abs(l['score']-t['score'])>tol)


def report(s):
    return f'''Stationary margin: {s['status']}\nValue and endpoint audit: {json.dumps(s['value_audit'])}
Tested {s['tested_subpieces']} / {s['expected_subpieces']} bracketed archived subpieces: {s['by_scope']}.
Same safeguarded scipy.optimize.brentq for teacher derivative and learned margin: xtol=5e-324, rtol=1e-15, maxiter=100.
Root found/not-found mismatches: {s['root_found_mismatches']}.
Teacher / learned solver failures: {s['teacher_not_found']} / {s['learned_not_found']}.
Root displacement: median {s['root_displacement_median']:.17g}, p99 {s['root_displacement_p99']:.17g}, max {s['root_displacement_max']:.17g}.
Maximum root displacement in local cut ULPs: {s['max_root_displacement_ulps']}. Absolute displacement is the relevant scale here; small cuts have much finer ULPs.
Maximum original teacher score difference at roots: {s['max_score_difference']:.17g}.
Stationary scores outside 1e-12*max(1,teacher score): {s['stationary_scores_outside_equivalence']}.
Subpiece minimum identity changes: {s['subpiece_minimum_identity_changes']}.
Subpiece minimum changes outside equivalence: {s['subpiece_minimum_changes_outside_equivalence']}.
Raw subpiece canonical cut-coordinate changes: {s['subpiece_raw_canonical_coordinate_changes']}; candidate identity and coordinate equality are reported separately.
Canonical allocation-interval and parent comparisons: {json.dumps(s['canonical_pool_comparisons'])}.
Multiple local sign changes within +/-32 adjacent floats around recovered roots: teacher {s['teacher_multiple_local_sign_changes']}, learned {s['learned_multiple_local_sign_changes']}.
The existing solver's selected root is retained; no sign smoothing, snapping or repair.
{s['global_pool_scope']}
{s['geometric_score']}
All {s['protected_checkpoints']} prior checkpoint hashes unchanged: {s['all_checkpoint_hashes_unchanged']}.
Evaluation wall time: {s['elapsed_seconds']:.3f} seconds.
No training, recursion, global search or threshold model. Underlying checkpoints remain unchanged. Pointwise cancellation sign fidelity is distinct from teacher-equivalent minimum recovery.
'''


if __name__=='__main__': run()
