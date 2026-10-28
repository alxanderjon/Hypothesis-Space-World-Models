"""Read-only root/minimum impact audit of the current frozen derivative."""
import ast
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

from experiments.geometry.validate_full_derivative import FullDerivative, Inputs, case_matrix, teacher
from experiments.geometry.validate_sqrt_derivative_branch import SqrtDerivativeBranch, COMPONENTS
from experiments.geometry.validate_log_derivative_branch import LogDerivativeBranch
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.train_subpiece_stationary_solver import score as generic_score

OUT = ROOT / 'stationary_point_impact_audit'


def load_model():
    refs = {}
    branches = {}
    for key, folder, filename, cls, source in (
        ('sqrt', 'sqrt_derivative_branch_assembly', 'sqrt_derivative_branch_validated.json', SqrtDerivativeBranch, 'validate_sqrt_derivative_branch.py'),
        ('log', 'log_derivative_branch_assembly', 'log_derivative_branch_validated.json', LogDerivativeBranch, 'validate_log_derivative_branch.py'),
    ):
        path = ROOT / folder / filename
        manifest = json.loads(path.read_text())
        assert manifest['status'] == 'VALIDATED_ASSEMBLY'
        assert sha(path) == json.loads((path.parent/'results.json').read_text())['assembly_sha256']
        assert sha(Path(__file__).parent/source) == manifest['implementation_sha256']
        models = {}
        for name, ref in manifest['components'].items():
            cp = Path(ref['checkpoint']); assert sha(cp) == ref['sha256']
            m = COMPONENTS[name][1]()
            m.load_state_dict(torch.load(cp, weights_only=False)['state_dict'])
            models[name] = m.eval().requires_grad_(False)
        branches[key] = cls(models).eval().requires_grad_(False)
        refs[key] = dict(manifest=str(path), sha256=sha(path))
    path = ROOT/'addition_operator_smoke/candidate_manifest.json'
    ref = json.loads(path.read_text()); cp = Path(ref['checkpoint'])
    assert sha(cp) == ref['sha256']
    addition = Addition(); addition.load_state_dict(torch.load(cp, weights_only=False)['state_dict'])
    refs['addition'] = ref
    return FullDerivative(branches['sqrt'], branches['log'], addition).eval().requires_grad_(False), refs


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
    summary=dict(status='EVALUATION_ONLY',tested_subpieces=len(rows),expected_subpieces=len(expected),
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
        components=refs,protected_checkpoints=len(protected),elapsed_seconds=time.monotonic()-started)
    summary['all_checkpoint_hashes_unchanged']=all(sha(Path(p))==h for p,h in protected.items())
    assert summary['all_checkpoint_hashes_unchanged']
    meaningful=failures or any(r['outside_equivalence'] or r['identity_changed'] for r in comparisons) or summary['subpiece_minimum_identity_changes']
    summary['decision']='STOP_MEANINGFUL_CHANGE' if meaningful else 'NO_MEANINGFUL_MINIMUM_CHANGE_ON_AUDITED_POOL'
    summary['full_derivative_promoted']=False
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
    (OUT/'stationary_point_impact_summary.txt').write_text(report(summary))
    print(json.dumps(summary,indent=2),flush=True)


def compare_pool(scope,case,candidates):
    t,l=canonical(candidates['teacher']),canonical(candidates['learned'])
    best=min(x['score'] for x in candidates['teacher']); tol=1e-12*max(1.,best)
    return dict(scope=scope,case=case,teacher_axis=t['axis'],learned_axis=l['axis'],teacher_cut=t['c'],learned_cut=l['c'],
        teacher_identity=t['identity'],learned_identity=l['identity'],identity_changed=t['identity']!=l['identity'],
        raw_cut_changed=t['c']!=l['c'] or t['axis']!=l['axis'],teacher_score=t['score'],learned_cut_teacher_score=l['score'],
        score_difference=abs(l['score']-t['score']),outside_equivalence=abs(l['score']-t['score'])>tol)


def report(s):
    return f'''Stationary-point impact audit: {s['decision']}
Tested {s['tested_subpieces']} / {s['expected_subpieces']} bracketed archived subpieces: {s['by_scope']}.
Same safeguarded scipy.optimize.brentq for both derivatives: xtol=5e-324, rtol=1e-15, maxiter=100.
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
No training, recursion, global search or checkpoint promotion. The full derivative still has pointwise sign discrepancies; this audit measures their stationary/minimum impact only.
'''


if __name__=='__main__': run()
