"""Saved-state ULP offset and local-state separability audit; no fitting/recursion."""
import csv
import json
import time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.sho.audit_fixed_allocation_intervals import Audit

OUT=ROOT/'brent_branch_switch_offset_audit'
BASE=ROOT/'branch_boundary_candidate_baseline_validation'


def read(p):return list(csv.DictReader(p.open()))
def bits(x):
    # All cut coordinates are strictly positive. IEEE uint64 is monotone here.
    assert 0<float(x)<1
    return int(np.float64(x).view(np.uint64))


def saved_geometries():
    plan=json.loads((BASE/'planned_cases.json').read_text());out=dict(plan['parents']);recorded={}
    for name,root in plan['recursive']:
        recorded[(name,'r','teacher')]=root;recorded[(name,'r','baseline')]=root
    # Unroll saved transitions only. No selector, allocation, root solver or new
    # split decision is called; this restores metadata omitted from the CSV.
    for row in read(BASE/'recursive_fidelity.csv'):
        name,path=row['label'].split(':',1)
        for arm,prefix in [('teacher','teacher'),('baseline','learned')]:
            p=recorded[(name,path,arm)]
            out[row['label']+('' if arm=='teacher' else ':baseline_geometry')]=p
            axis=int(row[prefix+'_axis']);cut=float(row[prefix+'_cut']);counts=json.loads(row[prefix+'_counts'])
            lo=np.array(p['lower']);hi=np.array(p['upper']);coord=lo[axis]+cut*(hi[axis]-lo[axis])
            for j in (0,1):
                lower=lo.copy();upper=hi.copy()
                if j==0:upper[axis]=coord
                else:lower[axis]=coord
                recorded[(name,path+str(j),arm)]=dict(d=p['d'],n=counts[j],lower=lower.tolist(),upper=upper.tolist())
    failure=json.loads((BASE/'first_failure.json').read_text())
    for arm,key in [('teacher','teacher_parent'),('baseline','baseline_parent')]:
        name=failure['context']['label'];path=failure['context']['path'];label=name+':'+path+('' if arm=='teacher' else ':baseline_geometry')
        assert out[label]==failure['context'][key]
    return out


def run():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('manifest','locked','validated'))})
    source=ROOT/'branch_switch_location_residual_training/location_cases.json'
    protected[str(source)]=sha(source)
    for p in (BASE/'branch_candidate_coordinates.csv',BASE/'first_failure.json',BASE/'recursive_fidelity.csv',BASE/'planned_cases.json'):protected[str(p)]=sha(p)
    saved=json.loads(source.read_text());assert len(saved)==550
    raw=[]
    for i,r in enumerate(saved):
        raw.append(dict(pool='saved_522' if i<522 else 'fresh_28',label=r['group']+':'+r['case'],parent=r['parent'],axis=r['axis'],count_left=r['k'],side=r['side'],
            scan_left=r['c_left'],scan_right=r['c_right'],boundary=r['boundary'],teacher_cut=r['teacher_cut'],
            margin_left=r['margin_1'],margin_right=r['margin_2']))
    geometry=saved_geometries()
    for r in read(BASE/'branch_candidate_coordinates.csv'):
        assert r['teacher_identity_exists']=='True'
        axis,k,_=map(int,r['piece'].split(':'))
        raw.append(dict(pool='end_to_end_calls',label=r['label']+':'+r['piece'],call_label=r['label'],piece=r['piece'],parent=geometry[r['label']],
            axis=axis,count_left=k,side=int(r['side']),scan_left=float(r['scan_left']),scan_right=float(r['scan_right']),boundary=float(r['boundary']),
            teacher_cut=float(r['teacher_cut']),margin_left=float(r['margin_left']),margin_right=float(r['margin_right'])))
    a=Audit.__new__(Audit);rows=[];seen={};unique=[];target_conflicts=[]
    for i,r in enumerate(raw):
        l,u=r['scan_left'],r['scan_right'];b=float(np.clip(r['boundary'],l,u));c=r['teacher_cut'];h=u-l
        assert l<=c<=u and h>0
        a.setup(r['parent'],r['axis']);f=a.formula(np.array([l,u]),r['count_left'])
        q=f[3];aspect=f[4];power=f[5][np.arange(2),aspect]
        qc=int(q[0]!=q[1]);ac=int(aspect[0]!=aspect[1]);pc=int(power[0]!=power[1])
        typ='simultaneous' if qc+ac+pc>1 else 'q_only' if qc else 'aspect_only' if ac or pc else 'none'
        assert typ!='none'
        k=bits(c)-bits(b);bracket_ulps=bits(u)-bits(l)
        key=(r['parent']['d'],r['parent']['n'],tuple(r['parent']['lower']),tuple(r['parent']['upper']),r['axis'],r['count_left'],r['side'],l,u,r['boundary'])
        uid=seen.get(key)
        if uid is None:uid=len(unique);seen[key]=uid
        elif unique[uid]['signed_k']!=k:target_conflicts.append(dict(unique_id=uid,first_k=unique[uid]['signed_k'],second_k=k))
        row=dict(index=i,unique_id=uid,pool=r['pool'],label=r['label'],call_label=r.get('call_label',''),piece=r.get('piece',''),
            dimension=r['parent']['d'],budget=r['parent']['n'],axis=r['axis'],count_left=r['count_left'],count_right=r['parent']['n']-r['count_left'],
            side=r['side'],scan_left=l,scan_right=u,boundary=r['boundary'],projected_boundary=b,teacher_cut=c,
            projected_hex=b.hex(),teacher_hex=float(c).hex(),signed_k=k,abs_k=abs(k),switch_type=typ,
            q_change=qc,aspect_change=ac,power_change=pc,q_left=int(q[0]),q_right=int(q[1]),aspect_left=int(aspect[0]),aspect_right=int(aspect[1]),
            power_left=int(power[0]),power_right=int(power[1]),margin_left=r['margin_left'],margin_right=r['margin_right'],
            teacher_derivative_left=float(f[1][0]),teacher_derivative_right=float(f[1][1]),
            normalized_boundary_position=(r['boundary']-l)/h,normalized_projected_position=(b-l)/h,
            scan_spacing=h,scan_exponent=int(np.frexp(h)[1]),local_exponent=int(np.frexp(b)[1]),bracket_float_steps=bracket_ulps,
            spacing_in_local_ULPs=h/np.spacing(b),normalized_B=a.B/a.a**2,normalized_h=a.h/a.a)
        derivative_scale=max(abs(row['teacher_derivative_left']),abs(row['teacher_derivative_right']))
        row['joint_derivative_left']=row['teacher_derivative_left']/derivative_scale
        row['joint_derivative_right']=row['teacher_derivative_right']/derivative_scale
        rows.append(row)
        if uid==len(unique):unique.append(row)
    assert not target_conflicts
    metrics=[];distribution=[];coverage=[]
    def report(group,rr):
        kk=np.array([r['signed_k'] for r in rr]);ab=abs(kk)
        metrics.append(dict(group=group,cases=len(rr),exact_boundary=int(np.sum(kk==0)),exact_fraction=float(np.mean(kk==0)),
            negative=int(np.sum(kk<0)),positive=int(np.sum(kk>0)),minimum_k=int(kk.min()),maximum_k=int(kk.max()),
            median_abs_k=float(np.median(ab)),p99_abs_k=float(np.quantile(ab,.99)),max_abs_k=int(ab.max()),distinct_k=len(set(kk))))
        for k,n in sorted(Counter(kk).items()):distribution.append(dict(group=group,signed_k=int(k),cases=n,fraction=n/len(kk)))
        for K in (0,1,2,4,8,16,32,64,128,256,341,512):coverage.append(dict(group=group,K=K,covered=int(np.sum(ab<=K)),missed=int(np.sum(ab>K)),coverage=float(np.mean(ab<=K))))
    report('all_occurrences',rows);report('unique_geometry_states',unique)
    for pool in sorted({r['pool'] for r in rows}):report(pool,[r for r in rows if r['pool']==pool])
    for typ in sorted({r['switch_type'] for r in unique}):report('unique:'+typ,[r for r in unique if r['switch_type']==typ])
    # Empirical grouping only: no predictive model or classifier is fitted.
    feature_sets={
        'type':['switch_type'],
        'margins_type':['margin_left','margin_right','switch_type'],
        'local_minimal':['margin_left','margin_right','normalized_boundary_position','switch_type'],
        'local_plus_spacing_exponents':['margin_left','margin_right','normalized_boundary_position','switch_type','scan_spacing','local_exponent','scan_exponent'],
        'local_plus_discrete_spacing':['margin_left','margin_right','normalized_boundary_position','switch_type','bracket_float_steps','local_exponent'],
        'local_plus_branch_ids':['margin_left','margin_right','normalized_boundary_position','switch_type','scan_spacing','local_exponent','q_left','q_right','aspect_left','aspect_right','power_left','power_right'],
        'local_plus_geometry':['margin_left','margin_right','normalized_boundary_position','switch_type','scan_spacing','local_exponent','normalized_B','normalized_h','count_left','count_right','dimension']}
    conflicts=[];separability=[]
    for name,fields in feature_sets.items():
        for digits in (None,6,12,15):
            groups=defaultdict(list)
            for r in unique:
                key=tuple(round(r[f],digits) if digits is not None and isinstance(r[f],float) else r[f] for f in fields)
                groups[key].append(r)
            mixed=[g for g in groups.values() if len({r['signed_k'] for r in g})>1]
            separability.append(dict(features=name,rounding='exact_float64' if digits is None else str(digits)+'_decimal',groups=len(groups),
                singleton_groups=sum(len(g)==1 for g in groups.values()),repeated_groups=sum(len(g)>1 for g in groups.values()),
                conflicting_groups=len(mixed),states_in_conflicts=sum(map(len,mixed)),
                minimum_unavoidable_errors=sum(len(g)-max(Counter(r['signed_k'] for r in g).values()) for g in mixed),
                maximum_k_range=max((max(r['signed_k'] for r in g)-min(r['signed_k'] for r in g) for g in mixed),default=0)))
            if digits is None:
                for gi,g in enumerate(mixed):
                    for r in g:conflicts.append(dict(feature_set=name,conflict_group=gi,**r))
    # Evaluate support extrapolation without fitting anything: offsets observed
    # on earlier 522 cases versus later fresh and recorded recursive states.
    old_support=sorted({r['signed_k'] for r in rows if r['pool']=='saved_522'})
    support=[]
    for pool in ('fresh_28','end_to_end_calls'):
        rr=[r for r in rows if r['pool']==pool];missing=[r for r in rr if r['signed_k'] not in old_support]
        support.append(dict(group=pool,cases=len(rr),outside_saved_offset_set=len(missing),new_offsets=sorted({r['signed_k'] for r in missing})))
    failure=[r for r in rows if r['call_label'] in ('recursive_d4_n16:r10','recursive_d4_n16:r100:baseline_geometry') and
             r['piece']==('2:2:1' if r['call_label'].endswith(':r10') else '1:1:1')]
    assert len(failure)==2
    write_csv(OUT/'brent_offsets_all.csv',rows);write_csv(OUT/'brent_offsets_unique.csv',unique)
    write_csv(OUT/'offset_metrics.csv',metrics);write_csv(OUT/'signed_offset_distribution.csv',distribution)
    write_csv(OUT/'neighborhood_coverage.csv',coverage);write_csv(OUT/'feature_separability.csv',separability)
    write_csv(OUT/'exact_feature_conflicts.csv',conflicts or [dict(status='none')]);write_csv(OUT/'recursive_failure_local_state.csv',failure)
    # Additional information-loss checks on the same saved sample values.
    # Still grouping only: no model fit and no intermediate derivative queries.
    extra_sets={
        'joint_scaled_derivatives_plus_local':['normalized_boundary_position','switch_type','scan_spacing','local_exponent','joint_derivative_left','joint_derivative_right'],
        'raw_derivatives_plus_local':['normalized_boundary_position','switch_type','scan_spacing','local_exponent','teacher_derivative_left','teacher_derivative_right'],
        'raw_derivatives_exact_scan_coordinates_boundary_branch_ids':['teacher_derivative_left','teacher_derivative_right','scan_left','scan_right','boundary','switch_type','q_left','q_right','aspect_left','aspect_right','power_left','power_right']}
    extra_reports=[]
    for name,fields in extra_sets.items():
        groups=defaultdict(list)
        for r in unique:groups[tuple(r[f] for f in fields)].append(r)
        mixed=[g for g in groups.values() if len({r['signed_k'] for r in g})>1]
        extra_reports.append(dict(features=name,groups=len(groups),singleton_groups=sum(len(g)==1 for g in groups.values()),
            conflicting_groups=len(mixed),states_in_conflicts=sum(map(len,mixed)),
            minimum_unavoidable_errors=sum(len(g)-max(Counter(r['signed_k'] for r in g).values()) for g in mixed)))
        if name=='raw_derivatives_exact_scan_coordinates_boundary_branch_ids':
            write_csv(OUT/'full_endpoint_state_conflicts.csv',[dict(conflict_group=i,**r) for i,g in enumerate(mixed) for r in g] or [dict(status='none')])
    write_csv(OUT/'derivative_information_check.csv',extra_reports)
    json_write(OUT/'offset_support.json',dict(saved_522_offsets=old_support,all_unique_offsets=sorted({r['signed_k'] for r in unique}),support_checks=support))
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='AUDIT_ONLY_COMPLETE',occurrences=len(rows),unique_states=len(unique),source_target_conflicts=target_conflicts,
        metrics=metrics,exact_feature_separability=[r for r in separability if r['rounding']=='exact_float64'],support_checks=support,
        recursive_failure=failure,extra_endpoint_information=extra_reports,no_training=True,no_root_solving=True,no_recursion=True,protected_files=len(protected),all_protected_hashes_unchanged=True,
        elapsed_seconds=time.monotonic()-start,implementation_sha256=sha(Path(__file__)))
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
