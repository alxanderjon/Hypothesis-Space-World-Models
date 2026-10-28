"""False-positive local teacher-scan audit. No model fitting or prediction."""
import csv
import json
import time
from collections import defaultdict,Counter
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from scipy.optimize import brentq
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_parent_split_selector import safe_json
from experiments.geometry.validate_terminal_candidate_full import original_parents
from experiments.geometry.audit_stationary_candidate_availability import math_roots
from experiments.sho.audit_fixed_allocation_intervals import Audit

ROOT=ref.ROOT;SOURCE=ROOT/'branch_switch_candidate_availability_training'
OUT=ROOT/'branch_availability_false_positive_audit'


def write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def run():
    started=time.monotonic();torch.set_num_threads(1);mp.mp.dps=80
    if OUT.exists():assert not any(OUT.iterdir()), 'Preserve prior completed audit artifacts.'
    else:OUT.mkdir(parents=True)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('manifest','locked','validated'))})
    original=original_parents();oldfresh=json.loads((ROOT/'stationary_terminal_candidate_training/fresh_parents.json').read_text())
    previous=json.loads((ROOT/'stationary_candidate_availability_audit/fresh_parents.json').read_text())
    independent=json.loads((SOURCE/'additional_fresh_parents.json').read_text())
    data=list(csv.DictReader((SOURCE/'availability_predictions.csv').open()))
    details=json.loads((SOURCE/'teacher_label_details.json').read_text())
    features=np.load(SOURCE/'boundary_features.npz')['features']
    a=Audit.__new__(Audit)  # formula only: no allocator/locator/selection calls.
    def parent(r):
        pi=int(r['case'].split(':')[0]);g=r['group']
        return original[pi] if g=='archived_geometry' else oldfresh[str(pi)] if g=='existing_fresh_geometry' else independent[pi] if g=='independent_fresh_geometry' else previous[pi]
    scans={}
    def grid(r,N=129,inset=1,midcell=False):
        l=float(r['lower']);u=float(r['upper']);axis=int(r['case'].split(':')[1]);k=int(r['child_left'])
        a.setup(parent(r),axis);ll=l;rr=u
        for _ in range(inset):ll=np.nextafter(ll,u);rr=np.nextafter(rr,l)
        xx=np.linspace(ll,rr,N)
        if midcell:xx=ll+(np.arange(N)+.5)*(rr-ll)/N
        return xx,a.formula(xx,k)
    def extract(r,index):
        cachekey=(r['group'],r['case']);side=int(r['boundary_side']);b=float(r['boundary'])
        if cachekey not in scans:scans[cachekey]=grid(r)
        xx,f=scans[cachekey];sg=np.sign(f[1]).astype(int);cross=f[1][:-1]*f[1][1:]<0
        # Nominal boundaries are endpoints: there is no actual scan point on
        # the exterior side. Report missing explicitly, not a fabricated sample.
        below=np.flatnonzero(xx<b);above=np.flatnonzero(xx>b)
        ib=int(below[-1]) if len(below) else None;ia=int(above[0]) if len(above) else None
        j=0 if side==0 else len(xx)-2
        qi=f[3][[j,j+1]];ai=f[4][[j,j+1]];powers=[int(f[5][h,f[4][h]]) for h in (j,j+1)]
        qchange=qi[0]!=qi[1];achange=ai[0]!=ai[1];pchange=powers[0]!=powers[1]
        types=[name for name,flag in [('q_max',qchange),('aspect_max',achange),('aspect_power',pchange)] if flag]
        out=dict(state_index=index,group=r['group'],case=r['case'],dimension=int(r['dimension']),budget=int(r['budget']),
            child_counts=json.dumps([int(r['child_left']),int(r['child_right'])]),boundary_side=side,boundary=b,
            lower=float(r['lower']),upper=float(r['upper']),teacher_label=int(r['label']),classifier_prediction=r.get('prediction','not_evaluated'),
            classifier_false_positive=r.get('error')=='True' and r.get('prediction')=='1',old_local_switch_type=r['switch_type'],
            nearest_scan_below=float(xx[ib]) if ib is not None else None,nearest_scan_above=float(xx[ia]) if ia is not None else None,
            distance_to_scan_below=b-xx[ib] if ib is not None else None,distance_to_scan_above=xx[ia]-b if ia is not None else None,
            phase_between_below_above=(b-xx[ib])/(xx[ia]-xx[ib]) if ib is not None and ia is not None else None,
            missing_scan_side='below' if ib is None else 'above' if ia is None else 'neither',
            pair_index=j,pair_left=float(xx[j]),pair_right=float(xx[j+1]),
            signed_phase_in_inward_pair=float((b-xx[j])/(xx[j+1]-xx[j])) if xx[j]!=xx[j+1] else None,
            normalized_nearest_sample_distance=float(min(abs(b-xx[j]),abs(b-xx[j+1]))/(xx[j+1]-xx[j])) if xx[j]!=xx[j+1] else None,
            collapsed_pair=bool(xx[j]==xx[j+1]),
            pair_left_distance=float(abs(b-xx[j])),pair_right_distance=float(abs(b-xx[j+1])),
            q_identity_left=int(qi[0]),q_identity_right=int(qi[1]),aspect_identity_left=int(ai[0]),aspect_identity_right=int(ai[1]),
            aspect_power_left=powers[0],aspect_power_right=powers[1],
            derivative_left=float(f[1][j]),derivative_right=float(f[1][j+1]),sign_left=int(sg[j]),sign_right=int(sg[j+1]),
            pair_strict_sign_transition=bool(cross[j]),pair_q_switch=bool(qchange),pair_aspect_switch=bool(achange),pair_power_switch=bool(pchange),
            pair_switch_count=len(types),pair_switch_types='+'.join(types) or 'none',
            full_scan_transition_count=int(cross.sum()),full_scan_transition_indices=json.dumps(np.flatnonzero(cross).tolist()),
            smooth_fixed_root_present=r.get('smooth_roots_present')=='True')
        for tag,kidx in [('below',ib),('above',ia)]:
            out['q_identity_'+tag]=int(f[3][kidx]) if kidx is not None else None
            out['aspect_identity_'+tag]=int(f[4][kidx]) if kidx is not None else None
            out['derivative_'+tag]=float(f[1][kidx]) if kidx is not None else None
        return out
    states=[extract(r,i) for i,r in enumerate(data)]
    fp=[i for i,r in enumerate(states) if r['classifier_false_positive']]
    tp=np.array([i for i,r in enumerate(states) if r['teacher_label']==1 and r['classifier_prediction']=='1'])
    tn=np.array([i for i,r in enumerate(states) if r['teacher_label']==0 and r['classifier_prediction']=='0'])
    assert len(fp)==120
    # Match on geometry, never on a scan-transition label or root coordinate.
    geom=features[:,:13];scale=np.std(geom,axis=0);scale[scale==0]=1;norm=geom/scale
    def match(i,pool):
        same_type=np.array([states[j]['old_local_switch_type']==states[i]['old_local_switch_type'] for j in pool])
        same_dim=np.array([states[j]['dimension']==states[i]['dimension'] for j in pool])
        same_counts=np.array([states[j]['child_counts']==states[i]['child_counts'] for j in pool])
        mask=same_type&same_dim&same_counts;level='type+dimension+counts'
        if not mask.any():mask=same_type&same_dim;level='type+dimension'
        if not mask.any():mask=same_type;level='type'
        if not mask.any():mask=np.ones(len(pool),bool);level='unrestricted'
        choices=pool[mask];distance=np.sum((norm[choices]-norm[i])**2,axis=1);at=int(np.argmin(distance))
        return int(choices[at]),float(np.sqrt(distance[at])),level
    matches=[];study=set(fp)
    for i in fp:
        t,dt,lt=match(i,tp);n,dn,ln=match(i,tn);study.update((t,n))
        row=dict(false_positive_index=i,positive_index=t,negative_index=n,positive_geometry_distance=dt,negative_geometry_distance=dn,
                 positive_match_level=lt,negative_match_level=ln)
        for prefix,j in [('FP',i),('TP',t),('TN',n)]:row.update({prefix+'_'+k:v for k,v in states[j].items()})
        matches.append(row)
    # Additional never-fit boundary states: perturb the oriented boundary only,
    # using +/-1 and +/-4 ULPs. No candidate-location prediction is performed.
    seeds=[]
    for typ in ('q_max','aspect_max','multiple'):
        ii=[i for i in fp if states[i]['old_local_switch_type']==typ][:8]
        seeds.extend(ii)
        seeds.extend([match(i,tp)[0] for i in ii])
    seeds=sorted(set(seeds));fresh=[];counterfactual=[]
    def label_grid(r,N=129,inset=1,midcell=False):
        xx,f=grid(r,N,inset,midcell);k=int(r['child_left']);side=int(r['boundary_side']);l=float(r['lower']);u=float(r['upper'])
        if xx[0]>=xx[-1]:return 0
        mid=a.formula(np.array([(l+u)/2]),k);iq=int(mid[3][0]);ia=int(mid[4][0]);power=int(mid[5][0,ia])
        coeff=(1/(a.h*np.power(float(k if iq==0 else a.n-k),1/a.d)),a.B,a.a,.05*power,iq,ia)
        cross=np.flatnonzero(f[1][:-1]*f[1][1:]<0);roots=[]
        for j in cross:roots.append(float(brentq(lambda c:float(a.formula(c,k)[1][0]),xx[j],xx[j+1],xtol=5e-324,rtol=1e-15)))
        smooth=[z for z in math_roots(coeff) if mp.mpf(l)<z<mp.mpf(u)] if roots else []
        r['smooth_roots_present']=str(bool(smooth))
        if smooth:return 0
        return int(any((0 if abs(c-l)<=abs(u-c) else 1)==side for c in roots))
    for i in seeds:
        source_row=data[i]
        for shift in (-4,-1,1,4):
            r=dict(source_row);b=float(r['boundary']);b=np.uint64(int(np.float64(b).view(np.uint64))+shift).view(np.float64)
            r['boundary']=str(b);r['lower' if int(r['boundary_side'])==0 else 'upper']=str(b)
            if not 0<float(r['lower'])<float(r['upper'])<1:continue
            # Clear cached identity: interval changed, despite same source case.
            scans.pop((r['group'],r['case']),None)
            label=label_grid(r);r['label']=str(label);r['prediction']='not_evaluated';r['error']='False'
            s=extract(r,f'fresh_{i}_{shift}');s.update(source_state=i,boundary_shift_ULPs=shift,source_label=int(source_row['label']))
            fresh.append(s);scans.pop((r['group'],r['case']),None)
    for i in sorted(study):
        r=data[i];baseline=label_grid(r)
        assert baseline==int(r['label']), 'Original teacher label did not replay.'
        for name,N,inset,mid in [('65_samples',65,1,False),('257_samples',257,1,False),('two_ULP_inset',129,2,False),('half_cell_phase_shift',129,1,True)]:
            y=label_grid(r,N,inset,mid)
            counterfactual.append(dict(state_index=i,group=r['group'],case=r['case'],variation=name,baseline_label=baseline,variation_label=y,changed=baseline!=y))
    def errors(rr,func):
        pred=np.array([func(r) for r in rr]);truth=np.array([r['teacher_label'] for r in rr])
        return dict(cases=len(rr),errors=int(np.sum(pred!=truth)),false_YES=int(np.sum((pred==1)&(truth==0))),false_NO=int(np.sum((pred==0)&(truth==1))))
    rulemetrics={}
    for name,rr in [('all_original',states),('all_false_positives',[states[i] for i in fp]),('matched_study',[states[i] for i in study]),('additional_fresh',fresh)]:
        rulemetrics[name]=dict(sampled_sign_pair=errors(rr,lambda r:r['pair_strict_sign_transition']),
            sampled_sign_and_branch_change=errors(rr,lambda r:r['pair_strict_sign_transition'] and r['pair_switch_count']>0),
            sampled_sign_and_no_smooth_root=errors(rr,lambda r:r['pair_strict_sign_transition'] and not r['smooth_fixed_root_present']))
    # Empirical conditional conflicts, not a learned decision rule/model.
    sets=dict(A_scan_phase=['boundary_side','missing_scan_side','phase_bucket'],B_sampled_signs=['sign_left','sign_right'],
              C_active_identities=['q_identity_left','q_identity_right','aspect_identity_left','aspect_identity_right','aspect_power_left','aspect_power_right'],
              D_switch_pattern=['pair_q_switch','pair_aspect_switch','pair_power_switch'])
    sets['B+C']=sets['B_sampled_signs']+sets['C_active_identities'];sets['A+B+C+D']=sets['A_scan_phase']+sets['B+C']+sets['D_switch_pattern']
    conditional={}
    for name,columns in sets.items():
        groups=defaultdict(list)
        for r in states:
            phase=r['normalized_nearest_sample_distance'];r['phase_bucket']=int(np.floor(np.log2(max(phase,np.finfo(float).tiny)))) if phase is not None else 999
            groups[tuple(r[c] for c in columns)].append(r['teacher_label'])
        mixed=[v for v in groups.values() if len(set(v))>1]
        conditional[name]=dict(groups=len(groups),mixed_label_groups=len(mixed),states_in_mixed_groups=sum(map(len,mixed)),
                               minority_labels_in_mixed_groups=sum(min(sum(v),len(v)-sum(v)) for v in mixed))
    write(OUT/'false_positives_matched.csv',matches)
    write(OUT/'all_scan_states.csv',states);write(OUT/'fresh_boundary_scan_states.csv',fresh)
    write(OUT/'scan_discretization_checks.csv',counterfactual)
    failure_rules=[r for r in states+fresh if bool(r['pair_strict_sign_transition'] and r['pair_switch_count']>0)!=bool(r['teacher_label'])]
    write(OUT/'local_pair_rule_exceptions.csv',failure_rules)
    assert all(ref.sha(Path(k))==v for k,v in protected.items())
    result=dict(status='AUDIT_ONLY_COMPLETE',original_states=len(states),false_positives=len(fp),matched_unique_states=len(study),
        additional_fresh_states=len(fresh),rules=rulemetrics,conditional_conflicts=conditional,
        scan_discretization={name:dict(cases=sum(r['variation']==name for r in counterfactual),changed=sum(r['variation']==name and r['changed'] for r in counterfactual),
             YES_to_NO=sum(r['variation']==name and r['baseline_label']==1 and r['variation_label']==0 for r in counterfactual),
             NO_to_YES=sum(r['variation']==name and r['baseline_label']==0 and r['variation_label']==1 for r in counterfactual)) for name in sorted({r['variation'] for r in counterfactual})},
        false_positive_sign_patterns=dict(Counter(str((states[i]['sign_left'],states[i]['sign_right'])) for i in fp)),
        original_no_scan_point_on_one_side=sum(r['missing_scan_side']!='neither' for r in states),
        elapsed_seconds=time.monotonic()-started,protected_files=len(protected),all_protected_hashes_unchanged=True,
        no_training=True,no_candidate_location_prediction=True,no_recursion=True,no_global_search=True,
        implementation_sha256=ref.sha(Path(__file__)))
    (OUT/'results.json').write_text(safe_json(result));(OUT/'protected_hashes.json').write_text(safe_json(protected))
    print(safe_json(result),flush=True)


if __name__=='__main__':run()
