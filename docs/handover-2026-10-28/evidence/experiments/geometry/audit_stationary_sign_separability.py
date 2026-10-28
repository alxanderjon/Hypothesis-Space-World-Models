"""Read-only reliability-gap separability audit; no threshold fitting or search."""
import ast
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_full_derivative import case_matrix

OUT=ref.ROOT/'stationary_sign_separability_audit'


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ref.ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ref.ROOT.rglob('*manifest.json')})
    manifest=json.loads((ref.OUT/'candidate_manifest.json').read_text());cp=Path(manifest['checkpoint'])
    assert manifest['status']=='FROZEN_CANDIDATE' and ref.sha(cp)==manifest['sha256']
    model=ref.ScalarPower();model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    _,router=ref.routers(model)
    source=ref.Inputs();source.collect();arrays=[];groups=[];pieces={};offset=0;excluded=0
    def add(x,group,cases):
        nonlocal offset,excluded
        x=np.asarray(x,dtype=np.float64).reshape(-1,7)
        names=np.full(len(x),cases,dtype=object) if isinstance(cases,str) else np.asarray(cases,dtype=object)
        opposite=np.sign(x[:,0])*np.where(x[:,5]==0,1,-1)*np.sign(x[:,3])*np.where(x[:,6]==0,1,-1)<0
        excluded+=int((~opposite).sum());x=x[opposite];names=names[opposite]
        if len(x):
            arrays.append(x);groups.append(dict(start=offset,end=offset+len(x),group=group,cases=names));offset+=len(x)
    for b in source.batches:
        add(np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov'])),
            'archived_opposite_sign',b['scope']+':'+b['case'])
        if b['scope']=='archived_subpiece_grids':
            key=ast.literal_eval(b['case']);r=source.piece_map[key]
            pieces[('geometric',b['case'])]=dict(coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),lo=float(r['lower']),hi=float(r['upper']),
                roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots']))
    for key,r in source.generic.items():pieces[('generic_coefficients',key)]=dict(coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),lo=r['lower'],hi=r['upper'],roots=list(r['roots']))
    del source.batches
    def read(path):
        protected[str(path)]=ref.sha(path);return list(csv.DictReader(path.open()))
    reference=read(ref.FAIL/'stationary_midpoint_recovery.csv')
    for r in reference:pieces[(r['scope'],r['case'])]['roots'].append(float(r['c_teacher']))
    for key,p in pieces.items():
        near=[]
        for center in set(p['roots']):
            if p['lo']<=center<=p['hi']:near.extend((np.float64(center).view(np.int64)+np.arange(-64,65,dtype=np.int64)).view(np.float64))
        if near:
            c=np.unique(near);c=c[(c>=p['lo'])&(c<=p['hi'])]
            add(case_matrix(p['coeff'],c),'one_ULP_root_neighborhood',':'.join(key))
    def direct(rows,keys,group):
        if rows:add([[float(r[k]) for k in keys] for r in rows],group,[r.get('scope','')+':'+r.get('case',str(i)) for i,r in enumerate(rows)])
    std=('A','B','a','beta','c','orientation_t','orientation_v')
    for folder,file in [('stationary_sign_assembly','stationary_sign_failures.csv'),('full_derivative_assembly','full_derivative_sign_failures.csv')]:
        rows=read(ref.ROOT/folder/file)
        zero=[r for r in rows if (r.get('routed_sign')=='0' or ('learned_derivative' in r and float(r['learned_derivative'])==0))]
        direct(zero,std,folder+'_previous_false_ZERO')
        direct([r for r in rows if not (r.get('routed_sign')=='0' or ('learned_derivative' in r and float(r['learned_derivative'])==0))],std,folder+'_saved_direction_failures')
    rows=read(ref.FAIL/'direction_and_zero_failures.csv')
    for event,group in [('false_learned_ZERO','original_2908_false_ZERO'),('wrong_direction','original_5330_direction_failures')]:
        direct([r for r in rows if r['event']==event],('A','B','a','beta','q','orientation_t','orientation_v'),group)
    rows=read(ref.OUT/'refined_query_failure_attribution.csv')
    direct([r for r in rows if r['false_ZERO']=='True'],('A','B','a','beta','c','ot','ov'),'refined_2944_false_ZERO')
    direct([r for r in rows if r['false_ZERO']!='True'],('A','B','a','beta','c','ot','ov'),'refined_saved_direction_failures')
    for file,group,coordinate in [('unresolved_zero_neighborhoods.csv','unresolved_509_midpoints','q'),
                                  ('zero_neighborhood_events.csv','previous_ZERO_neighborhood_events','q'),
                                  ('direction_disagreements.csv','ZERO_neighborhood_saved_direction_failures','c')]:
        rows=read(ref.ROOT/'stationary_zero_neighborhood_audit'/file);x=[];names=[];neighbors=[];nn=[]
        for r in rows:
            p=pieces[(r['scope'],r['case'])];x.extend(case_matrix(p['coeff'],[float(r[coordinate])]))
            names.append(r['scope']+':'+r['case'])
            if group=='unresolved_509_midpoints':
                neighbors.extend(case_matrix(p['coeff'],[float(r['q_minus']),float(r['q_plus'])]));nn.extend([names[-1]+':minus',names[-1]+':plus'])
        add(x,group,names)
        if neighbors:add(neighbors,'unresolved_509_neighbors',nn)
    direct(read(ref.ROOT/'stationary_squared_magnitude_audit/new_squared_disagreements.csv'),('A','B','a','beta','c','ot','ov'),'saved_new_squared_failures')
    fixtures=[[1.,0.,1.,-.5*(1 if ot==0 else -1)/(1 if ov==0 else -1),.5,ot,ov] for ot in (0,1) for ov in (0,1)]
    add(fixtures,'exact_stationary_fixtures',[f't={ot},v={ov}' for ot in (0,1) for ov in (0,1)])
    data=np.concatenate(arrays);del arrays;n=len(data)
    print(json.dumps(dict(event='pool_ready',cases=n,excluded_nonopposite=excluded)),flush=True)
    gap=np.empty(n);learned=np.empty(n,np.int8);teacher=np.empty(n,np.int8);terms=np.empty((n,2));fallback=0
    with torch.no_grad():
        for begin in range(0,n,16384):
            sl=slice(begin,min(begin+16384,n));tx=torch.from_numpy(data[sl]);term=router.terms(tx)
            terms[sl]=term.numpy();learned[sl]=router.route_terms(term).numpy()
            mag=abs(terms[sl]);scale=np.max(mag,axis=1)
            # Requested order: subtract actual learned magnitudes, then divide.
            gap[sl]=abs(mag[:,0]-mag[:,1])/scale
            teacher[sl],f=ref.mathematical_sign(data[sl]);fallback+=f
    assert np.isfinite(gap).all() and np.isfinite(terms).all() and np.all((gap>=0)&(gap<=1))
    ambiguous=(learned!=teacher)|(teacher==0);reliable=~ambiguous
    tau=float(gap[ambiguous].max());minrel=float(gap[reliable].min())
    # Deduplicate input tuples to expose the effect of repeated replay occurrences.
    _,unique=np.unique(data,axis=0,return_index=True)
    allids=np.arange(n);distribution=[]
    def stats(ids,group):
        for label,mask in [('RELIABLE',reliable),('AMBIGUOUS',ambiguous)]:
            v=gap[ids[mask[ids]]]
            if not len(v):distribution.append(dict(group=group,label=label,count=0,min_r='',max_r='',median='',p90='',p99='',p999=''));continue
            distribution.append(dict(group=group,label=label,count=len(v),min_r=float(v.min()),max_r=float(v.max()),
                median=float(np.median(v)),p90=float(np.quantile(v,.9)),p99=float(np.quantile(v,.99)),p999=float(np.quantile(v,.999))))
    stats(allids,'all_occurrences');stats(unique,'unique_inputs')
    groupids={}
    for group in sorted({g['group'] for g in groups}):
        ids=np.concatenate([np.arange(g['start'],g['end']) for g in groups if g['group']==group]);groupids[group]=ids;stats(ids,group)
    ref.write_csv(OUT/'sign_reliability_gap_distribution.csv',distribution)
    eps=np.finfo(float).eps
    thresholds=np.unique(np.r_[0.,.25*eps,.5*eps,eps,2*eps,4*eps,8*eps,.5*tau,.75*tau,.9*tau,
                              np.nextafter(tau,0.),tau,np.nextafter(tau,np.inf),1.25*tau,1.5*tau,2*tau])
    checks=[]
    def check(ids,population,value):
        rel=reliable[ids];amb=ambiguous[ids];g=gap[ids];deferred=int(np.sum(rel&(g<=value)))
        checks.append(dict(population=population,tau=float(value),tau_hex=float(value).hex(),
            ambiguous_failures_missed=int(np.sum(amb&(g>value))),reliable_cases_deferred=deferred,
            reliable_fraction_deferred=deferred/int(rel.sum()) if rel.any() else 0.,
            all_cases_deferred=int(np.sum(g<=value)),cases=len(ids)))
    for ids,pop in [(allids,'all_occurrences'),(unique,'unique_inputs')]:
        for value in thresholds:check(ids,pop,value)
    for name,ids in groupids.items():check(ids,name,tau)
    ref.write_csv(OUT/'sign_reliability_threshold_candidates.csv',checks)
    ends=np.array([g['end'] for g in groups]);examples=[]
    # Save every ambiguous occurrence plus 20 lowest-gap reliable examples and
    # 20 reliable examples nearest the proposed empirical boundary.
    ids=np.unique(np.r_[np.flatnonzero(ambiguous),np.flatnonzero(reliable)[np.argsort(gap[reliable])[:20]],
                       np.flatnonzero(reliable)[np.argsort(abs(gap[reliable]-tau))[:20]]])
    for i in ids:
        g=groups[int(np.searchsorted(ends,i,side='right'))];x=data[i]
        examples.append(dict(group=g['group'],case=str(g['cases'][i-g['start']]),A=x[0],B=x[1],a=x[2],beta=x[3],c=x[4],
            ot=int(x[5]),ov=int(x[6]),T1=terms[i,0],T2=terms[i,1],r=gap[i],r_hex=float(gap[i]).hex(),
            learned_sign=int(learned[i]),teacher_sign=int(teacher[i]),label='AMBIGUOUS' if ambiguous[i] else 'RELIABLE',
            reason='teacher_exact_zero' if teacher[i]==0 else 'learned_false_ZERO' if learned[i]==0 else 'wrong_sign' if ambiguous[i] else 'agrees',
            at_ambiguity_boundary=bool(ambiguous[i] and gap[i]==tau)))
    ref.write_csv(OUT/'sign_reliability_cases.csv',examples)
    np.savez_compressed(OUT/'sign_reliability_values.npz',inputs=data,gap=gap,learned_sign=learned,teacher_sign=teacher,ambiguous=ambiguous,unique_indices=unique)
    overlap=np.intersect1d(np.unique(gap[reliable]),np.unique(gap[ambiguous]))
    best=next(r for r in checks if r['population']=='all_occurrences' and r['tau']==tau)
    unique_best=next(r for r in checks if r['population']=='unique_inputs' and r['tau']==tau)
    result=dict(status='AUDIT_ONLY_NO_THRESHOLD_TRAINED',cases=n,unique_inputs=len(unique),excluded_nonopposite=excluded,
        reliable_count=int(reliable.sum()),ambiguous_count=int(ambiguous.sum()),teacher_exact_zero_count=int(np.sum(teacher==0)),
        false_ZERO_count=int(np.sum((teacher!=0)&(learned==0))),strict_sign_errors=int(np.sum(teacher*learned<0)),
        smallest_reliable_r=minrel,largest_ambiguous_r=tau,smallest_empirical_conservative_tau=tau,tau_hex=tau.hex(),tau_in_float64_epsilon=tau/eps,
        distributions_overlap=bool(minrel<=tau),overlap_interval=[minrel,tau] if minrel<=tau else None,overlap_width=max(0.,tau-minrel),
        exact_shared_r_values=len(overlap),reliable_occurrences_at_exact_shared_r=int(np.isin(gap[reliable],overlap).sum()),
        selected_threshold_occurrences=best,selected_threshold_unique_inputs=unique_best,
        next_lower_float_threshold_misses=int(np.sum(ambiguous&(gap>np.nextafter(tau,0.)))),
        nonfinite_gap_or_terms=0,exact_rational_fallback_occurrences=fallback,
        gap_execution='abs(abs(T1)-abs(T2))/max(abs(T1),abs(T2)), using refined-sqrt learned terms. Not abs(normalized comparison margin).',
        label='RELIABLE iff routed sign equals mathematical sign and teacher sign is nonzero; all teacher exact zeros AMBIGUOUS.',
        teacher='Exact rational signs on float64 inputs, with outward-rounded certification away from cancellation.',
        counting='Occurrence pools overlap; unique-input statistics remove duplicate seven-input tuples.',
        scope_limit='Tau is the smallest zero-miss threshold on this audited pool, selected retrospectively. It is not a proof or independent validation for unseen inputs.',
        sqrt_checkpoint=str(cp),sqrt_sha256=manifest['sha256'],implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started)
    assert all(ref.sha(Path(p))==h for p,h in {**protected,**source.inputs}.items())
    result['all_protected_hashes_unchanged']=True
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'sign_reliability_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
