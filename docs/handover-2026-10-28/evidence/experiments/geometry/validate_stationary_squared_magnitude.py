"""Offline squared stationary comparison audit; no root-location procedure."""
import ast
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_full_derivative import case_matrix

OUT=ref.ROOT/'stationary_squared_magnitude_audit'


class SquaredMagnitude(nn.Module):
    def __init__(self,components,margin):
        super().__init__()
        self.components=nn.ModuleDict({k:components[k] for k in ('orientation','multiply','quadratic')})
        self.margin=margin

    def forward(self,x,details=False):
        A,B,a,beta,c,ot,ov=x.unbind(1);m=self.components
        t=m['orientation'](torch.stack((c,ot),1));v=m['orientation'](torch.stack((c,ov),1))
        def mul(a,b):return m['multiply'](torch.stack((a,b),1))
        a2=mul(a,a);q=m['quadratic'](torch.stack((B,a,t),1))
        z1=mul(A,a2);z2=mul(z1,t);z3=mul(z2,v);X=mul(z3,z3)
        beta2=mul(beta,beta);Y=mul(beta2,q);scale=torch.maximum(X,Y)
        result=torch.zeros_like(scale);active=scale>0
        result[active]=self.margin(torch.stack((X[active]/scale[active],Y[active]/scale[active]),1))
        invalid=~torch.isfinite(X)|~torch.isfinite(Y)
        result[invalid]=float('nan')
        if details:return result,torch.stack((a2,q,z1,z2,z3,X,beta2,Y),1)
        return result


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ref.ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ref.ROOT.rglob('*manifest.json')})
    manifest=json.loads((ref.OUT/'candidate_manifest.json').read_text());cp=Path(manifest['checkpoint']);assert ref.sha(cp)==manifest['sha256']
    sqrt=ref.ScalarPower();sqrt.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);sqrt.eval().requires_grad_(False)
    original,previous=ref.routers(sqrt)
    squared=SquaredMagnitude(previous.components,previous.margin).eval().requires_grad_(False)
    assert set(squared.components)=={'orientation','multiply','quadratic'}
    source=ref.Inputs();source.collect();arrays=[];groups=[];offset=0;excluded=0;pieces={}
    def add(x,group,cases):
        nonlocal offset,excluded
        x=np.asarray(x,dtype=np.float64).reshape(-1,7)
        names=np.full(len(x),cases,dtype=object) if isinstance(cases,str) else np.asarray(cases,dtype=object)
        opposite=np.sign(x[:,0])*np.where(x[:,5]==0,1,-1)*np.sign(x[:,3])*np.where(x[:,6]==0,1,-1)<0
        excluded+=int((~opposite).sum());x=x[opposite];names=names[opposite]
        if not len(x):return
        arrays.append(x);groups.append(dict(start=offset,end=offset+len(x),group=group,cases=names));offset+=len(x)
    for b in source.batches:
        x=np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov']))
        add(x,'archived_opposite_sign',b['scope']+':'+b['case'])
        if b['scope']=='archived_subpiece_grids':
            key=ast.literal_eval(b['case']);r=source.piece_map[key]
            pieces[('geometric',b['case'])]=dict(coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),lo=float(r['lower']),hi=float(r['upper']),
                roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots']))
    for key,r in source.generic.items():pieces[('generic_coefficients',key)]=dict(coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),lo=r['lower'],hi=r['upper'],roots=list(r['roots']))
    del source.batches
    reference=list(csv.DictReader((ref.FAIL/'stationary_midpoint_recovery.csv').open()))
    for r in reference:pieces[(r['scope'],r['case'])]['roots'].append(float(r['c_teacher']))
    for key,p in pieces.items():
        neighbors=[]
        for center in set(p['roots']):
            if p['lo']<=center<=p['hi']:
                neighbors.extend((np.float64(center).view(np.int64)+np.arange(-64,65,dtype=np.int64)).view(np.float64))
        if neighbors:
            c=np.unique(neighbors);c=c[(c>=p['lo'])&(c<=p['hi'])]
            add(case_matrix(p['coeff'],c),'one_ULP_root_neighborhood',':'.join(key))
    def read(path):
        protected[str(path)]=ref.sha(path);return list(csv.DictReader(path.open()))
    def from_rows(rows,keys,group,name='case'):
        if rows:add([[float(r[k]) for k in keys] for r in rows],group,[r.get(name,str(i)) for i,r in enumerate(rows)])
    standard=('A','B','a','beta','c','orientation_t','orientation_v')
    r=read(ref.ROOT/'stationary_sign_assembly/stationary_sign_failures.csv')
    from_rows([v for v in r if v['routed_sign']=='0'],'A B a beta c orientation_t orientation_v'.split(),'prior_sign_router_false_ZERO')
    r=read(ref.ROOT/'full_derivative_assembly/full_derivative_sign_failures.csv')
    from_rows([v for v in r if float(v['learned_derivative'])==0],standard,'prior_derivative_false_ZERO')
    r=read(ref.FAIL/'direction_and_zero_failures.csv')
    from_rows([v for v in r if v['event']=='false_learned_ZERO'],('A','B','a','beta','q','orientation_t','orientation_v'),'original_2908_false_ZERO')
    from_rows([v for v in r if v['event']=='wrong_direction'],('A','B','a','beta','q','orientation_t','orientation_v'),'original_5330_wrong_direction')
    r=read(ref.OUT/'refined_query_failure_attribution.csv')
    from_rows([v for v in r if v['false_ZERO']=='True'],('A','B','a','beta','c','ot','ov'),'refined_2944_false_ZERO')
    refined=read(ref.OUT/'refined_sqrt_midpoint_audit.csv')
    # Replay exact saved q coordinates, without performing any bracket iterations.
    for folder,filename,group in [('stationary_zero_neighborhood_audit','unresolved_zero_neighborhoods.csv','unresolved_509_midpoints'),
                                  ('stationary_zero_neighborhood_audit','zero_neighborhood_events.csv','previous_ZERO_neighborhood_events')]:
        r=read(ref.ROOT/folder/filename)
        x=[];names=[];adj=[];adj_names=[]
        for v in r:
            p=pieces[(v['scope'],v['case'])];x.extend(case_matrix(p['coeff'],[float(v['q'])]));names.append(v['scope']+':'+v['case'])
            if group=='unresolved_509_midpoints':
                adj.extend(case_matrix(p['coeff'],[float(v['q_minus']),float(v['q_plus'])]));adj_names.extend([names[-1]+':minus',names[-1]+':plus'])
        add(x,group,names)
        if adj:add(adj,'unresolved_509_neighbors',adj_names)
    fixtures=[]
    for ot in (0,1):
        for ov in (0,1):fixtures.append([1.,0.,1.,-.5*(1 if ot==0 else -1)/(1 if ov==0 else -1),.5,ot,ov])
    add(fixtures,'four_exact_stationary_fixtures',[f't={ot},v={ov}' for ot in (0,1) for ov in (0,1)])
    data=np.concatenate(arrays);del arrays;n=len(data)
    print(json.dumps(dict(event='pool_ready',opposite_sign_occurrences=n,excluded_nonopposite=excluded)),flush=True)
    pred=np.empty(n);prior=np.empty(n,np.int8);older=np.empty(n,np.int8);target=np.empty(n,np.int8)
    intermediate=np.empty((n,8));fallback=0;rawcmp=np.empty(n,np.int8);prior_terms=np.empty((n,2));last=time.monotonic()
    with torch.no_grad():
        for begin in range(0,n,16384):
            end=min(begin+16384,n);sl=slice(begin,end);x=data[sl];tx=torch.from_numpy(x)
            y,detail=squared(tx,details=True);pred[sl]=y.numpy();intermediate[sl]=detail.numpy()
            def previous_cmp(model):
                terms=model.terms(tx);m=abs(terms);scale=torch.maximum(m[:,0],m[:,1]);return torch.sign(model.margin(m/scale[:,None])).numpy(),terms.numpy()
            prior[sl],prior_terms[sl]=previous_cmp(previous);older[sl],_=previous_cmp(original)
            real,nfallback=ref.mathematical_sign(x);fallback+=nfallback
            s1=np.sign(x[:,0])*np.where(x[:,5]==0,1,-1);target[sl]=real*s1
            X,Y=detail[:,5].numpy(),detail[:,7].numpy();rawcmp[sl]=np.where(X>Y,1,np.where(X<Y,-1,0))
            if time.monotonic()-last>20:
                print(json.dumps(dict(event='progress',evaluated=end,seconds=time.monotonic()-start)),flush=True);last=time.monotonic()
    output=np.sign(pred);finite=np.isfinite(pred);names=['a2','q','z1','z2','z3','X','beta2','Y']
    def metric(ids,group):
        p=output[ids];t=target[ids];b=prior[ids];old=older[ids];v=intermediate[ids];bad=p!=t;bb=b!=t
        return dict(group=group,cases=len(ids),teacher_positive=int(np.sum(t>0)),teacher_negative=int(np.sum(t<0)),teacher_zero=int(np.sum(t==0)),
            squared_disagreements=int(bad.sum()),strict_opposite_comparisons=int(np.sum(p*t<0)),
            teacher_zero_learned_nonzero=int(np.sum((t==0)&(p!=0))),teacher_nonzero_learned_zero=int(np.sum((t!=0)&(p==0))),
            refined_sqrt_disagreements=int(bb.sum()),refined_sqrt_teacher_nonzero_learned_zero=int(np.sum((t!=0)&(b==0))),
            fixed_vs_refined_sqrt=int(np.sum(bb&~bad)),introduced_vs_refined_sqrt=int(np.sum(~bb&bad)),both_wrong=int(np.sum(bb&bad)),
            original_sqrt_disagreements=int(np.sum(old!=t)),fixed_vs_original_sqrt=int(np.sum((old!=t)&~bad)),introduced_vs_original_sqrt=int(np.sum((old==t)&bad)),
            normalization_changed_raw_XY_comparison=int(np.sum(p!=rawcmp[ids])),nonfinite_margin=int(np.sum(~finite[ids])),
            overflow_cases=int(np.sum(np.isinf(v).any(1))),underflow_to_zero_cases=int(np.sum((v==0).any(1))),
            subnormal_intermediate_cases=int(np.sum(((abs(v)>0)&(abs(v)<np.finfo(float).tiny)).any(1))))
    rows=[metric(np.arange(n),'all_occurrences')]
    for group in sorted({g['group'] for g in groups}):
        ids=np.concatenate([np.arange(g['start'],g['end']) for g in groups if g['group']==group]);rows.append(metric(ids,group))
    ref.write_csv(OUT/'squared_magnitude_comparison.csv',rows)
    # Store exact inputs and both comparison results for every disagreement in
    # either representation, including new failures and exact fixtures.
    ends=np.array([g['end'] for g in groups]);failures=[];fixtures_out=[]
    for i in np.flatnonzero((output!=target)|(prior!=target)|(target==0)):
        g=groups[int(np.searchsorted(ends,i,side='right'))];x=data[i]
        row=dict(group=g['group'],case=str(g['cases'][i-g['start']]),A=x[0],B=x[1],a=x[2],beta=x[3],c=x[4],c_hex=float(x[4]).hex(),ot=int(x[5]),ov=int(x[6]),
            X=intermediate[i,5],X_hex=float(intermediate[i,5]).hex(),Y=intermediate[i,7],Y_hex=float(intermediate[i,7]).hex(),
            squared_margin=pred[i],mathematical_comparison=int(target[i]),squared_comparison=float(output[i]),refined_sqrt_comparison=int(prior[i]),
            original_sqrt_comparison=int(older[i]),refined_T1=prior_terms[i,0],refined_T2=prior_terms[i,1],
            fixed_vs_refined=bool(prior[i]!=target[i] and output[i]==target[i]),introduced_vs_refined=bool(prior[i]==target[i] and output[i]!=target[i]))
        failures.append(row)
        if g['group']=='four_exact_stationary_fixtures':fixtures_out.append(row)
    ref.write_csv(OUT/'comparison_failure_cases.csv',failures)
    ref.write_csv(OUT/'new_squared_disagreements.csv',[r for r in failures if r['introduced_vs_refined']])
    ref.write_csv(OUT/'exact_stationary_fixtures.csv',fixtures_out)
    ranges=[dict(intermediate=name,min=float(np.min(abs(intermediate[:,j]))),max=float(np.max(abs(intermediate[:,j]))),
        zeros=int(np.sum(intermediate[:,j]==0)),nonfinite=int(np.sum(~np.isfinite(intermediate[:,j]))),subnormals=int(np.sum((abs(intermediate[:,j])>0)&(abs(intermediate[:,j])<np.finfo(float).tiny)))) for j,name in enumerate(names)]
    ref.write_csv(OUT/'numerical_range_audit.csv',ranges)
    overall=rows[0];per={r['group']:r for r in rows}
    # Stop on new ambiguity: exact-zero/root-neighborhood recovery is the goal,
    # not merely a better average while silently introducing wrong comparisons.
    passed=bool(overall['squared_disagreements']<overall['refined_sqrt_disagreements'] and overall['introduced_vs_refined_sqrt']==0
        and overall['overflow_cases']==0 and overall['underflow_to_zero_cases']==0 and overall['nonfinite_margin']==0
        and per['four_exact_stationary_fixtures']['squared_disagreements']==0)
    unchanged=all(ref.sha(Path(p))==h for p,h in {**protected,**source.inputs}.items());assert unchanged
    result=dict(operator='stationary_squared_magnitude_margin',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        metrics=rows,excluded_nonopposite_occurrences=excluded,counts='Occurrences; named replay pools overlap the archive and each other. Same-sign/zero-term inputs excluded.',
        teacher='Outward-rounded rational interval certificate; ambiguous cases use exact Fraction arithmetic on the original float64 coefficients and exact real 1-c.',
        exact_rational_fallback_occurrences=fallback,architecture='Specified learned-multiply square chain + learned quadratic + deterministic normalization + frozen magnitude margin. No sqrt in this representation.',
        prior_baseline='refined sqrt comparator; original unrefined comparator additionally reported',
        no_root_search=True,no_midpoint_iteration=True,no_training=True,protected_hashes_unchanged=unchanged,
        implementation_sha256=ref.sha(Path(__file__)),elapsed_seconds=time.monotonic()-start)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    if passed:(OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2))
    (OUT/'stationary_squared_magnitude_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
