"""Sign routing only. No training, root location, recursion, or search."""
import ast
import csv
import json
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.validate_full_derivative import Inputs,case_matrix,teacher
from experiments.geometry.validate_stationary_margin import load_model as load_terms,margin_teacher
from experiments.geometry.train_stationary_magnitude_margin import MagnitudeMargin
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv

OUT=ROOT/'stationary_sign_assembly'
EPS=np.finfo(float).eps


class StationarySign(nn.Module):
    def __init__(self,components,margin,threshold):
        super().__init__(); self.components=nn.ModuleDict({k:components[k] for k in ('orientation','multiply','quadratic','sqrt')})
        self.margin=margin; self.threshold=threshold

    def terms(self,x):
        A,B,a,beta,c,ot,ov=x.unbind(1); m=self.components
        t=m['orientation'](torch.stack((c,ot),1)); v=m['orientation'](torch.stack((c,ov),1))
        w=m['orientation'].weights; tp=w[2]+w[3]*ot; vp=w[2]+w[3]*ov
        r=m['sqrt'](m['quadratic'](torch.stack((B,a,t),1)))
        def mul(x,y):return m['multiply'](torch.stack((x,y),1))
        a2=mul(a,a); z1=mul(A,a2); z2=mul(z1,t); z3=mul(z2,tp)
        return torch.stack((mul(z3,v),mul(mul(beta,vp),r)),1)

    def route_terms(self,terms):
        a,b=terms.unbind(1); sa,sb=torch.sign(a),torch.sign(b)
        opposite=(sa*sb)<0
        out=torch.where(a==0,sb,sa)
        # On equal-sign inputs or a single zero the above is already the sign.
        if torch.any(opposite):
            mag=torch.abs(terms[opposite]); scale=torch.maximum(mag[:,0],mag[:,1])
            cmp=self.margin(mag/scale[:,None]); decision=self.threshold.decision(cmp)
            out[opposite]=torch.where(decision>0,sa[opposite],torch.where(decision<0,sb[opposite],torch.zeros_like(decision)))
        return out

    def forward(self,x):return self.route_terms(self.terms(x))


def term_teacher(terms):
    a,b=terms.T; sa,sb=np.sign(a),np.sign(b); out=np.where(a==0,sb,sa)
    opposite=sa*sb<0
    out[opposite]=np.where(abs(a[opposite])>abs(b[opposite]),sa[opposite],
        np.where(abs(a[opposite])<abs(b[opposite]),sb[opposite],0))
    return out.astype(np.int8)


@lru_cache(maxsize=32768)
def rational_coeff(A,B,a,beta):
    return tuple(Fraction.from_float(float(v)) for v in (A,B,a,beta))


@lru_cache(maxsize=700000)
def exact_sign(row):
    A,B,a,beta,c,ot,ov=row; A,B,a,beta=rational_coeff(A,B,a,beta); c=Fraction.from_float(c)
    t=c if ot==0 else 1-c; v=c if ov==0 else 1-c
    s1=(1 if ot==0 else -1)*(1 if A>0 else -1 if A<0 else 0)
    s2=(1 if ov==0 else -1)*(1 if beta>0 else -1 if beta<0 else 0)
    if s1==0:return s2
    if s2==0 or s1==s2:return s1
    # Positive terms: squared comparison removes sqrt without approximation.
    diff=(A*a*a*t*v)**2-beta*beta*(B+a*a*t*t)
    return s1*(1 if diff>0 else -1 if diff<0 else 0)


def mathematical_sign(x):
    """Outward-rounded interval certificate; exact rational fallback if ambiguous.

    Treat input float64 coefficients/c as exact reals, including real 1-c.
    Positive square-root sign reduces exactly to comparing two rational squares.
    This teacher runs only in validation, never in the learned router.
    """
    A,B,a,beta,c,ot,ov=x.T
    assert np.all(B>=0) and np.all(a>0) and np.all((c>0)&(c<1))
    def down(v):return np.maximum(0.,np.nextafter(v,-np.inf))
    def up(v):return np.nextafter(v,np.inf)
    def mul(u,v):return down(u[0]*v[0]),up(u[1]*v[1])
    def add(u,v):return down(u[0]+v[0]),up(u[1]+v[1])
    def point(v):return v,v
    rev=(down(1-c),up(1-c))
    t=(np.where(ot==0,c,rev[0]),np.where(ot==0,c,rev[1]))
    v=(np.where(ov==0,c,rev[0]),np.where(ov==0,c,rev[1]))
    a2=mul(point(a),point(a)); q=add(point(B),mul(a2,mul(t,t)))
    left=mul(mul(mul(point(abs(A)),a2),t),v); left2=mul(left,left)
    right2=mul(mul(point(abs(beta)),point(abs(beta))),q)
    s1=np.sign(A)*np.where(ot==0,1,-1); s2=np.sign(beta)*np.where(ov==0,1,-1)
    result=np.where(s1==0,s2,s1).astype(np.int8); opposite=s1*s2<0
    larger=left2[0]>right2[1]; smaller=left2[1]<right2[0]
    result[opposite&smaller]=s2[opposite&smaller]
    ambiguous=np.flatnonzero(opposite&~(larger|smaller))
    for i in ambiguous:result[i]=exact_sign(tuple(float(v) for v in x[i]))
    return result,len(ambiguous)


def endpoints_class(signs):
    a,b=map(int,signs)
    return 'endpoint_zero' if a==0 or b==0 else f"({'+' if a>0 else '-'},{'+' if b>0 else '-'})"


def run():
    start=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}; references={}
    path=ROOT/'stationary_margin_assembly/stationary_margin_validated.json'; sm=json.loads(path.read_text())
    assert sm['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE'
    assert sha(Path('experiments/geometry/validate_stationary_margin.py'))==sm['implementation_sha256']
    reference,refs=load_terms();references['term_assembly']=dict(manifest=str(path),sha256=sha(path))
    protected[str(path)]=sha(path)
    models=[]
    for name,folder,cls in [('magnitude_margin','stationary_magnitude_margin_smoke',MagnitudeMargin),
        ('threshold','stationary_magnitude_threshold_smoke/zero_anchor_converged',MagnitudeThreshold)]:
        p=ROOT/folder/'candidate_manifest.json';r=json.loads(p.read_text());assert r['status']=='FROZEN_CANDIDATE'
        cp=Path(r['checkpoint']);assert sha(cp)==r['sha256']
        model=cls();model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);models.append(model.eval().requires_grad_(False))
        references[name]=dict(checkpoint=str(cp),sha256=sha(cp),manifest=str(p));protected[str(p)]=sha(p)
    model=StationarySign(reference.components,*models).eval().requires_grad_(False)
    source=Inputs();source.collect(); arrays=[];groups=[]; offset=0; pieces=[]
    def add(x,kind,case):
        nonlocal offset
        arrays.append(x);groups.append(dict(start=offset,end=offset+len(x),kind=kind,case=case));offset+=len(x)
    for b in source.batches:
        add(np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov'])),
            'archive',b['scope']+':'+b['case'])
        if b['scope']=='archived_subpiece_grids':
            r=source.piece_map[ast.literal_eval(b['case'])]
            pieces.append(dict(case='geometric:'+b['case'],lo=float(r['lower']),hi=float(r['upper']),
                coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),
                roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots'])))
    for key,r in source.generic.items():pieces.append(dict(case='generic_coefficients:'+key,lo=r['lower'],hi=r['upper'],
        coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),roots=r['roots']))
    for sp in pieces:
        lo,hi,coeff=sp['lo'],sp['hi'],sp['coeff'];add(case_matrix(coeff,[lo,hi]),'endpoints',sp['case'])
        grid=np.unique(np.r_[np.linspace(lo,hi,129),[r for r in sp['roots'] if lo<=r<=hi]])
        x=case_matrix(coeff,grid);td=teacher(x)[:,0];add(x,'dense_subpiece',sp['case'])
        local=[];ulp=[]
        for j in np.flatnonzero(np.sign(td[:-1])*np.sign(td[1:])<0):local.extend(np.linspace(grid[j],grid[j+1],257))
        for center in np.unique(np.r_[[r for r in sp['roots'] if lo<=r<=hi],grid[td==0]]):
            width=max((hi-lo)*1e-6,128*np.spacing(center))
            local.extend(np.linspace(max(lo,center-width),min(hi,center+width),257))
            ulp.extend((np.float64(center).view(np.int64)+np.arange(-64,65,dtype=np.int64)).view(np.float64))
        for name,values in [('near_root',local),('one_ulp_root_neighborhood',ulp)]:
            if values:
                values=np.unique(values);values=values[(values>=lo)&(values<=hi)]
                add(case_matrix(coeff,values),name,sp['case'])
    # Replay every saved direct-derivative failure and saved worst margin failures.
    for file,kind in [('full_derivative_assembly/full_derivative_sign_failures.csv','previous_derivative_failure'),
        ('stationary_margin_assembly/stationary_margin_sign_worst20.csv','previous_margin_failure')]:
        p=ROOT/file;protected[str(p)]=sha(p)
        for r in csv.DictReader(p.open()):
            add(np.array([[float(r[k]) for k in ('A','B','a','beta','c','orientation_t','orientation_v')]]),kind,r['case'])
    # Exact stationary examples give an unambiguous end-to-end ZERO audit.
    fixtures=[]
    for ot in (0,1):
        for ov in (0,1):
            tp=1 if ot==0 else -1;vp=1 if ov==0 else -1
            fixtures.append([1.,0.,1.,-.5*tp/vp,.5,ot,ov])
    add(np.array(fixtures),'exact_stationary_fixture','A=a=1,B=0,c=0.5')
    data=np.concatenate(arrays);del arrays,source.batches
    print(json.dumps(dict(event='pool_ready',cases=len(data),subpieces=len(pieces))),flush=True)
    output=np.empty(len(data),np.int8); intended=np.empty_like(output);same_terms=np.empty_like(output);float_teacher=np.empty_like(output)
    derivative_sign=np.empty_like(output);outside=np.empty(len(data),bool);learned_terms=np.empty((len(data),2));teacher_terms=np.empty_like(learned_terms)
    fallback=0;terms_match=True; last=time.monotonic()
    with torch.no_grad():
        for start_i in range(0,len(data),16384):
            end=min(start_i+16384,len(data));sl=slice(start_i,end);x=data[sl];tx=torch.from_numpy(x)
            terms=model.terms(tx);learned_terms[sl]=terms.numpy();output[sl]=model.route_terms(terms).numpy().astype(np.int8)
            # The prior assembly is called only in validation to check copied term execution.
            terms_match &= np.array_equal(reference(tx,details=True).numpy()[:,1:],learned_terms[sl])
            t=margin_teacher(x);teacher_terms[sl]=t[:,1:];float_teacher[sl]=term_teacher(t[:,1:]);same_terms[sl]=term_teacher(learned_terms[sl])
            intended[sl],n=mathematical_sign(x);fallback+=n
            derivative_sign[sl]=np.sign(teacher(x)[:,0]).astype(np.int8)
            scale=abs(t[:,1])+abs(t[:,2]);outside[sl]=abs(t[:,0])>32*EPS*scale
            if time.monotonic()-last>20:
                print(json.dumps(dict(event='audit_progress',evaluated=end,exact_fallbacks=fallback,seconds=time.monotonic()-start)),flush=True);last=time.monotonic()
    assert terms_match
    def metric(ids,kind):
        p=output[ids];y=intended[ids];s=same_terms[ids];ft=float_teacher[ids];terms=teacher_terms[ids]
        return dict(group=kind,cases=len(p),teacher_positive=int(np.sum(y>0)),teacher_negative=int(np.sum(y<0)),teacher_zero=int(np.sum(y==0)),
            routed_positive=int(np.sum(p>0)),routed_negative=int(np.sum(p<0)),routed_zero=int(np.sum(p==0)),
            sign_disagreements=int(np.sum(p!=y)),opposite_signs=int(np.sum(p*y<0)),
            teacher_exact_zero_routed_nonzero=int(np.sum((y==0)&(p!=0))),routed_zero_teacher_nonzero=int(np.sum((y!=0)&(p==0))),
            meaningful_sign_disagreements=int(np.sum((p!=y)&outside[ids])),
            identical_term_routing_disagreements=int(np.sum(p!=s)),float_teacher_term_sign_disagreements=int(np.sum(p!=ft)),
            float_teacher_zero_routed_nonzero=int(np.sum((ft==0)&(p!=0))),routed_zero_float_teacher_nonzero=int(np.sum((ft!=0)&(p==0))),
            opposite_term_cases=int(np.sum(np.sign(terms[:,0])*np.sign(terms[:,1])<0)))
    rows=[metric(np.arange(len(data)),'all')]
    for kind in sorted(set(g['kind'] for g in groups)):
        ids=np.concatenate([np.arange(g['start'],g['end']) for g in groups if g['kind']==kind]);rows.append(metric(ids,kind))
    for name,mask in [('ordinary_away_from_cancellation',outside),('cancellation_roundoff_region',~outside),
        ('opposite_sign_terms',np.sign(teacher_terms[:,0])*np.sign(teacher_terms[:,1])<0)]:
        rows.append(metric(np.flatnonzero(mask),name))
    endpoints=[]
    for g in groups:
        if g['kind']!='endpoints':continue
        sl=slice(g['start'],g['end']);t=intended[sl];p=output[sl];d=derivative_sign[sl]
        tc,pc,dc=map(endpoints_class,(t,p,d));tb=t[0]*t[1]<0;pb=p[0]*p[1]<0;db=d[0]*d[1]<0
        endpoints.append(dict(case=g['case'],lower=data[g['start'],4],upper=data[g['end']-1,4],teacher_class=tc,routed_class=pc,prior_derivative_class=dc,
            teacher_left=int(t[0]),teacher_right=int(t[1]),routed_left=int(p[0]),routed_right=int(p[1]),classification_changed=tc!=pc,
            missed_bracket=bool(tb and not pb),added_bracket=bool(pb and not tb),prior_bracket=bool(db),
            prior_bracket_direction_changed=bool(db and not np.array_equal(d,p))))
    # All routing branches, zero handling, magnitude ties and adjacent magnitudes.
    tiny=np.nextafter(0.,1.);cases=[]
    for a in (-1.,-tiny,-0.,0.,tiny,1.):
        for b in (-1.,-tiny,-0.,0.,tiny,1.):cases.append([a,b])
    for a in (tiny,np.finfo(float).tiny,1.,np.finfo(float).max):
        for b in (a,np.nextafter(a,0.)):
            if b>0:cases.extend([[a,-b],[-a,b],[b,-a],[-b,a]])
    pair=np.array(cases)
    with torch.no_grad():routed=model.route_terms(torch.from_numpy(pair)).numpy()
    expected=term_teacher(pair);unit_errors=int(np.sum(routed!=expected))
    write_csv(OUT/'routing_truth_table.csv',[dict(term1=a,term2=b,teacher=int(t),routed=int(p)) for (a,b),t,p in zip(pair,expected,routed)])
    failures=[];ends=np.array([g['end'] for g in groups])
    for i in np.flatnonzero(output!=intended):
        g=groups[int(np.searchsorted(ends,i,side='right'))];x=data[i]
        failures.append(dict(group=g['kind'],case=g['case'],A=x[0],B=x[1],a=x[2],beta=x[3],c=x[4],c_hex=float(x[4]).hex(),orientation_t=int(x[5]),orientation_v=int(x[6]),
            mathematical_teacher_sign=int(intended[i]),routed_sign=int(output[i]),float_teacher_sign=int(float_teacher[i]),
            teacher_term1=teacher_terms[i,0],teacher_term2=teacher_terms[i,1],learned_term1=learned_terms[i,0],learned_term2=learned_terms[i,1],
            same_term_routing_correct=bool(output[i]==same_terms[i]),meaningful=bool(outside[i])))
    write_csv(OUT/'stationary_sign_validation.csv',rows);write_csv(OUT/'stationary_sign_endpoints.csv',endpoints)
    write_csv(OUT/'stationary_sign_failures.csv',failures or [dict(status='none')])
    write_csv(OUT/'exact_zero_failures.csv',[f for f in failures if f['mathematical_teacher_sign']==0 or f['routed_sign']==0] or [dict(status='none')])
    unchanged=all(sha(Path(p))==h for p,h in {**protected,**source.inputs}.items());assert unchanged
    overall=rows[0];changes=sum(r['classification_changed'] for r in endpoints)
    # Do not waive an exact mathematical-zero failure merely because it is small.
    passed=bool(overall['meaningful_sign_disagreements']==0 and changes==0 and unit_errors==0
        and overall['identical_term_routing_disagreements']==0 and overall['teacher_exact_zero_routed_nonzero']==0
        and overall['routed_zero_teacher_nonzero']==0 and np.all(np.isfinite(learned_terms)))
    result=dict(operator='stationary_sign_operator',status='VALIDATED_ASSEMBLY' if passed else 'DIAGNOSTIC_ONLY_STOP',gate_passed=passed,
        metrics=rows,endpoint_subpieces=len(endpoints),endpoint_classification_changes=changes,
        missed_brackets=sum(r['missed_bracket'] for r in endpoints),added_brackets=sum(r['added_bracket'] for r in endpoints),
        previously_bracketed_subpieces=sum(r['prior_bracket'] for r in endpoints),prior_bracket_direction_changes=sum(r['prior_bracket_direction_changed'] for r in endpoints),
        routing_truth_table_cases=len(pair),routing_truth_table_errors=unit_errors,learned_terms_match_existing_assembly=terms_match,
        exact_rational_fallback_occurrences=fallback,exact_rational_unique=exact_sign.cache_info().misses,
        oracle='Outward-rounded positive polynomial intervals with exact Fraction comparison of squared magnitudes for ambiguous cases; no root solving.',
        teacher_semantics='Input float64 coefficients and c interpreted as exact reals, t/v=c or real 1-c. Float64 teacher-term and identical learned-term controls reported separately.',
        meaningful_definition='abs(float64 teacher M)>32*eps*(abs(teacher T1)+abs(teacher T2)); reporting only; never changes router outputs.',
        zero_gate='Exact-zero routing must match both directions; do not waive false zeros or lost mathematical zeros.',
        nonfinite_terms=int(np.sum(~np.isfinite(learned_terms))),references=references,
        all_prior_hashes_unchanged=True,no_training=True,no_root_search=True,elapsed_seconds=time.monotonic()-start)
    if passed:
        manifest=dict(operator='stationary_sign_operator',status=result['status'],references=references,module='experiments.geometry.validate_stationary_sign',
            model='StationarySign',implementation_sha256=sha(Path(__file__)),evaluation=str(OUT/'results.json'))
        path=OUT/'stationary_sign_validated.json';path.write_text(json.dumps(manifest,indent=2));result['assembly_sha256']=sha(path)
        path.with_suffix('.sha256').write_text(result['assembly_sha256']+'  '+path.name+'\n')
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(source.inputs,indent=2))
    (OUT/'stationary_sign_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','references')},indent=2),flush=True)
    print(json.dumps(rows[0],indent=2),flush=True)


if __name__=='__main__':run()
