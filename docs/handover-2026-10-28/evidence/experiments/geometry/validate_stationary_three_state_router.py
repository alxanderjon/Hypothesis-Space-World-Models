"""Frozen three-state sign assembly. No ZERO output or root-location behavior."""
import ast
import csv
import json
import time
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_stationary_reliability_threshold import ReliabilityThreshold
from experiments.geometry.validate_full_derivative import case_matrix

OUT=ref.ROOT/'stationary_three_state_sign_router_assembly'
TRUST_POSITIVE=1
TRUST_NEGATIVE=2
DEFER=3
NAMES={1:'TRUST_POSITIVE',2:'TRUST_NEGATIVE',3:'DEFER'}


class StationaryThreeStateRouter(nn.Module):
    def __init__(self,term_assembly,reliability):
        super().__init__();self.term_assembly=term_assembly;self.reliability=reliability

    def route_terms(self,terms,details=False):
        a,b=terms.unbind(1)
        out=torch.full_like(a,DEFER,dtype=torch.int8)
        positive=((a>0)&(b>=0))|((a==0)&(b>0))
        negative=((a<0)&(b<=0))|((a==0)&(b<0))
        out[positive]=TRUST_POSITIVE;out[negative]=TRUST_NEGATIVE
        opposite=((a>0)&(b<0))|((a<0)&(b>0));gap=torch.full_like(a,float('nan'))
        ids=torch.nonzero(opposite).flatten()
        if len(ids):
            mags=abs(terms[ids]);scale=torch.maximum(mags[:,0],mags[:,1])
            r=abs(mags[:,0]-mags[:,1])/scale;gap[ids]=r
            trusted=self.reliability.decision(r);ti=ids[trusted]
            if len(ti):
                norm=mags[trusted]/scale[trusted,None]
                margin=self.term_assembly.margin(norm)
                decision=self.term_assembly.threshold.decision(margin)
                chosen=torch.where(decision>0,a[ti],b[ti])
                out[ti]=torch.where(decision==0,DEFER,torch.where(chosen>0,TRUST_POSITIVE,TRUST_NEGATIVE)).to(torch.int8)
        return (out,gap,opposite) if details else out

    def forward(self,x,details=False):
        terms=self.term_assembly.terms(x)
        result=self.route_terms(terms,details=details)
        return (*result,terms) if details else result


def load_router():
    references={}
    for name,folder in [('sqrt_refined','sqrt_operator_failure_refine'),('reliability','stationary_reliability_threshold_smoke'),
                        ('magnitude_margin','stationary_magnitude_margin_smoke'),('magnitude_threshold','stationary_magnitude_threshold_smoke/zero_anchor_converged')]:
        path=ref.ROOT/folder/'candidate_manifest.json';m=json.loads(path.read_text());cp=Path(m['checkpoint'])
        assert m['status']=='FROZEN_CANDIDATE' and ref.sha(cp)==m['sha256']
        references[name]=dict(manifest=str(path),manifest_sha256=ref.sha(path),checkpoint=str(cp),sha256=m['sha256'])
    sqrt=ref.ScalarPower();sqrt.load_state_dict(torch.load(references['sqrt_refined']['checkpoint'],weights_only=False)['state_dict']);sqrt.eval().requires_grad_(False)
    _,terms=ref.routers(sqrt)
    reliability=ReliabilityThreshold();reliability.load_state_dict(torch.load(references['reliability']['checkpoint'],weights_only=False)['state_dict']);reliability.eval().requires_grad_(False)
    _,primitives=ref.load_terms()
    references['term_primitive_references']={k:primitives[k] for k in ('orientation','multiply','quadratic')}
    return StationaryThreeStateRouter(terms,reliability).eval().requires_grad_(False),references


def metric(out,teacher,group):
    p=out==TRUST_POSITIVE;n=out==TRUST_NEGATIVE;d=out==DEFER;trusted=p|n
    wrongpos=p&(teacher!=1);wrongneg=n&(teacher!=-1);wrong=wrongpos|wrongneg
    return dict(group=group,cases=len(out),TRUST_POSITIVE=int(p.sum()),TRUST_NEGATIVE=int(n.sum()),DEFER=int(d.sum()),
        wrong_TRUST_POSITIVE=int(wrongpos.sum()),wrong_TRUST_NEGATIVE=int(wrongneg.sum()),trusted_disagreements=int(wrong.sum()),
        exact_zero_incorrectly_trusted=int(np.sum(trusted&(teacher==0))),trusted_failure_rate=float(wrong.sum()/trusted.sum()) if trusted.any() else None,
        positive_deferred=int(np.sum(d&(teacher>0))),negative_deferred=int(np.sum(d&(teacher<0))),exact_zero_deferred=int(np.sum(d&(teacher==0))),
        defer_fraction=float(d.mean()) if len(d) else None)


def run():
    started=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ref.ROOT.rglob('*.pt')};protected.update({str(p):ref.sha(p) for p in ref.ROOT.rglob('*manifest.json')})
    router,references=load_router();source=ref.Inputs();source.collect();pieces={};archive=[];archtypes=[]
    for b in source.batches:
        x=np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['beta'],b['c'],b['o'],b['ov']))
        archive.append(x);archtypes.extend([b['scope']]*len(x))
        if b['scope']=='archived_subpiece_grids':
            key=ast.literal_eval(b['case']);r=source.piece_map[key]
            pieces[('geometric',b['case'])]=dict(coeff=(b['A'][0],b['B'],b['a'],b['beta'][0],b['o'][0],b['ov'][0]),lo=float(r['lower']),hi=float(r['upper']),
                roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots']))
    for key,r in source.generic.items():pieces[('generic_coefficients',key)]=dict(coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),lo=r['lower'],hi=r['upper'],roots=list(r['roots']))
    del source.batches
    metrics=[];failures=[];endpoint_rows=[];all_out=[];all_teacher=[];totals=dict(nonfinite_terms=0,nonfinite_opposite_gaps=0,opposite_terms=0,same_sign_terms=0,one_zero_term=0,both_zero_terms=0)
    def evaluate(x,group,cases=None,subgroups=None):
        x=np.asarray(x,dtype=np.float64).reshape(-1,7);out=np.empty(len(x),np.int8);teacher=np.empty(len(x),np.int8);gaps=np.empty(len(x));opps=np.empty(len(x),bool)
        for begin in range(0,len(x),16384):
            end=min(begin+16384,len(x));sl=slice(begin,end)
            with torch.no_grad():oo,gg,op,term=router(torch.from_numpy(x[sl]),details=True)
            out[sl]=oo.numpy();gaps[sl]=gg.numpy();opps[sl]=op.numpy();t=term.numpy();teacher[sl],_=ref.mathematical_sign(x[sl])
            totals['nonfinite_terms']+=int(np.sum(~np.isfinite(t).all(1)));totals['nonfinite_opposite_gaps']+=int(np.sum(op.numpy()&~np.isfinite(gg.numpy())))
            totals['opposite_terms']+=int(op.sum());totals['same_sign_terms']+=int(np.sum(np.sign(t[:,0])*np.sign(t[:,1])>0))
            totals['one_zero_term']+=int(np.sum((t[:,0]==0)^(t[:,1]==0)));totals['both_zero_terms']+=int(np.sum((t[:,0]==0)&(t[:,1]==0)))
            bad=((out[sl]==TRUST_POSITIVE)&(teacher[sl]!=1))|((out[sl]==TRUST_NEGATIVE)&(teacher[sl]!=-1))
            for j in np.flatnonzero(bad):
                i=begin+j;v=x[i]
                failures.append(dict(group=group,case=str(cases[i]) if cases is not None else str(i),A=v[0],B=v[1],a=v[2],beta=v[3],c=v[4],ot=int(v[5]),ov=int(v[6]),
                    T1=t[j,0],T2=t[j,1],r=gaps[i],prediction=NAMES[int(out[i])],mathematical_sign=int(teacher[i])))
        assert np.isin(out,[TRUST_POSITIVE,TRUST_NEGATIVE,DEFER]).all()
        metrics.append(metric(out,teacher,group));all_out.append(out);all_teacher.append(teacher)
        # Overlapping diagnostic slices, not additional overall observations.
        ordinary=~opps|(gaps>1e-12)
        metrics.append(metric(out[ordinary],teacher[ordinary],group+':ordinary_gap_slice'))
        metrics.append(metric(out[~ordinary],teacher[~ordinary],group+':near_cancellation_slice'))
        if subgroups is not None:
            for name in sorted(set(subgroups)):
                mask=np.asarray(subgroups)==name;metrics.append(metric(out[mask],teacher[mask],group+':'+name))
        return out,teacher,gaps
    evaluate(np.concatenate(archive),'full_archived_stationary_pool',subgroups=np.array(archtypes));del archive,archtypes
    def read(path):
        protected[str(path)]=ref.sha(path);return list(csv.DictReader(path.open()))
    baseline=read(ref.FAIL/'stationary_midpoint_recovery.csv')
    for row in baseline:pieces[(row['scope'],row['case'])]['roots'].append(float(row['c_teacher']))
    near=[];ulp=[]
    for key,p in pieces.items():
        local=[];adj=[]
        for c in set(p['roots']):
            if not p['lo']<=c<=p['hi']:continue
            width=max((p['hi']-p['lo'])*1e-6,128*np.spacing(c))
            local.extend(np.linspace(max(p['lo'],c-width),min(p['hi'],c+width),257))
            adj.extend((np.float64(c).view(np.int64)+np.arange(-64,65,dtype=np.int64)).view(np.float64))
        for values,store in [(local,near),(adj,ulp)]:
            if values:
                c=np.unique(values);c=c[(c>=p['lo'])&(c<=p['hi'])];store.append(case_matrix(p['coeff'],c))
    evaluate(np.concatenate(near),'dense_archived_near_root');del near
    evaluate(np.concatenate(ulp),'archived_one_ULP_neighborhood');del ulp
    def replay(rows,keys,group):
        if rows:evaluate([[float(r[k]) for k in keys] for r in rows],group,cases=[r.get('case',str(i)) for i,r in enumerate(rows)])
    standard=('A','B','a','beta','c','orientation_t','orientation_v')
    for folder,file in [('stationary_sign_assembly','stationary_sign_failures.csv'),('full_derivative_assembly','full_derivative_sign_failures.csv')]:
        rows=read(ref.ROOT/folder/file)
        zero=lambda r:r.get('routed_sign')=='0' or ('learned_derivative' in r and float(r['learned_derivative'])==0)
        replay([r for r in rows if zero(r)],standard,folder+':previous_false_ZERO')
        replay([r for r in rows if not zero(r)],standard,folder+':previous_direction_failures')
    rows=read(ref.FAIL/'direction_and_zero_failures.csv')
    for event,group in [('false_learned_ZERO','original_2908_false_ZERO'),('wrong_direction','original_5330_direction_failures')]:
        replay([r for r in rows if r['event']==event],('A','B','a','beta','q','orientation_t','orientation_v'),group)
    rows=read(ref.OUT/'refined_query_failure_attribution.csv')
    replay([r for r in rows if r['false_ZERO']=='True'],('A','B','a','beta','c','ot','ov'),'refined_2944_false_ZERO')
    replay([r for r in rows if r['false_ZERO']!='True'],('A','B','a','beta','c','ot','ov'),'refined_previous_direction_failures')
    for file,group,column in [('unresolved_zero_neighborhoods.csv','previous_509_unresolved_neighborhoods','q'),
                             ('direction_disagreements.csv','previous_ZERO_rule_direction_failures','c')]:
        rows=read(ref.ROOT/'stationary_zero_neighborhood_audit'/file);x=[]
        for r in rows:
            p=pieces[(r['scope'],r['case'])]
            cuts=[float(r[column])]
            if column=='q':cuts.extend([float(r['q_minus']),float(r['q_plus'])])
            x.extend(case_matrix(p['coeff'],cuts))
        evaluate(x,group)
    for folder,file,label in [('fixed_sign_reliability_fresh_holdout','evaluated_holdout_results.npz','fresh_holdout_stationary_cases')]:
        path=ref.ROOT/folder/file;protected[str(path)]=ref.sha(path);data=np.load(path)
        evaluate(data['inputs'],label,subgroups=data['group'])
    fixture=[[1.,0.,1.,-.5*(1 if ot==0 else -1)/(1 if ov==0 else -1),.5,ot,ov] for ot in (0,1) for ov in (0,1)]
    fo,ft,fg=evaluate(fixture,'four_exact_stationary_fixtures')
    ref.write_csv(OUT/'exact_stationary_fixtures.csv',[dict(orientation_t=int(x[5]),orientation_v=int(x[6]),prediction=NAMES[int(o)],teacher_sign=int(t),gap=float(g)) for x,o,t,g in zip(fixture,fo,ft,fg)])
    keys=list(pieces);endpoints=np.concatenate([case_matrix(pieces[k]['coeff'],[pieces[k]['lo'],pieces[k]['hi']]) for k in keys])
    eo,et,eg=evaluate(endpoints,'all_archived_subpiece_endpoints');eo=eo.reshape(-1,2);et=et.reshape(-1,2);eg=eg.reshape(-1,2)
    existing={(r['scope'],r['case']) for r in baseline}
    for k,o,t,g in zip(keys,eo,et,eg):
        expected=np.where(t>0,TRUST_POSITIVE,np.where(t<0,TRUST_NEGATIVE,DEFER));isbracket=k in existing
        endpoint_rows.append(dict(scope=k[0],case=k[1],lower=pieces[k]['lo'],upper=pieces[k]['hi'],left=NAMES[int(o[0])],right=NAMES[int(o[1])],
            teacher_left=int(t[0]),teacher_right=int(t[1]),left_gap=float(g[0]) if np.isfinite(g[0]) else '',right_gap=float(g[1]) if np.isfinite(g[1]) else '',
            left_deferred=bool(o[0]==DEFER),right_deferred=bool(o[1]==DEFER),
            trusted_sign_errors=int(np.sum((o!=DEFER)&(o!=expected))),existing_stationary_bracket=isbracket,
            bracket_lost_either_trusted_endpoint=bool(isbracket and np.any(o==DEFER)),bracket_lost_both_trusted_endpoints=bool(isbracket and np.all(o==DEFER)),
            bracket_retains_two_correct_directions=bool(isbracket and np.all(o==expected) and np.all(o!=DEFER))))
    ref.write_csv(OUT/'three_state_endpoint_brackets.csv',endpoint_rows)
    ref.write_csv(OUT/'deferred_endpoints.csv',[r for r in endpoint_rows if r['left_deferred'] or r['right_deferred']] or [dict(status='none')])
    # Direct term fixtures exercise same-sign, one-zero, both-zero and magnitude
    # ties. These are separate from stationary input-pool totals.
    pairs=np.array([[a,b] for a in (-1.,-0.,0.,1.) for b in (-1.,-0.,0.,1.)]+[[1.,-np.nextafter(1.,0.)],[-1.,np.nextafter(1.,0.)],[1.,-1.],[-1.,1.]])
    with torch.no_grad():po=router.route_terms(torch.from_numpy(pairs)).numpy()
    tt=[]
    for (a,b),o in zip(pairs,po):
        value=Fraction.from_float(a)+Fraction.from_float(b);sg=1 if value>0 else -1 if value<0 else 0
        opposite=a*b<0;r=abs(abs(a)-abs(b))/max(abs(a),abs(b)) if opposite else None
        must_defer=(a==b==0) or (opposite and r<=float(router.reliability.theta))
        expected=DEFER if must_defer else TRUST_POSITIVE if sg>0 else TRUST_NEGATIVE if sg<0 else DEFER
        tt.append(dict(T1=a,T2=b,prediction=NAMES[int(o)],expected=NAMES[expected],correct=bool(o==expected)))
    ref.write_csv(OUT/'three_state_term_routing_fixtures.csv',tt)
    overall=metric(np.concatenate(all_out),np.concatenate(all_teacher),'all_occurrences');metrics.insert(0,overall)
    ref.write_csv(OUT/'three_state_sign_validation.csv',metrics)
    if failures:ref.write_csv(OUT/'wrong_trusted_sign_cases.csv',failures)
    bracket=dict(subpieces=len(keys),endpoint_cases=len(endpoints),deferred_endpoints=sum(r['left_deferred']+r['right_deferred'] for r in endpoint_rows),
        trusted_endpoint_errors=sum(r['trusted_sign_errors'] for r in endpoint_rows),existing_stationary_brackets=len(existing),
        brackets_losing_either_trusted_endpoint=sum(r['bracket_lost_either_trusted_endpoint'] for r in endpoint_rows),
        brackets_losing_both_trusted_endpoints=sum(r['bracket_lost_both_trusted_endpoints'] for r in endpoint_rows),
        brackets_retaining_two_correct_directions=sum(r['bracket_retains_two_correct_directions'] for r in endpoint_rows))
    passed=bool(overall['trusted_disagreements']==0 and bracket['trusted_endpoint_errors']==0 and bracket['brackets_losing_either_trusted_endpoint']==0
        and totals['nonfinite_terms']==totals['nonfinite_opposite_gaps']==0 and all(t['correct'] for t in tt))
    assert all(ref.sha(Path(p))==h for p,h in {**protected,**source.inputs}.items())
    result=dict(operator='stationary_three_state_sign_router',status='VALIDATED_ASSEMBLY' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        metrics=metrics,bracket_check=bracket,term_categories=totals,term_fixture_errors=sum(not t['correct'] for t in tt),
        output_codes={str(k):v for k,v in NAMES.items()},meaning='DEFER is numerical uncertainty; never ZERO, root found, or search termination.',
        references=references,module='experiments.geometry.validate_stationary_three_state_router',model='StationaryThreeStateRouter',loader='load_router',
        implementation_sha256=ref.sha(Path(__file__)),all_protected_hashes_unchanged=True,
        counting='Overall counts are evaluated occurrences; archives and named failure replays overlap. Ordinary/near-cancellation and fresh subgroup metric rows are slices, not extra observations.',
        no_training=True,no_root_iteration=True,no_recursion_or_search=True,elapsed_seconds=time.monotonic()-started)
    if passed:
        manifest={k:v for k,v in result.items() if k!='metrics'};path=OUT/'stationary_three_state_sign_router_validated.json'
        path.write_text(json.dumps(manifest,indent=2));digest=ref.sha(path);path.with_suffix('.sha256').write_text(digest+'  '+path.name+'\n')
        result['assembly_manifest']=str(path);result['assembly_sha256']=digest
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(source.inputs,indent=2));(OUT/'three_state_sign_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','references')},indent=2),flush=True)
    print(json.dumps(overall,indent=2),flush=True)


if __name__=='__main__':run()
