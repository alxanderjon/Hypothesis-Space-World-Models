"""Exact-state identifiability audit; no classifier fit or locator replay."""
import itertools
import json
import time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read,bits
from experiments.sho.audit_fixed_allocation_intervals import Audit

SOURCE=ROOT/'branch_switch_locator_assembly'
OUT=ROOT/'terminal_ulp_correction_state_audit'


def key(x):
    if isinstance(x,(float,np.floating)):return float(x).hex()
    if isinstance(x,(list,tuple,np.ndarray)):return tuple(key(v) for v in x)
    return x


def partition(rows,fields):
    groups=defaultdict(list)
    for r in rows:groups[tuple(key(r[f]) for f in fields)].append(r)
    return list(groups.values())


def evaluate(rows,fields,name,scope):
    groups=partition(rows,fields);bad=[g for g in groups if len({r['k'] for r in g})>1]
    return dict(scope=scope,feature_set=name,cases=len(rows),groups=len(groups),
        singleton_groups=sum(len(g)==1 for g in groups),repeated_groups=sum(len(g)>1 for g in groups),
        repeated_cases=sum(len(g) for g in groups if len(g)>1),conflicting_groups=len(bad),
        conflicting_cases=sum(map(len,bad)),minimum_unavoidable_errors=sum(len(g)-max(Counter(r['k'] for r in g).values()) for g in bad)),bad


def run():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    rec=json.loads((SOURCE/'records.json').read_text());states=json.loads((SOURCE/'case_states.json').read_text())
    trace=defaultdict(list)
    for r in read(SOURCE/'dynamic_query_trace.csv'):trace[int(r['case_id'])].append(r)
    for name in ('records.json','case_states.json','dynamic_query_trace.csv'):protected[str(SOURCE/name)]=sha(SOURCE/name)
    endpoint_rows=[];rows=[];a=Audit.__new__(Audit)
    for i,(r,s) in enumerate(zip(rec,states)):
        L,R=r['final_L'],r['final_R'];assert bits(R)==bits(L)+1
        # Saved sequence: initial inspection (2), actual locate calls (queries),
        # then terminal-feature inspection. The locator verifies both endpoints
        # in its last two calls. No new query or root iteration is performed.
        calls=trace[i][2:2+r['queries']]
        assert len(calls)==r['queries']
        assert float(calls[-2]['c'])==L and float(calls[-1]['c'])==R
        work=calls[2:-2];last=work[-3:]
        pattern=[int(v['learned_state'])==3 for v in last]
        pattern=[False]*(3-len(pattern))+pattern
        a.setup(s['parent'],s['axis']);f=a.formula(np.array([L,R]),s['count'])
        branches=[];learned_terms=[];margins=[];teacher_values=[];teacher_margins=[];competing=[]
        for j,(side,c,cached) in enumerate(zip(('L','R'),(L,R),calls[-2:])):
            iq=int(f[3][j]);ia=int(f[4][j]);power=int(f[5][j,ia]);branches.extend((iq,ia,power))
            assert (iq,ia,power)==tuple(int(cached[k]) for k in ('q_active','aspect_active','aspect_power'))
            lt=np.array([float(cached['T1']),float(cached['T2'])]);lm=float(lt.sum());den=float(abs(lt).sum());ln=lm/den if den else 0.
            learned_terms.extend(lt.tolist());margins.append(ln);teacher_values.append(float(f[1][j]))
            A=1/(a.h*np.power(float(s['count'] if iq==0 else a.n-s['count']),1/a.d))
            t=c if iq==0 else 1-c;v=c if ia==0 else 1-c
            t1=A*a.a**2*t*(1 if iq==0 else -1)*v
            t2=.05*power*(1 if ia==0 else -1)*np.sqrt(a.B+a.a**2*t**2)
            tm=t1+t2;td=abs(t1)+abs(t2);teacher_margins.append(float(tm/td if td else 0.))
            competing.extend(f[6][j].tolist()+f[7][j].tolist())
            endpoint_rows.append(dict(case_id=i,label=s['label'],endpoint=side,c=c,q_active=iq,aspect_active=ia,aspect_power=power,
                learned_T1=lt[0],learned_T2=lt[1],learned_margin=lm,learned_normalized_margin=ln,learned_margin_magnitude=abs(lm),
                learned_state=int(cached['learned_state']),teacher_derivative=f[1][j],teacher_derivative_magnitude=abs(f[1][j]),
                teacher_margin=tm,teacher_normalized_margin=teacher_margins[-1],
                q0=f[6][j,0],q1=f[6][j,1],aspect0=f[7][j,0],aspect1=f[7][j,1]))
        k=bits(s['teacher_cut'])-bits(L)
        rows.append(dict(case_id=i,uid=s['uid'],label=s['label'],source=s['source'],k=k,teacher_cut=s['teacher_cut'],
            relation='LEFT_OF_L' if k<0 else 'EQUAL_L' if k==0 else 'EQUAL_R' if k==1 else 'RIGHT_OF_R',
            endpoints=[L,R],L=L,R=R,normalized_margins=margins,learned_terms=learned_terms,
            teacher_derivatives=teacher_values,teacher_normalized_margins=teacher_margins,
            branches=branches,competing_values=competing,geometry=[a.a,a.B,a.h,a.omin,a.omax],
            a=a.a,B=a.B,h=a.h,omin=a.omin,omax=a.omax,dimension=a.d,counts=[s['count'],a.n-s['count']],
            dimension_counts=[a.d,s['count'],a.n-s['count']],switch_type=s['switch_type'],scan_pair=[s['left'],s['right']],
            scan_left=s['left'],scan_right=s['right'],projected_boundary=s['b'],
            stop_state=[r['termination'],r['final_left_state'],r['final_right_state'],*pattern],
            final_defer_pattern=pattern,locator_defer_queries=sum(int(v['learned_state'])==3 for v in work),
            total_iterations=r['iterations']))
    # Progressive declared feature blocks. No target/candidate coordinate/ID is
    # included in grouping. Teacher derivative features are an audit-only tier.
    tiers={
        'discrete_final_state':['switch_type','stop_state','branches'],
        'add_normalized_margins':['switch_type','stop_state','branches','normalized_margins'],
        'add_adjacent_coordinates':['switch_type','stop_state','branches','normalized_margins','endpoints'],
        'add_learned_term_magnitudes':['switch_type','stop_state','branches','normalized_margins','endpoints','learned_terms'],
        'add_competing_branch_values':['switch_type','stop_state','branches','normalized_margins','endpoints','learned_terms','competing_values'],
        'add_geometry_counts':['switch_type','stop_state','branches','normalized_margins','endpoints','learned_terms','competing_values','geometry','dimension_counts'],
        'full_static_final_state':['switch_type','stop_state','branches','normalized_margins','endpoints','learned_terms','competing_values','geometry','dimension_counts','scan_pair'],
        'full_plus_teacher_diagnostics':['switch_type','stop_state','branches','normalized_margins','endpoints','learned_terms','competing_values','geometry','dimension_counts','scan_pair','teacher_derivatives','teacher_normalized_margins'],
        'compact_function_state':['L','a','B','h','omin','omax','dimension_counts','scan_pair'],
    }
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv')
    groups={g:sorted({int(v['unique_id']) for v in conflicts if v['feature_set']=='local_plus_spacing_exponents' and int(v['conflict_group'])==g}) for g in range(21)}
    oldids=sorted(set(sum(groups.values(),[])))
    scopes={'all_818':rows,'saved_774':rows[:774],'fresh_44':rows[774:],'previous_21_groups': [rows[i] for i in oldids]}
    stats=[];badrows=[];groupstats=[]
    for scope,rr in scopes.items():
        for name,fields in tiers.items():
            stat,bad=evaluate(rr,fields,name,scope);stats.append(stat)
            if scope=='all_818':
                for gi,g in enumerate(bad):
                    badrows.extend(dict(feature_set=name,conflict_group=gi,case_id=r['case_id'],label=r['label'],k=r['k']) for r in g)
    for g,ids in groups.items():
        for name,fields in tiers.items():groupstats.append(evaluate([rows[i] for i in ids],fields,name,str(g))[0])
    # Exact exhaustive cardinality search over these nine INPUT BLOCKS. A vector
    # block counts as one, so this is not claimed as a minimum scalar dimension.
    vocab=['endpoints','normalized_margins','learned_terms','branches','geometry','dimension_counts','scan_pair','switch_type','stop_state']
    minima=[];cardinality=None
    for size in range(1,len(vocab)+1):
        for fs in itertools.combinations(vocab,size):
            stat,bad=evaluate(rows,fs,'subset','all_818')
            if not bad:minima.append(dict(fields=list(fs),**stat))
        if minima:cardinality=size;break
    # Groupwise first-resolving tier, recording actual prior conflicting members.
    first=[]
    prior=evaluate(rows,tiers['add_normalized_margins'],'prior','all_818')[1]
    for gi,g in enumerate(prior):
        resolved=next((name for name,fs in tiers.items() if not evaluate(g,fs,name,'group')[1]),None)
        first.append(dict(original_group=gi,cases=len(g),case_ids=json.dumps([r['case_id'] for r in g]),
                          offsets=json.dumps(sorted({r['k'] for r in g})),first_resolving_tier=resolved))
    dist=[dict(k=k,cases=v) for k,v in sorted(Counter(r['k'] for r in rows).items())]
    full=next(r for r in stats if r['scope']=='all_818' and r['feature_set']=='full_static_final_state')
    compact=next(r for r in stats if r['scope']=='all_818' and r['feature_set']=='compact_function_state')
    assert all(sha(Path(p))==v for p,v in protected.items())
    write_csv(OUT/'terminal_ULP_distribution.csv',dist);write_csv(OUT/'feature_sufficiency.csv',stats)
    write_csv(OUT/'feature_conflicts.csv',badrows);write_csv(OUT/'first_resolving_static_state.csv',first)
    write_csv(OUT/'previous_21_group_sufficiency.csv',groupstats);write_csv(OUT/'endpoint_states.csv',endpoint_rows)
    write_csv(OUT/'terminal_state_cases.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows])
    json_write(OUT/'terminal_state_cases.json',rows);json_write(OUT/'minimal_feature_blocks.json',minima)
    result=dict(status='FULL_STATIC_STATE_IDENTIFIES_OBSERVED_OFFSETS' if full['conflicting_groups']==0 else 'STOP_FULL_STATE_CONFLICTS',
        cases=len(rows),k_min=min(r['k'] for r in rows),k_max=max(r['k'] for r in rows),distinct_k=len(dist),distribution=dist,
        relations=dict(Counter(r['relation'] for r in rows)),full_static=full,compact_function=compact,
        smallest_empirical_block_count=cardinality,smallest_empirical_feature_sets=minima,
        recursive_fixture=rows[765],defer_query_total=sum(r['locator_defer_queries'] for r in rows),
        final_defer_patterns=dict(Counter(str(r['final_defer_pattern']) for r in rows)),
        training_performed=False,locator_replayed=False,terminal_operator_used=False,recursion_run=False,
        protected_files=len(protected),protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    print(json.dumps(dict(status=result['status'],distribution=dist,relations=result['relations'],full=full,minima=minima,
                         feature_tiers=[r for r in stats if r['scope']=='all_818'],fixture_k=rows[765]['k'],
                         final_defer_patterns=result['final_defer_patterns'],seconds=result['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':run()
