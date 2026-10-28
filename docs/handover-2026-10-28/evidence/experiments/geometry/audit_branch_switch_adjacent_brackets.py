"""Saved adjacent-pair audit only; no inference, fitting, or root iteration."""
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read,bits
from experiments.sho.audit_fixed_allocation_intervals import Audit

SOURCE=ROOT/'branch_switch_locator_assembly'
OUT=ROOT/'branch_switch_adjacent_bracket_audit'


def metrics(rows,name):
    outside=[r for r in rows if r['relation'].startswith('OUTSIDE')]
    dist=np.array([r['outside_distance_ULPs'] for r in outside],dtype=np.int64)
    return dict(group=name,cases=len(rows),teacher_equals_L=sum(r['relation']=='LEFT' for r in rows),
                teacher_equals_R=sum(r['relation']=='RIGHT' for r in rows),outside=len(outside),
                outside_left=sum(r['relation']=='OUTSIDE_LEFT' for r in rows),
                outside_right=sum(r['relation']=='OUTSIDE_RIGHT' for r in rows),
                outside_ULP_min=int(dist.min()) if len(dist) else None,
                outside_ULP_median=float(np.median(dist)) if len(dist) else None,
                outside_ULP_p99=float(np.quantile(dist,.99)) if len(dist) else None,
                outside_ULP_max=int(dist.max()) if len(dist) else None)


def run():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    for name in ('records.json','case_states.json','dynamic_query_trace.csv'):
        p=SOURCE/name;protected[str(p)]=sha(p)
    records=json.loads((SOURCE/'records.json').read_text())
    states=json.loads((SOURCE/'case_states.json').read_text());assert len(records)==len(states)==818
    # All final endpoints were already queried and validated. Reuse their saved
    # learned terms and routed signs; do not invoke any locator/terminal model.
    wanted={(i,bits(r[end])) for i,r in enumerate(records) for end in ('final_L','final_R')}
    endpoint_trace={}
    for row in read(SOURCE/'dynamic_query_trace.csv'):
        key=(int(row['case_id']),bits(float(row['c'])))
        if key in wanted:endpoint_trace[key]=row
    assert len(endpoint_trace)==1636
    rows=[];details=[];audit=Audit.__new__(Audit)
    for i,(r,s) in enumerate(zip(records,states)):
        assert r['case_id']==i and r['termination']=='ADJACENT'
        L=float(r['final_L']);R=float(r['final_R']);c=float(s['teacher_cut'])
        il,ir,ic=bits(L),bits(R),bits(c)
        assert ir-il==1 and np.nextafter(L,R)==R
        relation='LEFT' if c==L else 'RIGHT' if c==R else 'OUTSIDE_LEFT' if c<L else 'OUTSIDE_RIGHT'
        distance=max(il-ic,ic-ir,0)
        row=dict(case_id=i,uid=s['uid'],label=s['label'],source=s['source'],switch_type=s['switch_type'],
                 final_L=L,final_R=R,teacher_cut=c,L_hex=L.hex(),R_hex=R.hex(),teacher_hex=c.hex(),
                 relation=relation,teacher_offset_from_L=ic-il,teacher_offset_from_R=ic-ir,
                 outside_distance_ULPs=distance,outside_coordinate_distance=max(L-c,c-R,0.),
                 projected_boundary=s['b'],teacher_offset_from_boundary=ic-bits(s['b']))
        rows.append(row)
        if relation.startswith('OUTSIDE'):continue
        audit.setup(s['parent'],s['axis']);f=audit.formula(np.array([L,R]),s['count'])
        for j,(side,coord) in enumerate((('L',L),('R',R))):
            cached=endpoint_trace[(i,bits(coord))]
            iq=int(f[3][j]);ia=int(f[4][j]);power=int(f[5][j,ia])
            assert iq==int(cached['q_active']) and ia==int(cached['aspect_active']) and power==int(cached['aspect_power'])
            A=1/(audit.h*np.power(float(s['count'] if iq==0 else audit.n-s['count']),1/audit.d))
            beta=.05*power;t=coord if iq==0 else 1-coord;v=coord if ia==0 else 1-coord
            tp=1. if iq==0 else -1.;vp=1. if ia==0 else -1.
            # Direct float64 stationary margin in the original multiplication
            # order; separately record the teacher's authoritative derivative.
            t1=A*audit.a**2*t*tp*v
            t2=beta*vp*np.sqrt(audit.B+audit.a**2*t**2)
            teacher_margin=t1+t2;scale=abs(t1)+abs(t2)
            l1=float(cached['T1']);l2=float(cached['T2']);learned_margin=l1+l2;lscale=abs(l1)+abs(l2)
            details.append(dict(**row,endpoint=side,coordinate=coord,q_active=iq,aspect_active=ia,aspect_power=power,
                competing_q_0=f[6][j,0],competing_q_1=f[6][j,1],competing_log_aspect_0=f[7][j,0],competing_log_aspect_1=f[7][j,1],
                teacher_derivative=f[1][j],teacher_abs_derivative=abs(f[1][j]),
                teacher_T1=t1,teacher_T2=t2,teacher_margin=teacher_margin,teacher_abs_margin=abs(teacher_margin),
                teacher_normalized_margin=teacher_margin/scale if scale else 0.,
                learned_T1=l1,learned_T2=l2,learned_margin=learned_margin,learned_abs_margin=abs(learned_margin),
                learned_normalized_margin=learned_margin/lscale if lscale else 0.,learned_routed_sign=int(cached['learned_state']),
                teacher_score=f[0][j],a=audit.a,B=audit.B,h=audit.h,omin=audit.omin,omax=audit.omax,
                dimension=audit.d,parent_budget=audit.n,count_left=s['count'],count_right=audit.n-s['count'],axis=s['axis'],
                active_A=A,active_beta=beta,orientation_t=iq,orientation_v=ia,
                parent_lower=json.dumps(s['parent']['lower']),parent_upper=json.dumps(s['parent']['upper'])))
    outside=[r for r in rows if r['relation'].startswith('OUTSIDE')]
    summary=[metrics(rows,'all_818'),metrics(rows[:774],'saved_unique_774'),metrics(rows[774:],'fresh_44')]
    for typ in sorted({r['switch_type'] for r in rows}):summary.append(metrics([r for r in rows if r['switch_type']==typ],typ))
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv');group_rows=[];members=[]
    for group in range(21):
        ids=sorted({int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==group})
        group_rows.append(metrics([rows[i] for i in ids],str(group)))
        members.extend(dict(conflicting_group=group,**rows[i]) for i in ids)
    distribution=[dict(outside_ULP_distance=k,cases=v) for k,v in sorted(Counter(r['outside_distance_ULPs'] for r in outside).items())]
    write_csv(OUT/'adjacent_bracket_relations.csv',rows)
    write_csv(OUT/'outside_bracket_cases.csv',outside)
    write_csv(OUT/'matched_endpoint_local_states.csv',details)
    write_csv(OUT/'adjacent_bracket_summary.csv',summary)
    write_csv(OUT/'conflicting_group_summary.csv',group_rows)
    write_csv(OUT/'conflicting_group_cases.csv',members)
    write_csv(OUT/'fresh_44_cases.csv',rows[774:])
    write_csv(OUT/'outside_ULP_distribution.csv',distribution)
    fixture=rows[765];assert fixture['teacher_offset_from_boundary']==-6
    json_write(OUT/'recursive_k_minus6_fixture.json',dict(relation=fixture,saved_state=states[765],
        endpoint_trace=[endpoint_trace[(765,bits(fixture[k]))] for k in ('final_L','final_R')]))
    assert all(sha(Path(p))==v for p,v in protected.items())
    result=dict(status='STOP_TEACHER_CANDIDATES_OUTSIDE_ADJACENT_PAIRS' if outside else 'ALL_TEACHER_CANDIDATES_ARE_ENDPOINTS',
        summary=summary,conflicting_groups=group_rows,fixture=fixture,
        outside_ULP_distribution=distribution,matched_cases=len(details)//2,matched_endpoint_records=len(details),
        side_selection_identifiability_audit_run=False,
        stop_reason='A LEFT/RIGHT-only operator cannot represent the 613 saved teacher candidates outside the adjacent pair.',
        no_training=True,no_checkpoint_modifications=True,no_locator_rerun=True,no_terminal_operator_used=True,no_recursion=True,
        protected_files=len(protected),protected_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    print(json.dumps(dict(summary=summary,fixture=fixture,distribution=distribution),indent=2),flush=True)


if __name__=='__main__':run()
