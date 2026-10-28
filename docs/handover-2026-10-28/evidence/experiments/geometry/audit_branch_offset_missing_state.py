"""Missing static state in 21 saved offset-conflict groups; no model fitting."""
import itertools
import json
import time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from experiments.geometry.audit_brent_branch_offsets import saved_geometries,read,bits
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.sho.audit_fixed_allocation_intervals import Audit

SOURCE=ROOT/'brent_branch_switch_offset_audit'
OUT=ROOT/'branch_switch_offset_missing_state_audit'


def exact_key(x):
    if isinstance(x,(float,np.floating)):return float(x).hex()
    if isinstance(x,(list,tuple,np.ndarray)):return tuple(exact_key(v) for v in x)
    return x.item() if isinstance(x,np.generic) else x


def conflicts(rows,fields):
    groups=defaultdict(list)
    for r in rows:groups[tuple(exact_key(r[f]) for f in fields)].append(r)
    return [g for g in groups.values() if len({r['offset_k'] for r in g})>1]


def run():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('manifest','locked','validated'))})
    conflict_file=SOURCE/'exact_feature_conflicts.csv';source_file=ROOT/'branch_switch_location_residual_training/location_cases.json'
    protected.update({str(p):sha(p) for p in (conflict_file,source_file)})
    records=[r for r in read(conflict_file) if r['feature_set']=='local_plus_spacing_exponents']
    assert len(records)==117 and len({r['conflict_group'] for r in records})==21
    saved=json.loads(source_file.read_text());geometry=saved_geometries()
    a=Audit.__new__(Audit);rows=[]
    for r in records:
        i=int(r['index']);parent=saved[i]['parent'] if i<550 else geometry[r['call_label']]
        axis=int(r['axis']);k=int(r['count_left']);a.setup(parent,axis)
        left=float(r['scan_left']);right=float(r['scan_right']);boundary=float(r['boundary']);c=np.array([left,right]);f=a.formula(c,k)
        analytic=np.tile(a.s,(2,2,1));analytic[:,0,axis]=a.a*c;analytic[:,1,axis]=a.a*(1-c)
        physical=a.geometry(c)
        identities=np.column_stack((f[3],f[4],f[5][np.arange(2),f[4]]))
        row=dict(original_group=int(r['conflict_group']),unique_id=int(r['unique_id']),label=r['label'],offset_k=int(r['signed_k']),
            teacher_cut=float(r['teacher_cut']),projected_boundary=float(r['projected_boundary']),
            parent_lower=parent['lower'],parent_upper=parent['upper'],parent_side_lengths=a.s.tolist(),
            analytic_child_sides=analytic.tolist(),physical_child_sides=physical.tolist(),
            dimension=a.d,parent_budget=a.n,counts=[k,a.n-k],axis=axis,side=int(r['side']),
            scan_coordinates=[left,right],exact_boundary=boundary,
            original_margins=[float(r['margin_left']),float(r['margin_right'])],
            normalized_boundary_position=float(r['normalized_boundary_position']),switch_type=r['switch_type'],
            scan_spacing=float(r['scan_spacing']),local_exponent=int(r['local_exponent']),scan_exponent=int(r['scan_exponent']),
            branch_identities=identities.tolist(),competing_q=f[6].tolist(),competing_log_aspect=f[7].tolist(),
            teacher_derivatives=f[1].tolist(),a=a.a,B=a.B,h=a.h,omin=a.omin,omax=a.omax,
            normalized_B=a.B/a.a**2,normalized_h=a.h/a.a)
        rows.append(row)
    by=defaultdict(list)
    for r in rows:by[r['original_group']].append(r)
    sets={
        'original_state':['original_margins','normalized_boundary_position','switch_type','scan_spacing','local_exponent','scan_exponent'],
        'add_counts_axis_identities_coordinates':['counts','axis','branch_identities','scan_coordinates','exact_boundary','dimension'],
        'add_competing_branch_values':['counts','axis','branch_identities','scan_coordinates','exact_boundary','dimension','competing_q','competing_log_aspect'],
        'full_parent_static':['parent_side_lengths','counts','dimension','axis','scan_coordinates','exact_boundary'],
        'full_child_static':['analytic_child_sides','physical_child_sides','counts','dimension','axis','scan_coordinates','exact_boundary','branch_identities','competing_q','competing_log_aspect'],
        'compact_executed_geometry':['a','B','h','omin','omax','counts','dimension','scan_coordinates','exact_boundary'],
        'normalized_geometry_only':['normalized_B','normalized_h','counts','dimension','scan_coordinates','exact_boundary','branch_identities']}
    reports=[];group_reports=[];minimal=[]
    original=sets['original_state']
    for name,fields in sets.items():
        use=list(dict.fromkeys(original+fields));bad=[];subgroups=[]
        for gi,g in by.items():
            mixed=conflicts(g,use)
            if mixed:bad.append(gi);subgroups.extend(mixed)
            group_reports.append(dict(original_group=gi,feature_set=name,cases=len(g),resolved=not mixed,remaining_conflicting_subgroups=len(mixed),
                remaining_conflicting_states=sum(map(len,mixed))))
        reports.append(dict(feature_set=name,original_groups=21,resolved_groups=21-len(bad),unresolved_groups=len(bad),
            unresolved_group_ids=json.dumps(bad),conflicting_subgroups=len(subgroups),conflicting_states=sum(map(len,subgroups))))
    # Exhaustive cardinality-minimum separating additions from an explicitly
    # declared finite vocabulary. This is label grouping, not classifier fitting.
    vocabulary=['dimension','counts','axis','a','B','h','omin','omax','scan_coordinates','exact_boundary',
                'branch_identities','competing_q','competing_log_aspect','parent_side_lengths','analytic_child_sides']
    for gi,g in sorted(by.items()):
        solutions=[];cardinality=None
        for n in range(1,len(vocabulary)+1):
            for fields in itertools.combinations(vocabulary,n):
                if not conflicts(g,fields):solutions.append(list(fields))
            if solutions:cardinality=n;break
        minimal.append(dict(original_group=gi,cases=len(g),signed_offsets=json.dumps(sorted({r['offset_k'] for r in g})),
            minimum_additional_feature_blocks=cardinality,all_minimal_separators=json.dumps(solutions),
            qualification='Empirical exact partition of this group; vector blocks count as one block. Not a proven sufficient generalization feature set.'))
    # Scalar-only global minimum additions on top of the original state; counts
    # and full vectors are excluded here to keep the size interpretation honest.
    scalar_vocab=['dimension','axis','a','B','h','omin','omax','exact_boundary']
    global_min=[]
    for n in range(1,len(scalar_vocab)+1):
        for fields in itertools.combinations(scalar_vocab,n):
            if all(not conflicts(g,fields) for g in by.values()):global_min.append(list(fields))
        if global_min:break
    unresolved=[gi for gi,g in by.items() if conflicts(g,sets['compact_executed_geometry'])]
    # Re-run only the teacher's local Brent solve to check sufficiency of the
    # compact exact function state. No recursion or learned inference involved.
    replay=[]
    if not unresolved:
        for r in rows:
            compact=Audit.__new__(Audit)
            compact.d=r['dimension'];compact.n=sum(r['counts'])
            for field in ('a','B','h','omin','omax'):setattr(compact,field,r[field])
            left,right=r['scan_coordinates'];count=r['counts'][0]
            calls=[]
            def fun(c):
                f=compact.formula(c,count);calls.append(float(c));return float(f[1][0])
            root,info=brentq(fun,left,right,xtol=5e-324,rtol=1e-15,full_output=True)
            offset=bits(root)-bits(r['projected_boundary'])
            replay.append(dict(original_group=r['original_group'],unique_id=r['unique_id'],label=r['label'],
                saved_cut=r['teacher_cut'],replayed_cut=root,saved_k=r['offset_k'],replayed_k=offset,
                exact_cut_match=root==r['teacher_cut'],exact_k_match=offset==r['offset_k'],function_calls=info.function_calls,iterations=info.iterations,
                converged=info.converged))
    else:
        # Refuse to fabricate internal brackets/actions from callback coordinates.
        json_write(OUT/'unresolved_requires_instrumented_brent.json',dict(groups=unresolved))
    # Failure case and its formerly indistinguishable peers, from original group 17.
    failure=next(r for r in rows if r['unique_id']==765);peers=by[failure['original_group']]
    json_write(OUT/'recursive_k_minus6_full_static_state.json',dict(failure=failure,original_group_peers=peers))
    json_write(OUT/'static_states.json',rows)
    write_csv(OUT/'static_feature_levels.csv',reports);write_csv(OUT/'group_feature_resolution.csv',group_reports)
    write_csv(OUT/'minimal_static_separators.csv',minimal)
    if replay:write_csv(OUT/'compact_static_brent_replay.csv',replay)
    flat=[]
    for r in rows:
        flat.append({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
    write_csv(OUT/'missing_state_comparison.csv',flat)
    protected_unchanged=all(sha(Path(p))==v for p,v in protected.items());assert protected_unchanged
    result=dict(status='AUDIT_ONLY_COMPLETE' if not unresolved else 'STATIC_CONFLICTS_REMAIN',original_conflicting_groups=21,cases=len(rows),
        richer_static_resolved=21-len(unresolved),remaining_conflicting_groups=unresolved,levels=reports,
        smallest_empirical_global_scalar_additions=global_min,
        exact_compact_state_replay_cases=len(replay),exact_compact_state_replay_matches=sum(r['exact_cut_match'] and r['exact_k_match'] for r in replay),
        brent_trace_required=bool(unresolved),brent_trace_note='No unresolved full-static conflicts, so conditional Level 2 not run.' if not unresolved else 'Instrumented trace required.',
        recursive_failure_group=failure['original_group'],recursive_failure_k=failure['offset_k'],
        no_training=True,no_recursion=True,no_global_search=True,protected_files=len(protected),all_protected_hashes_unchanged=True,
        elapsed_seconds=time.monotonic()-start,implementation_sha256=sha(Path(__file__)))
    json_write(OUT/'results.json',result);json_write(OUT/'protected_hashes.json',protected)
    print(json.dumps(result,indent=2),flush=True)
    print(json.dumps(dict(failure_group=[{k:r[k] for k in ('label','offset_k','a','B','h','omin','omax','parent_side_lengths')} for r in peers]),indent=2),flush=True)


if __name__=='__main__':run()
