"""Add hard historical geometry acceptance to the unchanged CDF proposal rule."""
import gzip
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.sho import validate_shift_aware_quota_realizer as base

OUT = base.ROOT / 'geometry_guarded_shift_aware_realizer'


def reference_points():
    grid = (np.arange(512)+.5)/512
    return np.vstack((np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2),
                      np.random.default_rng(27093001).random((262144,2))))


class GeometryGuard:
    def __init__(self, points, selected, cover, holes, nn_p01):
        self.points=points; self.cover=cover; self.hole_limit=holes; self.p01=nn_p01
        ref=reference_points(); tree=cKDTree(ref)
        self.edges=[]; self.holes=[]
        for p in points:
            ix=np.asarray(tree.query_ball_point(p,np.nextafter(cover,np.inf)),dtype=int)
            dist=np.sqrt(np.sum((ref[ix]-p)**2,axis=1))
            self.edges.append(ix[dist<=cover]); self.holes.append(ix[dist<=1/np.sqrt(base.N)])
        boundary=base.boundary_faces(base.N,points.shape[1])
        bd=cKDTree(boundary).query(ref)[0]
        self.count=(bd<=cover).astype(np.int32); self.hcount=(bd<=1/np.sqrt(base.N)).astype(np.int32)
        for i in selected:
            self.count[self.edges[i]]+=1; self.hcount[self.holes[i]]+=1
        self.hole_count=int(np.sum(self.hcount==0))
        assert self.count.min()>0 and self.hole_count<=holes
        self.used=np.zeros(len(points),bool); self.used[selected]=True
        dd,ii=cKDTree(points).query(points,k=min(65,len(points)))
        self.neighbors=ii[:,1:]; self.distances=dd[:,1:]
        self.reverse=[[] for _ in points]
        for row in range(len(points)):
            for j in self.neighbors[row]: self.reverse[j].append(row)
        self.reverse=[np.asarray(v,dtype=int) for v in self.reverse]
        self.near=np.min(np.where(self.used[self.neighbors],self.distances,np.inf),axis=1)
        assert np.isfinite(self.near).all()

    def attempt(self,old,new):
        lost=np.setdiff1d(self.edges[old],self.edges[new],assume_unique=False)
        uncovered=int(np.sum(self.count[lost]==1))
        hlost=np.setdiff1d(self.holes[old],self.holes[new])
        hgain=np.setdiff1d(self.holes[new],self.holes[old])
        holes=self.hole_count+int(np.sum(self.hcount[hlost]==1))-int(np.sum(self.hcount[hgain]==0))
        info=dict(holes_before=self.hole_count,holes_after=holes,uncovered_after=uncovered)
        if uncovered: return False,'coverage',info
        if holes>self.hole_limit: return False,'holes',info
        self.used[old]=False; self.used[new]=True
        rows=np.union1d(self.reverse[old],self.reverse[new])
        prev=self.near[rows].copy()
        self.near[rows]=np.min(np.where(self.used[self.neighbors[rows]],self.distances[rows],np.inf),axis=1)
        assert np.isfinite(self.near).all(), 'Nearest reservoir prefix exhausted; exact check required'
        nn=self.near[self.used]; cv=float(nn.std()/nn.mean()); p01=float(np.quantile(nn,.01))
        info.update(nn_cv_after=cv,nn_p01_after=p01)
        if cv<.08 or p01<self.p01:
            self.near[rows]=prev; self.used[old]=True; self.used[new]=False
            return False,'nn_cv' if cv<.08 else 'nn_p01',info
        self.count[self.edges[old]]-=1; self.count[self.edges[new]]+=1
        self.hcount[self.holes[old]]-=1; self.hcount[self.holes[new]]+=1
        self.hole_count=holes
        return True,'accepted',info


def realize(points,classes,selected,limits,path,expected=None):
    start=time.monotonic(); selected=selected.copy(); state=base.CDFState(points,selected)
    guard=GeometryGuard(points,selected,**limits)
    groups={int(c):np.flatnonzero(classes==c) for c in np.unique(classes)}
    quotas=np.bincount(classes[selected],minlength=int(classes.max())+1)
    counts={}; sweeps=[]; events=0
    compare=gzip.open(expected,'rt') if expected else None
    with gzip.open(path,'wt') as stream:
        def emit(event):
            line=json.dumps(event,separators=(',',':'))
            if compare: assert json.loads(next(compare))==event
            stream.write(line+'\n')
        emit(dict(type='initial',selected=selected.tolist(),limits=limits,cdf_hash=state.fingerprint(),
                  ordering='saved slots; negative CDF effect ascending, reservoir ID ascending; first admissible'))
        for sweep in range(base.MAX_SWEEPS):
            accepted=0
            for slot,old in enumerate(selected):
                old=int(old); group=groups[int(classes[old])]
                alternatives=group[(~guard.used[group])|(group==old)]
                costs=state.effects(old,alternatives)
                order=np.lexsort((alternatives,costs)); attempts=[]; new=old
                for j in order:
                    if costs[j]>=0: break
                    candidate=int(alternatives[j])
                    ok,reason,info=guard.attempt(old,candidate)
                    counts[reason]=counts.get(reason,0)+1
                    attempts.append(dict(candidate=candidate,cdf_scaled_change=int(costs[j]),accept=ok,reason=reason,**info))
                    if ok: new=candidate; break
                emit(dict(event=events,sweep=sweep,slot=slot,quota_class=int(classes[old]),old=old,
                          attempts=attempts,selected=new,remaining_policy='next ascending score/ID until accepted or exhausted'))
                events+=1
                if new!=old:
                    state.swap(old,new);selected[slot]=new;accepted+=1
            assert np.array_equal(np.bincount(classes[selected],minlength=len(quotas)),quotas)
            assert base.CDFState(points,selected).fingerprint()==state.fingerprint()
            row=dict(sweep=sweep+1,accepted=accepted,cdf_loss=state.loss(),holes=guard.hole_count)
            sweeps.append(row);print(json.dumps(dict(pool=path.name,**row)),flush=True)
            if not accepted: break
    if compare: assert compare.read()=='';compare.close()
    return selected,dict(events=events,counts=counts,sweeps=sweeps,runtime_seconds=time.monotonic()-start)


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    protected=json.loads((base.OUT/'protected_hashes.json').read_text())
    protected.update({str(p):base.sha(p) for p in (base.OUT).iterdir() if p.is_file()})
    # Existing source hashes remain immutable; the new wrapper is separate.
    protected={p:base.sha(p) for p in protected}
    base.dump(OUT/'protected_hashes.json',protected)
    source=json.loads((base.SOURCE/'results.json').read_text())
    inputs=np.load(base.ROOT/'geometry_preserving_quota_realizer/input_reservoirs.npz')
    initial=np.load(base.SOURCE/'interior_pools.npz')
    arrays=np.load(base.OUT/'state_arrays.npz')
    historical=list(__import__('csv').DictReader((base.ROOT/'progressive_spacefill_smoke/progressive_spacefill_geometry.csv').open()))
    sb=[r for r in historical if int(float(r['N']))==base.N and r['generator']=='Sobol']
    pools={};rows=[];settings={};chosen_ids={}
    for name in initial.files:
        cohort='fresh' if int(name.split('_')[0].removeprefix('iteration'))>=10000 else 'archived'
        entry=next(r['entry'] for r in source['cohort_health'] if r['cohort']==cohort)
        cover=1.05*entry['cover']; holes=(entry['hole_fraction']+.0005)*len(reference_points())
        if cohort=='archived':
            it=int(name.split('_')[0].removeprefix('iteration'));kind=name.rsplit('_',1)[-1]
            paired=next(r for r in sb if int(float(r['iteration']))==it and r['pool_type']==kind)
            cover=min(cover,1.1*max(float(r['complete_cover']) for r in sb),1.15*float(paired['complete_cover']))
            holes=min(holes,max(float(r['large_hole_count']) for r in sb)+.001*len(reference_points()))
        limits=dict(cover=cover,holes=holes,nn_p01=.95*entry['nn_p01']);settings[name]=limits
        points=inputs[name];classes=arrays[name+'_classes'];selected=arrays[name+'_initial_selected']
        assert cKDTree(points).query(points,k=2)[0][:,1].min()>=.4/np.sqrt(base.N)
        chosen,stats=realize(points,classes,selected,limits,OUT/(name+'.jsonl.gz'))
        pools[name]=points[chosen];chosen_ids[name]=chosen
        assert np.array_equal(base.capacity(pools[name]),arrays[name+'_quotas'])
        assert len(np.unique(pools[name],axis=0))==base.N and np.isfinite(pools[name]).all()
        rows.append(dict(condition=name,**stats))
        np.savez_compressed(OUT/'realized_pools.npz',**pools)
        base.dump(OUT/'progress.json',rows)
    first=initial.files[0]
    replay,rs=realize(inputs[first],arrays[first+'_classes'],arrays[first+'_initial_selected'],settings[first],
                      OUT/'determinism_replay.jsonl.gz',expected=OUT/(first+'.jsonl.gz'))
    assert np.array_equal(replay,chosen_ids[first])
    np.savez_compressed(OUT/'selected_candidate_ids.npz',**chosen_ids)
    base.OUT=OUT
    result=base.evaluate(pools,initial,source);fresh=base.fresh_baseline_check()
    passed=bool(result['passed'] and fresh['passed'])
    assert all(base.sha(p)==h for p,h in protected.items())
    result.update(passed=passed,status='VALIDATED_DETERMINISTIC_TEACHER' if passed else 'STOPPED_HISTORICAL_GATE_FAILURE',
                  rows=rows,fresh_discrepancy=fresh,limits=settings,deterministic_replay_events=rs['events'],
                  quota_errors=0,budget_errors=0,protected_files_unchanged=len(protected),runtime_seconds=time.monotonic()-start)
    base.dump(OUT/'results.json',result)
    base.dump(OUT/'manifest.json',dict(geometry_guarded_shift_aware_realizer=result['status'],promoted=passed,trained=False,
              implementation_sha256=base.sha(__file__),results_sha256=base.sha(OUT/'results.json'),pool_sha256=base.sha(OUT/'realized_pools.npz')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('geometry','rows','fresh_discrepancy','scale_comparison')},indent=2),flush=True)


def finish_report():
    """Independent final-state geometry check and human-readable audit."""
    result=json.loads((OUT/'results.json').read_text())
    pools=np.load(OUT/'realized_pools.npz')
    inputs=np.load(base.ROOT/'geometry_preserving_quota_realizer/input_reservoirs.npz')
    chosen=np.load(OUT/'selected_candidate_ids.npz');ref=reference_points()
    checks=[]
    for name in pools.files:
        g=GeometryGuard(inputs[name],chosen[name],**result['limits'][name])
        support=base.boundary_faces(base.N,pools[name].shape[1])
        dd=cKDTree(np.concatenate((pools[name],support))).query(ref)[0]
        nn=cKDTree(pools[name]).query(pools[name],k=2)[0][:,1]
        assert np.array_equal(dd<=g.cover,g.count>0)
        assert np.array_equal(dd<=1/np.sqrt(base.N),g.hcount>0)
        assert np.array_equal(nn,g.near[chosen[name]])
        checks.append(dict(condition=name,coverage_membership_errors=0,hole_membership_errors=0,nearest_distance_errors=0))
    base.write_csv(OUT/'independent_guard_validation.csv',checks)
    totals={}
    for row in result['rows']:
        for reason,n in row['counts'].items(): totals[reason]=totals.get(reason,0)+n
    result['acceptance_counts']=totals;result['independent_guard_validation']=checks
    base.dump(OUT/'results.json',result)
    sources=[base.ROOT/'geometry_preserving_quota_realizer/input_reservoirs.npz',
             base.ROOT/'shift_aware_quota_realizer/state_arrays.npz',
             base.SOURCE/'interior_pools.npz',base.SOURCE/'quota_assignment_trace.json']
    base.dump(OUT/'trace_sources.json',{str(p):base.sha(p) for p in sources})
    lines=['# Geometry-guarded shift-aware CDF exchange teacher','',result['status'],'',
      'The previous CDF state, exact integer effect scores, three-sweep schedule, fixed slot order, and score/ID tie convention are unchanged. Every negative-effect alternative is considered in ascending effect then reservoir ID order until the first admissible exchange is found. If none passes, retain the incumbent. No weighted objective, new reservoir, quota solve, training, or downstream N1 run.', '',
      '## Acceptance rule','',
      'Require unused same-incidence-class replacement. On the exact historical reference points, require zero uncovered references at the existing coverage radius and a large-hole count within the existing limit (distance > 1/sqrt(N)). Require NN CV >= .08 and NN p01 >= .95 times its entry baseline. The full reservoir separation floor .4/sqrt(N) proves every subset has zero points below the historical clustering cutoff .1/sqrt(N). Budget/quota preservation follows from exchanging exactly one point in the identical class. Fixed boundary coordinates are retained.', '',
      'Coverage limit is the minimum of applicable existing cohort-entry, global Sobol, and paired-Sobol limits. Hole limit is the minimum of applicable existing entry and Sobol limits. No limits were retuned. Cross-pool overlap, union coverage, boundary behavior, and all discrepancy gates are rechecked jointly after realization.', '',
      'Minimal incremental state: selected candidate mask; current reference covering multiplicities at the coverage and hole radii; total uncovered-hole count; selected nearest-neighbor distances; fixed per-candidate reference memberships and reservoir neighbor lists; immutable class IDs and quotas. Only references losing their last cover or gaining their first cover change admissibility. A 64-neighbor cache is exact whenever a selected neighbor exists in that prefix, asserted at every decision. Every final cached distance and coverage membership is independently checked against direct SciPy queries.', '',
      'These geometry reference points now enter acceptance as explicitly requested; the historical gate is unchanged but no longer an independent unseen-reference test. Actual validation comprises five archived and three saved fresh 2D pools of 8,600 points. The rule has no dimension-specific routing; higher-dimensional pool validation is not claimed.', '',
      '## Results','',f'Proposal outcomes: {json.dumps(totals)}.',
      f"Archived discrepancy ratios: {json.dumps(result['archived_one_count_ratio'])}.",
      f"Fresh discrepancy ratios: {json.dumps(result['fresh_discrepancy']['worst_one_count_ratios'])}.",
      f"Historical gates: {json.dumps(result['historical_gate'])}.",
      f"Fresh discrepancy gates: {json.dumps(result['fresh_discrepancy']['gates'])}.",'',
      '| Cohort | Coverage | Coverage limit | Hole fraction | Hole limit | Health |','|---|---:|---:|---:|---:|---|']
    for row in result['cohort_health']:
        v,e=row['value'],row['entry']
        lines.append(f"| {row['cohort']} | {v['cover']:.10g} | {1.05*e['cover']:.10g} | {v['hole_fraction']:.10g} | {e['hole_fraction']+.0005:.10g} | {row['passed']} |")
    lines += ['', 'Per-scale maximum/RMS discrepancy and occupancy/Fano are in `scale_comparison.csv`; individual pools in `geometry.csv`; exact projection/dyadic counts in `counts.csv`.',
      f"Zero budget, quota, duplicate, nonfinite, and protected-file regressions. {result['deterministic_replay_events']:,} first-pool decision events exactly rerun, including every rejected alternative and geometry state. {result['protected_files_unchanged']:,} protected files unchanged.", '',
      'Trace JSONL records the initial selection and thresholds; each slot records all proposed exchanges in canonical order, exact CDF improvement, geometry before/after hole counts, coverage violations, applicable NN statistics, acceptance/rejection reason, and final selection. Full CDF and geometry states reconstruct from the immutable reservoir plus preceding accepted swaps. Rejected states do not modify counts or CDF.', '',
      f"Runtime: {result['runtime_seconds']:.2f} seconds (including exact first-pool replay and historical validation). CDF proposal complexity is unchanged. Geometry preprocessing stores sparse candidate/reference incidence; exchange checks cost the affected reference memberships plus affected nearest-neighbor rows, and O(N) nearest-neighbor summaries. No combinatorial search or alternate ordering is added.", '',
      ('All historical gates pass. Ready as a validated deterministic teacher for this tested scope; not a learned operator or full N1 validation.' if result['passed'] else 'Stopped at the failed historical gate. The tested ordering/sweep policy has not established a jointly feasible discrepancy/geometry endpoint; this is not a proof of reservoir infeasibility. No second policy or gate adjustment was attempted.')]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest.update(implementation_sha256=base.sha(__file__),results_sha256=base.sha(OUT/'results.json'),report_sha256=base.sha(OUT/'report.md'))
    base.dump(OUT/'manifest.json',manifest)


if __name__=='__main__':
    main()
    finish_report()
