"""Fixed-quota adaptation of existing deterministic farthest-point selection.

No quota solve, training, new candidates, or gate changes. Canonical ties retain
input candidate order. Boundary support is mandatory initial geometric support;
only interior selections consume quotas or participate in the separation guard.
"""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import distance,cKDTree
from experiments.sho.audit_online_quota_reservoir import (OUT as PREVIOUS,ROOT,N,A,META,ids,capacity,guarded_buffer,geometric)
from experiments.sho.progressive_spacefill_generator import boundary_faces
from experiments.sho.train_dyadic_spacefill_curriculum import health
from experiments.sho.evaluate_progressive_spacefill_smoke import overlap
from experiments.sho.validate_joint_quota_assignment_teacher import metrics as count_metrics
OUT=ROOT/'geometry_preserving_quota_realizer'

def initialize_distance(candidates,boundary):
    nearest=np.full(len(candidates),np.inf,dtype=np.float64)
    for start in range(0,len(candidates),2048):
        nearest[start:start+2048]=distance.cdist(candidates[start:start+2048],boundary).min(axis=1)
    return nearest

def realize(candidates,class_ids,quotas,boundary,trace_path):
    remaining=quotas.copy();chosen=np.zeros(len(candidates),bool);nearest=initialize_distance(candidates,boundary)
    selected=[];budget=int(quotas.sum())
    assert np.all(np.bincount(class_ids,minlength=len(quotas))>=quotas)
    with gzip.open(trace_path,'wt') as out:
        out.write(json.dumps({'type':'initial','input_reservoir_ref':Path(trace_path).name.replace('.jsonl.gz',''),'quotas':quotas.tolist(),'candidate_class_ids':class_ids.tolist(),'boundary':boundary.tolist(),'rule':'largest nearest Euclidean distance among unselected candidates whose class has remaining quota; first input index wins equal scores','initial_budget':budget})+'\n')
        for step in range(budget):
            eligible=(~chosen)&(remaining[class_ids]>0)
            scores=np.where(eligible,nearest,-np.inf);j=int(np.argmax(scores));cl=int(class_ids[j])
            assert eligible[j] and remaining[cl]>0
            out.write(json.dumps({'input_state_ref':step,'selected_id':j,'class_id':cl,'nearest_distance':float(nearest[j]),'class_quota_before':int(remaining[cl]),'class_quota_after':int(remaining[cl]-1),'remaining_budget':budget-step-1,'tie_break':'first input candidate index'})+'\n')
            chosen[j]=True;selected.append(j);remaining[cl]-=1
            update=np.linalg.norm(candidates-candidates[j][None,:],axis=1)
            nearest=np.minimum(nearest,update)
        assert np.all(remaining==0)
    return np.asarray(selected,dtype=np.int64)

def replay(candidates,quota,boundary,path):
    class_ids=ids(candidates);remaining=quota.copy();chosen=np.zeros(len(candidates),bool);near=initialize_distance(candidates,boundary);steps=0
    with gzip.open(path,'rt') as f:
        first=json.loads(next(f));assert first['quotas']==quota.tolist()
        for line in f:
            row=json.loads(line);eligible=(~chosen)&(remaining[class_ids]>0)
            expected=int(np.argmax(np.where(eligible,near,-np.inf)));assert expected==row['selected_id']
            cl=int(class_ids[expected]);assert row['input_state_ref']==steps and row['class_id']==cl
            assert row['nearest_distance']==near[expected] and row['class_quota_before']==remaining[cl]
            chosen[expected]=True;remaining[cl]-=1;steps+=1
            near=np.minimum(near,np.linalg.norm(candidates-candidates[expected],axis=1))
            assert row['class_quota_after']==remaining[cl] and row['remaining_budget']==int(remaining.sum())
        assert np.all(remaining==0) and steps==int(quota.sum())
    return steps

def cohort(pools,names,pairs,reference):
    g=[geometric(pools[n],reference) for n in names];ov=[overlap(pools[names[i]],pools[names[j]],N) for i,j in pairs]
    return dict(cover=max(v['cover'] for v in g),hole_fraction=max(v['hole_fraction'] for v in g),min_nn_CV=min(v['min_nn_CV'] for v in g),nn_p01=min(v['nn_p01'] for v in g),clustering=max(v['clustering'] for v in g),exact_overlap=max(v['exact_overlap_fraction'] for v in ov),near_overlap=max(v['near_overlap_fraction_005_spacing'] for v in ov),boundary=all(v['boundary_preserved'] for v in g),finite=all(v['finite'] for v in g)),ov

def run():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    protect=list(ROOT.rglob('*.pt'))+[Path('experiments/sho/validate_joint_quota_assignment_teacher.py'),Path('experiments/sho/audit_online_quota_reservoir.py')]
    for folder in [ROOT/'joint_quota_assignment_teacher',PREVIOUS]:protect.extend(p for p in folder.iterdir() if p.is_file())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protect}
    old=json.load(open(PREVIOUS/'results.json'));saved=np.load(PREVIOUS/'captured_pools.npz');boundary=boundary_faces(N,2)
    t=(np.arange(512)+.5)/512;reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    rows=[];pools={};inputs={};selections={};traces=[]
    for state in old['screens']:
        name=state['condition'];begin=time.monotonic();base=saved[name+'_base'];raw=saved[name+'_raw'];previous=saved[name+'_screen']
        candidates=guarded_buffer(base,raw);cl=ids(candidates);quota=capacity(previous)
        expected=next(r for r in old['rows'] if r['source']=='guarded_single_context_buffer' and r['condition']==name)
        assert len(candidates)==expected['reservoir_size'] and capacity(candidates).tolist()==expected['capacity']
        # The full existing guarded candidate set is mutually separation-compatible.
        nearest=cKDTree(candidates).query(candidates,k=2)[0][:,1]
        assert np.all(nearest>=.4/np.sqrt(N))
        path=OUT/(name+'.jsonl.gz');selected=realize(candidates,cl,quota,boundary,path);p=candidates[selected]
        assert len(p)==N and len(np.unique(p,axis=0))==N
        assert np.array_equal(capacity(p),quota) and np.array_equal(A@capacity(p),A@capacity(previous))
        elapsed=time.monotonic()-begin
        row=dict(condition=name,archived=state['archived'],candidate_count=len(candidates),quotas=quota.tolist(),selection_seconds=elapsed,quota_errors=0,budget_error=0,coarse_count_errors=0,geometry=geometric(p,reference),count_metrics=count_metrics(A@capacity(p),META,N),previous_geometry=state['screen_geometry'])
        pools[name]=p;inputs[name]=candidates;selections[name]=selected;traces.append(str(path));rows.append(row)
        print(name,'cover',row['geometry']['cover'],'seconds',round(elapsed,3),flush=True)
        (OUT/'progress.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(OUT/'input_reservoirs.npz',**inputs);np.savez_compressed(OUT/'selected_candidate_ids.npz',**selections);np.savez_compressed(OUT/'realized_pools.npz',**pools)
    groups=[]
    for group,names,pairs in [('archived',[f'iteration{100+j}_cell' for j in range(4)]+['iteration100_greedy'],[(0,1),(1,2),(2,3),(0,4)]),('fresh',['iteration12000_cell','iteration12001_cell','iteration12000_greedy'],[(0,1),(0,2)])]:
        entry=next(v['entry'] for v in old['historical_cohort_gates'] if v['group']==group)
        value,ov=cohort(pools,names,pairs,reference)
        checks=dict(cover=value['cover']<=1.05*entry['cover'],holes=value['hole_fraction']<=entry['hole_fraction']+.0005,nn_CV=value['min_nn_CV']>=.08,nn_p01=value['nn_p01']>=.95*entry['nn_p01'],clustering=value['clustering']<=.005,exact_freshness=value['exact_overlap']==0,near_freshness=value['near_overlap']<=min(.03,entry['near_overlap']+.005),boundary=value['boundary'],finite=value['finite'])
        assert health(value,entry)==all(v for k,v in checks.items() if k not in ('boundary','finite'))
        groups.append(dict(group=group,entry=entry,result=value,checks=checks,freshness=ov,passed=all(checks.values())))
    passed=all(g['passed'] for g in groups)
    trace_events=0
    # Independent replay verifies every action and its exact score, not just final counts.
    for name in inputs:trace_events+=replay(inputs[name],capacity(saved[name+'_screen']),boundary,OUT/(name+'.jsonl.gz'))
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());assert unchanged
    report=dict(status='VALIDATED_DETERMINISTIC_TEACHER' if passed else 'STOPPED_GEOMETRY_GATE_FAILURE',rule='quota-masked farthest-point selection seeded with unchanged boundary support',existing_rule_source='experiments/sho/run_stage1b_enhanced_edirect_boundary_rescue.py:214-244',adaptation='fixed quota eligibility mask and mandatory boundary initialization; no objective/loss labels',scope='eight existing 2D pools, original full guarded buffers and exact previously chosen class quotas',rows=rows,groups=groups,replayed_selection_events=trace_events,trace_files=traces,runtime_seconds=time.monotonic()-start,protected_files=len(hashes),protected_unchanged=unchanged,trained=False)
    (OUT/'results.json').write_text(json.dumps(report,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(hashes,indent=2))
    manifest=dict(geometry_preserving_quota_realizer=report['status'],trained=False,implementation=str(Path(__file__)),implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),candidate_source='unchanged full guarded buffers from previous audit, reconstructed and capacity-checked; no new proposals',quota_source='previous captured_pools.npz *_screen incidence counts, never re-solved')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['rows','groups','trace_files']},indent=2),flush=True)
    for g in groups:print(g['group'],g['checks'],flush=True)

if __name__=='__main__':run()
