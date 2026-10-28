"""Enumerate saved trusted brackets; frozen arithmetic, no root iteration/training."""
import ast
import csv
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_mantissa_log import MonotoneMantissaLog
from experiments.geometry.train_exponent_log2_operator import ExponentLog2

OUT=ref.ROOT/'stationary_float64_minimizer_full_audit'
SOURCE=ref.ROOT/'defer_aware_stationary_locator_audit'


def bits(x): return int(np.float64(x).view(np.uint64))


def run():
    start=time.monotonic();torch.set_num_threads(1);mp.mp.dps=80
    assert not (OUT/'results.json').exists(), 'Do not overwrite a completed audit.'
    OUT.mkdir(parents=True,exist_ok=True)
    protected={str(p):ref.sha(p) for p in ref.ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ref.ROOT.rglob('*manifest.json')})
    locator=SOURCE/'defer_aware_stationary_locator_validated.json'
    assert json.loads(locator.read_text())['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE'
    assert ref.sha(locator)==json.loads((SOURCE/'results.json').read_text())['assembly_sha256']
    bracket_path=SOURCE/'defer_aware_root_recovery.csv'
    for p in (locator,bracket_path):protected[str(p)]=ref.sha(p)
    brackets=list(csv.DictReader(bracket_path.open()));assert len(brackets)==7093
    base,refs=ref.load_terms();models=base.components
    for name,folder,cls in [('sqrt','sqrt_operator_failure_refine',ref.ScalarPower),
                           ('mantissa','mantissa_log_operator_support_refine',MonotoneMantissaLog),
                           ('exponent','exponent_log2_operator_smoke',ExponentLog2)]:
        path=ref.ROOT/folder/'candidate_manifest.json';manifest=json.loads(path.read_text());cp=Path(manifest['checkpoint'])
        assert ref.sha(cp)==manifest['sha256']
        model=cls();model.load_state_dict(torch.load(cp,weights_only=False)['state_dict'])
        models[name]=model.eval().requires_grad_(False)
        refs[name]=dict(manifest=str(path),checkpoint=str(cp),sha256=ref.sha(cp))
    source=ref.Inputs();source.collect();pieces={};audit=ref.Audit.__new__(ref.Audit)
    for b in source.batches:
        if b['scope']!='archived_subpiece_grids':continue
        key=ast.literal_eval(b['case']);r=source.piece_map[key];parent,axis,k,_=key
        audit.setup(source.parents[parent],axis)
        power=int(r['aspect_log_power'])
        if audit.d==1:C=0.
        elif power==1:C=.05*np.log(audit.a/audit.omin)
        elif power==-1:C=.05*np.log(audit.omax/audit.a)
        else:C=.05*np.log(audit.omax/audit.omin)
        p=dict(scope='geometric',case=b['case'],key=key,lo=float(r['lower']),hi=float(r['upper']),C=float(C),
               coeff=(float(b['A'][0]),b['B'],b['a'],float(b['beta'][0]),float(b['o'][0]),float(b['ov'][0])),
               roots=json.loads(r['stationary_points'])+json.loads(r['quartic_stationary_roots']))
        A,B,a,beta,t,v=p['coeff'];p['record']=dict(A=A,B=B,a=a,beta=beta,t=t,v=v)
        pieces[(p['scope'],p['case'])]=p
    for key,r in source.generic.items():
        p=dict(scope='generic_coefficients',case=key,lo=r['lower'],hi=r['upper'],C=float(r.get('C',0)),
               coeff=(r['A'],r['B'],r['a'],r['beta'],r['t'],r['v']),record=r,roots=r['roots'])
        pieces[(p['scope'],p['case'])]=p
    del source.batches
    arrays=[];truth=[];original_geometry=[];offsets=[0];chosen=[]
    for b in brackets:
        p=pieces[(b['scope'],b['case'])];chosen.append(p)
        L,R=float(b['final_L']),float(b['final_R']);assert 0<L<R<1
        codes=np.arange(bits(L),bits(R)+1,dtype=np.uint64);c=codes.view(np.float64)
        assert len(c)==int(b['final_width_in_float_steps'])+1 and c[0]==L and c[-1]==R
        assert np.array_equal(np.nextafter(c[:-1],np.inf),c[1:])
        A,B,a,beta,t,v=p['coeff']
        arrays.append(np.column_stack([np.full(len(c),z) for z in (A,B,a,beta)]+[c]+[np.full(len(c),z) for z in (t,v,p['C'])]))
        # Preserve the existing generic subpiece teacher's evaluation order.
        y=ref.generic_score(p['record'],c)+p['C'];truth.append(y)
        if p['scope']=='geometric':
            parent,axis,k,_=p['key'];audit.setup(source.parents[parent],axis);original_geometry.append(audit.formula(c,k)[0])
        else:original_geometry.append(y)
        offsets.append(offsets[-1]+len(c))
    x=np.concatenate(arrays);teacher=np.concatenate(truth);geometry=np.concatenate(original_geometry);del arrays,truth,original_geometry
    predicted=np.empty(len(x))
    with torch.no_grad():
        for start_block in range(0,len(x),16384):
            z=torch.from_numpy(x[start_block:start_block+16384]);A,B,a,beta,c,ot,ov,C=z.unbind(1)
            def mul(a,b):return models['multiply'](torch.stack((a,b),1))
            def add(a,b):return models['addition'](torch.stack((a,b),1))
            t=models['orientation'](torch.stack((c,ot),1));v=models['orientation'](torch.stack((c,ov),1))
            q=models['quadratic'](torch.stack((B,a,t),1));first=mul(A,models['sqrt'](q))
            # Exact deterministic decomposition retained from the assembled-log audit.
            m,e=np.frexp(v.numpy());m=2*m;e=e-1
            assert np.array_equal(np.ldexp(m,e),v.numpy())
            logv=add(models['mantissa'](torch.from_numpy(m)),models['exponent'](torch.from_numpy(e.astype(np.float64))))
            predicted[start_block:start_block+len(z)]=add(add(first,mul(beta,logv)),C).numpy()
    assert np.isfinite(teacher).all() and np.isfinite(predicted).all()
    offsets=np.array(offsets);np.savez_compressed(OUT/'all_float64_candidates.npz',inputs=x,teacher_score=teacher,
        learned_score=predicted,original_geometry_score=geometry,offsets=offsets)
    print(json.dumps(dict(enumerated=len(x),brackets=len(brackets),elapsed=time.monotonic()-start)),flush=True)
    rows=[];solved={};subpieces=[]
    for i,(b,p) in enumerate(zip(brackets,chosen)):
        s,e=offsets[i:i+2];c=x[s:e,4];yt=teacher[s:e];yl=predicted[s:e]
        # Strict discrete argmin, first ascending cut for exact equal scores.
        it=int(np.argmin(yt));il=int(np.argmin(yl));ct=float(c[it]);cl=float(c[il]);tol=1e-12*max(1.,float(yt[it]))
        root=mp.mpf(b['c_teacher_decimal']);rn=float(root)
        below=rn if mp.mpf(rn)<=root else np.nextafter(rn,-np.inf)
        above=rn if mp.mpf(rn)>=root else np.nextafter(rn,np.inf)
        relation='exact_representable_root' if mp.mpf(ct)==root else 'nearest_below' if ct==below else 'nearest_above' if ct==above else 'other_nearby_float'
        row=dict(index=i,scope=p['scope'],case=p['case'],count=int(e-s),L=c[0],R=c[-1],
            c_teacher_min=ct,c_learned_min=cl,teacher_min_hex=ct.hex(),learned_min_hex=cl.hex(),
            exact_match=ct==cl,ulp_distance=abs(bits(ct)-bits(cl)),coordinate_difference=abs(ct-cl),
            teacher_at_teacher_min=float(yt[it]),teacher_at_learned_min=float(yt[il]),
            learned_at_teacher_min=float(yl[it]),learned_at_learned_min=float(yl[il]),
            teacher_score_penalty=float(yt[il]-yt[it]),learned_score_penalty=float(yl[it]-yl[il]),
            teacher_equivalent=bool(abs(yt[il]-yt[it])<=tol),teacher_exact_minimizers=int(np.sum(yt==yt[it])),
            learned_exact_minimizers=int(np.sum(yl==yl[il])),teacher_equivalent_minimizers=int(np.sum(yt<=yt[it]+tol)),
            root_decimal=b['c_teacher_decimal'],root_relation=relation,root_nearest_below=float(below),root_nearest_above=float(above),
            root_neighborhood_ulp_distance=min(abs(bits(ct)-bits(below)),abs(bits(ct)-bits(above))),
            teacher_min_root_distance=float(abs(mp.mpf(ct)-root)))
        rows.append(row)
        cuts=np.array([p['lo'],ct,cl,p['hi'],float(root)])
        if p['scope']=='geometric':
            parent,axis,k,_=p['key'];audit.setup(source.parents[parent],axis);scores=audit.formula(cuts,k)[0]
        else:axis=0;scores=ref.generic_score(p['record'],cuts)+p['C']
        def candidates(cut,value):return [dict(axis=axis,c=float(c),score=float(v),identity=p['case']+':'+kind,kind=kind)
            for c,v,kind in [(p['lo'],scores[0],'left'),(cut,value,'stationary'),(p['hi'],scores[3],'right')]]
        tc,lc,rc=candidates(ct,scores[1]),candidates(cl,scores[2]),candidates(float(root),scores[4])
        cmp=ref.compare_pool('subpiece',p['case'],dict(teacher=tc,learned=lc));cmp['source_scope']=p['scope']
        rootcmp=ref.compare_pool('subpiece_root_baseline',p['case'],dict(teacher=rc,learned=lc))
        cmp['identity_changed_vs_root']=rootcmp['identity_changed'];cmp['outside_equivalence_vs_root']=rootcmp['outside_equivalence']
        subpieces.append(cmp);solved[(p['scope'],p['case'])]=(tc,lc,rc)
    pools={};parents={};comparisons=[];root_comparisons=[]
    for p in pieces.values():
        if p['scope']!='geometric':continue
        parent,axis,k,_=p['key'];pair=solved.get((p['scope'],p['case']))
        if pair is None:
            audit.setup(source.parents[parent],axis);cuts=sorted(set([p['lo'],p['hi']]+[c for c in p['roots'] if p['lo']<=c<=p['hi']]))
            scores=audit.formula(np.array(cuts),k)[0]
            cs=[dict(axis=axis,c=c,score=float(v),identity=p['case']+f':saved{j}') for j,(c,v) in enumerate(zip(cuts,scores))];pair=cs,cs,cs
        for arm,cs in zip(('teacher','learned','root'),pair):pools.setdefault((parent,axis,k),{'teacher':[],'learned':[],'root':[]})[arm].extend(cs)
    for key,pool in pools.items():
        comparisons.append(ref.compare_pool('allocation_interval',str(key),pool))
        root_comparisons.append(ref.compare_pool('allocation_interval',str(key),dict(teacher=pool['root'],learned=pool['learned'])))
        for arm in pool:parents.setdefault(key[0],{'teacher':[],'learned':[],'root':[]})[arm].extend(pool[arm])
    for parent,pool in parents.items():
        comparisons.append(ref.compare_pool('archived_parent',str(parent),pool))
        root_comparisons.append(ref.compare_pool('archived_parent',str(parent),dict(teacher=pool['root'],learned=pool['learned'])))
    assert len(parents)==106
    def aggregate(cs):return dict(comparisons=len(cs),identity_changes=sum(r['identity_changed'] for r in cs),
        raw_coordinate_changes=sum(r['raw_cut_changed'] for r in cs),outside_equivalence=sum(r['outside_equivalence'] for r in cs),
        maximum_score_difference=max(r['score_difference'] for r in cs))
    intervals=[r for r in comparisons if r['scope']=='allocation_interval'];par=[r for r in comparisons if r['scope']=='archived_parent']
    root_stats={scope:aggregate([r for r in root_comparisons if r['scope']==scope]) for scope in ('allocation_interval','archived_parent')}
    result=dict(operator='stationary_float64_minimizer',status='DIAGNOSTIC_ONLY_STOP',brackets=len(rows),total_float64_candidates=len(x),
        exact_minimizer_matches=sum(r['exact_match'] for r in rows),exact_match_fraction=float(np.mean([r['exact_match'] for r in rows])),
        one_ulp_disagreements=sum(r['ulp_distance']==1 for r in rows),larger_coordinate_disagreements=sum(r['ulp_distance']>1 for r in rows),
        maximum_ulp_distance=max(r['ulp_distance'] for r in rows),maximum_coordinate_difference=max(r['coordinate_difference'] for r in rows),
        maximum_teacher_score_penalty=max(r['teacher_score_penalty'] for r in rows),maximum_learned_score_penalty=max(r['learned_score_penalty'] for r in rows),
        score_differences_outside_equivalence=sum(not r['teacher_equivalent'] for r in rows),
        teacher_tied_minimum_brackets=sum(r['teacher_exact_minimizers']>1 for r in rows),
        learned_tied_minimum_brackets=sum(r['learned_exact_minimizers']>1 for r in rows),
        total_teacher_exact_minimizers=sum(r['teacher_exact_minimizers'] for r in rows),
        total_teacher_equivalent_minimizers=sum(r['teacher_equivalent_minimizers'] for r in rows),
        root_relation={v:sum(r['root_relation']==v for r in rows) for v in sorted(set(r['root_relation'] for r in rows))},
        root_neighborhood_ulp_distance=dict(median=float(np.median([r['root_neighborhood_ulp_distance'] for r in rows])),
            p99=float(np.quantile([r['root_neighborhood_ulp_distance'] for r in rows],.99)),max=max(r['root_neighborhood_ulp_distance'] for r in rows)),
        subpiece_minima=aggregate(subpieces),allocation_interval_minima=aggregate(intervals),archived_parent_minima=aggregate(par),
        versus_previous_high_precision_root_candidates=root_stats,
        max_subpiece_vs_original_geometry_arithmetic_difference=float(np.max(abs(teacher-geometry))),
        selection_semantics='Exact minimum score; equal float64 scores retain first ascending cut. Downstream: original 1e-12*max(1,best) tolerance, then axis/cut order.',
        teacher_evaluation='Archived generic_score: A*sqrt(B+(a**2*t)*t)+beta*log(v), then +C. Geometric C reconstructed from archived active aspect branch.',
        downstream_evaluation='Original geometric Audit.formula on both selected coordinates, unchanged saved candidates elsewhere; isolates replacement of the stationary candidate.',
        log_assembly='Frozen mantissa log + frozen exponent-log2 + frozen addition, exact deterministic frexp decomposition; no log approximation repair.',
        components=refs,source_locator=str(locator),source_locator_sha256=ref.sha(locator),
        source_brackets_sha256=ref.sha(bracket_path),implementation_sha256=ref.sha(Path(__file__)),no_training=True,no_root_iteration=True,
        elapsed_seconds=time.monotonic()-start)
    passed=(result['score_differences_outside_equivalence']==0 and result['subpiece_minima']['outside_equivalence']==0
        and result['allocation_interval_minima']['identity_changes']==0 and result['allocation_interval_minima']['outside_equivalence']==0
        and result['archived_parent_minima']['raw_coordinate_changes']==0 and result['archived_parent_minima']['outside_equivalence']==0
        and root_stats['allocation_interval']['identity_changes']==0 and root_stats['archived_parent']['raw_coordinate_changes']==0)
    result['passed']=passed
    assert all(ref.sha(Path(p))==h for p,h in {**protected,**source.inputs}.items());result['all_protected_hashes_unchanged']=True
    ref.write_csv(OUT/'stationary_float64_minimizers.csv',rows)
    ref.write_csv(OUT/'subpiece_minimum_comparison.csv',subpieces)
    ref.write_csv(OUT/'interval_parent_comparison.csv',comparisons)
    ref.write_csv(OUT/'previous_root_candidate_comparison.csv',root_comparisons)
    disagreements=[r for r in rows if not r['exact_match']]
    ref.write_csv(OUT/'worst_20_disagreements.csv',sorted(disagreements,key=lambda r:(r['teacher_score_penalty'],r['ulp_distance']),reverse=True)[:20])
    if passed:
        result['status']='VALIDATED_UNDER_TEACHER_EQUIVALENCE'
        path=OUT/'stationary_float64_minimizer_validated.json';path.write_text(json.dumps(result,indent=2));digest=ref.sha(path)
        path.with_suffix('.sha256').write_text(digest+'  '+path.name+'\n');result['assembly_manifest']=str(path);result['assembly_sha256']=digest
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'stationary_float64_minimizer_summary.txt').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
