"""Frozen selector replay with unavailable locator candidates explicitly absent."""
import ast
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from experiments.geometry.validate_branch_switch_locator import OUT
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read
from experiments.geometry.verify_learned_continuous_recursion import Selector
from experiments.geometry.train_subpiece_minimum_selector import canonical,teacher_select
from experiments.sho.train_finite_split_score_smoke import upstream,predict


def run():
    torch.set_num_threads(1)
    records=json.loads((OUT/'records.json').read_text());raw=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_all.csv')
    saved=json.loads((ROOT/'branch_switch_location_residual_training/location_cases.json').read_text())
    fresh=json.loads((ROOT/'branch_switch_ulp_offset_training/fresh_candidate_states.json').read_text())
    rows=saved+fresh
    mapped=[records[int(raw[i]['unique_id'])] for i in range(550)]+records[774:]
    assert len(rows)==len(mapped)==594
    sel=Selector()
    def reduce(pool,teacher=False):
        by=defaultdict(list)
        for e in pool:by[(e.get('group','fixture'),e.get('parent',0),e['piece'])].append(e)
        slots=[]
        for pp in by.values():
            named={e.get('slot',e['kind']):e for e in pp};slots.append([named['left'],named['right'],named.get('stationary')])
        scores=np.array([[e['teacher_score' if teacher else 'score'] if e else 0 for e in row] for row in slots])
        valid=np.array([[e is not None for e in row] for row in slots])
        ids=teacher_select(scores,valid) if teacher else sel.sub.select(sel.sub.normalize(scores,valid),valid)
        sp={};ip=defaultdict(list)
        for key,row,idx in zip(by,slots,ids):
            e=row[idx];assert e is not None;sp[key]=e
            ip[(e.get('group','fixture'),e.get('parent',0),e['axis'],e['k'])].append(e)
        order=lambda e:(e['axis'],e['cut'],e['identity'])
        keys=sorted(ip);ordered=[sorted(ip[k],key=order) for k in keys]
        if teacher:iw=[canonical(p) for p in ordered]
        else:
            ii,_,_=sel.interval.reduce([np.array([e['score'] for e in p]) for p in ordered]);iw=[p[j] for p,j in zip(ordered,ii)]
        pp=defaultdict(list)
        for e in iw:pp[(e.get('group','fixture'),e.get('parent',0))].append(e)
        pk=sorted(pp);pl=[sorted(pp[k],key=order) for k in pk]
        if teacher:pw=[canonical(p) for p in pl]
        else:
            pi,_,_=sel.parent.reduce([np.array([e['score'] for e in p]) for p in pl]);pw=[p[j] for p,j in zip(pl,pi)]
        return dict(subpiece=sp,interval=dict(zip(keys,iw)),parent=dict(zip(pk,pw)))
    def compare(t,l,scope):
        out=[]
        for level in t:
            for key,tt in t[level].items():
                ll=l[level][key];tol=1e-12*max(1.,tt['teacher_score'])
                out.append(dict(scope=scope,level=level,key=str(key),teacher_identity=tt['identity'],learned_identity=ll['identity'],
                    identity_changed=tt['identity']!=ll['identity'],teacher_cut=tt['cut'],learned_cut=ll['cut'],coordinate_changed=tt['cut']!=ll['cut'],
                    outside_equivalence=ll['teacher_score']>tt['teacher_score']+tol,
                    score_difference=abs(ll['teacher_score']-tt['teacher_score'])))
        return out
    subresults=[]
    for i,(r,result) in enumerate(zip(rows,mapped)):
        base=dict(axis=r['axis'],k=r['k'],piece=r['case'])
        tc=[dict(base,identity=r['case']+':'+kind,kind=kind,cut=c) for kind,c in [('left',r['lower']),('right',r['upper']),('stationary',r['teacher_cut'])]]
        lc=[dict(e) for e in tc[:2]]
        if result['candidate_available']:lc.append(dict(tc[2],cut=result['candidate']))
        sel.score(r['parent'],tc);sel.score(r['parent'],lc)
        comparison=compare(reduce(tc,True),reduce(lc),'positive_subpiece')
        subresults.append(dict(case=r['case'],group=r['group'],candidate_available=result['candidate_available'],
                              **{k:v for k,v in comparison[0].items() if k!='scope'}))
    # Preserve every other cached learned candidate; replace or remove ONLY the
    # branch-switch slot under test. Missing is never replaced by midpoint/boundary.
    lookup={(('archive_geometry' if r['group']=='archived_geometry' else 'fresh_geometry'),tuple(map(int,r['case'].split(':')))):(r,m)
            for r,m in zip(rows,mapped) if r['group'] in ('archived_geometry','existing_fresh_geometry')}
    poolmap=defaultdict(list)
    for e in read(ROOT/'subpiece_minimum_selector_training/frozen_candidate_scores.csv'):
        if e['group'] not in ('archive_geometry','fresh_geometry'):continue
        r=dict(e);r.update(parent=int(e['parent']),axis=int(e['axis']),k=int(e['k']),cut=float(e['cut']),teacher_score=float(e['teacher_score']),score=float(e['score']))
        case=ast.literal_eval(r['case']) if r['case'].startswith('(') else tuple(map(int,r['case'].split(':')))
        r['piece']=str(case);r['slot']='left' if r['kind'] in ('left','saved0') else 'right' if r['kind'] in ('right','saved1') else 'stationary'
        poolmap[(r['group'],case)].append(r)
    allteacher=[];alllearned=[];missing_cached=0
    for key,pool in poolmap.items():
        tc=[dict(e) for e in pool];lc=[dict(e) for e in pool]
        if key in lookup:
            r,result=lookup[key];cur=next((j for j,e in enumerate(tc) if e['slot']=='stationary'),None)
            if cur is None:
                tc.append(dict(tc[0],identity=str(key[1])+':stationary',kind='stationary',slot='stationary',cut=r['teacher_cut']));cur=len(tc)-1
            tc[cur]['cut']=r['teacher_cut']
            lc=[e for e in lc if e['slot']!='stationary']
            if result['candidate_available']:lc.append(dict(tc[cur],cut=result['candidate']))
            else:missing_cached+=1
            updated=[tc[cur]]+([lc[-1]] if result['candidate_available'] else [])
            a=sel.audit;a.setup(r['parent'],r['axis'])
            for e in updated:
                e['teacher_score']=float(a.formula(e['cut'],e['k'])[0][0])
                side=a.s;ell=np.tile(side,(1,2,1));ell[0,0,e['axis']]*=e['cut'];ell[0,1,e['axis']]*=1-e['cut']
                e['score']=float(predict(a.finite,upstream(a.models,a.execution,side[None,:],np.array([a.n]),ell,np.array([[e['k'],a.n-e['k']]])))[0])
        allteacher.extend(tc);alllearned.extend(lc)
    changes=compare(reduce(allteacher,True),reduce(alllearned),'cached_archived_and_fresh')
    fixture=json.loads((ROOT/'learned_continuous_recursive_verification/first_failure.json').read_text())['selector_context']
    ft=fixture['teacher_candidates'];fl=[dict(e) for e in fixture['learned_candidates']]
    for r,result in zip(rows,mapped):
        if r['group']!='saved_failure_geometry' or not result['candidate_available']:continue
        piece=':'.join(r['case'].split(':')[1:]);fl.append(dict(axis=r['axis'],k=r['k'],piece=piece,identity=piece+':stationary0',kind='stationary',cut=result['candidate']))
    sel.score(fixture['parent'],ft);sel.score(fixture['parent'],fl)
    fixture_changes=compare(reduce(ft,True),reduce(fl),'d2_n8_fixture')
    write_csv(OUT/'positive_subpiece_downstream.csv',subresults)
    write_csv(OUT/'cached_selector_replay.csv',changes);write_csv(OUT/'d2_n8_fixture_replay.csv',fixture_changes)
    json_write(OUT/'d2_n8_candidate_sets.json',dict(teacher=ft,learned=fl))
    stats={level:dict(cases=sum(r['level']==level for r in changes),
                     identity_changes=sum(r['level']==level and r['identity_changed'] for r in changes),
                     raw_coordinate_changes=sum(r['level']==level and r['coordinate_changed'] for r in changes),
                     outside_equivalence=sum(r['level']==level and r['outside_equivalence'] for r in changes)) for level in ('subpiece','interval','parent')}
    protected=json.loads((OUT/'protected_hashes.json').read_text());assert all(sha(Path(p))==v for p,v in protected.items())
    result=json.loads((OUT/'assembly_result.json').read_text())
    result.update(status='DIAGNOSTIC_ONLY_STOP',promoted=False,
        downstream=dict(positive_subpieces=len(subresults),missing_positive_candidates=sum(not r['candidate_available'] for r in subresults),
            positive_subpiece_identity_changes=sum(r['identity_changed'] for r in subresults),
            positive_subpiece_score_failures=sum(r['outside_equivalence'] for r in subresults),
            cached=stats,missing_cached_branch_candidates=missing_cached,
            fixture_parent=[r for r in fixture_changes if r['level']=='parent'],
            scope='594 positive subpieces including all 44 fresh; 106 archived + 8 cached fresh full parent pools; original d2 N8 fixture. Missing terminal outputs remain absent; no fallback; no recursion.'),
        protected_hashes_unchanged=True)
    json_write(OUT/'final_result.json',result)
    print(json.dumps(result['downstream'],indent=2),flush=True)


if __name__=='__main__':run()
